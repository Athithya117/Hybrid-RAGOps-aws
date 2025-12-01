import os
import sys
import json
import time
import logging
import hashlib
import re
import unicodedata
import tempfile
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, Optional, Tuple, List
import boto3
import requests
from botocore.exceptions import ClientError
try:
    import trafilatura
except Exception:
    trafilatura = None
try:
    import tiktoken
except Exception:
    tiktoken = None
try:
    import spacy
    from spacy.pipeline import Sentencizer
except Exception:
    spacy = None
    Sentencizer = None
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("html_trafilatura")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
PARSER_VERSION = os.getenv("PARSER_VERSION_HTML", "trafilatura-only-v2")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "3"))
FETCH_BACKOFF = float(os.getenv("FETCH_BACKOFF", "0.5"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
NUMBER_OF_OVERLAPPING_SENTENCES = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", "2"))
s3 = boto3.client("s3")
ENCODER = None
if tiktoken is not None:
    try:
        ENCODER = tiktoken.get_encoding(ENC_NAME)
    except Exception:
        ENCODER = None
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def canonicalize_text(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s
def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False
def fetch_html_with_retries(url: str, timeout: int = REQUEST_TIMEOUT, retries: int = FETCH_RETRIES, backoff: float = FETCH_BACKOFF) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_err
def upload_snapshot_to_s3(snapshot_html: str, doc_id: str) -> Optional[str]:
    if not SAVE_SNAPSHOT or not S3_BUCKET:
        return None
    key = f"{S3_CHUNKED_PREFIX}{doc_id}.snapshot.html"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=snapshot_html.encode("utf-8"), ContentType="text/html")
    return f"s3://{S3_BUCKET}/{key}"
def trafilatura_extract_markdown(html_text: str):
    if trafilatura is None:
        return None, {}
    md = trafilatura.extract(html_text, output_format="markdown", with_metadata=True)
    parsed = {}
    try:
        json_doc = trafilatura.extract(html_text, output_format="json", with_metadata=True)
        if json_doc:
            parsed = json.loads(json_doc)
    except Exception:
        parsed = {}
    return md, parsed
def token_count_for(text: str) -> int:
    if not text:
        return 0
    if ENCODER is not None:
        try:
            return len(ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split())
def _make_sentencizer():
    if spacy is None:
        return None
    try:
        nlp = spacy.blank("en")
        try:
            nlp.add_pipe("sentencizer")
        except Exception:
            if Sentencizer is not None:
                nlp.add_pipe(Sentencizer())
            else:
                nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None
_NLP_SENTENCIZER = _make_sentencizer()
def _regex_sentences_with_offsets(text: str):
    spans = []
    pattern = re.compile(r'(.+?[\.\?\!]["\']?\s+)|(.+?$)', re.DOTALL)
    cursor = 0
    for m in pattern.finditer(text):
        s = (m.group(1) or m.group(2) or "").strip()
        if not s:
            continue
        start = text.find(s, cursor)
        if start == -1:
            start = cursor
        end = start + len(s)
        spans.append((s, start, end))
        cursor = end
    return spans
def _sentences_with_offsets(text: str):
    if _NLP_SENTENCIZER is not None:
        doc = _NLP_SENTENCIZER(text)
        return [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
    return _regex_sentences_with_offsets(text)
def _make_encoder_clients():
    if ENCODER is not None:
        return (lambda txt: ENCODER.encode(txt), lambda toks: ENCODER.decode(toks), "tiktoken")
    return (lambda txt: txt.split(), lambda toks: " ".join(toks), "whitespace")
_ENCODER_ENCODE, _ENCODER_DECODE, _ENCODER_BACKEND = _make_encoder_clients()
def split_into_token_windows(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, min_tokens: int = MIN_TOKENS_PER_CHUNK, overlap_sentences: int = NUMBER_OF_OVERLAPPING_SENTENCES) -> Iterator[Dict[str, Any]]:
    if not text:
        yield {"window_index": 0, "text": "", "token_count": 0, "token_start": 0, "token_end": 0}
        return
    text = canonicalize_text(text)
    sentences = _sentences_with_offsets(text)
    sent_items = []
    for s, sc, ec in sentences:
        toks = _ENCODER_ENCODE(s)
        tok_len = len(toks)
        sent_items.append({"text": s, "start_char": sc, "end_char": ec, "token_len": tok_len, "tokens": toks})
    total_sent = len(sent_items)
    if total_sent == 0:
        all_tokens = _ENCODER_ENCODE(text)
        yield {"window_index": 0, "text": text, "token_count": len(all_tokens), "token_start": 0, "token_end": len(all_tokens)}
        return
    token_cursor = 0
    for idx, si in enumerate(sent_items):
        if "token_start_idx" not in si:
            si["token_start_idx"] = token_cursor
            si["token_end_idx"] = token_cursor + si["token_len"]
            token_cursor = si["token_end_idx"]
    windows = []
    i = 0
    window_index = 0
    while i < len(sent_items):
        cur_token_count = 0
        chunk_sent_texts: List[str] = []
        chunk_token_start = sent_items[i]["token_start_idx"]
        chunk_token_end = chunk_token_start
        is_truncated_sentence = False
        start_i = i
        while i < len(sent_items):
            sent = sent_items[i]
            sent_tok_len = sent["token_len"]
            if cur_token_count + sent_tok_len > max_tokens:
                if not chunk_sent_texts:
                    if sent_tok_len > 0:
                        if _ENCODER_BACKEND == "tiktoken":
                            prefix_tok_ids = sent["tokens"][:max_tokens]
                            prefix_text = _ENCODER_DECODE(prefix_tok_ids)
                            chunk_sent_texts.append(prefix_text)
                            cur_token_count = len(prefix_tok_ids)
                            is_truncated_sentence = True
                            remainder_tok_ids = sent["tokens"][max_tokens:]
                            if remainder_tok_ids:
                                remainder_text = _ENCODER_DECODE(remainder_tok_ids)
                                sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "token_len": len(remainder_tok_ids), "tokens": remainder_tok_ids, "token_start_idx": None, "token_end_idx": None}
                            else:
                                i += 1
                            chunk_token_end = chunk_token_start + cur_token_count
                            break
                        else:
                            tokens = sent["tokens"]
                            prefix = tokens[:max_tokens]
                            prefix_text = " ".join(prefix)
                            chunk_sent_texts.append(prefix_text)
                            cur_token_count = len(prefix)
                            is_truncated_sentence = True
                            remainder = tokens[max_tokens:]
                            if remainder:
                                remainder_text = " ".join(remainder)
                                sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "token_len": len(remainder), "tokens": remainder, "token_start_idx": None, "token_end_idx": None}
                            else:
                                i += 1
                            chunk_token_end = chunk_token_start + cur_token_count
                            break
                    else:
                        i += 1
                        break
                else:
                    break
            else:
                chunk_sent_texts.append(sent["text"])
                cur_token_count += sent_tok_len
                chunk_token_end = sent.get("token_end_idx", chunk_token_start + cur_token_count)
                i += 1
        if not chunk_sent_texts:
            i += 1
            continue
        chunk_text = " ".join(chunk_sent_texts).strip()
        chunk_meta = {"window_index": window_index, "text": chunk_text, "token_count": cur_token_count, "token_start": chunk_token_start, "token_end": chunk_token_end, "start_sentence_idx": start_i, "end_sentence_idx": i, "is_truncated_sentence": is_truncated_sentence}
        window_index += 1
        new_start = max(start_i + 1, chunk_meta["end_sentence_idx"] - overlap_sentences)
        if windows and chunk_meta["token_count"] < min_tokens:
            prev = windows[-1]
            prev["text"] = prev["text"] + " " + chunk_meta["text"]
            prev["token_count"] = prev["token_count"] + chunk_meta["token_count"]
            prev["token_end"] = chunk_meta["token_end"]
            prev["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
            prev["is_truncated_sentence"] = prev.get("is_truncated_sentence", False) or chunk_meta.get("is_truncated_sentence", False)
        else:
            windows.append(chunk_meta)
        i = new_start
    for w in windows:
        yield w
class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: Optional[str], ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f".{ext}", dir="/tmp")
        self.count = 0
        self._first = True
        if self.ext == "json":
            self.temp.write(b"[\n")
            self.temp.flush()
    def write_payload(self, payload: Dict[str, Any]) -> int:
        self.count += 1
        if self.ext == "jsonl":
            line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.temp.write(line)
        else:
            pretty = json.dumps(payload, ensure_ascii=False, indent=2)
            indented = ("\n".join("  " + ln for ln in pretty.splitlines()) + "\n").encode("utf-8")
            if not self._first:
                self.temp.write(b",\n")
            self.temp.write(indented)
            self._first = False
        self.temp.flush()
        return 1
    def finalize_and_upload(self, out_key: str) -> Tuple[int, str]:
        if self.ext == "json":
            self.temp.write(b"]\n")
        self.temp.flush()
        self.temp.close()
        try:
            s3.upload_file(self.temp.name, S3_BUCKET, out_key, ExtraArgs={"ContentType": self.content_type})
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            return self.count, out_key
        except Exception:
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            raise
def _derive_file_name_from_source(source: Optional[str], s3_key: str) -> str:
    if source:
        try:
            base = source.split("?")[0].rstrip("/")
            base_name = os.path.basename(base)
            if base_name:
                return base_name
        except Exception:
            pass
    return os.path.basename(s3_key)
def sanitize_payload_for_weaviate(payload: Dict[str, Any]) -> None:
    for k in list(payload.keys()):
        v = payload.get(k)
        if k == "tags":
            if v is None:
                payload[k] = []
            elif isinstance(v, (list, tuple)):
                payload[k] = [str(x) for x in v]
            else:
                payload[k] = [str(v)]
            continue
        if v is None:
            payload.pop(k, None)
            continue
        if isinstance(v, (list, tuple, dict)):
            try:
                payload[k] = json.dumps(v)
            except Exception:
                payload[k] = str(v)
            continue
        if not isinstance(v, (str, int, float, bool)):
            payload[k] = str(v)
def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        logger.error("Could not head S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    last_modified = head.get("LastModified", "")
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(last_modified or ""))
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Skipping entire file because chunked file exists: %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        logger.error("Could not get S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    raw_body = obj["Body"].read()
    try:
        raw_text = raw_body.decode("utf-8", errors="replace")
    except Exception:
        raw_text = raw_body.decode("latin-1", errors="replace")
    s3_path = f"s3://{S3_BUCKET}/{s3_key}" if S3_BUCKET else None
    stripped = raw_text.strip()
    use_remote_fetch = False
    remote_url = None
    if stripped.startswith("http://") or stripped.startswith("https://"):
        use_remote_fetch = True
        remote_url = stripped.splitlines()[0].strip()
    if use_remote_fetch:
        try:
            html_text = fetch_html_with_retries(remote_url)
            source_url = remote_url
        except Exception:
            html_text = raw_text
            source_url = s3_path
    else:
        html_text = raw_text
        source_url = s3_path
    try:
        _ = upload_snapshot_to_s3(html_text, doc_id)
    except Exception:
        pass
    t0_extract = time.perf_counter()
    md, parsed = trafilatura_extract_markdown(html_text)
    extract_duration_ms = int((time.perf_counter() - t0_extract) * 1000)
    if not md:
        fallback = re.sub(r'(?is)<(script|style).*?>.*?</\1>', '', html_text)
        fallback = re.sub(r'(?is)<.*?>', ' ', fallback)
        md = re.sub(r'\s+', ' ', fallback).strip()
    title = parsed.get("title") if isinstance(parsed, dict) else None
    canonical_full = canonicalize_text(md or "")
    token_ct = token_count_for(canonical_full)
    saved = 0
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_path, ext=ext)
    file_name = _derive_file_name_from_source(source_url, s3_key)
    try:
        windows = list(split_into_token_windows(canonical_full))
        if len(windows) <= 1:
            chunk_index = 1
            chunk_id = f"{doc_id}_{chunk_index}"
            payload = {
                "document_id": doc_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "chunk_type": "page",
                "text": canonical_full,
                "token_count": int(token_ct),
                "figures": "[]",
                "embedding": None,
                "file_type": "text/html",
                "source_url": source_url,
                "page_number": None,
                "slide_range": None,
                "row_range": None,
                "token_range": None,
                "audio_range": None,
                "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "parser_version": PARSER_VERSION,
                "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                "layout_tags": ["page"],
                "used_ocr": False,
                "heading_path": [],
                "headings": [title] if title else [],
                "line_range": None
            }
            sanitize_payload_for_weaviate(payload)
            writer.write_payload(payload)
            saved += 1
        else:
            for w in windows:
                window_idx = int(w.get("window_index", 0))
                chunk_index = window_idx + 1
                chunk_id = f"{doc_id}_{chunk_index}"
                wtext = w.get("text", "")
                w_token_count = int(w.get("token_count", 0))
                token_range = [int(w.get("token_start", 0)), int(w.get("token_end", 0))]
                payload = {
                    "document_id": doc_id,
                    "file_name": file_name,
                    "chunk_id": chunk_id,
                    "chunk_type": "token_window",
                    "text": wtext,
                    "token_count": int(w_token_count),
                    "figures": "[]",
                    "embedding": None,
                    "file_type": "text/html",
                    "source_url": source_url,
                    "page_number": None,
                    "slide_range": None,
                    "row_range": None,
                    "token_range": token_range,
                    "audio_range": None,
                    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "parser_version": PARSER_VERSION,
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": ["page"],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": [title] if title else [],
                    "line_range": None
                }
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved += 1
    except Exception as e:
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.error("Error while buffering chunks for %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if saved == 0:
            try:
                if writer and writer.temp:
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("No chunks produced for %s", s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Wrote %d chunks for %s → %s (%d ms)", count, s3_key, uploaded_key, total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        logger.error("Failed to upload chunked file for %s error=%s", s3_key, str(e_up))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}
if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.lower().endswith(".html") or key.lower().endswith(".htm")):
                continue
            logger.info("Routing parse_file for s3://%s/%s", S3_BUCKET, key)
            manifest_key = key + ".manifest.json"
            try:
                mf_obj = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf_obj["Body"])
            except Exception:
                manifest = {}
            parse_file(key, manifest)
