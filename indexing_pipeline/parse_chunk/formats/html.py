import os
import sys
import json
import time
import logging
import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, Optional, List
from tempfile import NamedTemporaryFile
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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("html_trafilatura")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
PARSER_VERSION = os.getenv("PARSER_VERSION_HTML", "trafilatura-only-v2")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"
WINDOW_SIZE = int(os.getenv("HTML_WINDOW_SIZE", "2000"))
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "3"))
FETCH_BACKOFF = float(os.getenv("FETCH_BACKOFF", "0.5"))
STORE_ONE_FILE_PER_CHUNK = os.getenv("STORE_ONE_FILE_PER_CHUNK", "true").lower() == "true"
S3_PUT_RETRIES = int(os.getenv("S3_PUT_RETRIES", "3"))
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", "0.3"))

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

def split_into_token_windows(text: str, window_size: int = WINDOW_SIZE) -> Iterator[Dict[str, Any]]:
    if not text:
        yield {"window_index": 0, "text": "", "token_count": 0, "token_start": 0, "token_end": 0}
        return
    overlap = max(1, int(window_size * 0.1))
    if ENCODER is None:
        tokens = text.split()
        total = len(tokens)
        step = max(1, window_size - overlap)
        idx = 0
        for start in range(0, total, step):
            end = min(start + window_size, total)
            window_text = " ".join(tokens[start:end])
            yield {"window_index": idx, "text": canonicalize_text(window_text), "token_count": end - start, "token_start": start, "token_end": end}
            idx += 1
            if end >= total:
                break
        return
    tokens = ENCODER.encode(text)
    total = len(tokens)
    step = max(1, window_size - overlap)
    idx = 0
    for start in range(0, total, step):
        end = start + window_size
        slice_tokens = tokens[start:end]
        window_text = ENCODER.decode(slice_tokens)
        yield {"window_index": idx, "text": canonicalize_text(window_text), "token_count": len(slice_tokens), "token_start": start, "token_end": min(end, total)}
        idx += 1
        if end >= total:
            break

class LocalChunkAppender:
    def __init__(self, chunk_format: str, doc_id: str):
        self.chunk_format = chunk_format
        self.doc_id = doc_id
        suffix = f".{chunk_format}"
        self.temp = NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=suffix)
        self.path = self.temp.name
        self.count = 0
    def append(self, payload: dict):
        if self.chunk_format == "jsonl":
            line = json.dumps(payload, ensure_ascii=False)
            self.temp.write(line + "\n")
        else:
            pretty = json.dumps(payload, indent=2, ensure_ascii=False)
            self.temp.write(pretty + "\n")
        self.count += 1
        self.temp.flush()
    def finalize_and_upload(self, s3_bucket: str, s3_key: str):
        self.temp.close()
        extra = {"ContentType": "application/json"}
        try:
            s3.upload_file(self.path, s3_bucket, s3_key, ExtraArgs=extra)
            logger.info("Uploaded combined chunks for %s → s3://%s/%s (%d chunks)", self.doc_id, s3_bucket, s3_key, self.count)
        finally:
            try:
                os.remove(self.path)
            except Exception:
                pass

def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
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
    last_modified = obj.get("LastModified", "")
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(last_modified or ""))
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
    snapshot = None
    try:
        snapshot = upload_snapshot_to_s3(html_text, doc_id)
    except Exception:
        snapshot = None
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
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    combined_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    combined_appender = None
    if not STORE_ONE_FILE_PER_CHUNK:
        if not FORCE_OVERWRITE and s3_object_exists(combined_key):
            logger.info("Skipping file %s because combined target exists and FORCE_OVERWRITE is false → %s", s3_key, combined_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": int((time.perf_counter() - start_all) * 1000)}
        combined_appender = LocalChunkAppender(ext, doc_id)
        logger.info("Using combined chunk file mode for %s → s3://%s/%s", doc_id, S3_BUCKET, combined_key)
    if token_ct <= WINDOW_SIZE:
        t0_chunk = time.perf_counter()
        chunk_index = 1
        chunk_id = f"{doc_id}_{chunk_index}"
        parse_ms = extract_duration_ms + int((time.perf_counter() - t0_chunk) * 1000) or 1
        payload = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_type": "page",
            "text": canonical_full,
            "token_count": int(token_ct),
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
            "tags": [],
            "layout_tags": ["page"],
            "used_ocr": False,
            "parse_chunk_duration_ms": int(parse_ms),
            "heading_path": [],
            "headings": [title] if title else [],
            "line_range": None,
            "chunk_duration_ms": None,
            "snapshot_url": snapshot
        }
        out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
        if STORE_ONE_FILE_PER_CHUNK:
            if not FORCE_OVERWRITE and s3_object_exists(out_key):
                logger.info("Skipping existing chunk %s", payload["chunk_id"])
            else:
                body = (json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode()
                s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json")
                logger.info("Wrote page chunk %s → %s", payload["chunk_id"], out_key)
                saved += 1
        else:
            try:
                combined_appender.append(payload)
                logger.info("Appended page chunk %s → %s", payload["chunk_id"], combined_key)
                saved += 1
            except Exception as e:
                logger.error("Failed to append payload for %s: %s", payload["chunk_id"], e)
    else:
        windows = list(split_into_token_windows(canonical_full))
        for w in windows:
            t0_chunk = time.perf_counter()
            window_idx = int(w.get("window_index", 0))
            chunk_index = window_idx + 1
            chunk_id = f"{doc_id}_{chunk_index}"
            wtext = w.get("text", "")
            w_token_count = int(w.get("token_count", 0))
            parse_ms = extract_duration_ms + int((time.perf_counter() - t0_chunk) * 1000) or 1
            token_range = [int(w.get("token_start")), int(w.get("token_end"))]
            payload = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_type": "token_window",
                "text": wtext,
                "token_count": int(w_token_count),
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
                "tags": [],
                "layout_tags": ["page"],
                "used_ocr": False,
                "parse_chunk_duration_ms": int(parse_ms),
                "heading_path": [],
                "headings": [title] if title else [],
                "line_range": None,
                "chunk_duration_ms": None,
                "snapshot_url": snapshot
            }
            out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
            if STORE_ONE_FILE_PER_CHUNK:
                if not FORCE_OVERWRITE and s3_object_exists(out_key):
                    logger.info("Skipping existing window chunk %s", payload["chunk_id"])
                    continue
                body = (json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode()
                s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json")
                logger.info("Wrote window chunk %s (window %d) → %s", payload["chunk_id"], window_idx, out_key)
                saved += 1
            else:
                try:
                    combined_appender.append(payload)
                    logger.info("Appended window chunk %s (window %d) → %s", payload["chunk_id"], window_idx, combined_key)
                    saved += 1
                except Exception as e:
                    logger.error("Failed to append payload for %s: %s", payload["chunk_id"], e)
    if not STORE_ONE_FILE_PER_CHUNK and combined_appender is not None:
        try:
            combined_appender.finalize_and_upload(S3_BUCKET, combined_key)
        except Exception as e:
            logger.error("Failed uploading combined file for %s: %s", doc_id, e)
    total_ms = int((time.perf_counter() - start_all) * 1000)
    logger.info("Completed parsing %d chunks for %s in %d ms total", saved, s3_key, total_ms)
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}

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
