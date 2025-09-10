import os
import sys
import json
import time
import logging
import hashlib
import re
import unicodedata
from datetime import datetime
from typing import Iterator, Dict, Any
import boto3
import requests
from botocore.exceptions import ClientError
import trafilatura
import tiktoken

try:
    import colorama
    colorama.init()
except Exception:
    pass

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[36m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;41m"
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"

logger = logging.getLogger("html_trafilatura")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers[:] = [handler]
log = logger

S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
PARSER_VERSION = os.getenv("PARSER_VERSION_HTML", "trafilatura-only-v2")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"

WINDOW_SIZE = int(os.getenv("HTML_WINDOW_SIZE", "2000"))
OVERLAP_TOKENS = int(os.getenv("HTML_OVERLAP_TOKENS", "200"))
SPLIT_THRESHOLD = int(os.getenv("SPLIT_THRESHOLD", str(WINDOW_SIZE)))

ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "3"))
FETCH_BACKOFF = float(os.getenv("FETCH_BACKOFF", "0.5"))

s3 = boto3.client("s3")
enc = tiktoken.get_encoding(ENC_NAME)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

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

def upload_snapshot_to_s3(snapshot_html: str, doc_id: str, s3_key_prefix: str) -> str:
    if not SAVE_SNAPSHOT:
        return ""
    key = f"{s3_key_prefix}{doc_id}.snapshot.html"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=snapshot_html.encode("utf-8"), ContentType="text/html")
    return f"s3://{S3_BUCKET}/{key}"

def trafilatura_extract_markdown(html_text: str):
    md = trafilatura.extract(html_text, output_format="markdown", with_metadata=True)
    parsed = {}
    try:
        json_doc = trafilatura.extract(html_text, output_format="json", with_metadata=True)
        if json_doc:
            parsed = json.loads(json_doc)
    except Exception:
        parsed = {}
    return md, parsed

def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False

def token_count_for(text: str) -> int:
    return len(enc.encode(text))

def split_into_token_windows(text: str, doc_id: str, source_url: str, parser_version: str = PARSER_VERSION) -> Iterator[Dict[str, Any]]:
    tokens = enc.encode(text)
    total = len(tokens)
    step = WINDOW_SIZE - OVERLAP_TOKENS
    if step <= 0:
        raise ValueError("OVERLAP_TOKENS must be smaller than WINDOW_SIZE")
    window_idx = 0
    for start in range(0, total, step):
        end = start + WINDOW_SIZE
        window_tokens = tokens[start:end]
        window_text = enc.decode(window_tokens)
        yield {
            "window_index": window_idx,
            "text": canonicalize_text(window_text),
            "token_count": len(window_tokens),
            "token_start": start,
            "token_end": min(end, total)
        }
        window_idx += 1
        if end >= total:
            break

def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    raw_body = obj["Body"].read()
    raw_text = raw_body.decode("utf-8", errors="replace")
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(obj.get("LastModified", "")))
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    stripped = raw_text.strip()
    use_remote_fetch = False
    remote_url = None
    if stripped.startswith("http://") or stripped.startswith("https://"):
        use_remote_fetch = True
        remote_url = stripped.splitlines()[0].strip()
    if use_remote_fetch:
        html_text = fetch_html_with_retries(remote_url)
        source_url = remote_url
    else:
        html_text = raw_text
        source_url = s3_path
    snapshot_path = ""
    try:
        snapshot_path = upload_snapshot_to_s3(html_text, doc_id, S3_CHUNKED_PREFIX)
    except Exception:
        snapshot_path = ""
    t0_extract = time.perf_counter()
    md, parsed = trafilatura_extract_markdown(html_text)
    extract_duration_ms = int((time.perf_counter() - t0_extract) * 1000)
    if not md:
        fallback = re.sub(r'(?is)<(script|style).*?>.*?</\1>', '', html_text)
        fallback = re.sub(r'(?is)<.*?>', ' ', fallback)
        md = re.sub(r'\s+', ' ', fallback).strip()
    title = parsed.get("title") if isinstance(parsed, dict) else None
    canonical_full = canonicalize_text(md)
    token_ct = token_count_for(canonical_full)
    saved = 0
    if token_ct <= SPLIT_THRESHOLD:
        t0_chunk = time.perf_counter()
        chunk_index = 1
        chunk_id = f"{doc_id}_{chunk_index}"
        payload = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_type": "page",
            "text": canonical_full,
            "embedding": None,
            "source": {
                "file_type": "text/html",
                "source_url": source_url,
                "snapshot_path": snapshot_path,
                "text_checksum": sha256_hex(canonical_full)
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "title": title,
                "parser_version": PARSER_VERSION,
                "token_count": token_ct,
                "token_encoder": ENC_NAME,
                "parse_chunk_duration_ms": int((time.perf_counter() - t0_chunk) * 1000) + extract_duration_ms
            }
        }
        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
        if not FORCE_OVERWRITE and s3_object_exists(out_key):
            log.info(f"Skipping existing chunk {payload['chunk_id']}")
        else:
            body = (json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode()
            s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json")
            log.info(f"Wrote page chunk {payload['chunk_id']} → {out_key}")
            saved += 1
    else:
        t0_split_total = time.perf_counter()
        windows = list(split_into_token_windows(canonical_full, doc_id, source_url, PARSER_VERSION))
        for w in windows:
            t0_chunk = time.perf_counter()
            window_idx = w.get("window_index", 0)
            chunk_index = window_idx + 1
            chunk_id = f"{doc_id}_{chunk_index}"
            wtext = w["text"]
            w_token_count = w.get("token_count")
            payload = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_type": "token_window",
                "text": wtext,
                "embedding": None,
                "source": {
                    "file_type": "text/html",
                    "source_url": source_url,
                    "snapshot_path": snapshot_path,
                    "text_checksum": sha256_hex(wtext),
                    "token_start": w.get("token_start"),
                    "token_end": w.get("token_end")
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "title": title,
                    "parser_version": PARSER_VERSION,
                    "token_count": w_token_count,
                    "window_index": window_idx,
                    "token_encoder": ENC_NAME,
                    "parse_chunk_duration_ms": extract_duration_ms + int((time.perf_counter() - t0_chunk) * 1000)
                }
            }
            ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
            out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
            if not FORCE_OVERWRITE and s3_object_exists(out_key):
                log.info(f"Skipping existing window chunk {payload['chunk_id']}")
                continue
            body = (json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode()
            s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json")
            log.info(f"Wrote window chunk {payload['chunk_id']} (window {window_idx}) → {out_key}")
            saved += 1
        split_duration_ms = int((time.perf_counter() - t0_split_total) * 1000)
        log.info(f"Split into windows in {split_duration_ms} ms (extraction took {extract_duration_ms} ms)")
    total_ms = int((time.perf_counter() - start_all) * 1000)
    log.info(f"Completed parsing {saved} chunks for {s3_key} in {total_ms} ms total")
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}

if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.lower().endswith(".html") or key.lower().endswith(".htm")):
                continue
            log.info(f"Routing parse_file for s3://{S3_BUCKET}/{key}")
            manifest_key = key + ".manifest.json"
            try:
                mf_obj = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf_obj["Body"])
            except Exception:
                manifest = {}
            parse_file(key, manifest)
