#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import hashlib
import boto3
import unicodedata
import re
import tempfile
from datetime import datetime
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Tuple

try:
    import colorama
    colorama.init()
except Exception:
    pass

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[37m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;41m"
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"

log = logging.getLogger("txt_parser")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
log.handlers[:] = [handler]

REQUIRED = ["S3_BUCKET", "S3_RAW_PREFIX", "S3_CHUNKED_PREFIX", "CHUNK_FORMAT"]
missing = [v for v in REQUIRED if os.getenv(v) is None]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl")
TXT_MAX_TOKENS_PER_CHUNK = int(os.getenv("TXT_MAX_TOKENS_PER_CHUNK", os.getenv("MD_MAX_TOKENS_PER_CHUNK", "600")))
DEFAULT_OVERLAP = max(1, int(TXT_MAX_TOKENS_PER_CHUNK * 0.1))
OVERLAP_TOKENS = int(os.getenv("TXT_OVERLAP_TOKENS", str(DEFAULT_OVERLAP)))
if OVERLAP_TOKENS >= TXT_MAX_TOKENS_PER_CHUNK:
    OVERLAP_TOKENS = max(1, TXT_MAX_TOKENS_PER_CHUNK - 1)
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
PARSER_VERSION = os.getenv("PARSER_VERSION_TXT", "plain-txt-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
S3_PUT_RETRIES = int(os.getenv("S3_PUT_RETRIES", "3"))
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", "0.3"))

s3 = boto3.client("s3")

try:
    import tiktoken
    enc = tiktoken.get_encoding(ENC_NAME)
except Exception:
    enc = None

def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r'[ \t]+$', '', ln) for ln in s.split("\n")]
    return "\n".join(lines).strip()

def try_decode_bytes(b: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return b.decode(encoding)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")

def token_count_for(text: str) -> int:
    if not text:
        return 0
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())

def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False

def s3_put_object_with_retries(key: str, body: bytes, content_type: str = "application/json") -> None:
    for attempt in range(1, S3_PUT_RETRIES + 1):
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)
            return
        except Exception:
            if attempt < S3_PUT_RETRIES:
                time.sleep(S3_PUT_BACKOFF * attempt)
    raise Exception("s3 put failed after retries")

def _derive_source_key_from_path(s3_path: str) -> str:
    prefix = f"s3://{S3_BUCKET}/"
    if s3_path.startswith(prefix):
        return s3_path[len(prefix):]
    return ""

def split_long_line_into_char_windows(line: str, max_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
    pieces = []
    approx_char_per_token = max(1, len(line) // max(1, token_count_for(line)))
    window_chars = max(200, approx_char_per_token * max_tokens)
    step_chars = max(1, window_chars - approx_char_per_token * overlap_tokens)
    start = 0
    idx = 1
    cap = 1000
    while start < len(line) and idx <= cap:
        end = min(len(line), start + window_chars)
        piece = line[start:end]
        pieces.append({"text": piece, "token_count": token_count_for(piece), "subchunk_index": idx})
        idx += 1
        if end >= len(line):
            break
        start = start + step_chars
    return pieces

def split_lines_by_tokens(lines: List[str], base_start_line: int, overlap_tokens: int, max_tokens: int) -> List[Dict[str, Any]]:
    line_token_cache: Dict[int, int] = {}
    token_counts = []
    for idx, l in enumerate(lines):
        abs_idx = base_start_line + idx
        if abs_idx in line_token_cache:
            cnt = line_token_cache[abs_idx]
        else:
            try:
                cnt = len(enc.encode(l)) if enc else len(l.split())
            except Exception:
                cnt = len(l.split())
            line_token_cache[abs_idx] = cnt
            cnt = line_token_cache[abs_idx]
        token_counts.append(cnt)
    n = len(lines)
    chunks: List[Dict[str, Any]] = []
    ptr = 0
    sub_idx = 1
    while ptr < n:
        current_tokens = 0
        j = ptr
        while j < n:
            next_tokens = token_counts[j]
            if current_tokens + next_tokens > max_tokens and current_tokens > 0:
                break
            current_tokens += next_tokens
            j += 1
        if j == ptr:
            line_idx = ptr
            line_text = lines[line_idx]
            long_pieces = split_long_line_into_char_windows(line_text, max_tokens, overlap_tokens)
            for p in long_pieces:
                chunk_start_line = base_start_line + line_idx
                chunk_end_line = chunk_start_line + 1
                chunk_text = p["text"]
                chunks.append({"text": canonicalize_text(chunk_text), "token_count": p["token_count"], "start_line": chunk_start_line, "end_line": chunk_end_line, "subchunk_index": p["subchunk_index"]})
                sub_idx += 1
            ptr = ptr + 1
            continue
        chunk_start_line = base_start_line + ptr
        chunk_end_line = base_start_line + j
        chunk_text = "".join(lines[ptr:j]).strip()
        chunks.append({"text": canonicalize_text(chunk_text), "token_count": current_tokens, "start_line": chunk_start_line, "end_line": chunk_end_line, "subchunk_index": sub_idx})
        sub_idx += 1
        if overlap_tokens <= 0:
            next_ptr = j
        else:
            back_sum = 0
            back_idx = j - 1
            min_back_idx = ptr
            while back_idx >= min_back_idx and back_sum < overlap_tokens:
                back_sum += token_counts[back_idx]
                back_idx -= 1
            overlap_start = max(ptr, back_idx + 1)
            next_ptr = overlap_start
            if next_ptr <= ptr:
                next_ptr = j
        ptr = next_ptr
    return chunks

class S3DocWriter:
    """
    Aggregates chunks per raw file into a single temp file in /tmp, then uploads once.
    Writes clean JSONL (one JSON object per line) or pretty JSON array (multi-line, indented objects).
    """
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
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

def _derive_doc_id_from_head(s3_key: str, head_obj: dict, manifest: dict) -> str:
    """
    Derive a stable doc_id without downloading content.
    Priority:
      1. manifest['file_hash'] (if present)
      2. ETag (from HEAD) - stripped of quotes
      3. LastModified (string)
      4. fallback to basename(s3_key)
      5. fallback to sha256(s3_key)
    """
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        return manifest.get("file_hash")
    etag = head_obj.get("ETag", "")
    if isinstance(etag, str):
        etag = etag.strip('"')
    if etag:
        return sha256_hex(s3_key + str(etag))
    lm = head_obj.get("LastModified", "")
    if lm:
        return sha256_hex(s3_key + str(lm))
    base = os.path.basename(s3_key)
    if base:
        return base
    return sha256_hex(s3_key)

def parse_file(s3_key: str, manifest: dict) -> dict:
    """
    Fast, idempotent parsing for plain .txt files:
    - HEAD to derive doc_id and check whether the aggregated chunk file exists
    - if chunk exists (and not FORCE_OVERWRITE) skip fast without downloading the raw object
    - otherwise GET and produce chunked output
    """
    start_all = time.perf_counter()

    # HEAD first to avoid downloading when possible
    try:
        head_obj = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        # If HEAD fails, try to continue with GET (but return an error if GET also fails)
        log.warning("HEAD failed for %s: %s -- will try GET", s3_key, e)
        head_obj = {}

    doc_id = _derive_doc_id_from_head(s3_key, head_obj or {}, manifest or {})
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"

    # fast skip if aggregated output exists
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Skipping entire file because chunked file exists: %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}

    # get object (we need to parse)
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("Could not read S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

    raw_body = obj["Body"].read()
    raw_text = try_decode_bytes(raw_body)

    # prefer explicit manifest.file_hash if present (keeps backward compatibility)
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash")
        out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"

    # race-check after download: another process might have written the aggregated file between HEAD and GET
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Skipping entire file because chunked file exists (post-download): %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}

    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    canonical_full = canonicalize_text(raw_text)
    # keep newline at line ends so line ranges are meaningful
    lines = [ln + ("\n" if not ln.endswith("\n") else "") for ln in canonical_full.split("\n")]
    full_token_count = token_count_for(canonical_full)
    saved = 0
    chunk_index = 1
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_path, ext=ext)
    file_name = os.path.basename(s3_key)

    try:
        # small document -> single chunk
        if full_token_count <= TXT_MAX_TOKENS_PER_CHUNK:
            chunk_id = f"{doc_id}_{chunk_index}"
            chunk_index += 1
            payload = {
                "document_id": doc_id or "",
                "file_name": file_name,
                "chunk_id": chunk_id or "",
                "chunk_type": "txt_subchunk",
                "text": canonical_full or "",
                "token_count": int(full_token_count or 0),
                "embedding": None,
                "file_type": "text/plain",
                "source_url": s3_path,
                "page_number": None,
                "slide_range": None,
                "row_range": None,
                "token_range": None,
                "audio_range": None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "parser_version": PARSER_VERSION,
                "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                "layout_tags": [],
                "used_ocr": False,
                "heading_path": [],
                "headings": [],
                "line_range": [1, len(lines)]
            }
            writer.write_payload(payload)
            log.info("Buffered single chunk %s", payload["chunk_id"])
            saved += 1
        else:
            # produce token-based subchunks
            subchunks = split_lines_by_tokens(lines, 0, OVERLAP_TOKENS, TXT_MAX_TOKENS_PER_CHUNK)
            for sub in subchunks:
                chunk_text = sub.get("text", "")
                token_ct = sub.get("token_count", 0)
                sline = sub.get("start_line", 0)
                eline = sub.get("end_line", sline)
                chunk_id = f"{doc_id}_{chunk_index}"
                chunk_index += 1
                start_line = sline + 1
                end_line = eline
                payload = {
                    "document_id": doc_id or "",
                    "file_name": file_name,
                    "chunk_id": chunk_id or "",
                    "chunk_type": "txt_subchunk",
                    "text": chunk_text or "",
                    "token_count": int(token_ct or 0),
                    "embedding": None,
                    "file_type": "text/plain",
                    "source_url": s3_path,
                    "page_number": None,
                    "slide_range": None,
                    "row_range": None,
                    "token_range": None,
                    "audio_range": None,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION,
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": [],
                    "line_range": [int(start_line), int(end_line)]
                }
                writer.write_payload(payload)
                log.info("Buffered subchunk %s (lines %d-%d)", payload["chunk_id"], start_line, end_line)
                saved += 1
    except Exception as e:
        try:
            if writer and getattr(writer, "temp", None):
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.exception("Error while buffering chunks for %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

    try:
        if saved == 0:
            try:
                if writer and getattr(writer, "temp", None):
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("No chunks produced for %s", s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Wrote %d chunks for %s → %s (%d ms)", count, s3_key, uploaded_key, total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("Failed to upload chunked file for %s error=%s", s3_key, str(e_up))
        try:
            if writer and getattr(writer, "temp", None):
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".txt"):
                continue
            log.info("Routing parse_file for s3://%s/%s", S3_BUCKET, key)
            manifest_key = key + ".manifest.json"
            try:
                mf = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf["Body"])
            except Exception:
                manifest = {}
            try:
                parse_file(key, manifest)
            except Exception as e:
                log.exception("Failed to parse %s: %s", key, e)
