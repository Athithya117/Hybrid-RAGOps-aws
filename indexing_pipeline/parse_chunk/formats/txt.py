#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import hashlib
import boto3
import tiktoken
import unicodedata
import re
from datetime import datetime
from botocore.exceptions import ClientError
from typing import List, Dict, Any

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
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"
S3_PUT_RETRIES = int(os.getenv("S3_PUT_RETRIES", "3"))
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", "0.3"))

s3 = boto3.client("s3")

try:
    enc = tiktoken.get_encoding(ENC_NAME)
except Exception:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

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
                chunks.append({"text": canonicalize_text(chunk_text), "token_count": p["token_count"], "start_line": chunk_start_line, "end_line": chunk_end_line, "subchunk_index": sub_idx})
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

def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    raw_body = obj["Body"].read()
    raw_text = try_decode_bytes(raw_body)
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(obj.get("LastModified", "")))
    commit_sha = manifest.get("commit_sha") or manifest.get("git_commit") or ""
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    snapshot_path = ""
    if SAVE_SNAPSHOT:
        try:
            key = f"{S3_CHUNKED_PREFIX}{doc_id}.snapshot.txt"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=raw_text.encode("utf-8"), ContentType="text/plain")
            snapshot_path = f"s3://{S3_BUCKET}/{key}"
        except Exception:
            snapshot_path = ""
    canonical_full = canonicalize_text(raw_text)
    lines = [ln + ("\n" if not ln.endswith("\n") else "") for ln in canonical_full.split("\n")]
    full_token_count = token_count_for(canonical_full)
    saved = 0
    chunk_index = 1
    if full_token_count <= TXT_MAX_TOKENS_PER_CHUNK:
        chunk_id = f"{doc_id}_{chunk_index}"
        chunk_index += 1
        checksum = sha256_hex(canonical_full)
        chunk_build_start = time.perf_counter()
        payload = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_type": "txt_section",
            "text": canonical_full,
            "embedding": None,
            "source": {
                "file_type": "text/plain",
                "source_url": s3_path,
                "snapshot_path": snapshot_path,
                "text_checksum": checksum
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "parser_version": PARSER_VERSION,
                "token_count": full_token_count,
                "token_encoder": ENC_NAME,
                "line_range": [1, len(lines)],
                "commit_sha": commit_sha
            }
        }
        payload["metadata"]["parse_chunk_duration_ms"] = int((time.perf_counter() - chunk_build_start) * 1000)
        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
        if not FORCE_OVERWRITE and s3_object_exists(out_key):
            log.info("Skipping existing chunk %s", payload["chunk_id"])
        else:
            body = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())
            s3_put_object_with_retries(out_key, body)
            log.info("Wrote chunk %s → %s", payload["chunk_id"], out_key)
            saved += 1
    else:
        subchunks = split_lines_by_tokens(lines, 0, OVERLAP_TOKENS, TXT_MAX_TOKENS_PER_CHUNK)
        for sub in subchunks:
            chunk_build_start = time.perf_counter()
            chunk_text = sub.get("text", "")
            token_ct = sub.get("token_count", 0)
            sline = sub.get("start_line", 0)
            eline = sub.get("end_line", sline)
            chunk_id = f"{doc_id}_{chunk_index}"
            chunk_index += 1
            start_line = sline + 1
            end_line = eline
            checksum = sha256_hex(chunk_text)
            payload = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_type": "txt_subchunk",
                "text": chunk_text,
                "embedding": None,
                "source": {
                    "file_type": "text/plain",
                    "source_url": s3_path,
                    "snapshot_path": snapshot_path,
                    "text_checksum": checksum
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION,
                    "token_count": token_ct,
                    "token_encoder": ENC_NAME,
                    "line_range": [start_line, end_line],
                    "subchunk_index": sub.get("subchunk_index", 0),
                    "commit_sha": commit_sha
                }
            }
            payload["metadata"]["parse_chunk_duration_ms"] = int((time.perf_counter() - chunk_build_start) * 1000)
            ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
            out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
            if not FORCE_OVERWRITE and s3_object_exists(out_key):
                log.info("Skipping existing subchunk %s", payload["chunk_id"])
                continue
            body = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())
            s3_put_object_with_retries(out_key, body)
            log.info("Wrote subchunk %s (lines %d-%d) → %s", payload["chunk_id"], start_line, end_line, out_key)
            saved += 1
    total_ms = int((time.perf_counter() - start_all) * 1000)
    log.info("Completed parsing %d chunks for %s in %d ms total", saved, s3_key, total_ms)
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}

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
