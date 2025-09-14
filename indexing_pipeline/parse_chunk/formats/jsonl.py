import os
import io
import json
import time
import math
import logging
import hashlib
import unicodedata
from datetime import datetime
from typing import Any, Dict, Iterator, Tuple, List
import boto3
import botocore
import ray
import pandas as pd
import tiktoken

try:
    import colorama
    colorama.init()
except Exception:
    pass

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[97m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;41m",
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"

logger = logging.getLogger("jsonl_parser")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers[:] = [handler]
log = logger

S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
PARSER_VERSION = os.getenv("PARSER_VERSION_JSONL", "ray-jsonl-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
TARGET_TOKENS_PER_CHUNK = int(os.getenv("JSONL_TARGET_TOKENS_PER_CHUNK", os.getenv("CSV_TARGET_TOKENS_PER_CHUNK", "1000")))
ROWS_PER_CHUNK_OVERRIDE = os.getenv("JSONL_ROWS_PER_CHUNK", os.getenv("CSV_ROWS_PER_CHUNK", ""))
MIN_ROWS_PER_CHUNK = int(os.getenv("JSONL_MIN_ROWS_PER_CHUNK", os.getenv("CSV_MIN_ROWS_PER_CHUNK", "1")))
MAX_ROWS_PER_CHUNK = int(os.getenv("JSONL_MAX_ROWS_PER_CHUNK", os.getenv("CSV_MAX_ROWS_PER_CHUNK", "100")))
S3_PUT_RETRIES = int(os.getenv("S3_PUT_RETRIES", "3"))
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", "0.5"))
S3_RANGE_BYTES = int(os.getenv("S3_RANGE_BYTES", "131072"))

s3 = boto3.client("s3")
enc = tiktoken.get_encoding(ENC_NAME)

_RAY_CONNECTED = False

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split()).strip()

def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False
    except Exception:
        return False

def s3_put_object_with_retries(key: str, body: bytes, content_type: str = "application/json") -> Tuple[int, int]:
    attempt = 0
    t_start = time.perf_counter()
    last_exc = None
    while True:
        try:
            attempt += 1
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)
            t_end = time.perf_counter()
            ms = int(round((t_end - t_start) * 1000))
            if ms == 0:
                ms = 1
            return attempt, ms
        except Exception as e:
            last_exc = e
            if attempt >= max(1, S3_PUT_RETRIES):
                raise
            time.sleep(S3_PUT_BACKOFF * attempt)

def token_count_for(text: str) -> int:
    return len(enc.encode(text))

def split_into_token_windows(text: str, window_tokens: int, overlap: int = 0) -> Iterator[Dict[str, Any]]:
    tokens = enc.encode(text)
    total = len(tokens)
    if window_tokens <= overlap:
        raise ValueError("window_tokens must be greater than overlap")
    step = window_tokens - overlap
    idx = 0
    for start in range(0, total, step):
        end = start + window_tokens
        window_slice = tokens[start:end]
        window_text = enc.decode(window_slice)
        yield {
            "window_index": idx,
            "text": canonicalize_text(window_text),
            "token_count": len(window_slice),
            "token_start": start,
            "token_end": min(end, total),
        }
        idx += 1
        if end >= total:
            break

def row_to_schema_text(row: Any) -> str:
    parts: List[str] = []
    if isinstance(row, pd.Series):
        for c, v in row.items():
            parts.append(f"{c}: {'' if pd.isna(v) else v}")
    elif isinstance(row, dict):
        for k, v in row.items():
            parts.append(f"{k}: {'' if v is None else v}")
    else:
        parts.append(str(row))
    return canonicalize_text(" | ".join(parts))

def ensure_ray() -> None:
    global _RAY_CONNECTED
    if _RAY_CONNECTED:
        return
    addr = os.getenv("RAY_ADDRESS", "auto")
    try:
        ray.init(address=addr, ignore_reinit_error=True)
        log.info(f"Connected to Ray via address={addr}")
    except Exception:
        ray.init(ignore_reinit_error=True, include_dashboard=False, configure_logging=False)
        log.info("Started local Ray instance")
    _RAY_CONNECTED = True

def get_header_and_sample_tokens(s3_key: str) -> Tuple[str, int]:
    try:
        range_header = {"Range": f"bytes=0-{S3_RANGE_BYTES - 1}"}
        resp = s3.get_object(Bucket=S3_BUCKET, Key=s3_key, Range=range_header["Range"])
        body_bytes = resp.get("Body").read()
    except Exception:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            body_bytes = obj.get("Body").read()
        except Exception:
            return "", 32
    try:
        text = body_bytes.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        parsed = []
        for ln in lines[:4]:
            try:
                parsed.append(json.loads(ln))
            except Exception:
                continue
        if not parsed:
            return "", 32
        keys = sorted(set().union(*(list(p.keys()) for p in parsed if isinstance(p, dict))))
        header_text = canonicalize_text(" | ".join(keys))
        sample_obj = parsed[0] if parsed else parsed[0]
        sample_text = row_to_schema_text(sample_obj)
        sample_tokens = max(1, token_count_for(sample_text))
        return header_text, sample_tokens
    except Exception:
        return "", 32

def make_doc_id(s3_key: str, last_modified: Any) -> str:
    return sha256_hex(s3_key + str(last_modified or ""))

def _derive_source_key_from_path(s3_path: str) -> str:
    prefix = f"s3://{S3_BUCKET}/"
    if s3_path.startswith(prefix):
        return s3_path[len(prefix):]
    return ""

def _flush_rows_chunk(
    doc_id: str,
    s3_path: str,
    chunk_index: int,
    header_text: str,
    rows_text: list,
    start_row_num: int,
) -> Tuple[int, int]:
    if not rows_text:
        return 0, chunk_index
    chunk_index += 1
    chunk_id = f"{doc_id}_{chunk_index}"
    t0_chunk = time.perf_counter()
    chunk_text = header_text + "\n" + "\n".join(rows_text) if header_text else "\n".join(rows_text)
    token_ct = token_count_for(chunk_text)
    end_row_num = start_row_num + len(rows_text) - 1

    source_key = _derive_source_key_from_path(s3_path)

    # universal schema payload
    payload = {
        "document_id": doc_id or "",
        "chunk_id": chunk_id or "",
        "chunk_type": "row_group",
        "text": canonicalize_text(chunk_text) or "",
        "token_count": int(token_ct or 0),
        "embedding": None,
        "file_type": "application/x-ndjson",
        "source_path": s3_path,
        "source_url": None,
        "snapshot_path": "",
        "text_checksum": sha256_hex(chunk_text),
        "page_number": None,
        "slide_range_start": None,
        "slide_range_end": None,
        "row_range_start": int(start_row_num),
        "row_range_end": int(end_row_num),
        "token_start": None,
        "token_end": None,
        "audio_range_start": "",
        "audio_range_end": "",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parser_version": PARSER_VERSION or "",
        "token_encoder": ENC_NAME or "",
        "tags": [],
        "layout_tags": [],
        "used_ocr": False,
        "parse_chunk_duration_ms": None,
        "window_index": None,
        "heading_path": [],
        "headings": [],
        "line_range_start": None,
        "line_range_end": None,
        "subchunk_index": None,
        "commit_sha": "",
        "model_compute": "",
        "cpu_threads": None,
        "beam_size": None,
        "chunk_duration_ms": None,
        "token_window_index": None,
        "snapshot_id": "",
        "source_bucket": S3_BUCKET,
        "source_key": source_key,
        "source_format_hint": "application/x-ndjson"
    }

    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
    body = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())

    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        t_after = time.perf_counter()
        parse_ms = int(round((t_after - t0_chunk) * 1000))
        if parse_ms == 0:
            parse_ms = 1
        payload["parse_chunk_duration_ms"] = parse_ms
        log.info(f"Skipping existing chunk {payload['chunk_id']} (assembly {parse_ms} ms)")
        return 0, chunk_index

    # write initial payload
    t_put_start = time.perf_counter()
    s3_put_object_with_retries(out_key, body, content_type="application/json")
    t_put_end = time.perf_counter()
    total_ms = int(round((t_put_end - t0_chunk) * 1000))
    if total_ms == 0:
        total_ms = 1
    payload["parse_chunk_duration_ms"] = total_ms
    payload["chunk_duration_ms"] = total_ms

    # write payload again with duration metadata to preserve the original behavior
    body_with_meta = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())
    s3_put_object_with_retries(out_key, body_with_meta, content_type="application/json")
    log.info(f"Wrote JSONL chunk {payload['chunk_id']} → {out_key} ({total_ms} ms)")
    return 1, chunk_index

def _process_batch_rows(rows_iterable, doc_id, s3_path, chunk_index, header_text, next_row_num):
    saved = 0
    rows_text = []
    start_row_of_current = next_row_num
    for _, row in rows_iterable:
        if isinstance(row, pd.Series):
            row_text = row_to_schema_text(row)
        elif isinstance(row, dict):
            row_text = canonicalize_text(" | ".join([f"{k}: {v}" for k, v in row.items()]))
        else:
            row_text = canonicalize_text(str(row))
        row_num = next_row_num
        next_row_num += 1
        row_tokens = token_count_for(row_text)
        header_tokens = token_count_for(header_text) if header_text else 0

        source_key = _derive_source_key_from_path(s3_path)

        if row_tokens > TARGET_TOKENS_PER_CHUNK:
            if rows_text:
                wrote, chunk_index = _flush_rows_chunk(doc_id, s3_path, chunk_index, header_text, rows_text, start_row_of_current)
                saved += wrote
                rows_text = []
            windows = list(split_into_token_windows(row_text, TARGET_TOKENS_PER_CHUNK, overlap=int(TARGET_TOKENS_PER_CHUNK * 0.1)))
            for w in windows:
                chunk_index += 1
                chunk_id = f"{doc_id}_{chunk_index}"
                candidate_text = header_text + "\n" + w["text"] if header_text and (header_tokens + w["token_count"] <= TARGET_TOKENS_PER_CHUNK) else w["text"]
                token_ct = token_count_for(candidate_text)
                # universal schema for token window
                payload = {
                    "document_id": doc_id or "",
                    "chunk_id": chunk_id or "",
                    "chunk_type": "token_window",
                    "text": canonicalize_text(candidate_text) or "",
                    "token_count": int(token_ct or 0),
                    "embedding": None,
                    "file_type": "application/x-ndjson",
                    "source_path": s3_path,
                    "source_url": None,
                    "snapshot_path": "",
                    "text_checksum": sha256_hex(candidate_text),
                    "page_number": None,
                    "slide_range_start": None,
                    "slide_range_end": None,
                    "row_range_start": int(row_num),
                    "row_range_end": int(row_num),
                    "token_start": int(w.get("token_start")),
                    "token_end": int(w.get("token_end")),
                    "audio_range_start": "",
                    "audio_range_end": "",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION or "",
                    "token_encoder": ENC_NAME or "",
                    "tags": [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "parse_chunk_duration_ms": None,
                    "window_index": int(w.get("window_index")),
                    "heading_path": [],
                    "headings": [],
                    "line_range_start": None,
                    "line_range_end": None,
                    "subchunk_index": None,
                    "commit_sha": "",
                    "model_compute": "",
                    "cpu_threads": None,
                    "beam_size": None,
                    "chunk_duration_ms": None,
                    "token_window_index": int(w.get("window_index")),
                    "snapshot_id": "",
                    "source_bucket": S3_BUCKET,
                    "source_key": source_key,
                    "source_format_hint": "application/x-ndjson"
                }

                ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
                out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ext}"
                body = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())
                t0_chunk = time.perf_counter()
                if not FORCE_OVERWRITE and s3_object_exists(out_key):
                    t_after = time.perf_counter()
                    parse_ms = int(round((t_after - t0_chunk) * 1000))
                    if parse_ms == 0:
                        parse_ms = 1
                    payload["parse_chunk_duration_ms"] = parse_ms
                    log.info(f"Skipping existing window chunk {payload['chunk_id']} (assembly {parse_ms} ms)")
                    continue
                s3_put_object_with_retries(out_key, body, content_type="application/json")
                t_put_end = time.perf_counter()
                total_ms = int(round((t_put_end - t0_chunk) * 1000))
                if total_ms == 0:
                    total_ms = 1
                payload["parse_chunk_duration_ms"] = total_ms
                payload["chunk_duration_ms"] = total_ms
                body_with_meta = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())
                s3_put_object_with_retries(out_key, body_with_meta, content_type="application/json")
                log.info(f"Wrote JSONL token window {payload['chunk_id']} → {out_key} ({total_ms} ms)")
                saved += 1
            start_row_of_current = next_row_num
            continue

        candidate_text = header_text + "\n" + "\n".join(rows_text + [row_text]) if header_text else "\n".join(rows_text + [row_text])
        candidate_tokens = token_count_for(candidate_text)
        if candidate_tokens <= TARGET_TOKENS_PER_CHUNK:
            if not rows_text:
                start_row_of_current = row_num
            rows_text.append(row_text)
            continue
        else:
            wrote, chunk_index = _flush_rows_chunk(doc_id, s3_path, chunk_index, header_text, rows_text, start_row_of_current)
            saved += wrote
            rows_text = [row_text]
            start_row_of_current = row_num
    if rows_text:
        wrote, chunk_index = _flush_rows_chunk(doc_id, s3_path, chunk_index, header_text, rows_text, start_row_of_current)
        saved += wrote
    return saved, chunk_index, next_row_num

def parse_file(s3_key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    start_all = time.perf_counter()
    try:
        head_obj = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        log.error(f"Could not head S3 object {s3_key}: {e}")
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": str(e)}
    last_modified = head_obj.get("LastModified", "")
    doc_id = manifest.get("file_hash") or make_doc_id(s3_key, last_modified)
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    header_text, sample_row_tokens = get_header_and_sample_tokens(s3_key)
    header_tokens = token_count_for(header_text) if header_text else 0
    if header_tokens >= TARGET_TOKENS_PER_CHUNK:
        log.warning("JSONL header token count >= target chunk size. Header will not be prepended to row_group chunks to avoid exceeding target.")
        header_text = ""
        header_tokens = 0
    if ROWS_PER_CHUNK_OVERRIDE:
        rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, int(ROWS_PER_CHUNK_OVERRIDE)))
    else:
        available_for_rows = max(1, TARGET_TOKENS_PER_CHUNK - header_tokens)
        estimated_rows = max(1, int(available_for_rows / max(1, sample_row_tokens)))
        rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, estimated_rows))
    log.info(f"{s3_key} sample_row_tokens={sample_row_tokens} header_tokens={header_tokens} rows_per_chunk={rows_per_chunk}")
    saved = 0
    ensure_ray()
    chunk_index = 0
    next_row_num = 1
    try:
        ds = ray.data.read_json(f"s3://{S3_BUCKET}/{s3_key}", file_extensions=["jsonl"])
        batch_iter = ds.iter_batches(batch_size=rows_per_chunk, batch_format="pandas", prefetch_batches=2)
        for batch in batch_iter:
            if not isinstance(batch, pd.DataFrame) or batch.shape[0] == 0:
                continue
            saved_batch, chunk_index, next_row_num = _process_batch_rows(batch.iterrows(), doc_id, s3_path, chunk_index, header_text, next_row_num)
            saved += saved_batch
    except Exception as e_ray:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            body_bytes = obj.get("Body").read()
            text = body_bytes.decode("utf-8", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            buffer: List[Dict[str, Any]] = []
            for ln in lines:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                buffer.append(rec)
                if len(buffer) >= rows_per_chunk:
                    indexed_iter = ((i, row) for i, row in enumerate(buffer))
                    saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, s3_path, chunk_index, header_text, next_row_num)
                    saved += saved_chunk
                    buffer = []
            if buffer:
                indexed_iter = ((i, row) for i, row in enumerate(buffer))
                saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, s3_path, chunk_index, header_text, next_row_num)
                saved += saved_chunk
        except Exception as e_pd:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.error(f"Skipping malformed or unreadable JSONL {s3_key} error_ray={str(e_ray)} error_pd={str(e_pd)}")
            return {
                "saved_chunks": 0,
                "total_parse_duration_ms": total_ms,
                "skipped": True,
                "error": f"{str(e_ray)} | {str(e_pd)}",
            }
    total_ms = int((time.perf_counter() - start_all) * 1000)
    log.info(f"{s3_key} sample_row_tokens={sample_row_tokens} header_tokens={header_tokens} rows_per_chunk={rows_per_chunk}")
    log.info(f"Completed parsing {saved} chunks for {s3_key} in {total_ms} ms total")
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}
