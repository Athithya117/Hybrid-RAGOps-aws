import os
import io
import json
import time
import logging
import hashlib
import unicodedata
import tempfile
from datetime import datetime
from typing import Any, Dict, Iterator, Tuple, List
import boto3
import botocore

try:
    import ray
except Exception:
    ray = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import tiktoken
except Exception:
    tiktoken = None

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
ENCODER = None
if tiktoken is not None:
    try:
        ENCODER = tiktoken.get_encoding(ENC_NAME)
    except Exception:
        ENCODER = None

_RAY_CONNECTED = False

def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def canonicalize_text(s: Any) -> str:
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
    while True:
        try:
            attempt += 1
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)
            ms = int(round((time.perf_counter() - t_start) * 1000))
            if ms == 0:
                ms = 1
            return attempt, ms
        except Exception:
            if attempt >= max(1, S3_PUT_RETRIES):
                raise
            time.sleep(S3_PUT_BACKOFF * attempt)

def token_count_for(text: str) -> int:
    if not text:
        return 0
    if ENCODER is not None:
        try:
            return len(ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split())

def split_into_token_windows(text: str, window_tokens: int, overlap: int = 0) -> Iterator[Dict[str, Any]]:
    if ENCODER is None:
        tokens = text.split()
        total = len(tokens)
        if window_tokens <= overlap:
            raise ValueError("window_tokens must be greater than overlap")
        step = window_tokens - overlap
        idx = 0
        for start in range(0, total, step):
            end = min(start + window_tokens, total)
            window_text = " ".join(tokens[start:end])
            yield {
                "window_index": idx,
                "text": canonicalize_text(window_text),
                "token_count": len(tokens[start:end]),
                "token_start": start,
                "token_end": end,
            }
            idx += 1
            if end >= total:
                break
        return
    tokens = ENCODER.encode(text)
    total = len(tokens)
    if window_tokens <= overlap:
        raise ValueError("window_tokens must be greater than overlap")
    step = window_tokens - overlap
    idx = 0
    for start in range(0, total, step):
        end = start + window_tokens
        window_slice = tokens[start:end]
        window_text = ENCODER.decode(window_slice)
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
    if pd is not None and isinstance(row, pd.Series):
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
    if ray is None:
        return
    addr = os.getenv("RAY_ADDRESS", "auto")
    try:
        ray.init(address=addr, ignore_reinit_error=True)
        log.info(f"Connected to Ray via address={addr}")
    except Exception:
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False, configure_logging=False)
            log.info("Started local Ray instance")
        except Exception:
            pass
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
        sample_obj = parsed[0]
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

class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        self.count = 0
        self._first = True
        if self.ext == "json":
            self.temp.write(b"[")
    def write_payload(self, payload: Dict[str, Any]) -> int:
        self.count += 1
        if self.ext == "jsonl":
            line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.temp.write(line)
        else:
            j = json.dumps(payload, ensure_ascii=False)
            if not self._first:
                self.temp.write(b",")
            self.temp.write(j.encode("utf-8"))
            self._first = False
        return 1
    def finalize_and_upload(self, out_key: str) -> Tuple[int, str]:
        if self.ext == "json":
            self.temp.write(b"]")
        self.temp.flush()
        self.temp.close()
        try:
            s3.upload_file(self.temp.name, S3_BUCKET, out_key)
            os.unlink(self.temp.name)
            return self.count, out_key
        except Exception:
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            raise

def _flush_rows_chunk(writer: S3DocWriter, doc_id: str, chunk_index: int, header_text: str, rows_text: List[str], start_row_num: int, manifest_tags: List[str] = None) -> Tuple[int, int]:
    if not rows_text:
        return 0, chunk_index
    chunk_index += 1
    chunk_id = f"{doc_id}_{chunk_index}"
    chunk_text = header_text + "\n" + "\n".join(rows_text) if header_text else "\n".join(rows_text)
    token_ct = token_count_for(chunk_text)
    end_row_num = start_row_num + len(rows_text) - 1
    payload = {
        "document_id": doc_id or "",
        "chunk_id": chunk_id or "",
        "chunk_type": "row_group",
        "text": canonicalize_text(chunk_text) or "",
        "token_count": int(token_ct or 0),
        "embedding": None,
        "file_type": "application/x-ndjson",
        "source_url": f"s3://{S3_BUCKET}/{writer.s3_path}",
        "page_number": None,
        "slide_range": None,
        "row_range": [int(start_row_num), int(end_row_num)],
        "token_range": None,
        "audio_range": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parser_version": PARSER_VERSION or "",
        "tags": manifest_tags or [],
        "layout_tags": [],
        "used_ocr": False,
        "heading_path": [],
        "headings": [],
        "line_range": None,
    }
    writer.write_payload(payload)
    log.info(f"Buffered chunk {payload['chunk_id']}")
    return 1, chunk_index

def _process_batch_rows(rows_iterable, doc_id, s3_path, chunk_index, header_text, next_row_num, writer: S3DocWriter, manifest_tags: List[str] = None):
    saved = 0
    rows_text: List[str] = []
    start_row_of_current = next_row_num
    for _, row in rows_iterable:
        if pd is not None and isinstance(row, pd.Series):
            row_text = row_to_schema_text(row)
        elif isinstance(row, dict):
            row_text = canonicalize_text(" | ".join([f"{k}: {v}" for k, v in row.items()]))
        else:
            row_text = canonicalize_text(str(row))
        row_num = next_row_num
        next_row_num += 1
        row_tokens = token_count_for(row_text)
        header_tokens = token_count_for(header_text) if header_text else 0
        if row_tokens > TARGET_TOKENS_PER_CHUNK:
            if rows_text:
                wrote, chunk_index = _flush_rows_chunk(writer, doc_id, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags)
                saved += wrote
                rows_text = []
            windows = list(split_into_token_windows(row_text, TARGET_TOKENS_PER_CHUNK, overlap=int(TARGET_TOKENS_PER_CHUNK * 0.1)))
            for w in windows:
                chunk_index += 1
                chunk_id = f"{doc_id}_{chunk_index}"
                candidate_text = header_text + "\n" + w["text"] if header_text and (header_tokens + w["token_count"] <= TARGET_TOKENS_PER_CHUNK) else w["text"]
                token_ct = token_count_for(candidate_text)
                payload = {
                    "document_id": doc_id or "",
                    "chunk_id": chunk_id or "",
                    "chunk_type": "token_window",
                    "text": canonicalize_text(candidate_text) or "",
                    "token_count": int(token_ct or 0),
                    "embedding": None,
                    "file_type": "application/x-ndjson",
                    "source_url": f"s3://{S3_BUCKET}/{s3_path}",
                    "page_number": None,
                    "slide_range": None,
                    "row_range": [int(row_num), int(row_num)],
                    "token_range": [int(w.get("token_start")), int(w.get("token_end"))],
                    "audio_range": None,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION or "",
                    "tags": manifest_tags or [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": [],
                    "line_range": None,
                }
                writer.write_payload(payload)
                log.info(f"Buffered token_window {payload['chunk_id']}")
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
            wrote, chunk_index = _flush_rows_chunk(writer, doc_id, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags)
            saved += wrote
            rows_text = [row_text]
            start_row_of_current = row_num
    if rows_text:
        wrote, chunk_index = _flush_rows_chunk(writer, doc_id, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags)
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
    s3_path = f"{s3_key}"
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info(f"Skipping entire file because chunked file exists: {out_key}")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
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
    manifest_tags = manifest.get("tags", []) if isinstance(manifest, dict) else []
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_path, ext=ext)
    try:
        if ray is not None:
            ds = ray.data.read_json(f"s3://{S3_BUCKET}/{s3_key}", file_extensions=["jsonl"])
            batch_iter = ds.iter_batches(batch_size=rows_per_chunk, batch_format="pandas", prefetch_batches=2)
            for batch in batch_iter:
                if pd is None or not isinstance(batch, pd.DataFrame) or batch.shape[0] == 0:
                    continue
                saved_batch, chunk_index, next_row_num = _process_batch_rows(batch.iterrows(), doc_id, s3_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                saved += saved_batch
        else:
            raise Exception("ray-unavailable")
    except Exception:
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
                    saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, s3_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                    saved += saved_chunk
                    buffer = []
            if buffer:
                indexed_iter = ((i, row) for i, row in enumerate(buffer))
                saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, s3_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                saved += saved_chunk
        except Exception as e_pd:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.error(f"Skipping malformed or unreadable JSONL {s3_key} error={str(e_pd)}")
            try:
                if writer and writer.temp:
                    try:
                        os.unlink(writer.temp.name)
                    except Exception:
                        pass
            except Exception:
                pass
            return {
                "saved_chunks": 0,
                "total_parse_duration_ms": total_ms,
                "skipped": True,
                "error": str(e_pd),
            }
    try:
        if saved == 0:
            try:
                if writer and writer.temp:
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info(f"No chunks produced for {s3_key}")
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": False}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info(f"Wrote {count} chunks for {s3_key} → {uploaded_key} ({total_ms} ms)")
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error(f"Failed to upload chunked file for {s3_key} error={str(e_up)}")
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}
