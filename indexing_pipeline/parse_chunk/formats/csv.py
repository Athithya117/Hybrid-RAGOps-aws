import os
import io
import json
import time
import logging
import hashlib
import unicodedata
import tempfile
from datetime import datetime
from typing import Any, Dict, Iterator, Tuple, List, Optional
import boto3
import botocore
import urllib.parse
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
logger = logging.getLogger("csv_parser")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers[:] = [handler]
log = logger
S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
PARSER_VERSION = os.getenv("PARSER_VERSION_CSV", "ray-csv-firstrow-auto-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
TARGET_TOKENS_PER_CHUNK = int(os.getenv("CSV_TARGET_TOKENS_PER_CHUNK", "1000"))
ROWS_PER_CHUNK_OVERRIDE = os.getenv("CSV_ROWS_PER_CHUNK", "")
MIN_ROWS_PER_CHUNK = int(os.getenv("CSV_MIN_ROWS_PER_CHUNK", "1"))
MAX_ROWS_PER_CHUNK = int(os.getenv("CSV_MAX_ROWS_PER_CHUNK", "100"))
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
def s3_put_object_with_retries(key: str, body: bytes, content_type: str = "application/json") -> None:
    attempt = 0
    while True:
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)
            return
        except Exception:
            attempt += 1
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
    if not text:
        yield {"window_index": 0, "text": "", "token_count": 0, "token_start": 0, "token_end": 0}
        return
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
            yield {"window_index": idx, "text": canonicalize_text(window_text), "token_count": end - start, "token_start": start, "token_end": end}
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
        slice_tokens = tokens[start:end]
        window_text = ENCODER.decode(slice_tokens)
        yield {"window_index": idx, "text": canonicalize_text(window_text), "token_count": len(slice_tokens), "token_start": start, "token_end": min(end, total)}
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
    except Exception:
        ray.init(ignore_reinit_error=True, include_dashboard=False, configure_logging=False)
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
        stream = io.StringIO(text)
        if pd is None:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return "", 32
            header = lines[0]
            sample_line = lines[1] if len(lines) > 1 else ""
            header_cols = [c.strip() for c in header.split(",")] if header else []
            header_text = canonicalize_text(" | ".join(header_cols))
            sample_tokens = max(1, token_count_for(canonicalize_text(sample_line)))
            return header_text, sample_tokens
        else:
            try:
                df = pd.read_csv(stream, dtype=str, nrows=2, on_bad_lines="skip")
            except TypeError:
                df = pd.read_csv(stream, dtype=str, nrows=2)
            if df.shape[1] == 0 and df.shape[0] == 0:
                return "", 32
            header_cols = list(df.columns)
            header_text = canonicalize_text(" | ".join(header_cols))
            sample_row_index = 1 if df.shape[0] > 1 else 0
            if df.shape[0] == 0:
                sample_tokens = 32
            else:
                sample_series = df.iloc[sample_row_index]
                sample_text = row_to_schema_text(sample_series)
                sample_tokens = max(1, token_count_for(sample_text))
            return header_text, sample_tokens
    except Exception:
        return "", 32
def make_doc_id(s3_key: str, last_modified: Any) -> str:
    return sha256_hex(s3_key + str(last_modified or ""))
def filename_from_source_url(source_url: Optional[str]) -> str:
    if not source_url:
        return ""
    try:
        if source_url.startswith("s3://"):
            return os.path.basename(source_url)
        parsed = urllib.parse.urlparse(source_url)
        if parsed.path:
            return os.path.basename(parsed.path)
        return os.path.basename(source_url)
    except Exception:
        return os.path.basename(str(source_url))
class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path
        self.ext = ext
        self.content_type = content_type
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False, dir="/tmp", prefix=f"{doc_id}_", suffix=f".{ext}")
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
            s3.upload_file(self.temp.name, S3_BUCKET, out_key, ExtraArgs={"ContentType": "application/json"})
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
def _flush_rows_chunk(writer: S3DocWriter, doc_id: str, chunk_index: int, header_text: str, rows_text: List[str], start_row_num: int, manifest_tags: List[str] = None) -> Tuple[int, int]:
    if not rows_text:
        return 0, chunk_index
    chunk_index += 1
    chunk_id = f"{doc_id}_{chunk_index}"
    chunk_text = header_text + "\n" + "\n".join(rows_text) if header_text else "\n".join(rows_text)
    token_ct = token_count_for(chunk_text)
    end_row_num = start_row_num + len(rows_text) - 1
    source_url = f"s3://{S3_BUCKET}/{writer.s3_path}" if S3_BUCKET else None
    payload: Dict[str, Any] = {
        "document_id": doc_id or "",
        "chunk_id": chunk_id or "",
        "chunk_type": "row_group",
        "text": canonicalize_text(chunk_text) or "",
        "token_count": int(token_ct or 0),
        "embedding": None,
        "file_type": "text/csv",
        "source_url": source_url,
        "file_name": filename_from_source_url(source_url) if source_url else "",
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
    sanitize_payload_for_weaviate(payload)
    writer.write_payload(payload)
    log.info(f"Buffered CSV row_group chunk {payload.get('chunk_id')}")
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
                payload: Dict[str, Any] = {
                    "document_id": doc_id or "",
                    "chunk_id": chunk_id or "",
                    "chunk_type": "token_window",
                    "text": canonicalize_text(candidate_text) or "",
                    "token_count": int(token_ct or 0),
                    "embedding": None,
                    "file_type": "text/csv",
                    "source_url": f"s3://{S3_BUCKET}/{s3_path}" if S3_BUCKET else None,
                    "file_name": filename_from_source_url(f"s3://{S3_BUCKET}/{s3_path}") if S3_BUCKET else "",
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
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                log.info(f"Buffered CSV token_window {payload.get('chunk_id')}")
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
        logger.error("Could not head S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": str(e)}
    last_modified = head_obj.get("LastModified", "")
    doc_id = manifest.get("file_hash") or make_doc_id(s3_key, last_modified)
    s3_path = f"{s3_key}"
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Skipping entire file because chunked file exists: %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    header_text, sample_row_tokens = get_header_and_sample_tokens(s3_key)
    header_tokens = token_count_for(header_text) if header_text else 0
    if header_tokens >= TARGET_TOKENS_PER_CHUNK:
        logger.warning("CSV header token count >= target chunk size. Header will not be prepended to row_group chunks.")
        header_text = ""
        header_tokens = 0
    if ROWS_PER_CHUNK_OVERRIDE:
        rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, int(ROWS_PER_CHUNK_OVERRIDE)))
    else:
        available_for_rows = max(1, TARGET_TOKENS_PER_CHUNK - header_tokens)
        estimated_rows = max(1, int(available_for_rows / max(1, sample_row_tokens)))
        rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, estimated_rows))
    logger.info("%s sample_row_tokens=%d header_tokens=%d rows_per_chunk=%d", s3_key, sample_row_tokens, header_tokens, rows_per_chunk)
    saved = 0
    ensure_ray()
    chunk_index = 0
    next_row_num = 1
    manifest_tags = manifest.get("tags", []) if isinstance(manifest, dict) else []
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_path, ext=ext)
    try:
        if ray is not None:
            ds = ray.data.read_csv(f"s3://{S3_BUCKET}/{s3_key}", file_extensions=["csv"])
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
            body = obj.get("Body")
            if pd is not None:
                try:
                    text_stream = io.TextIOWrapper(body, encoding="utf-8", errors="replace", newline="")
                    reader = pd.read_csv(text_stream, dtype=str, chunksize=rows_per_chunk, on_bad_lines="skip")
                except TypeError:
                    text_stream = io.TextIOWrapper(body, encoding="utf-8", errors="replace", newline="")
                    reader = pd.read_csv(text_stream, dtype=str, chunksize=rows_per_chunk)
                for chunk in reader:
                    if not isinstance(chunk, pd.DataFrame) or chunk.shape[0] == 0:
                        continue
                    saved_chunk, chunk_index, next_row_num = _process_batch_rows(chunk.iterrows(), doc_id, s3_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                    saved += saved_chunk
            else:
                body_bytes = body.read()
                text = body_bytes.decode("utf-8", errors="replace")
                stream = io.StringIO(text)
                reader = csv_reader_fallback(stream)
                buffer = []
                for row in reader:
                    buffer.append(row)
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
            logger.error("Skipping malformed or unreadable CSV %s error=%s", s3_key, str(e_pd))
            try:
                if writer and writer.temp:
                    try:
                        os.unlink(writer.temp.name)
                    except Exception:
                        pass
            except Exception:
                pass
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_pd)}
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
        logger.error("Failed to upload chunked file for %s error=%s", s3_key, str(e_up))
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}
def csv_reader_fallback(text_stream):
    import csv
    reader = csv.DictReader(text_stream)
    for row in reader:
        yield row
if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.lower().endswith(".csv") or key.lower().endswith(".tsv") or key.lower().endswith(".txt")):
                continue
            log.info("Routing parse_file for s3://%s/%s", S3_BUCKET, key)
            manifest_key = key + ".manifest.json"
            try:
                mf_obj = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf_obj["Body"])
            except Exception:
                manifest = {}
            try:
                parse_file(key, manifest)
            except Exception as e:
                log.exception("Failed to parse %s: %s", key, e)
