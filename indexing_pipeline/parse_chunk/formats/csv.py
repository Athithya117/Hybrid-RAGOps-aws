import os
import io
import json
import time
import logging
import hashlib
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Tuple, List
from tempfile import NamedTemporaryFile
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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("csv_parser")

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
STORE_ONE_FILE_PER_CHUNK = os.getenv("STORE_ONE_FILE_PER_CHUNK", "true").lower() == "true"

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
        df = pd.read_csv(stream, dtype=str, nrows=2, on_bad_lines="skip")
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

def _derive_source_key_from_path(s3_path: str) -> str:
    prefix = f"s3://{S3_BUCKET}/"
    if s3_path.startswith(prefix):
        return s3_path[len(prefix):]
    return ""

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

def write_or_append(out_key: str, payload: Dict[str, Any], combined_appender: LocalChunkAppender | None):
    if not STORE_ONE_FILE_PER_CHUNK:
        try:
            combined_appender.append(payload)
            logger.info("Appended chunk %s → s3://%s/%s", payload.get("chunk_id"), S3_BUCKET, os.path.basename(out_key))
        except Exception as e:
            logger.error("Failed to append payload %s: %s", payload.get("chunk_id"), e)
        return
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        logger.info("Skipping existing chunk %s", payload.get("chunk_id"))
        return
    body = ((json.dumps(payload, ensure_ascii=False) + "\n").encode() if CHUNK_FORMAT == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode())
    s3_put_object_with_retries(out_key, body)
    logger.info("Wrote chunk %s → %s", payload.get("chunk_id"), out_key)

def _flush_rows_chunk(doc_id: str, s3_path: str, chunk_index: int, header_text: str, rows_text: List[str], start_row_num: int, manifest_tags: List[str] | None, combined_appender: LocalChunkAppender | None) -> Tuple[int, int]:
    if not rows_text:
        return 0, chunk_index
    chunk_index += 1
    chunk_id = f"{doc_id}_{chunk_index}"
    t0_chunk = time.perf_counter()
    chunk_text = header_text + "\n" + "\n".join(rows_text) if header_text else "\n".join(rows_text)
    token_ct = token_count_for(chunk_text)
    end_row_num = start_row_num + len(rows_text) - 1
    payload: Dict[str, Any] = {
        "document_id": doc_id or "",
        "chunk_id": chunk_id or "",
        "chunk_type": "row_group",
        "text": canonicalize_text(chunk_text) or "",
        "token_count": int(token_ct or 0),
        "embedding": None,
        "file_type": "text/csv",
        "source_url": f"s3://{S3_BUCKET}/{s3_path}" if S3_BUCKET else None,
        "page_number": None,
        "slide_range": None,
        "row_range": [int(start_row_num), int(end_row_num)],
        "token_range": None,
        "audio_range": None,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "parser_version": PARSER_VERSION or "",
        "tags": manifest_tags or [],
        "layout_tags": [],
        "used_ocr": False,
        "parse_chunk_duration_ms": None,
        "heading_path": [],
        "headings": [],
        "line_range": None,
        "chunk_duration_ms": None,
    }
    out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ 'jsonl' if CHUNK_FORMAT == 'jsonl' else 'json'}"
    if STORE_ONE_FILE_PER_CHUNK:
        if not FORCE_OVERWRITE and s3_object_exists(out_key):
            t_after = time.perf_counter()
            parse_ms = int(round((t_after - t0_chunk) * 1000)) or 1
            payload["parse_chunk_duration_ms"] = parse_ms
            payload["chunk_duration_ms"] = parse_ms
            logger.info("Skipping existing chunk %s (assembly %d ms)", payload["chunk_id"], parse_ms)
            return 0, chunk_index
        t_put_start = time.perf_counter()
        write_or_append(out_key, payload, None)
        t_put_end = time.perf_counter()
        total_ms = int(round((t_put_end - t0_chunk) * 1000)) or 1
        payload["parse_chunk_duration_ms"] = total_ms
        payload["chunk_duration_ms"] = total_ms
        write_or_append(out_key, payload, None)
        logger.info("Wrote CSV chunk %s → %s (%d ms)", payload["chunk_id"], out_key, total_ms)
        return 1, chunk_index
    else:
        t_after = time.perf_counter()
        parse_ms = int(round((t_after - t0_chunk) * 1000)) or 1
        payload["parse_chunk_duration_ms"] = parse_ms
        payload["chunk_duration_ms"] = parse_ms
        write_or_append(out_key, payload, combined_appender)
        return 1, chunk_index

def _process_batch_rows(rows_iterable, doc_id, s3_path, chunk_index, header_text, next_row_num, manifest_tags: List[str] | None, combined_appender: LocalChunkAppender | None):
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
                wrote, chunk_index = _flush_rows_chunk(doc_id, s3_path, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags, combined_appender)
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
                    "page_number": None,
                    "slide_range": None,
                    "row_range": [int(row_num), int(row_num)],
                    "token_range": [int(w.get("token_start")), int(w.get("token_end"))],
                    "audio_range": None,
                    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "parser_version": PARSER_VERSION or "",
                    "tags": manifest_tags or [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "parse_chunk_duration_ms": None,
                    "heading_path": [],
                    "headings": [],
                    "line_range": None,
                    "chunk_duration_ms": None,
                }
                out_key = f"{S3_CHUNKED_PREFIX}{payload['chunk_id']}.{ 'jsonl' if CHUNK_FORMAT == 'jsonl' else 'json'}"
                t0_chunk = time.perf_counter()
                if STORE_ONE_FILE_PER_CHUNK:
                    if not FORCE_OVERWRITE and s3_object_exists(out_key):
                        t_after = time.perf_counter()
                        parse_ms = int(round((t_after - t0_chunk) * 1000)) or 1
                        payload["parse_chunk_duration_ms"] = parse_ms
                        logger.info("Skipping existing window chunk %s (assembly %d ms)", payload["chunk_id"], parse_ms)
                        continue
                    write_or_append(out_key, payload, None)
                    t_put_end = time.perf_counter()
                    total_ms = int(round((t_put_end - t0_chunk) * 1000)) or 1
                    payload["parse_chunk_duration_ms"] = total_ms
                    write_or_append(out_key, payload, None)
                    logger.info("Wrote CSV token window %s → %s (%d ms)", payload["chunk_id"], out_key, total_ms)
                    saved += 1
                else:
                    t_after = time.perf_counter()
                    parse_ms = int(round((t_after - t0_chunk) * 1000)) or 1
                    payload["parse_chunk_duration_ms"] = parse_ms
                    payload["chunk_duration_ms"] = parse_ms
                    write_or_append(out_key, payload, combined_appender)
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
            wrote, chunk_index = _flush_rows_chunk(doc_id, s3_path, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags, combined_appender)
            saved += wrote
            rows_text = [row_text]
            start_row_of_current = row_num
    if rows_text:
        wrote, chunk_index = _flush_rows_chunk(doc_id, s3_path, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags, combined_appender)
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
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    combined_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    combined_appender = None
    if not STORE_ONE_FILE_PER_CHUNK:
        if not FORCE_OVERWRITE and s3_object_exists(combined_key):
            logger.info("Skipping file %s because combined target exists and FORCE_OVERWRITE is false → %s", s3_key, combined_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": int((time.perf_counter() - start_all) * 1000)}
        combined_appender = LocalChunkAppender(ext, doc_id)
        logger.info("Using combined chunk file mode for %s → s3://%s/%s", doc_id, S3_BUCKET, combined_key)
    try:
        if ray is not None:
            ds = ray.data.read_csv(f"s3://{S3_BUCKET}/{s3_key}", file_extensions=["csv"])
            batch_iter = ds.iter_batches(batch_size=rows_per_chunk, batch_format="pandas", prefetch_batches=2)
            for batch in batch_iter:
                if pd is None or not isinstance(batch, pd.DataFrame) or batch.shape[0] == 0:
                    continue
                saved_batch, chunk_index, next_row_num = _process_batch_rows(batch.iterrows(), doc_id, s3_path, chunk_index, header_text, next_row_num, manifest_tags, combined_appender)
                saved += saved_batch
        else:
            raise Exception("ray-unavailable")
    except Exception:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            body_bytes = obj.get("Body").read()
            text = body_bytes.decode("utf-8", errors="replace")
            stream = io.StringIO(text)
            reader = pd.read_csv(stream, dtype=str, chunksize=rows_per_chunk, on_bad_lines="skip")
            for chunk in reader:
                if not isinstance(chunk, pd.DataFrame) or chunk.shape[0] == 0:
                    continue
                saved_chunk, chunk_index, next_row_num = _process_batch_rows(chunk.iterrows(), doc_id, s3_path, chunk_index, header_text, next_row_num, manifest_tags, combined_appender)
                saved += saved_chunk
        except Exception as e_pd:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.error("Skipping malformed or unreadable CSV %s error=%s | %s", s3_key, str(e_pd), "")
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_pd)}
    if not STORE_ONE_FILE_PER_CHUNK and combined_appender is not None:
        try:
            combined_appender.finalize_and_upload(S3_BUCKET, combined_key)
        except Exception as e:
            logger.error("Failed uploading combined file for %s: %s", doc_id, e)
    total_ms = int((time.perf_counter() - start_all) * 1000)
    logger.info("Completed parsing %d chunks for %s in %d ms total", saved, s3_key, total_ms)
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}
