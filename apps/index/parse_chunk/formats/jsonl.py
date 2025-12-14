#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import io
import json
import time
import hashlib
import tempfile
import unicodedata
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Iterator, Tuple, List, Optional

# ---------- small logger shim ----------
class LoggerShim:
    def __init__(self, name: str):
        self.name = name

    def _emit(self, level: str, event: str, msg: str = "", **extra):
        out = {"ts": datetime.utcnow().isoformat() + "Z", "level": level, "event": event, "msg": msg}
        if extra:
            out.update(extra)
        text = json.dumps(out, ensure_ascii=False)
        if level in ("error", "warn", "warning", "exception"):
            print(text, file=sys.stderr, flush=True)
        else:
            print(text, flush=True)

    def _unpack(self, a, b, fmt_args, kwargs, default_event):
        if b is None:
            event = kwargs.pop("event", default_event)
            msg = a
        else:
            event = a
            msg = b
        if fmt_args:
            try:
                msg = msg % fmt_args
            except Exception:
                try:
                    msg = msg.format(*fmt_args)
                except Exception:
                    pass
        return event, msg, kwargs

    def info(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "info")
        self._emit("info", event, msg, **kw)

    def warning(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "warn")
        self._emit("warn", event, msg, **kw)

    def warn(self, a, b=None, *fmt_args, **kwargs):
        self.warning(a, b, *fmt_args, **kwargs)

    def error(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "error")
        self._emit("error", event, msg, **kw)

    def exception(self, a, b=None, *fmt_args, **kwargs):
        tb = traceback.format_exc()
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "exception")
        kw.update({"traceback": tb})
        self._emit("error", event, msg, **kw)

log = LoggerShim("jsonl_parser")

# ---------- env helpers (defensive) ----------
def parse_int_env(name: str, default: int) -> int:
    v = os.getenv(name, "")
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def parse_float_env(name: str, default: float) -> float:
    v = os.getenv(name, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

# ---------- config ----------
USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", "").strip().lower() in ("1", "true", "yes")

AZURE_CONTAINER = os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
if not AZURE_CONTAINER:
    log.error("startup_missing_container", "AZURE_CONTAINER (or STORAGE_CONTAINER or AZ_CONTAINER) must be set")
    sys.exit(1)

STORAGE_RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
STORAGE_CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"
PARSER_VERSION = os.getenv("PARSER_VERSION_JSONL", "polars-jsonl-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
TARGET_TOKENS_PER_CHUNK = parse_int_env("JSONL_TARGET_TOKENS_PER_CHUNK", parse_int_env("CSV_TARGET_TOKENS_PER_CHUNK", 1000))
ROWS_PER_CHUNK_OVERRIDE = os.getenv("JSONL_ROWS_PER_CHUNK", os.getenv("CSV_ROWS_PER_CHUNK", ""))
MIN_ROWS_PER_CHUNK = parse_int_env("JSONL_MIN_ROWS_PER_CHUNK", parse_int_env("CSV_MIN_ROWS_PER_CHUNK", 1))
MAX_ROWS_PER_CHUNK = parse_int_env("JSONL_MAX_ROWS_PER_CHUNK", parse_int_env("CSV_MAX_ROWS_PER_CHUNK", 100))
PUT_RETRIES = parse_int_env("PUT_RETRIES", 3)
PUT_BACKOFF = parse_float_env("PUT_BACKOFF", 0.5)
RANGE_BYTES = parse_int_env("RANGE_BYTES", 131072)

# ---------- optional libraries ----------
try:
    import fsspec
    from fsspec.spec import AbstractFileSystem
except Exception:
    fsspec = None  # type: ignore
    AbstractFileSystem = object  # type: ignore

try:
    import polars as pl
except Exception:
    pl = None

_tiktoken = None
try:
    import tiktoken as _tiktoken
except Exception:
    _tiktoken = None

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.storage.blob import BlobServiceClient, ContainerClient  # type: ignore
    AZURE_SDK_AVAILABLE = True
except Exception:
    DefaultAzureCredential = None  # type: ignore
    BlobServiceClient = None  # type: ignore
    ContainerClient = None  # type: ignore
    AZURE_SDK_AVAILABLE = False

# ---------- validation ----------
def fail(msg: str, code: int = 2):
    log.error("fatal", msg)
    sys.stderr.write(msg + "\n")
    sys.exit(code)

def validate_env_and_libs():
    if USE_MANAGED_IDENTITY:
        if not AZURE_SDK_AVAILABLE:
            fail("PROD/Managed-Identity mode requires 'azure-identity' and 'azure-storage-blob' (pip install azure-identity azure-storage-blob)")
        if not (os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")):
            fail("AZURE_STORAGE_ACCOUNT_NAME (or AZURE_ACCOUNT_NAME) must be set for managed identity mode")
    else:
        if fsspec is None:
            fail("STAGING/non-managed mode requires 'fsspec' and 'adlfs' (pip install fsspec adlfs)")
        if not (os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or os.getenv("AZURE_SAS_TOKEN") or os.getenv("AZURE_ANON")):
            fail("non-managed identity mode requires AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_KEY or AZURE_SAS_TOKEN or AZURE_ANON")

validate_env_and_libs()

def build_storage_options() -> Dict[str, Any]:
    if USE_MANAGED_IDENTITY:
        return {}
    opts: Dict[str, Any] = {}
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if conn:
        opts["connection_string"] = conn
        return opts
    acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
    sas = os.environ.get("AZURE_SAS_TOKEN")
    eps = os.environ.get("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"
    if acct and key:
        opts["account_name"] = acct
        opts["account_key"] = key
        opts["endpoint_suffix"] = eps
        return opts
    if acct and sas:
        opts["account_name"] = acct
        opts["sas_token"] = sas
        opts["endpoint_suffix"] = eps
        return opts
    if os.environ.get("AZURE_ANON"):
        if acct:
            opts["account_name"] = acct
        opts["anon"] = True
        return opts
    return opts

FS_OPTS = build_storage_options()

# ---------- storage client bootstrap ----------
BLOB_SERVICE_CLIENT = None
if USE_MANAGED_IDENTITY:
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
    endpoint_suffix = os.environ.get("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
    account_url = f"https://{account_name}.{endpoint_suffix}"
    try:
        mi_client_id = os.getenv("UAI_RAG_RW_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
        if mi_client_id and DefaultAzureCredential is not None:
            CREDENTIAL = DefaultAzureCredential(managed_identity_client_id=mi_client_id)
        elif DefaultAzureCredential is not None:
            CREDENTIAL = DefaultAzureCredential()
        else:
            CREDENTIAL = None
        if BlobServiceClient is not None and CREDENTIAL is not None:
            BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=account_url, credential=CREDENTIAL, connection_timeout=60)
            try:
                container_client = BLOB_SERVICE_CLIENT.get_container_client(AZURE_CONTAINER)
                try:
                    container_client.get_container_properties()
                except Exception as e_smoke:
                    log.warning("mi_smoke", "managed identity client created, but smoke-check failed (may be normal in restricted env)", error=str(e_smoke))
            except Exception:
                pass
    except Exception as e:
        fail(f"Failed to initialize BlobServiceClient with managed identity: {e}")
    FS: Optional[AbstractFileSystem] = None
else:
    try:
        FS = fsspec.filesystem("az", **FS_OPTS)  # type: ignore
    except Exception as e:
        fail(f"Failed to initialize fsspec 'az' filesystem: {e}")
    BLOB_SERVICE_CLIENT = None

STORAGE_ROOT = f"az://{AZURE_CONTAINER.rstrip('/')}/"

def full_path_from_key(key: str) -> str:
    return STORAGE_ROOT + key.lstrip("/")

def strip_root_from_path(full: str) -> str:
    if full.startswith(STORAGE_ROOT):
        return full[len(STORAGE_ROOT):]
    proto_prefix = "az://"
    if full.startswith(proto_prefix):
        rest = full[len(proto_prefix):]
        if rest.startswith(AZURE_CONTAINER + "/"):
            return rest[len(AZURE_CONTAINER) + 1:]
        if rest == AZURE_CONTAINER:
            return ""
    if full.startswith(AZURE_CONTAINER + "/"):
        return full[len(AZURE_CONTAINER) + 1:]
    return full

def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canonicalize_text(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split()).strip()

# ---------- encoder ----------
ENCODER = None
if _tiktoken is not None:
    try:
        ENCODER = _tiktoken.get_encoding(ENC_NAME)
    except Exception:
        try:
            ENCODER = _tiktoken.encoding_for_model("gpt2")
        except Exception:
            ENCODER = None

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
        try:
            window_text = ENCODER.decode(slice_tokens)
        except Exception:
            window_text = " ".join(str(x) for x in slice_tokens)
        yield {"window_index": idx, "text": canonicalize_text(window_text), "token_count": len(slice_tokens), "token_start": start, "token_end": min(end, total)}
        idx += 1
        if end >= total:
            break

def row_to_schema_text(row: Any) -> str:
    parts: List[str] = []
    if pl is not None and hasattr(pl, "Series") and isinstance(row, pl.Series):
        try:
            vals = row.to_list()
            cols = getattr(row, "_dtype_names", None) or []
            for i, v in enumerate(vals):
                name = cols[i] if i < len(cols) else str(i)
                parts.append(f"{name}: {'' if v is None else v}")
        except Exception:
            parts.append(str(row))
    elif isinstance(row, dict):
        for k, v in row.items():
            parts.append(f"{k}: {'' if v is None else v}")
    else:
        parts.append(str(row))
    return canonicalize_text(" | ".join(parts))

# ---------- memory heuristics ----------
def detect_total_memory_bytes() -> int:
    try:
        path_v2 = "/sys/fs/cgroup/memory.max"
        if os.path.exists(path_v2):
            with open(path_v2, "r") as f:
                val = f.read().strip()
                if val.isdigit():
                    v = int(val)
                    if v > 0 and v < 2 ** 60:
                        return v
        path_v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(path_v1):
            with open(path_v1, "r") as f:
                v = int(f.read().strip())
                if v > 0:
                    return v
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return 512 * (1024 ** 2)

def compute_streaming_chunk_size() -> int:
    total = detect_total_memory_bytes()
    size = max(32_000_000, min(256_000_000, max(16_000_000, int(total // 8))))
    return int(size)

try:
    if pl is not None and hasattr(pl, "Config"):
        pl.Config.set_streaming_chunk_size(compute_streaming_chunk_size())
except Exception:
    pass

# ---------- storage client wrapper ----------
class AzureStorageClient:
    def __init__(self, fs_obj: Optional[AbstractFileSystem], root: str, container: str, blob_service_client=None):
        self.fs = fs_obj
        self.root = root
        self.container = container
        self.blob_service_client = blob_service_client

    def _container_client(self) -> "ContainerClient":
        if self.blob_service_client is None:
            raise RuntimeError("blob_service_client not initialized for managed-identity mode")
        return self.blob_service_client.get_container_client(self.container)

    def exists(self, Bucket, Key) -> bool:
        try:
            if self.fs is not None:
                return self.fs.exists(full_path_from_key(Key))
            else:
                blob_client = self._container_client().get_blob_client(Key)
                return blob_client.exists()
        except Exception:
            return False

    def head_object(self, Bucket, Key):
        if self.fs is not None:
            full = full_path_from_key(Key)
            info = self.fs.info(full)
            out = {}
            out["ContentLength"] = int(info.get("size", 0))
            etag = info.get("etag") or info.get("ETag") or ""
            out["ETag"] = etag
            lm = info.get("Last-Modified") or info.get("last_modified") or info.get("LastModified") or ""
            out["LastModified"] = lm
            metadata = info.get("metadata") or info.get("meta") or {}
            out["Metadata"] = metadata
            return out
        else:
            container = self._container_client()
            blob_client = container.get_blob_client(Key)
            props = blob_client.get_blob_properties()
            out = {
                "ContentLength": getattr(props, "size", 0),
                "ETag": getattr(props, "etag", ""),
                "LastModified": getattr(props, "last_modified", ""),
                "Metadata": getattr(props, "metadata", {}) or {},
            }
            return out

    def get_object(self, Bucket, Key):
        if self.fs is not None:
            full = full_path_from_key(Key)
            with self.fs.open(full, "rb") as f:
                data = f.read()
            return {"Body": io.BytesIO(data)}
        else:
            container = self._container_client()
            blob_client = container.get_blob_client(Key)
            stream = blob_client.download_blob()
            data = stream.readall()
            return {"Body": io.BytesIO(data)}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        if self.fs is not None:
            full = full_path_from_key(Key)
            if isinstance(Body, (bytes, bytearray)):
                b = bytes(Body)
            elif isinstance(Body, str):
                b = Body.encode("utf-8")
            elif hasattr(Body, "read"):
                b = Body.read()
                if isinstance(b, str):
                    b = b.encode("utf-8")
            else:
                b = bytes(Body)
            with self.fs.open(full, "wb") as f:
                f.write(b)
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}
        else:
            container = self._container_client()
            blob_client = container.get_blob_client(Key)
            if isinstance(Body, (bytes, bytearray)):
                data = Body
            elif isinstance(Body, str):
                data = Body.encode("utf-8")
            elif hasattr(Body, "read"):
                data = Body.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
            else:
                data = str(Body).encode("utf-8")
            blob_client.upload_blob(data, overwrite=True)
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def upload_file(self, LocalFile, Bucket, Key, ExtraArgs=None):
        if self.fs is not None:
            full = full_path_from_key(Key)
            if hasattr(self.fs, "put"):
                self.fs.put(LocalFile, full)
            else:
                with open(LocalFile, "rb") as lf:
                    data = lf.read()
                with self.fs.open(full, "wb") as f:
                    f.write(data)
            return
        else:
            container = self._container_client()
            blob_client = container.get_blob_client(Key)
            with open(LocalFile, "rb") as lf:
                blob_client.upload_blob(lf, overwrite=True)

    def copy_object(self, CopySource, Bucket, Key):
        src = CopySource.get("Key")
        if self.fs is not None:
            full_src = full_path_from_key(src)
            full_dst = full_path_from_key(Key)
            with self.fs.open(full_src, "rb") as rf:
                data = rf.read()
            with self.fs.open(full_dst, "wb") as wf:
                wf.write(data)
            return
        else:
            src_blob_client = self._container_client().get_blob_client(src)
            dst_blob_client = self._container_client().get_blob_client(Key)
            src_url = src_blob_client.url
            dst_blob_client.start_copy_from_url(src_url)

    def delete_object(self, Bucket, Key):
        if self.fs is not None:
            full = full_path_from_key(Key)
            try:
                self.fs.rm(full)
            except Exception:
                try:
                    self.fs.delete(full)
                except Exception:
                    pass
            return
        else:
            blob_client = self._container_client().get_blob_client(Key)
            try:
                blob_client.delete_blob()
            except Exception:
                pass

    def get_paginator(self, name):
        if self.fs is not None:
            class P:
                def __init__(self, fs, root):
                    self.fs = fs
                    self.root = root
                def paginate(self, Bucket, Prefix, PaginationConfig=None):
                    base = (Prefix.rstrip("/")) + "/"
                    root_path = self.root + base
                    try:
                        if hasattr(self.fs, "find"):
                            found = self.fs.find(root_path)
                        else:
                            found = self.fs.glob(root_path + "**", recursive=True)
                    except Exception:
                        found = []
                    page = {"Contents": []}
                    for f in found:
                        try:
                            info = self.fs.info(f)
                        except Exception:
                            continue
                        if info.get("type") == "directory":
                            continue
                        rel = strip_root_from_path(f)
                        page["Contents"].append({"Key": rel})
                        if len(page["Contents"]) >= 1000:
                            yield page
                            page = {"Contents": []}
                    if page["Contents"]:
                        yield page
            return P(self.fs, self.root)
        else:
            class Pblob:
                def __init__(self, container_client):
                    self.container_client = container_client
                def paginate(self, Bucket, Prefix, PaginationConfig=None):
                    blobs = self.container_client.list_blobs(name_starts_with=Prefix)
                    page = {"Contents": []}
                    for b in blobs:
                        page["Contents"].append({"Key": b.name})
                        if len(page["Contents"]) >= 1000:
                            yield page
                            page = {"Contents": []}
                    if page["Contents"]:
                        yield page
            return Pblob(self._container_client())

# singleton
_storage_client: Optional[AzureStorageClient] = None
_storage_lock = threading.Lock()

def get_storage_client_singleton():
    global _storage_client
    if _storage_client is None:
        with _storage_lock:
            if _storage_client is None:
                if USE_MANAGED_IDENTITY:
                    _storage_client = AzureStorageClient(None, STORAGE_ROOT, AZURE_CONTAINER, blob_service_client=BLOB_SERVICE_CLIENT)
                else:
                    _storage_client = AzureStorageClient(FS, STORAGE_ROOT, AZURE_CONTAINER, blob_service_client=None)
    return _storage_client

def storage_upload_file_atomic(local_path: str, key: str, content_type: str = "application/octet-stream"):
    full = full_path_from_key(key)
    tmp = f"{full}.tmp.{os.getpid()}.{int(time.time())}"
    client = get_storage_client_singleton()
    for attempt in range(1, PUT_RETRIES + 1):
        try:
            if client.fs is not None:
                if hasattr(client.fs, "put"):
                    client.fs.put(local_path, tmp)
                else:
                    with open(local_path, "rb") as lf:
                        d = lf.read()
                    with client.fs.open(tmp, "wb") as f:
                        f.write(d)
                if hasattr(client.fs, "mv"):
                    client.fs.mv(tmp, full)
                else:
                    with client.fs.open(tmp, "rb") as rf:
                        data = rf.read()
                    with client.fs.open(full, "wb") as wf:
                        wf.write(data)
                    try:
                        client.fs.rm(tmp)
                    except Exception:
                        pass
                return
            else:
                client.upload_file(local_path, AZURE_CONTAINER, key)
                return
        except Exception as e:
            log.warning("upload_retry", "attempt=%d key=%s error=%s", attempt, key, str(e))
            time.sleep(PUT_BACKOFF * attempt)
    raise Exception(f"atomic upload failed for {key} after {PUT_RETRIES} attempts")

# ---------- parquet writer ----------
PA_AVAILABLE = False
_pa = None
_pq = None

def _ensure_pyarrow():
    global PA_AVAILABLE, _pa, _pq
    if PA_AVAILABLE:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        _pa = pa
        _pq = pq
        PA_AVAILABLE = True
    except Exception:
        PA_AVAILABLE = False
        _pa = None
        _pq = None

class ParquetWriter:
    def __init__(self, doc_id: str, source_path: str):
        self.doc_id = doc_id
        self.source_path = source_path
        self._rows: List[Dict[str, Any]] = []

    def _normalize_for_parquet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        fields["document_id"] = payload.get("document_id") or ""
        fields["chunk_id"] = payload.get("chunk_id") or ""
        fields["chunk_type"] = payload.get("chunk_type") or ""
        fields["text"] = str(payload.get("text") or "")
        try:
            fields["token_count"] = int(payload.get("token_count") or 0)
        except Exception:
            fields["token_count"] = 0
        for k in ("figures", "tags", "layout_tags", "heading_path", "headings"):
            v = payload.get(k, None)
            try:
                fields[k] = json.dumps(v, ensure_ascii=False, sort_keys=True) if v is not None else "[]"
            except Exception:
                try:
                    fields[k] = json.dumps([], ensure_ascii=False)
                except Exception:
                    fields[k] = "[]"
        fields["file_type"] = payload.get("file_type") or ""
        fields["source_url"] = payload.get("source_url") or ""
        fields["file_name"] = payload.get("file_name") or ""
        rr = payload.get("row_range")
        if isinstance(rr, (list, tuple)) and len(rr) >= 2:
            try:
                fields["row_start"] = int(rr[0])
                fields["row_end"] = int(rr[1])
            except Exception:
                fields["row_start"] = None
                fields["row_end"] = None
        else:
            fields["row_start"] = None
            fields["row_end"] = None
        tr = payload.get("token_range")
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            try:
                fields["token_start"] = int(tr[0])
                fields["token_end"] = int(tr[1])
            except Exception:
                fields["token_start"] = None
                fields["token_end"] = None
        else:
            fields["token_start"] = None
            fields["token_end"] = None
        fields["timestamp"] = payload.get("timestamp") or ""
        fields["parser_version"] = payload.get("parser_version") or PARSER_VERSION
        fields["used_ocr"] = bool(payload.get("used_ocr", False))
        return fields

    def write_payload(self, payload: Dict[str, Any]) -> int:
        self._rows.append(self._normalize_for_parquet(payload))
        return 1

    def finalize_and_upload(self, out_basename: str) -> Tuple[int, str, str, int]:
        if not self._rows:
            return 0, "", "", 0
        _ensure_pyarrow()
        if not PA_AVAILABLE or _pa is None or _pq is None:
            raise RuntimeError("pyarrow is required to finalize parquet output (install pyarrow)")
        pa = _pa
        pq = _pq
        schema = pa.schema([
            pa.field("document_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("text", pa.string()),
            pa.field("token_count", pa.int64()),
            pa.field("figures", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("layout_tags", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("headings", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("row_start", pa.int64()),
            pa.field("row_end", pa.int64()),
            pa.field("token_start", pa.int64()),
            pa.field("token_end", pa.int64()),
            pa.field("timestamp", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("used_ocr", pa.bool_())
        ])
        cols: Dict[str, List[Any]] = {name: [] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table = pa.Table.from_pydict(cols, schema=schema)
        existing_md = table.schema.metadata or {}
        new_md = dict(existing_md)
        new_md.update({
            b"schema_version": os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1").encode("utf-8"),
            b"parser_version": PARSER_VERSION.encode("utf-8"),
            b"producer": b"jsonl_parser",
            b"created_at": datetime.utcnow().isoformat().encode("utf-8")
        })
        table = table.replace_schema_metadata(new_md)
        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".parquet", dir="/tmp")
        tmpfile.close()
        pq.write_table(table, tmpfile.name, compression="zstd", flavor="spark")
        local_parquet_path = tmpfile.name
        with open(local_parquet_path, "rb") as fh:
            b = fh.read()
        sha = sha256_hex_bytes(b)
        size = os.path.getsize(local_parquet_path)
        parquet_key = out_basename + ".parquet"
        storage_upload_file_atomic(local_parquet_path, STORAGE_CHUNKED_PREFIX + parquet_key, content_type="application/octet-stream")
        try:
            os.unlink(local_parquet_path)
        except Exception:
            pass
        return len(self._rows), STORAGE_CHUNKED_PREFIX + parquet_key, sha, size

# ---------- helpers ----------
def sanitize_payload(payload: Dict[str, Any]) -> None:
    if "text" in payload:
        payload["text"] = canonicalize_text(payload.get("text") or "")
    else:
        payload["text"] = ""
    for k in ("tags", "figures", "layout_tags", "heading_path", "headings"):
        v = payload.get(k)
        if v is None:
            payload[k] = []
        elif isinstance(v, (list, tuple)):
            payload[k] = [x for x in v]
        else:
            payload[k] = [v]
    for rk in ("row_range", "token_range", "audio_range"):
        v = payload.get(rk)
        if v is None:
            payload[rk] = None
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                payload[rk] = [int(v[0]), int(v[1])]
            except Exception:
                payload[rk] = None
        else:
            payload[rk] = None
    try:
        payload["token_count"] = int(payload.get("token_count") or 0)
    except Exception:
        payload["token_count"] = 0
    payload["file_name"] = payload.get("file_name") or ""
    payload["source_url"] = payload.get("source_url") or ""
    payload["file_type"] = payload.get("file_type") or ""
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

def _flush_rows_chunk(writer: ParquetWriter, doc_id: str, chunk_index: int, header_text: str, rows_text: List[str], start_row_num: int, manifest_tags: List[str] = None) -> Tuple[int, int]:
    if not rows_text:
        return 0, chunk_index
    chunk_index += 1
    chunk_id = f"{doc_id}_{chunk_index}"
    chunk_text = header_text + "\n" + "\n".join(rows_text) if header_text else "\n".join(rows_text)
    token_ct = token_count_for(chunk_text)
    end_row_num = start_row_num + len(rows_text) - 1
    source_url = f"az://{AZURE_CONTAINER}/{writer.source_path}" if writer.source_path else None
    payload: Dict[str, Any] = {
        "document_id": doc_id or "",
        "chunk_id": chunk_id or "",
        "chunk_type": "row_group",
        "text": canonicalize_text(chunk_text) or "",
        "token_count": int(token_ct or 0),
        "figures": [],
        "embedding": None,
        "file_type": "application/x-ndjson",
        "source_url": source_url or "",
        "file_name": filename_from_source_url(source_url) if source_url else "",
        "row_range": [int(start_row_num), int(end_row_num)],
        "token_range": None,
        "audio_range": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parser_version": PARSER_VERSION or "",
        "tags": manifest_tags or [],
        "layout_tags": [],
        "used_ocr": False,
        "heading_path": [],
        "headings": []
    }
    sanitize_payload(payload)
    writer.write_payload(payload)
    log.info("buffered_chunk", "Buffered chunk", chunk_id=payload["chunk_id"])
    return 1, chunk_index

def _process_batch_rows(rows_iterable, doc_id, source_path, chunk_index, header_text, next_row_num, writer: ParquetWriter, manifest_tags: List[str] = None):
    saved = 0
    rows_text: List[str] = []
    start_row_of_current = next_row_num
    for _, row in rows_iterable:
        if isinstance(row, dict):
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
                source_url = f"az://{AZURE_CONTAINER}/{source_path}" if source_path else None
                payload: Dict[str, Any] = {
                    "document_id": doc_id or "",
                    "chunk_id": chunk_id or "",
                    "chunk_type": "token_window",
                    "text": canonicalize_text(candidate_text) or "",
                    "figures": [],
                    "token_count": int(token_ct or 0),
                    "embedding": None,
                    "file_type": "application/x-ndjson",
                    "source_url": source_url or "",
                    "file_name": filename_from_source_url(source_url) if source_url else "",
                    "row_range": [int(row_num), int(row_num)],
                    "token_range": [int(w.get("token_start") or 0), int(w.get("token_end") or 0)],
                    "audio_range": None,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION or "",
                    "tags": manifest_tags or [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": []
                }
                sanitize_payload(payload)
                writer.write_payload(payload)
                log.info("buffered_token_window", "Buffered token window", chunk_id=payload["chunk_id"])
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

def get_header_and_sample_tokens(blob_key: str) -> Tuple[str, int]:
    try:
        client = get_storage_client_singleton()
        if client.fs is not None:
            full = full_path_from_key(blob_key)
            with client.fs.open(full, "rb") as fh:
                data = fh.read(min(int(RANGE_BYTES), 256 * 1024))
        else:
            obj = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
            data = obj.get("Body").read()
        text = data.decode("utf-8", errors="replace")
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

def make_doc_id(blob_key: str, last_modified: Any) -> str:
    return sha256_hex(blob_key + str(last_modified or ""))

def filename_from_source_url(source_url: Optional[str]) -> str:
    if not source_url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(source_url)
        if parsed.path:
            return os.path.basename(parsed.path)
        return os.path.basename(source_url)
    except Exception:
        return os.path.basename(str(source_url))

# ---------- parse_file (public) ----------
def parse_file(blob_key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    start_all = time.perf_counter()
    client = get_storage_client_singleton()
    try:
        head_obj = client.head_object(Bucket=AZURE_CONTAINER, Key=blob_key)
    except Exception as e:
        log.error("head_failed", "Could not head object", key=blob_key, error=str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": str(e)}
    last_modified = head_obj.get("LastModified", "")
    doc_id = manifest.get("file_hash") or make_doc_id(blob_key, last_modified)
    source_path = f"{blob_key}"
    out_basename = f"{doc_id}"
    raw_manifest_key = blob_key + ".manifest.json"
    try:
        if not FORCE_OVERWRITE and client.exists(AZURE_CONTAINER, STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"):
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("skip_parquet_exists", "parquet exists", key=out_basename + ".parquet")
            try:
                if not client.exists(AZURE_CONTAINER, raw_manifest_key):
                    head = client.head_object(Bucket=AZURE_CONTAINER, Key=STORAGE_CHUNKED_PREFIX + out_basename + ".parquet")
                    etag = head.get("ETag", "")
                    if isinstance(etag, str):
                        etag = etag.strip('"')
                    size = head.get("ContentLength", 0)
                    raw_manifest = {"storage_key": STORAGE_CHUNKED_PREFIX + out_basename + ".parquet", "doc_id": doc_id, "rows": 0, "sha256": etag, "size_bytes": size, "schema_version": os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1"), "parser_version": PARSER_VERSION, "created_at": datetime.utcnow().isoformat() + "Z"}
                    client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
            except Exception:
                pass
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    except Exception:
        pass

    header_text, sample_row_tokens = get_header_and_sample_tokens(blob_key)
    header_tokens = token_count_for(header_text) if header_text else 0
    if header_tokens >= TARGET_TOKENS_PER_CHUNK:
        log.warning("header_too_large", "Header token count >= target; omitting header in chunks", key=blob_key)
        header_text = ""
        header_tokens = 0
    if ROWS_PER_CHUNK_OVERRIDE:
        try:
            rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, int(ROWS_PER_CHUNK_OVERRIDE)))
        except Exception:
            rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, 10))
    else:
        available_for_rows = max(1, TARGET_TOKENS_PER_CHUNK - header_tokens)
        estimated_rows = max(1, int(available_for_rows / max(1, sample_row_tokens)))
        rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, estimated_rows))

    log.info("sampling", "sample info", key=blob_key, sample_row_tokens=sample_row_tokens, header_tokens=header_tokens, rows_per_chunk=rows_per_chunk)
    saved = 0
    chunk_index = 0
    next_row_num = 1
    manifest_tags = manifest.get("tags", []) if isinstance(manifest, dict) else []
    writer = ParquetWriter(doc_id=doc_id, source_path=source_path)
    try:
        resp = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
        body = resp.get("Body")
        # body may be a BytesIO or file-like
        try:
            # attempt streaming line iteration first
            if hasattr(body, "iter_lines"):
                iter_lines = body.iter_lines(chunk_size=4096, keepends=False)
            else:
                # body is BytesIO or file-like -> ensure bytes lines
                content = body.read()
                if isinstance(content, bytes):
                    text_lines = content.decode("utf-8", errors="replace").splitlines()
                else:
                    text_lines = str(content).splitlines()
                iter_lines = (ln.encode("utf-8") for ln in text_lines if ln.strip())
            buffer: List[Dict[str, Any]] = []
            for ln in iter_lines:
                if not ln:
                    continue
                try:
                    raw = ln.decode("utf-8") if isinstance(ln, (bytes, bytearray)) else ln
                    rec = json.loads(raw)
                except Exception:
                    # skip malformed line but continue
                    continue
                buffer.append(rec)
                if len(buffer) >= rows_per_chunk:
                    indexed_iter = ((i, row) for i, row in enumerate(buffer))
                    saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, source_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                    saved += saved_chunk
                    buffer = []
            if buffer:
                indexed_iter = ((i, row) for i, row in enumerate(buffer))
                saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, source_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                saved += saved_chunk
        except Exception as e_inner:
            # fallback: read all and attempt
            try:
                body_bytes = body.read()
                text = body_bytes.decode("utf-8", errors="replace")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                buffer = []
                for ln in lines:
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    buffer.append(rec)
                    if len(buffer) >= rows_per_chunk:
                        indexed_iter = ((i, row) for i, row in enumerate(buffer))
                        saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, source_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                        saved += saved_chunk
                        buffer = []
                if buffer:
                    indexed_iter = ((i, row) for i, row in enumerate(buffer))
                    saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, source_path, chunk_index, header_text, next_row_num, writer, manifest_tags)
                    saved += saved_chunk
            except Exception as inner2:
                raise inner2 from e_inner
    except Exception as e_pd:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("read_failed", "Skipping malformed or unreadable JSONL", key=blob_key, error=str(e_pd), traceback=traceback.format_exc())
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_pd)}

    try:
        if saved == 0:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("no_chunks", "No chunks produced", key=blob_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": False}
        count, uploaded_key, sha, size = writer.finalize_and_upload(out_basename)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        try:
            raw_manifest = {"storage_key": uploaded_key, "doc_id": doc_id, "rows": count, "sha256": sha, "size_bytes": size, "schema_version": os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1"), "parser_version": PARSER_VERSION, "created_at": datetime.utcnow().isoformat() + "Z"}
            client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
        except Exception:
            log.warning("manifest_write_failed", "Failed to write raw manifest", key=blob_key)
        log.info("write_complete", "Wrote chunks", count=count, raw=blob_key, chunked=uploaded_key, duration_ms=total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("upload_failed", "Failed to upload chunked file", key=blob_key, error=str(e_up), traceback=traceback.format_exc())
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

# ---------- CLI helper (safe manifest read) ----------
def _safe_read_manifest(client: AzureStorageClient, manifest_key: str) -> Dict[str, Any]:
    try:
        mf_obj = client.get_object(Bucket=AZURE_CONTAINER, Key=manifest_key)
        body = mf_obj.get("Body")
        if hasattr(body, "read"):
            raw = body.read()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {}
        # fallback
        return {}
    except Exception:
        return {}

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    log.info("startup", "JSONL parser start", use_managed_identity=str(USE_MANAGED_IDENTITY).lower(), token_encoder=os.getenv("TOKEN_ENCODER", ENC_NAME), tiktoken_present="yes" if ENCODER is not None else "no")
    client = get_storage_client_singleton()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AZURE_CONTAINER, Prefix=STORAGE_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key:
                continue
            lower = key.lower()
            if lower.endswith(".manifest.json"):
                continue
            if not (lower.endswith(".jsonl") or lower.endswith(".ndjson")):
                continue
            log.info("cli_route", "Routing parse_file", key=key)
            manifest_key = key + ".manifest.json"
            manifest = _safe_read_manifest(client, manifest_key)
            try:
                result = parse_file(key, manifest)
                log.info("cli_result", "Result for file", key=key, result=result)
            except Exception:
                log.exception("cli_parse_failed", "Failed to parse", key=key)
