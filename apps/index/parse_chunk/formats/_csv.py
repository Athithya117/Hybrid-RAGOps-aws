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
import csv
import urllib.parse
import threading
from datetime import datetime
from typing import Any, Dict, Iterator, Tuple, List, Optional

class LoggerShim:
    def __init__(self, name: str):
        self.name = name

    def _emit(self, level: str, event: str, msg: str = "", **extra):
        o = {"ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z", "level": level, "event": event, "msg": msg}
        if extra:
            o.update(extra)
        print(json.dumps(o, ensure_ascii=False), flush=True)

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
        import traceback
        tb = traceback.format_exc()
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "exception")
        kw.update({"tb": tb})
        self._emit("error", event, msg, **kw)

log = LoggerShim("csv_parser")

ENV = os.getenv("ENV", "STAGING").upper()
USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower() in ("1", "true", "yes")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
STORAGE_RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("S3_RAW_PREFIX", "data/raw/")).rstrip("/") + "/"
STORAGE_CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")).rstrip("/") + "/"
PARSER_VERSION = os.getenv("PARSER_VERSION_CSV", "polars-csv-firstrow-auto-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
TARGET_TOKENS_PER_CHUNK = int(os.getenv("CSV_TARGET_TOKENS_PER_CHUNK", "1000"))
ROWS_PER_CHUNK_OVERRIDE = os.getenv("CSV_ROWS_PER_CHUNK", "")
MIN_ROWS_PER_CHUNK = int(os.getenv("CSV_MIN_ROWS_PER_CHUNK", "1"))
MAX_ROWS_PER_CHUNK = int(os.getenv("CSV_MAX_ROWS_PER_CHUNK", "100"))
PUT_RETRIES = int(os.getenv("PUT_RETRIES", "3"))
PUT_BACKOFF = float(os.getenv("PUT_BACKOFF", "0.5"))
RANGE_BYTES = int(os.getenv("CSV_RANGE_BYTES", "131072"))

def _norm_prefix(p: str) -> str:
    p = (p or "").rstrip("/")
    return p + "/" if p else ""

STORAGE_RAW_PREFIX = _norm_prefix(STORAGE_RAW_PREFIX)
STORAGE_CHUNKED_PREFIX = _norm_prefix(STORAGE_CHUNKED_PREFIX)

try:
    import fsspec
    from fsspec.spec import AbstractFileSystem  # type: ignore
except Exception:
    fsspec = None

try:
    import polars as pl
except Exception:
    pl = None

try:
    import tiktoken
except Exception:
    tiktoken = None

DefaultAzureCredential = None
BlobServiceClient = None
ContainerClient = None
AZURE_SDK_AVAILABLE = False
if USE_MANAGED_IDENTITY:
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        from azure.storage.blob import BlobServiceClient, ContainerClient  # type: ignore
        AZURE_SDK_AVAILABLE = True
    except Exception as e:
        AZURE_SDK_AVAILABLE = False
        log.error("azure_sdk_missing", "azure-identity and azure-storage-blob required for managed identity mode", error=str(e))

def build_storage_options() -> Dict[str, str]:
    if USE_MANAGED_IDENTITY:
        return {}
    opts: Dict[str, str] = {}
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
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

BLOB_CLIENT = None
FS = None
STORAGE_ROOT = ""
if USE_MANAGED_IDENTITY:
    if not AZURE_SDK_AVAILABLE:
        sys.stderr.write("ERROR: AZURE_USE_MANAGED_IDENTITY=1 but azure-identity/azure-storage-blob is not installed.\n")
        sys.stderr.write("Install: pip install azure-identity azure-storage-blob\n")
        sys.exit(2)
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    if not account_name:
        sys.stderr.write("ERROR: AZURE_STORAGE_ACCOUNT_NAME required when AZURE_USE_MANAGED_IDENTITY=1\n")
        sys.exit(2)
    endpoint_suffix = os.environ.get("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
    account_url = f"https://{account_name}.{endpoint_suffix}"
    mi_client_id = os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID") or os.getenv("UAI_RAG_RW_CLIENT_ID")
    try:
        if mi_client_id:
            CREDENTIAL = DefaultAzureCredential(managed_identity_client_id=mi_client_id)
            log.info("mi.chosen", "Using managed identity client id", client_id=mi_client_id)
        else:
            CREDENTIAL = DefaultAzureCredential()
            log.info("mi.chosen", "Using system default managed identity (no client id provided)")
        BLOB_CLIENT = BlobServiceClient(account_url=account_url, credential=CREDENTIAL, connection_timeout=60)
    except Exception as e:
        sys.stderr.write(f"ERROR: failed to create BlobServiceClient for account '{account_name}': {e}\n")
        sys.exit(2)
    FS = None
    STORAGE_ROOT = f"az://{AZURE_CONTAINER.rstrip('/')}/" if AZURE_CONTAINER else ""
else:
    if fsspec is None:
        sys.stderr.write("ERROR: AZURE_USE_MANAGED_IDENTITY=0 requires fsspec/adlfs. Install: pip install fsspec adlfs\n")
        sys.exit(2)
    if not FS_OPTS:
        sys.stderr.write("ERROR: non-managed identity mode requires AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_CONNECTION_STRING or AZURE_SAS_TOKEN (or AZURE_ANON).\n")
        sys.exit(2)
    try:
        FS = fsspec.filesystem("az", **FS_OPTS)  # type: ignore
        log.info("fs.init", "Initialized fsspec az filesystem", opts_keys=list(FS_OPTS.keys()))
    except Exception as e:
        sys.stderr.write(f"ERROR: failed to initialize fsspec az filesystem: {e}\n")
        sys.exit(2)
    BLOB_CLIENT = None
    STORAGE_ROOT = f"az://{AZURE_CONTAINER.rstrip('/')}/" if AZURE_CONTAINER else ""

def full_path_from_key(key: str) -> str:
    return STORAGE_ROOT + key.lstrip("/")

def strip_root_from_path(full: str) -> str:
    if full.startswith(STORAGE_ROOT):
        return full[len(STORAGE_ROOT):]
    proto_prefix = "az://"
    if full.startswith(proto_prefix):
        rest = full[len(proto_prefix):]
        if rest.startswith((AZURE_CONTAINER or "") + "/"):
            return rest[len(AZURE_CONTAINER) + 1 : ]
        if rest == (AZURE_CONTAINER or ""):
            return ""
    if full.startswith((AZURE_CONTAINER or "") + "/"):
        return full[len(AZURE_CONTAINER) + 1 : ]
    return full

def sha256_hex_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canonicalize_text(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split()).strip()

def token_count_for(text: str) -> int:
    if not text:
        return 0
    try:
        if tiktoken is None:
            raise RuntimeError("no tiktoken")
        enc = None
        try:
            enc = tiktoken.get_encoding(ENC_NAME)
        except Exception:
            try:
                enc = tiktoken.encoding_for_model("gpt2")
            except Exception:
                enc = None
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
    except Exception:
        pass
    return len(text.split())

class AzureStorageClient:
    def __init__(self, fs_obj=None, root=None, container=None, blob_client=None):
        self.fs = fs_obj
        self.root = root
        self.container = container
        self.blob_client = blob_client

    def _container_client(self):
        if self.blob_client is None:
            raise RuntimeError("blob_client not initialized for managed-identity mode")
        return self.blob_client.get_container_client(self.container)

    def head_object(self, Bucket, Key):
        if self.fs is not None:
            full = full_path_from_key(Key)
            info_obj = self.fs.info(full)
            out = {}
            out["ContentLength"] = int(info_obj.get("size", 0))
            etag = info_obj.get("etag") or info_obj.get("ETag") or info_obj.get("eTag") or ""
            out["ETag"] = etag
            lm = info_obj.get("Last-Modified") or info_obj.get("last_modified") or info_obj.get("LastModified") or ""
            out["LastModified"] = lm
            metadata = info_obj.get("metadata") or info_obj.get("meta") or {}
            out["Metadata"] = metadata
            return out
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
            props = blob_client.get_blob_properties()
            out = {
                "ContentLength": props.size,
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
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
            stream = blob_client.download_blob()
            data = stream.readall()
            return {"Body": io.BytesIO(data)}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        if self.fs is not None:
            full = full_path_from_key(Key)
            if isinstance(Body, bytes):
                b = Body
            elif isinstance(Body, str):
                b = Body.encode("utf-8")
            elif hasattr(Body, "read"):
                b = Body.read()
                if isinstance(b, str):
                    b = b.encode("utf-8")
            else:
                try:
                    b = bytes(Body)
                except Exception:
                    b = str(Body).encode("utf-8")
            with self.fs.open(full, "wb") as f:
                f.write(b)
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
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
                    d = lf.read()
                with self.fs.open(full, "wb") as f:
                    f.write(d)
            return
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
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
                            info_obj = self.fs.info(f)
                        except Exception:
                            continue
                        if info_obj.get("type") == "directory":
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

_storage_client = None
_storage_lock = threading.Lock()

def get_storage_client():
    global _storage_client
    if _storage_client is None:
        with _storage_lock:
            if _storage_client is None:
                if USE_MANAGED_IDENTITY:
                    _storage_client = AzureStorageClient(fs_obj=None, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=BLOB_CLIENT)
                else:
                    _storage_client = AzureStorageClient(fs_obj=FS, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=None)
    return _storage_client

def storage_file_exists(key: str) -> bool:
    try:
        client = get_storage_client()
        try:
            client.head_object(Bucket=AZURE_CONTAINER, Key=key)
            return True
        except Exception:
            return False
    except Exception:
        return False

def storage_upload_file_atomic(local_path: str, key: str, content_type: str = "application/octet-stream"):
    full = full_path_from_key(key)
    tmp = f"{full}.tmp.{os.getpid()}.{int(time.time())}"
    client = get_storage_client()
    for attempt in range(1, PUT_RETRIES + 1):
        try:
            if client.fs is not None and hasattr(client.fs, "put"):
                client.fs.put(local_path, tmp)
            elif client.fs is not None:
                with open(local_path, "rb") as lf:
                    d = lf.read()
                with client.fs.open(tmp, "wb") as f:
                    f.write(d)
            else:
                client.upload_file(local_path, AZURE_CONTAINER, key)
                return
            if client.fs is not None and hasattr(client.fs, "mv"):
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
        except Exception as e:
            log.warning("upload_retry", "attempt=%d key=%s error=%s", attempt, key, str(e))
            time.sleep(PUT_BACKOFF * attempt)
    raise Exception(f"atomic upload failed for {key} after {PUT_RETRIES} attempts")

def split_into_token_windows(text: str, max_tokens: int, overlap_fraction: float = 0.1) -> Iterator[Dict[str, Any]]:
    if not text:
        return
        yield
    words = text.split()
    n = len(words)
    if n == 0:
        return
    token_counts = [max(1, token_count_for(w)) for w in words]
    cum = [0] * (n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + token_counts[i]
    start = 0
    while start < n:
        end = start + 1
        while end <= n and (cum[end] - cum[start]) <= max_tokens:
            end += 1
        end = max(start + 1, end - 1)
        window_tokens = cum[end] - cum[start]
        window_text = " ".join(words[start:end])
        token_start = cum[start]
        token_end = token_start + window_tokens - 1 if window_tokens > 0 else token_start
        yield {"text": window_text, "token_start": token_start, "token_end": token_end, "token_count": window_tokens}
        overlap_tokens = int(max(1, max_tokens * overlap_fraction))
        desired_token_index = max(token_start, token_end - overlap_tokens + 1)
        new_start = start
        lo = start + 1
        hi = end
        while lo <= hi:
            mid = (lo + hi) // 2
            if cum[mid] < desired_token_index:
                lo = mid + 1
            else:
                new_start = mid
                hi = mid - 1
        if new_start <= start:
            start = end
        else:
            start = new_start

# ---------- NEW: semantic_region helper (CSV positional) ----------
def csv_positional_region(row_range: Optional[List[int]], total_rows: Optional[int]) -> str:
    """
    Deterministic mapping of row_range -> semantic region.
    Uses chunk midpoint relative to total_rows.

    Bands:
      intro  : 0%  - 5%
      early  : 5%  - 25%
      middle : 25% - 75%
      late   : 75% - 95%
      footer : 95% - 100%
    """
    if not row_range or total_rows is None or total_rows <= 0:
        return "middle"
    try:
        rs = float(row_range[0])
        re = float(row_range[1])
        mid = (rs + re) / 2.0
        frac = mid / float(total_rows)
    except Exception:
        return "middle"
    if frac <= 0.05:
        return "intro"
    if frac <= 0.25:
        return "early"
    if frac <= 0.75:
        return "middle"
    if frac <= 0.95:
        return "late"
    return "footer"

def count_data_rows(blob_key: str) -> int:
    """
    Count data rows (excluding header) efficiently.
    Works with fsspec (FS) or Azure SDK managed identity (BLOB_CLIENT).
    This is a streaming count; handles large files without loading whole file to memory.
    Returns number of data rows (lines after first non-empty header line).
    """
    client = get_storage_client()
    full = full_path_from_key(blob_key)
    # fsspec path
    if FS is not None:
        try:
            with FS.open(full, "rb") as fh:
                # read and skip header line
                first = fh.readline()
                if not first:
                    return 0
                count = 0
                for raw in fh:
                    if raw.strip():
                        count += 1
                return count
        except Exception as e:
            log.warning("count_rows_fs_failed", "fsspec counting failed, falling back", error=str(e))
            # fall through to try blob client if available
    # managed identity / azure sdk path
    if BLOB_CLIENT is not None:
        try:
            container_client = BLOB_CLIENT.get_container_client(AZURE_CONTAINER)
            blob_client = container_client.get_blob_client(blob_key)
            stream = blob_client.download_blob()
            rem = b""
            header_skipped = False
            count = 0
            for chunk in stream.chunks():
                if not chunk:
                    continue
                data = rem + chunk
                parts = data.split(b"\n")
                rem = parts.pop()
                for i, p in enumerate(parts):
                    if not header_skipped:
                        header_skipped = True
                        continue
                    if p.strip():
                        count += 1
            # remaining tail
            if rem:
                if not header_skipped:
                    # only header present; no data rows
                    pass
                else:
                    if rem.strip():
                        count += 1
            return count
        except Exception as e:
            log.warning("count_rows_blob_failed", "azure blob counting failed", error=str(e))
            # fallback below
    # last-resort fallback: try reading via storage client get_object (reads whole body)
    try:
        obj = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
        body = obj.get("Body")
        data = body.read()
        text = data.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return 0
        # consider first non-empty line as header
        return max(0, len(lines) - 1)
    except Exception:
        return 0

class ParquetWriter:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self._rows: List[Dict[str, Any]] = []

    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["document_id"] = payload.get("document_id") or ""
        out["file_name"] = payload.get("file_name") or ""
        out["chunk_id"] = payload.get("chunk_id") or ""
        out["chunk_type"] = payload.get("chunk_type") or ""
        out["text"] = payload.get("text") or ""
        try:
            out["token_count"] = int(payload.get("token_count") or 0)
        except Exception:
            out["token_count"] = 0
        out["figures"] = json.dumps(payload.get("figures", []), ensure_ascii=False)
        out["tags"] = json.dumps(payload.get("tags", []), ensure_ascii=False)
        out["layout_tags"] = json.dumps(payload.get("layout_tags", []), ensure_ascii=False)
        out["heading_path"] = json.dumps(payload.get("heading_path", []), ensure_ascii=False)
        out["headings"] = json.dumps(payload.get("headings", []), ensure_ascii=False)
        out["file_type"] = payload.get("file_type") or "text/csv"
        out["source_url"] = payload.get("source_url") or ""
        rs, re = None, None
        if payload.get("row_range") and isinstance(payload.get("row_range"), (list, tuple)):
            try:
                rs = int(payload["row_range"][0])
                re = int(payload["row_range"][1])
            except Exception:
                rs, re = None, None
        out["row_start"] = rs
        out["row_end"] = re
        ts, te = None, None
        if payload.get("token_range") and isinstance(payload.get("token_range"), (list, tuple)):
            try:
                ts = int(payload["token_range"][0])
                te = int(payload["token_range"][1])
            except Exception:
                ts, te = None, None
        out["token_start"] = ts
        out["token_end"] = te
        out["timestamp"] = payload.get("timestamp") or ""
        out["parser_version"] = payload.get("parser_version") or PARSER_VERSION
        out["used_ocr"] = bool(payload.get("used_ocr", False))
        # semantic_region field (string) - optional presence
        out["semantic_region"] = payload.get("semantic_region") or ""
        return out

    def write_payload(self, payload: Dict[str, Any]) -> int:
        self._rows.append(self._normalize(payload))
        return 1

    def finalize_and_upload(self, out_basename: str) -> Tuple[int, str, str, int]:
        if not self._rows:
            return 0, "", "", 0
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:
            log.error("pyarrow_missing", "pyarrow required for parquet writing", error=str(e))
            raise
        schema = pa.schema([
            pa.field("document_id", pa.string()),
            pa.field("file_name", pa.string()),
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
            pa.field("row_start", pa.int64()),
            pa.field("row_end", pa.int64()),
            pa.field("token_start", pa.int64()),
            pa.field("token_end", pa.int64()),
            pa.field("timestamp", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("used_ocr", pa.bool_()),
            pa.field("semantic_region", pa.string()),
        ])
        cols = {name: [] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table = pa.Table.from_pydict(cols, schema=schema)
        existing_md = table.schema.metadata or {}
        new_md = dict(existing_md)
        new_md.update({
            b"schema_version": os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1").encode("utf-8"),
            b"parser_version": PARSER_VERSION.encode("utf-8"),
            b"producer": b"csv_parser",
            b"created_at": datetime.utcnow().isoformat().encode("utf-8"),
        })
        table = table.replace_schema_metadata(new_md)
        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".parquet", dir="/tmp")
        tmpfile.close()
        pq.write_table(table, tmpfile.name, compression="zstd", flavor="spark")
        with open(tmpfile.name, "rb") as fh:
            data = fh.read()
        sha = sha256_hex_bytes(data)
        size = os.path.getsize(tmpfile.name)
        parquet_key = out_basename + ".parquet"
        storage_upload_file_atomic(tmpfile.name, STORAGE_CHUNKED_PREFIX + parquet_key, content_type="application/octet-stream")
        try:
            os.unlink(tmpfile.name)
        except Exception:
            pass
        return len(self._rows), STORAGE_CHUNKED_PREFIX + parquet_key, sha, size

def sanitize_payload_for_raw_manifest(doc_id: str, raw_key: str, chunked_key: str, rows: int, sha: str, size: int) -> Dict[str, Any]:
    return {
        "raw_key": raw_key,
        "doc_id": doc_id,
        "chunked_key": chunked_key,
        "rows": int(rows),
        "sha256": sha,
        "size_bytes": int(size),
        "schema_version": os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1"),
        "parser_version": PARSER_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

def filename_from_source_url(source_url: Optional[str]) -> str:
    if not source_url:
        return ""
    try:
        if source_url.startswith("az://"):
            return os.path.basename(source_url)
        p = urllib.parse.urlparse(source_url)
        if p.path:
            return os.path.basename(p.path)
        return os.path.basename(source_url)
    except Exception:
        return os.path.basename(str(source_url))

def get_header_and_sample_tokens(blob_key: str) -> Tuple[str, int]:
    try:
        full = full_path_from_key(blob_key)
        if FS is not None:
            with FS.open(full, "rb") as f:
                head_bytes = f.read(RANGE_BYTES)
        else:
            client = get_storage_client()
            obj = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
            head_bytes = obj.get("Body").read()
    except Exception:
        return "", 32
    try:
        text = head_bytes.decode("utf-8", errors="replace")
        if pl is not None:
            try:
                buf = io.BytesIO(head_bytes)
                df = pl.read_csv(buf, n_rows=2, dtype=str, ignore_errors=True)
                if df.shape[1] == 0 or df.shape[0] == 0:
                    return "", 32
                header_cols = df.columns
                header_text = canonicalize_text(" | ".join(header_cols))
                sample_tokens = 32
                if df.shape[0] > 0:
                    row = df.row(0)
                    sample_dict = {c: (row[i] if i < len(row) else "") for i, c in enumerate(header_cols)}
                    sample_text = " | ".join([f"{k}: {v}" for k, v in sample_dict.items()])
                    sample_tokens = max(1, token_count_for(sample_text))
                return header_text, sample_tokens
            except Exception:
                pass
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return "", 32
        header = lines[0]
        sample_line = lines[1] if len(lines) > 1 else ""
        header_cols = [c.strip() for c in header.split(",")] if header else []
        header_text = canonicalize_text(" | ".join(header_cols))
        sample_tokens = max(1, token_count_for(canonicalize_text(sample_line)))
        return header_text, sample_tokens
    except Exception:
        return "", 32

def _flush_rows_chunk(writer: ParquetWriter, doc_id: str, chunk_index: int, header_text: str, rows_text: List[str], start_row_num: int, manifest_tags: List[str] = None, total_rows: Optional[int] = None) -> Tuple[int, int]:
    if not rows_text:
        return 0, chunk_index
    chunk_index += 1
    chunk_id = f"{doc_id}_{chunk_index}"
    chunk_text = (header_text + "\n" + "\n".join(rows_text)) if header_text else "\n".join(rows_text)
    token_ct = token_count_for(chunk_text)
    end_row_num = start_row_num + len(rows_text) - 1
    src = writer._rows[0].get("source_url") if writer._rows else ""
    source_url = src if src else f"az://{AZURE_CONTAINER}/"
    row_range = [int(start_row_num), int(end_row_num)]
    semantic = csv_positional_region(row_range, total_rows)
    payload = {
        "document_id": doc_id or "",
        "chunk_id": chunk_id or "",
        "chunk_type": "row_group",
        "text": canonicalize_text(chunk_text) or "",
        "token_count": int(token_ct or 0),
        "figures": [],
        "embedding": None,
        "file_type": "text/csv",
        "source_url": source_url,
        "file_name": filename_from_source_url(source_url),
        "row_range": [int(start_row_num), int(end_row_num)],
        "token_range": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parser_version": PARSER_VERSION,
        "tags": manifest_tags or [],
        "layout_tags": [],
        "used_ocr": False,
        "heading_path": [],
        "headings": [],
        "line_range": None,
        "semantic_region": semantic,
    }
    writer.write_payload(payload)
    log.info("buffer_row_group", "Buffered CSV row_group chunk %s", payload.get("chunk_id"))
    return 1, chunk_index

def _process_batch_rows(rows_iterable, doc_id, blob_path, chunk_index, header_text, next_row_num, writer: ParquetWriter, manifest_tags: List[str] = None, total_rows: Optional[int] = None):
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
        if row_tokens > TARGET_TOKENS_PER_CHUNK:
            if rows_text:
                wrote, chunk_index = _flush_rows_chunk(writer, doc_id, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags, total_rows=total_rows)
                saved += wrote
                rows_text = []
            windows = list(split_into_token_windows(row_text, TARGET_TOKENS_PER_CHUNK, overlap_fraction=0.1))
            for w in windows:
                chunk_index += 1
                chunk_id = f"{doc_id}_{chunk_index}"
                candidate_text = header_text + "\n" + w["text"] if header_text and (token_count_for(header_text) + w["token_count"] <= TARGET_TOKENS_PER_CHUNK) else w["text"]
                token_ct = token_count_for(candidate_text)
                row_range = [int(row_num), int(row_num)]
                semantic = csv_positional_region(row_range, total_rows)
                payload = {
                    "document_id": doc_id or "",
                    "chunk_id": chunk_id or "",
                    "chunk_type": "token_window",
                    "text": canonicalize_text(candidate_text) or "",
                    "token_count": int(token_ct or 0),
                    "figures": [],
                    "embedding": None,
                    "file_type": "text/csv",
                    "source_url": f"az://{AZURE_CONTAINER}/{blob_path}",
                    "file_name": filename_from_source_url(f"az://{AZURE_CONTAINER}/{blob_path}"),
                    "row_range": [int(row_num), int(row_num)],
                    "token_range": [int(w.get("token_start")), int(w.get("token_end"))],
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION,
                    "tags": manifest_tags or [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": [],
                    "line_range": None,
                    "semantic_region": semantic,
                }
                writer.write_payload(payload)
                log.info("buffer_token_window", "Buffered CSV token_window %s", payload.get("chunk_id"))
                saved += 1
            start_row_of_current = next_row_num
            continue
        candidate_text = (header_text + "\n\n".join(rows_text + [row_text])) if header_text else "\n".join(rows_text + [row_text])
        candidate_tokens = token_count_for(candidate_text)
        if candidate_tokens <= TARGET_TOKENS_PER_CHUNK:
            if not rows_text:
                start_row_of_current = row_num
            rows_text.append(row_text)
            continue
        else:
            wrote, chunk_index = _flush_rows_chunk(writer, doc_id, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags, total_rows=total_rows)
            saved += wrote
            rows_text = [row_text]
            start_row_of_current = row_num
    if rows_text:
        wrote, chunk_index = _flush_rows_chunk(writer, doc_id, chunk_index, header_text, rows_text, start_row_of_current, manifest_tags, total_rows=total_rows)
        saved += wrote
    return saved, chunk_index, next_row_num

def parse_file(blob_key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    start_all = time.perf_counter()
    if not AZURE_CONTAINER:
        raise RuntimeError("AZURE_CONTAINER environment variable must be set")
    client = get_storage_client()
    try:
        head_obj = client.head_object(Bucket=AZURE_CONTAINER, Key=blob_key)
    except Exception as e:
        log.error("head_failed", "Could not HEAD blob %s: %s", blob_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": str(e)}
    last_modified = head_obj.get("LastModified", "")
    doc_id = manifest.get("file_hash") if isinstance(manifest, dict) and manifest.get("file_hash") else sha256_hex_str(blob_key + str(last_modified or ""))
    out_basename = f"{doc_id}"
    raw_manifest_key = blob_key + ".manifest.json"
    out_parquet_key = STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"
    if not FORCE_OVERWRITE:
        try:
            if storage_file_exists(raw_manifest_key):
                total_ms = int((time.perf_counter() - start_all) * 1000)
                log.info("skip_manifest_exists", "Skipping because raw manifest exists: %s", raw_manifest_key)
                return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
            if storage_file_exists(out_parquet_key):
                total_ms = int((time.perf_counter() - start_all) * 1000)
                log.info("skip_parquet_exists", "Skipping because parquet exists: %s", out_parquet_key)
                try:
                    if not storage_file_exists(raw_manifest_key):
                        head = client.head_object(Bucket=AZURE_CONTAINER, Key=out_parquet_key)
                        etag = head.get("ETag", "")
                        etag = etag.strip('"') if isinstance(etag, str) else etag
                        size = head.get("ContentLength", 0)
                        raw_manifest = sanitize_payload_for_raw_manifest(doc_id, blob_key, out_parquet_key, 0, etag, size)
                        client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
                except Exception:
                    pass
                return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
        except Exception:
            pass
    header_text, sample_row_tokens = get_header_and_sample_tokens(blob_key)
    header_tokens = token_count_for(header_text) if header_text else 0
    if header_tokens >= TARGET_TOKENS_PER_CHUNK:
        log.warning("header_too_large", "CSV header token count >= target chunk size. Header will not be prepended.")
        header_text = ""
        header_tokens = 0
    if ROWS_PER_CHUNK_OVERRIDE:
        try:
            rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, int(ROWS_PER_CHUNK_OVERRIDE)))
        except Exception:
            rows_per_chunk = MIN_ROWS_PER_CHUNK
    else:
        available_for_rows = max(1, TARGET_TOKENS_PER_CHUNK - header_tokens)
        estimated_rows = max(1, int(available_for_rows / max(1, sample_row_tokens)))
        rows_per_chunk = max(MIN_ROWS_PER_CHUNK, min(MAX_ROWS_PER_CHUNK, estimated_rows))
    log.info("sample_info", "%s sample_row_tokens=%d header_tokens=%d rows_per_chunk=%d", blob_key, sample_row_tokens, header_tokens, rows_per_chunk)

    # ---------- NEW: compute total_rows for semantic_region classification ----------
    total_rows = None
    try:
        total_rows = count_data_rows(blob_key)
        log.info("total_rows_counted", "Counted data rows", blob=blob_key, total_rows=total_rows)
    except Exception as e:
        log.warning("total_rows_count_failed", "Could not count total rows; semantic_region will default to 'middle'", error=str(e))
        total_rows = None

    saved = 0
    chunk_index = 0
    next_row_num = 1
    manifest_tags = manifest.get("tags", []) if isinstance(manifest, dict) else []
    writer = ParquetWriter(doc_id=doc_id)
    try:
        obj = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
        body = obj.get("Body")
        text_stream = io.TextIOWrapper(body, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text_stream, delimiter=",")
        buffer: List[Dict[str, Any]] = []
        for row in reader:
            buffer.append(row)
            if len(buffer) >= rows_per_chunk:
                indexed_iter = ((i, r) for i, r in enumerate(buffer))
                saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, blob_key, chunk_index, header_text, next_row_num, writer, manifest_tags, total_rows=total_rows)
                saved += saved_chunk
                buffer = []
        if buffer:
            indexed_iter = ((i, r) for i, r in enumerate(buffer))
            saved_chunk, chunk_index, next_row_num = _process_batch_rows(indexed_iter, doc_id, blob_key, chunk_index, header_text, next_row_num, writer, manifest_tags, total_rows=total_rows)
            saved += saved_chunk
    except Exception as e_pd:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("parse_failed", "Skipping malformed or unreadable CSV %s error=%s", blob_key, str(e_pd))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_pd)}
    try:
        if saved == 0:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("no_chunks", "No chunks produced for %s", blob_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key, sha, size = writer.finalize_and_upload(out_basename)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        try:
            raw_manifest = sanitize_payload_for_raw_manifest(doc_id, blob_key, uploaded_key, count, sha, size)
            client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
        except Exception:
            log.warning("manifest_write_failed", "Failed to write raw manifest for %s", blob_key)
        log.info("write_complete", "Wrote %d chunks for %s → %s (%d ms)", count, blob_key, uploaded_key, total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("upload_failed", "Failed to upload chunked file for %s error=%s", blob_key, str(e_up))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

if __name__ == "__main__":
    log.info("startup", "CSV parser start", env=ENV, use_managed_identity=str(USE_MANAGED_IDENTITY).lower(), token_encoder=os.getenv("TOKEN_ENCODER", ENC_NAME), tiktoken_present="yes" if tiktoken is not None else "no")
    if not AZURE_CONTAINER:
        log.error("startup_missing_container", "AZURE_CONTAINER env var required for CLI")
        sys.exit(1)
    client = get_storage_client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AZURE_CONTAINER, Prefix=STORAGE_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".csv"):
                continue
            log.info("cli_route", "Routing parse_file for %s", key)
            manifest_key = key + ".manifest.json"
            try:
                mf_obj = client.get_object(Bucket=AZURE_CONTAINER, Key=manifest_key)
                manifest = json.load(mf_obj["Body"])
            except Exception:
                manifest = {}
            try:
                result = parse_file(key, manifest)
                log.info("cli_result", "Result for %s: %s", key, result)
            except Exception as e:
                log.exception("cli_failed", "Failed to parse %s: %s", key, str(e))
