#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import io
import sys
import json
import time
import hashlib
import unicodedata
import re
import tempfile
import importlib
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import List, Tuple, Dict, Generator, Optional, Any

# ---------------- Logger shim ----------------
class LoggerShim:
    def __init__(self, name: str):
        self.name = name

    def _emit(self, level: str, event: str, msg: str = "", **extra):
        out = {"ts": datetime.utcnow().isoformat() + "Z", "level": level, "event": event, "msg": msg}
        if extra:
            out.update(extra)
        print(json.dumps(out, ensure_ascii=False), flush=True)

    def _unpack(self, a, b, fmt_args, kwargs, default_event):
        # Support both logger.info("event", "message") and logger.info("message %s", val)
        if b is None and not fmt_args:
            event = kwargs.pop("event", default_event)
            msg = a or ""
            fmt_tuple = ()
        else:
            a_is_fmt = isinstance(a, str) and ('%' in a or '{' in a)
            if a_is_fmt:
                event = kwargs.pop("event", default_event)
                msg = a or ""
                if b is None:
                    fmt_tuple = tuple(fmt_args)
                else:
                    fmt_tuple = (b,) + tuple(fmt_args)
            else:
                event = str(a or default_event)
                msg = b or ""
                fmt_tuple = tuple(fmt_args)
        if fmt_tuple:
            try:
                msg = msg % fmt_tuple
            except Exception:
                try:
                    msg = msg.format(*fmt_tuple)
                except Exception:
                    pass
        return event, msg, kwargs

    def info(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "info")
        self._emit("INFO", event, msg, **kw)

    def warning(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "warn")
        self._emit("WARN", event, msg, **kw)

    def warn(self, a, b=None, *fmt_args, **kwargs):
        self.warning(a, b, *fmt_args, **kwargs)

    def error(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "error")
        self._emit("ERROR", event, msg, **kw)

    def exception(self, a, b=None, *fmt_args, **kwargs):
        import traceback
        tb = traceback.format_exc()
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "exception")
        kw.update({"tb": tb})
        self._emit("ERROR", event, msg, **kw)

logger = LoggerShim("images_parser")

# ---------------- config / envs ----------------
_mi_val = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower()
USE_MANAGED_IDENTITY = _mi_val in ("1", "true", "yes")

AZURE_CONTAINER = os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
if not AZURE_CONTAINER:
    logger.error("startup_missing_container", "AZURE_CONTAINER (or STORAGE_CONTAINER/AZ_CONTAINER) must be set")
    sys.exit(1)

STORAGE_RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("S3_RAW_PREFIX", "data/raw/")).rstrip("/") + "/"
STORAGE_CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")).rstrip("/") + "/"
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"

IMAGE_DISABLE_OCR = os.getenv("IMAGE_DISABLE_OCR", "false").lower() in ("1", "true", "yes")
IMAGE_FORCE_OCR = os.getenv("IMAGE_FORCE_OCR", "false").lower() in ("1", "true", "yes")
IMAGE_OCR_ENGINE = os.getenv("IMAGE_OCR_ENGINE", "auto").lower()
IMAGE_TESSERACT_LANG = os.getenv("IMAGE_TESSERACT_LANG", "eng")
IMAGE_MIN_IMG_SIZE_BYTES = int(os.getenv("IMAGE_MIN_IMG_SIZE_BYTES", "3072"))
IMAGE_RENDER_DPI = int(os.getenv("IMAGE_RENDER_DPI", "300"))
IMAGE_UPSCALE_FACTOR = float(os.getenv("IMAGE_UPSCALE_FACTOR", "1.0"))
IMAGE_ENABLE_WORDSEGMENT = os.getenv("IMAGE_ENABLE_WORDSEGMENT", "false").lower() in ("1", "true", "yes")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6")
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
NUMBER_OF_OVERLAPPING_SENTENCES = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", "2"))
PARSER_VERSION_IMAGE = os.getenv("PARSER_VERSION_IMAGE", "image-v1")
PUT_RETRIES = int(os.getenv("S3_PUT_RETRIES", os.getenv("PUT_RETRIES", "3")))
PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", os.getenv("PUT_BACKOFF", "0.3")))
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
CHUNKED_SCHEMA_VERSION = os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1")
RANGE_BYTES = int(os.getenv("IMAGE_RANGE_BYTES", "131072"))

# ---------------- optional dependencies ----------------
try:
    import fsspec  # type: ignore
    from fsspec.spec import AbstractFileSystem  # type: ignore
    FSSPEC_AVAILABLE = True
except Exception:
    fsspec = None
    AbstractFileSystem = object  # type: ignore
    FSSPEC_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.storage.blob import BlobServiceClient, ContainerClient  # type: ignore
    AZURE_SDK_AVAILABLE = True
except Exception:
    DefaultAzureCredential = None  # type: ignore
    BlobServiceClient = None  # type: ignore
    ContainerClient = None  # type: ignore
    AZURE_SDK_AVAILABLE = False

def build_storage_options() -> Dict[str, Any]:
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        return {"connection_string": conn}
    acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
    sas = os.environ.get("AZURE_SAS_TOKEN")
    eps = os.environ.get("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"
    opts: Dict[str, Any] = {}
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

_FS_OPTS = build_storage_options()

def _validate_auth_envs():
    if USE_MANAGED_IDENTITY:
        if not AZURE_SDK_AVAILABLE:
            logger.error("azure_sdk_missing", "Managed identity requested but azure-identity/azure-storage-blob not installed. pip install azure-identity azure-storage-blob")
            sys.exit(2)
        acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
        if not acct:
            logger.error("missing_account_name", "Managed identity requested but AZURE_STORAGE_ACCOUNT_NAME is not set")
            sys.exit(2)
    else:
        if not FSSPEC_AVAILABLE:
            logger.error("fsspec_missing", "Key/SAS mode requested but fsspec/adlfs not installed. pip install fsspec adlfs")
            sys.exit(2)
        if not (_FS_OPTS and any(k in _FS_OPTS for k in ("connection_string", "account_name", "sas_token", "anon"))):
            logger.error("missing_key_creds", "Non-managed identity mode requires AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_CONNECTION_STRING or AZURE_SAS_TOKEN (or AZURE_ANON)")
            sys.exit(2)

_validate_auth_envs()

# ---------------- storage client abstraction ----------------
BLOB_CLIENT = None
FS = None
if USE_MANAGED_IDENTITY:
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    account_url = f"https://{account_name}.{os.environ.get('AZURE_ENDPOINT_SUFFIX','core.windows.net')}"
    try:
        CREDENTIAL = DefaultAzureCredential()
        BLOB_CLIENT = BlobServiceClient(account_url=account_url, credential=CREDENTIAL, connection_timeout=60)
        FS = None
        logger.info("storage_mode", "managed_identity", account=account_name)
    except Exception as e:
        logger.exception("blob_client_fail", "Failed to create BlobServiceClient", error=str(e))
        raise SystemExit(2)
else:
    try:
        FS = fsspec.filesystem("az", **_FS_OPTS)  # type: ignore
        BLOB_CLIENT = None
        logger.info("storage_mode", "fsspec_key_or_sas", opts=list(_FS_OPTS.keys()))
    except Exception as e:
        logger.exception("fsspec_init_fail", "Failed to initialize fsspec az filesystem", error=str(e))
        raise SystemExit(2)

STORAGE_ROOT = f"az://{AZURE_CONTAINER.rstrip('/')}/"

def full_path_from_key(key: str) -> str:
    return STORAGE_ROOT + key.lstrip("/")

def strip_root_from_path(full: str) -> str:
    if full.startswith(STORAGE_ROOT):
        return full[len(STORAGE_ROOT):]
    if full.startswith("az://"):
        rest = full[len("az://"):]
        if rest.startswith(AZURE_CONTAINER + "/"):
            return rest[len(AZURE_CONTAINER) + 1:]
        if rest == AZURE_CONTAINER:
            return ""
    if full.startswith(AZURE_CONTAINER + "/"):
        return full[len(AZURE_CONTAINER) + 1:]
    return full

def retry(func, retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning("retry_attempt", "attempt=%d error=%s", attempt + 1, str(e))
            time.sleep(delay)
            delay *= backoff

class AzureStorageClient:
    def __init__(self, fs_obj=None, root: str = "", container: str = "", blob_client=None):
        self.fs = fs_obj
        self.root = root
        self.container = container
        self.blob_client = blob_client

    def _container_client(self) -> "ContainerClient":
        if self.blob_client is None:
            raise RuntimeError("blob_client not initialized for managed-identity mode")
        return self.blob_client.get_container_client(self.container)

    def head_object(self, Bucket, Key):
        if self.fs is not None:
            full = full_path_from_key(Key)
            info = self.fs.info(full)
            out = {}
            out["ContentLength"] = int(info.get("size", 0))
            etag = info.get("etag") or info.get("ETag") or info.get("eTag") or ""
            out["ETag"] = etag
            lm = info.get("Last-Modified") or info.get("last_modified") or info.get("LastModified") or ""
            out["LastModified"] = lm
            metadata = info.get("metadata") or info.get("meta") or {}
            out["Metadata"] = metadata
            return out
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
            props = blob_client.get_blob_properties()
            out = {
                "ContentLength": int(getattr(props, "size", 0) or 0),
                "ETag": getattr(props, "etag", "") or "",
                "LastModified": getattr(props, "last_modified", "") or "",
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
            if isinstance(Body, (bytes, bytearray)):
                b = Body
            elif isinstance(Body, str):
                b = Body.encode("utf-8")
            elif hasattr(Body, "read"):
                b = Body.read()
                if isinstance(b, str):
                    b = b.encode("utf-8")
            else:
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
                try:
                    self.fs.put(LocalFile, full)
                    return
                except Exception:
                    pass
            with open(LocalFile, "rb") as lf:
                data = lf.read()
            with self.fs.open(full, "wb") as f:
                f.write(data)
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
                    self.fs = fs; self.root = root
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
                            yield page; page = {"Contents": []}
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
                            yield page; page = {"Contents": []}
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

def get_s3_client():
    return get_storage_client()

# ---------------- utilities ----------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

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
    try:
        import tiktoken
        enc = getattr(tiktoken, "encoding_for_model", None)
        if enc is not None:
            e = enc(ENC_NAME)
        else:
            e = tiktoken.get_encoding(ENC_NAME)
        return len(e.encode(text))
    except Exception:
        return len(text.split())

def storage_object_exists(key: str) -> bool:
    client = get_storage_client()
    try:
        client.head_object(Bucket=AZURE_CONTAINER, Key=key)
        return True
    except Exception:
        return False

def storage_upload_file_atomic(local_path: str, key: str, content_type: str = "application/octet-stream") -> None:
    client = get_storage_client()
    if client.fs is not None:
        full = full_path_from_key(key)
        tmp = f"{full}.tmp.{os.getpid()}.{int(time.time())}"
        for attempt in range(1, PUT_RETRIES + 1):
            try:
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
            except Exception as e:
                logger.warn("upload_retry", "attempt=%d key=%s error=%s", attempt, key, str(e))
                time.sleep(PUT_BACKOFF * attempt)
        raise Exception(f"atomic upload failed for {key} after {PUT_RETRIES} attempts")
    else:
        for attempt in range(1, PUT_RETRIES + 1):
            try:
                client.upload_file(local_path, AZURE_CONTAINER, key)
                return
            except Exception as e:
                logger.warn("upload_retry_blob", "attempt=%d key=%s error=%s", attempt, key, str(e))
                time.sleep(PUT_BACKOFF * attempt)
        raise Exception(f"blob upload failed for {key} after {PUT_RETRIES} attempts")

def download_blob_to_temp(blob_key: str, ext: str) -> str:
    suffix = f".{ext.lstrip('.')}" if ext else ".img"
    tmpdir = os.getenv("TMPDIR") or None
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmpdir)
    tf.close()
    client = get_storage_client()
    if client.fs is not None:
        full = full_path_from_key(blob_key)
        with client.fs.open(full, "rb") as f:
            with open(tf.name, "wb") as out:
                out.write(f.read())
        return tf.name
    else:
        try:
            obj = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
            data = obj.get("Body").read()
            with open(tf.name, "wb") as out:
                out.write(data)
            return tf.name
        except Exception as e:
            try:
                os.unlink(tf.name)
            except Exception:
                pass
            raise

def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for k in ("document_id", "file_name", "chunk_id", "chunk_type", "source_url", "parser_version"):
        v = payload.get(k)
        if v is None:
            payload[k] = ""
        elif isinstance(v, bytes):
            payload[k] = try_decode_bytes(v)
        else:
            payload[k] = str(v)
    txt = payload.get("text", "") or ""
    if isinstance(txt, bytes):
        txt = try_decode_bytes(txt)
    txt = canonicalize_text(txt)
    txt = re.sub(r'[\x00-\x1F]+', ' ', txt).strip()
    MAX_TEXT = int(os.getenv("SANITIZE_MAX_TEXT_CHARS", "200000"))
    if len(txt) > MAX_TEXT:
        txt = txt[:MAX_TEXT]
    payload["text"] = txt
    try:
        payload["token_count"] = int(payload.get("token_count") or 0)
    except Exception:
        payload["token_count"] = token_count_for(payload["text"])
    for listkey in ("figures", "tags", "layout_tags", "headings", "heading_path"):
        v = payload.get(listkey)
        if v is None:
            payload[listkey] = []
        elif isinstance(v, str):
            try:
                parsed = json.loads(v)
                payload[listkey] = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                payload[listkey] = [v]
        elif isinstance(v, (list, tuple)):
            payload[listkey] = list(v)
        else:
            payload[listkey] = [v]
    if payload.get("layout_bbox") is None:
        payload["layout_bbox"] = None
    else:
        try:
            payload["layout_bbox"] = json.loads(json.dumps(payload["layout_bbox"]))
        except Exception:
            payload["layout_bbox"] = None
    payload["used_ocr"] = bool(payload.get("used_ocr", False))
    return payload

# ---------------- Parquet writer ----------------
class S3ParquetWriter:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self._rows: List[Dict[str, Any]] = []

    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        fields["document_id"] = payload.get("document_id") or ""
        fields["file_name"] = payload.get("file_name") or ""
        fields["chunk_id"] = payload.get("chunk_id") or ""
        fields["chunk_type"] = payload.get("chunk_type") or ""
        fields["text"] = payload.get("text") or ""
        try:
            fields["token_count"] = int(payload.get("token_count") or 0)
        except Exception:
            fields["token_count"] = 0
        for k in ("figures", "tags", "layout_tags", "heading_path", "headings"):
            v = payload.get(k, None)
            try:
                fields[k] = json.dumps(v, ensure_ascii=False, sort_keys=True) if v is not None else "[]"
            except Exception:
                fields[k] = "[]"
        fields["file_type"] = payload.get("file_type") or ""
        fields["source_url"] = payload.get("source_url") or ""
        fields["page_number"] = None
        if payload.get("page_number") is not None:
            try:
                fields["page_number"] = int(payload.get("page_number"))
            except Exception:
                fields["page_number"] = None
        fields["timestamp"] = payload.get("timestamp") or ""
        fields["parser_version"] = payload.get("parser_version") or PARSER_VERSION_IMAGE
        fields["used_ocr"] = bool(payload.get("used_ocr", False))
        try:
            fields["layout_bbox"] = json.dumps(payload.get("layout_bbox")) if payload.get("layout_bbox") is not None else ""
        except Exception:
            fields["layout_bbox"] = ""
        fields["original_blob_key"] = payload.get("original_blob_key") or payload.get("source_blob") or ""
        fields["container"] = AZURE_CONTAINER
        return fields

    def write_payload(self, payload: Dict[str, Any]) -> int:
        try:
            sanitize_payload(payload)
        except Exception:
            pass
        self._rows.append(self._normalize(payload))
        return 1

    def finalize_and_upload(self, out_basename: str) -> Tuple[int, str, str, int]:
        if not self._rows:
            return 0, "", "", 0
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception:
            raise RuntimeError("pyarrow required (pip install pyarrow)")
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
            pa.field("page_number", pa.int64()),
            pa.field("timestamp", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("used_ocr", pa.bool_()),
            pa.field("layout_bbox", pa.string()),
            pa.field("original_blob_key", pa.string()),
            pa.field("container", pa.string())
        ])
        cols: Dict[str, List[Any]] = {name: [] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table = pa.Table.from_pydict(cols, schema=schema)
        existing_md = table.schema.metadata or {}
        new_md = dict(existing_md)
        new_md.update({
            b"schema_version": CHUNKED_SCHEMA_VERSION.encode("utf-8"),
            b"parser_version": PARSER_VERSION_IMAGE.encode("utf-8"),
            b"producer": b"images_parser",
            b"created_at": datetime.utcnow().isoformat().encode("utf-8")
        })
        table = table.replace_schema_metadata(new_md)
        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".parquet", dir="/tmp")
        tmpfile.close()
        pq.write_table(table, tmpfile.name, compression="zstd")
        local_parquet_path = tmpfile.name
        with open(local_parquet_path, "rb") as fh:
            b = fh.read()
        sha = sha256_hex_bytes(b)
        size = os.path.getsize(local_parquet_path)
        parquet_key = out_basename + ".parquet"
        target_key = STORAGE_CHUNKED_PREFIX + parquet_key
        storage_upload_file_atomic(local_parquet_path, target_key, content_type="application/octet-stream")
        try:
            os.unlink(local_parquet_path)
        except Exception:
            pass
        return len(self._rows), target_key, sha, size

# ---------------- helpers for OCR and chunking ----------------
def _derive_doc_id_from_head(blob_key: str, head_obj: dict, manifest: dict) -> str:
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        return manifest.get("file_hash")
    etag = head_obj.get("ETag", "") if isinstance(head_obj, dict) else ""
    if isinstance(etag, str):
        etag = etag.strip('"')
    if etag:
        return sha256_hex(blob_key + str(etag))
    lm = head_obj.get("LastModified", "") if isinstance(head_obj, dict) else ""
    if lm:
        return sha256_hex(blob_key + str(lm))
    base = os.path.basename(blob_key)
    if base:
        return base
    return sha256_hex(blob_key)

def _mime_type_for_ext(ext: str) -> str:
    e = ext.lower().lstrip(".")
    mapping = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "tif": "image/tiff", "tiff": "image/tiff", "bmp": "image/bmp", "gif": "image/gif"}
    return mapping.get(e, "application/octet-stream")

def reflow_and_clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[\x00-\x1F]+', ' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def postprocess_ocr_text(text: str) -> str:
    text = canonicalize_text(text)
    text = reflow_and_clean_text(text)
    if not IMAGE_ENABLE_WORDSEGMENT:
        return text
    try:
        from wordsegment import load, segment
        load()
        tokens = []
        for tok in text.split():
            if len(tok) > 8 and " " not in tok and tok.isalpha():
                seg = segment(tok)
                if seg and len(" ".join(seg)) < len(tok) + 5:
                    tokens.append(" ".join(seg))
                    continue
            tokens.append(tok)
        return " ".join(tokens)
    except Exception:
        return text

@contextmanager
def without_cwd_on_syspath():
    saved = list(sys.path)
    try:
        cwd = os.getcwd()
        sys.path = [p for p in sys.path if p not in ("", cwd)]
        yield
    finally:
        sys.path[:] = saved

def _create_rapidocr_engine(model_dir: Optional[str] = None):
    models_path = model_dir or os.getenv("RAPIDOCR_MODEL_DIR", "/opt/models/rapidocr")
    tried = []
    last_exc = None
    candidates = ("rapidocr_onnxruntime", "rapidocr")
    for module_name in candidates:
        try:
            with without_cwd_on_syspath():
                mod = importlib.import_module(module_name)
            RapidOCR = getattr(mod, "RapidOCR", None)
            if RapidOCR is None:
                raise ImportError(f"module {module_name} does not expose RapidOCR")
            try:
                eng = RapidOCR(model_dir=models_path)
            except TypeError:
                eng = RapidOCR(models_path)
            return eng
        except Exception as e:
            tried.append((module_name, repr(e))); last_exc = e
    raise ImportError("RapidOCR import failed; tried: " + "; ".join(f"{m}:{err}" for m, err in tried)) from last_exc

def get_image_ocr_engine():
    if IMAGE_DISABLE_OCR and not IMAGE_FORCE_OCR:
        logger.info("IMAGE_DISABLE_OCR", "skip", reason="IMAGE_DISABLE_OCR=true and IMAGE_FORCE_OCR=false")
        return "none", None
    choice = (IMAGE_OCR_ENGINE or "auto").lower()
    if choice == "rapidocr":
        try:
            eng = _create_rapidocr_engine()
            logger.info("rapidocr_selected", "Using RapidOCR", model_dir=os.getenv("RAPIDOCR_MODEL_DIR", "/opt/models/rapidocr"))
            return "rapidocr", eng
        except Exception as e:
            logger.exception("rapidocr_fail", "RapidOCR import/create failed", reason=str(e))
            return "none", None
    if choice == "tesseract":
        try:
            with without_cwd_on_syspath():
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
                logger.info("tesseract_selected", "Using Tesseract OCR")
                return "tesseract", pytesseract
        except Exception as e:
            logger.exception("tesseract_fail", "Tesseract import failed", reason=str(e))
            return "none", None
    if choice == "auto":
        try:
            eng = _create_rapidocr_engine()
            logger.info("auto_rapidocr", "Auto-selected RapidOCR", model_dir=os.getenv("RAPIDOCR_MODEL_DIR", "/opt/models/rapidocr"))
            return "rapidocr", eng
        except Exception as e_rapid:
            logger.warning("rapidocr_auto_fail", "RapidOCR auto-select failed", reason=repr(e_rapid))
            try:
                with without_cwd_on_syspath():
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
                    logger.info("auto_tesseract", "Auto-selected Tesseract")
                    return "tesseract", pytesseract
            except Exception as e_tess:
                logger.warning("tesseract_auto_fail", "Tesseract auto-select failed", reason=repr(e_tess))
                logger.error("ocr_none", "No OCR engine available. OCR will be skipped.")
                return "none", None
    try:
        eng = _create_rapidocr_engine()
        logger.info("rapidocr_fallback", "Fallback RapidOCR")
        return "rapidocr", eng
    except Exception as e:
        logger.exception("rapidocr_fallback_fail", "Fallback RapidOCR failed", reason=str(e))
        return "none", None

def run_ocr_on_pil_image(engine_name: str, engine_obj, pil_img) -> str:
    if engine_name == "rapidocr" and engine_obj is not None:
        try:
            import numpy as np  # optional
            # convert PIL image to numpy RGB (OpenCV BGR expected by some RapidOCRs)
            img_arr = None
            if hasattr(pil_img, "convert"):
                img_arr = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
            else:
                try:
                    img_arr = np.asarray(pil_img)
                except Exception:
                    img_arr = None
            if img_arr is None:
                logger.error("rapidocr_conv", "RapidOCR input image conversion failed")
                return ""
            res = engine_obj(img_arr)
            if isinstance(res, tuple) and len(res) >= 1:
                ocr_result = res[0]
            else:
                ocr_result = res
            lines: List[str] = []
            if isinstance(ocr_result, list):
                for item in ocr_result:
                    if isinstance(item, dict) and "text" in item:
                        txt = item.get("text") or item.get("rec") or ""
                        if txt:
                            lines.append(str(txt))
                            continue
                    if isinstance(item, (list, tuple)):
                        found = False
                        for element in item:
                            if isinstance(element, str) and element.strip():
                                lines.append(element.strip()); found = True; break
                            if isinstance(element, (list, tuple)) and element and isinstance(element[0], str):
                                lines.append(element[0].strip()); found = True; break
                        if found:
                            continue
                        try:
                            joined = " ".join([str(x) for x in item if x is not None])
                            if joined.strip():
                                lines.append(joined.strip()); continue
                        except Exception:
                            pass
                    try:
                        s = str(item)
                        if s and s.strip():
                            lines.append(s.strip())
                    except Exception:
                        pass
            else:
                try:
                    s = str(ocr_result)
                    if s and s.strip():
                        lines.append(s.strip())
                except Exception:
                    pass
            return "\n".join([ln for ln in lines if ln])
        except Exception:
            logger.exception("rapidocr_exec", "RapidOCR failed to OCR image")
            return ""
    if engine_name == "tesseract" and engine_obj is not None:
        try:
            pytesseract = engine_obj
            return pytesseract.image_to_string(pil_img, lang=IMAGE_TESSERACT_LANG, config=TESSERACT_CONFIG)
        except Exception:
            logger.exception("tesseract_exec", "Tesseract OCR failed to OCR image")
            return ""
    return ""

# ---------------- chunking helpers (SentenceChunker simplified) ----------------
def split_long_sentence_by_words(sent_text: str, max_tokens: int, encoder: Any) -> List[str]:
    words = sent_text.split()
    pieces: List[str] = []
    cur_words: List[str] = []
    cur_tok = 0
    for w in words:
        try:
            toks = encoder.encode(w)
        except Exception:
            toks = w.split()
        l = len(toks)
        if cur_tok + l > max_tokens:
            if cur_words:
                pieces.append(" ".join(cur_words)); cur_words = []; cur_tok = 0
            if l > max_tokens:
                try:
                    tok_ids = encoder.encode(w)
                except Exception:
                    tok_ids = list(w)
                i = 0
                while i < len(tok_ids):
                    chunk_ids = tok_ids[i:i+max_tokens]
                    try:
                        pieces.append(encoder.decode(chunk_ids))
                    except Exception:
                        pieces.append("".join(str(x) for x in chunk_ids))
                    i += max_tokens
                continue
        cur_words.append(w); cur_tok += l
    if cur_words:
        pieces.append(" ".join(cur_words))
    return pieces

class TokenEncoder:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name; self.backend = "whitespace"
        try:
            import tiktoken
            inc = getattr(tiktoken, "encoding_for_model", None)
            enc_local = inc(model_name) if inc is not None else tiktoken.get_encoding(model_name)
            self.encode = lambda txt: enc_local.encode(txt)
            self.decode = lambda toks: enc_local.decode(toks)
            self.backend = "tiktoken"
        except Exception:
            self.encode = lambda txt: txt.split()
            self.decode = lambda toks: " ".join(toks)

class SentenceChunker:
    def __init__(self, max_tokens_per_chunk: Optional[int] = None, overlap_sentences: Optional[int] = None, token_model: str = "gpt2", nlp=None, min_tokens_per_chunk: Optional[int] = None):
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", str(MAX_TOKENS_PER_CHUNK))) if max_tokens_per_chunk is None else int(max_tokens_per_chunk)
        self.overlap_sentences = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", str(NUMBER_OF_OVERLAPPING_SENTENCES))) if overlap_sentences is None else int(overlap_sentences)
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.min_tokens_per_chunk = int(os.getenv("MIN_TOKENS_PER_CHUNK", str(MIN_TOKENS_PER_CHUNK))) if min_tokens_per_chunk is None else int(min_tokens_per_chunk)
        if self.min_tokens_per_chunk < 0:
            raise ValueError("min_tokens_per_chunk must be >= 0")
        self.encoder = TokenEncoder(model_name=token_model)
        self.nlp = nlp or self._make_sentencizer()

    @staticmethod
    def _make_sentencizer():
        try:
            import spacy
            try:
                return spacy.load("en_core_web_sm")
            except Exception:
                nlp = spacy.blank("en")
                try:
                    from spacy.pipeline import Sentencizer as SentencizerCls
                    nlp.add_pipe("sentencizer")
                except Exception:
                    try:
                        nlp.add_pipe(SentencizerCls())
                    except Exception:
                        raise RuntimeError("Failed to add Sentencizer")
                return nlp
        except Exception:
            logger.warning("spacy_unavailable", "spaCy unavailable; falling back to regex")
            return None

    def _sentences_with_offsets_regex(self, text: str):
        pattern = re.compile(r'(?s).*?[\.\!\?]["\']?\s+|.+$')
        items = []; pos = 0
        for m in pattern.finditer(text):
            s = m.group(0)
            if not s or s.strip() == "":
                pos = m.end(); continue
            start = pos; end = pos + len(s); items.append((s.strip(), start, end)); pos = m.end()
        if not items and text.strip():
            items = [(text.strip(), 0, len(text))]
        return items

    def _sentences_with_offsets(self, text: str):
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                return [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
            except Exception:
                pass
        return self._sentences_with_offsets_regex(text)

    def chunk_document(self, text: str):
        sentences = self._sentences_with_offsets(text)
        sent_items = [{"text": s, "start_char": sc, "end_char": ec, "orig_idx": i, "is_remainder": False} for i, (s, sc, ec) in enumerate(sentences)]
        i = 0; n = len(sent_items); prev_chunk = None
        while i < n:
            cur_token_count = 0; chunk_sent_texts = []; chunk_start_idx = i
            chunk_start_char = sent_items[i]["start_char"] if i < n else None; chunk_end_char = None; is_truncated_sentence = False
            while i < n:
                sent_text = sent_items[i]["text"]; tok_ids = self.encoder.encode(sent_text); sent_tok_len = len(tok_ids)
                if sent_tok_len > self.max_tokens_per_chunk:
                    pieces = split_long_sentence_by_words(sent_text, self.max_tokens_per_chunk, self.encoder)
                    if not pieces:
                        pieces = [sent_text[:1000]]
                    sent_items[i]["text"] = pieces[0]
                    for j, rem in enumerate(pieces[1:], 1):
                        sent_items.insert(i + j, {"text": rem, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True})
                    n = len(sent_items); tok_ids = self.encoder.encode(sent_items[i]["text"]); sent_tok_len = len(tok_ids)
                if cur_token_count + sent_tok_len > self.max_tokens_per_chunk:
                    if not chunk_sent_texts:
                        prefix_tok_ids = tok_ids[: self.max_tokens_per_chunk]
                        try:
                            prefix_text = self.encoder.decode(prefix_tok_ids)
                        except Exception:
                            prefix_text = " ".join(str(x) for x in prefix_tok_ids)
                        chunk_sent_texts.append(prefix_text); cur_token_count = len(prefix_tok_ids); is_truncated_sentence = True
                        remainder_tok_ids = tok_ids[self.max_tokens_per_chunk :]
                        if remainder_tok_ids:
                            try:
                                remainder_text = self.encoder.decode(remainder_tok_ids)
                            except Exception:
                                remainder_text = " ".join(str(x) for x in remainder_tok_ids)
                            sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True}
                        else:
                            i += 1
                        break
                    break
                chunk_sent_texts.append(sent_text); cur_token_count += sent_tok_len; chunk_end_char = sent_items[i]["end_char"]; i += 1
            if not chunk_sent_texts:
                i += 1; continue
            chunk_meta = {"text": " ".join(chunk_sent_texts).strip(), "token_count": cur_token_count, "start_sentence_idx": chunk_start_idx, "end_sentence_idx": i, "start_char": chunk_start_char, "end_char": chunk_end_char, "is_truncated_sentence": is_truncated_sentence}
            new_start = max(chunk_start_idx + 1, chunk_meta["end_sentence_idx"] - self.overlap_sentences)
            if prev_chunk is None:
                prev_chunk = chunk_meta
            else:
                if chunk_meta["token_count"] < self.min_tokens_per_chunk:
                    prev_chunk["text"] = prev_chunk["text"] + " " + chunk_meta["text"]
                    prev_chunk["token_count"] = prev_chunk["token_count"] + chunk_meta["token_count"]
                    prev_chunk["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
                    prev_chunk["end_char"] = chunk_meta["end_char"]
                    prev_chunk["is_truncated_sentence"] = prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
                else:
                    yield prev_chunk; prev_chunk = chunk_meta
            i = new_start; n = len(sent_items)
        if prev_chunk is not None:
            yield prev_chunk

    @classmethod
    def from_env(cls, **kwargs):
        max_tokens = int(os.getenv("MAX_TOKENS_PER_CHUNK", str(MAX_TOKENS_PER_CHUNK)))
        overlap = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", str(NUMBER_OF_OVERLAPPING_SENTENCES)))
        min_tokens = int(os.getenv("MIN_TOKENS_PER_CHUNK", str(MIN_TOKENS_PER_CHUNK)))
        token_model = os.getenv("TOKEN_ENCODER_MODEL", os.getenv("TOKEN_ENCODER", "gpt2"))
        return cls(max_tokens_per_chunk=max_tokens, overlap_sentences=overlap, token_model=token_model, nlp=None, min_tokens_per_chunk=min_tokens)

# ---------------- core parse function ----------------
def process_image_s3_object(blob_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    client = get_storage_client()
    try:
        head_obj = client.head_object(Bucket=AZURE_CONTAINER, Key=blob_key)
    except Exception:
        head_obj = {}
    doc_id = _derive_doc_id_from_head(blob_key, head_obj or {}, manifest or {})
    out_basename = f"{doc_id}"
    raw_manifest_key = blob_key + ".manifest.json"
    if not FORCE_OVERWRITE:
        if storage_object_exists(raw_manifest_key):
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("skip_manifest_exists", "raw_manifest_exists", key=raw_manifest_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
        if storage_object_exists(STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"):
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("skip_parquet_exists", "parquet_exists", key=out_basename + ".parquet")
            try:
                if not storage_object_exists(raw_manifest_key):
                    head = client.head_object(Bucket=AZURE_CONTAINER, Key=STORAGE_CHUNKED_PREFIX + out_basename + ".parquet")
                    etag = head.get("ETag", "")
                    if isinstance(etag, str):
                        etag = etag.strip('"')
                    size = head.get("ContentLength", 0)
                    raw_manifest = {"raw_key": blob_key, "doc_id": doc_id, "chunked_key": STORAGE_CHUNKED_PREFIX + out_basename + ".parquet", "rows": 0, "sha256": etag, "size_bytes": size, "schema_version": CHUNKED_SCHEMA_VERSION, "parser_version": PARSER_VERSION_IMAGE, "created_at": datetime.utcnow().isoformat() + "Z", "container": AZURE_CONTAINER}
                    client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
            except Exception:
                pass
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}

    lower = blob_key.lower()
    allowed = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif")
    if not any(lower.endswith(a) for a in allowed):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.error("unsupported_ext", "Unsupported file extension", key=blob_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": "Unsupported file extension for images parser"}

    ext_on_disk = os.path.splitext(blob_key)[1] or ".img"
    try:
        local_img = download_blob_to_temp(blob_key, ext_on_disk)
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.error("download_failed", "Could not download blob", key=blob_key, error=str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash"); out_basename = f"{doc_id}"

    if not FORCE_OVERWRITE and storage_object_exists(STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"):
        try:
            os.unlink(local_img)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("skip_parquet_post_download", "parquet_exists_post_download", key=out_basename + ".parquet")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}

    img_ocr_name, img_ocr_obj = get_image_ocr_engine()
    try:
        chunker = SentenceChunker.from_env()
    except Exception as e:
        try:
            os.unlink(local_img)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.exception("chunker_init_failed", "Failed to initialise SentenceChunker")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

    try:
        from PIL import Image as PILImage
    except Exception as e:
        try:
            os.unlink(local_img)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.exception("pillow_missing", "Pillow required (pip install pillow)")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

    try:
        writer = S3ParquetWriter(doc_id=doc_id)
        saved = 0
        try:
            im = PILImage.open(local_img)
        except Exception as e:
            try:
                os.unlink(local_img)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.exception("open_failed", "PIL failed to open image", key=blob_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

        n_frames = getattr(im, "n_frames", 1)
        for frame_idx in range(n_frames):
            frame_start = time.perf_counter()
            try:
                if n_frames > 1:
                    im.seek(frame_idx)
                frame = im.convert("RGB")
            except Exception:
                frame = im.convert("RGB")

            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            used_ocr = False
            ocr_text = ""
            if img_bytes and len(img_bytes) >= IMAGE_MIN_IMG_SIZE_BYTES and img_ocr_name != "none":
                ocr_text = run_ocr_on_pil_image(img_ocr_name, img_ocr_obj, frame)
                if ocr_text and ocr_text.strip():
                    used_ocr = True
                    ocr_text = postprocess_ocr_text(ocr_text)

            source_url = f"az://{AZURE_CONTAINER}/{blob_key}"

            if not ocr_text:
                chunk_id = f"{doc_id}_f{frame_idx+1}_0"
                payload = {
                    "document_id": doc_id,
                    "file_name": os.path.basename(blob_key),
                    "chunk_id": chunk_id,
                    "chunk_type": "image_frame_chunk",
                    "text": "",
                    "token_count": 0,
                    "figures": [],
                    "file_type": _mime_type_for_ext(ext_on_disk),
                    "source_url": source_url,
                    "page_number": frame_idx+1,
                    "timestamp": datetime.utcnow().isoformat()+"Z",
                    "parser_version": PARSER_VERSION_IMAGE,
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": [],
                    "used_ocr": used_ocr,
                    "heading_path": [],
                    "headings": [],
                    "layout_bbox": None,
                    "original_blob_key": blob_key
                }
                sanitize_payload(payload)
                writer.write_payload(payload); saved += 1
                logger.info("buffer_empty_frame", "Buffered empty frame chunk", chunk=chunk_id)
                continue

            for idx, chunk in enumerate(chunker.chunk_document(ocr_text)):
                chunk_id = f"{doc_id}_f{frame_idx+1}_{idx}"
                payload = {
                    "document_id": doc_id,
                    "file_name": os.path.basename(blob_key),
                    "chunk_id": chunk_id,
                    "chunk_type": "image_frame_chunk",
                    "text": chunk["text"],
                    "token_count": int(chunk["token_count"]),
                    "figures": [],
                    "file_type": _mime_type_for_ext(ext_on_disk),
                    "source_url": source_url,
                    "page_number": frame_idx+1,
                    "timestamp": datetime.utcnow().isoformat()+"Z",
                    "parser_version": PARSER_VERSION_IMAGE,
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": [],
                    "used_ocr": used_ocr,
                    "heading_path": [],
                    "headings": [],
                    "layout_bbox": None,
                    "original_blob_key": blob_key
                }
                sanitize_payload(payload)
                writer.write_payload(payload); saved += 1

            frame_ms = int((time.perf_counter() - frame_start) * 1000)
            logger.info("frame_processed", "Processed frame", frame=frame_idx+1, total_frames=n_frames, frame_ms=frame_ms, chunks_so_far=saved)

        if saved == 0:
            try:
                os.unlink(local_img)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("no_chunks", "No chunks produced", key=blob_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}

        count, uploaded_key, sha, size = writer.finalize_and_upload(out_basename)
        try:
            os.unlink(local_img)
        except Exception:
            pass

        raw_manifest = {
            "raw_key": blob_key,
            "doc_id": doc_id,
            "chunked_key": uploaded_key,
            "rows": count,
            "sha256": sha,
            "size_bytes": size,
            "schema_version": CHUNKED_SCHEMA_VERSION,
            "parser_version": PARSER_VERSION_IMAGE,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "container": AZURE_CONTAINER,
            "original_file_name": os.path.basename(blob_key)
        }
        try:
            client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
        except Exception:
            logger.warning("manifest_write_failed", "Failed to write raw manifest", key=blob_key)

        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("write_complete", "Wrote chunks", count=count, raw=blob_key, chunked=uploaded_key, duration_ms=total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e:
        try:
            if 'local_img' in locals():
                try:
                    os.unlink(local_img)
                except Exception:
                    pass
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.exception("process_error", "Error while processing", key=blob_key, error=str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

def parse_file(s3_key: str, manifest: dict) -> dict:
    start = time.perf_counter()
    if not AZURE_CONTAINER:
        raise RuntimeError("AZURE_CONTAINER must be set in environment")
    try:
        return process_image_s3_object(s3_key, manifest or {})
    except Exception as e:
        total_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("parse_file_error", "parse_file error", key=s3_key, error=str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

if __name__ == "__main__":
    engine_name, engine_obj = get_image_ocr_engine()
    logger.info("cli_startup", "Engine result", engine=engine_name, loaded=bool(engine_obj))
