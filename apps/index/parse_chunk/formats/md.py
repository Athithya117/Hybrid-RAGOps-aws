#!/usr/bin/env python3
"""
md_parser_dualmode.py

Dual-mode markdown -> parquet parser for Azure blob storage.

Authentication selection is DETERMINISTIC and solely driven by:
  AZURE_USE_MANAGED_IDENTITY=1 (or USE_MANAGED_IDENTITY=1) -> Managed Identity mode
  otherwise -> key/SAS/connection-string mode (fsspec/adlfs)

Fail-fast on missing prerequisites and required env vars.
"""
from __future__ import annotations
import os
import sys
import time
import json
import hashlib
import tempfile
import re
import unicodedata
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# ---------- structured logger ----------
class LoggerShim:
    def __init__(self, name: str):
        self.name = name
    def _emit(self, level: str, event: str, msg: str = "", **extra):
        o = {"ts": datetime.utcnow().isoformat() + "Z", "level": level, "event": event, "msg": msg}
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
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "info"); self._emit("info", event, msg, **kw)
    def warning(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "warn"); self._emit("warn", event, msg, **kw)
    def warn(self, a, b=None, *fmt_args, **kwargs):
        self.warning(a, b, *fmt_args, **kwargs)
    def error(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "error"); self._emit("error", event, msg, **kw)
    def exception(self, a, b=None, *fmt_args, **kwargs):
        import traceback
        tb = traceback.format_exc()
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "exception"); kw.update({"tb": tb}); self._emit("error", event, msg, **kw)

log = LoggerShim("md_parser")

# ---------- deterministic runtime switch: USE_MANAGED_IDENTITY only ----------
# Accept either AZURE_USE_MANAGED_IDENTITY or USE_MANAGED_IDENTITY for convenience,
# but DO NOT infer from ENV (Prod/Stage) — user explicitly requested AUTH-mode switch only.
_USE_MI_RAW = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower()
USE_MANAGED_IDENTITY = _USE_MI_RAW in ("1", "true", "yes")

# ---------- runtime config (Azure-only) ----------
AZURE_CONTAINER = (os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER") or "").strip()
if not AZURE_CONTAINER:
    sys.stderr.write("ERROR: AZURE_CONTAINER (or STORAGE_CONTAINER / AZ_CONTAINER) environment variable must be set\n")
    sys.exit(2)

STORAGE_RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("AZURE_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
STORAGE_CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("AZURE_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
DEFAULT_OVERLAP = max(1, int(MAX_TOKENS_PER_CHUNK * 0.1))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", str(DEFAULT_OVERLAP)))
if OVERLAP_TOKENS >= MAX_TOKENS_PER_CHUNK:
    OVERLAP_TOKENS = max(1, MAX_TOKENS_PER_CHUNK - 1)
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
PARSER_VERSION = os.getenv("PARSER_VERSION_MD", "markdown-it-py-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"
PUT_RETRIES = int(os.getenv("PUT_RETRIES", "3"))
PUT_BACKOFF = float(os.getenv("PUT_BACKOFF", "0.3"))
CHUNKED_SCHEMA_VERSION = os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1")

# ---------- dependency checks (fail-fast with clear instructions) ----------
# For key/SAS/connstr mode we require fsspec + adlfs
FSSPEC_AVAILABLE = False
ADLFS_AVAILABLE = False
try:
    import fsspec  # type: ignore
    FSSPEC_AVAILABLE = True
except Exception:
    fsspec = None

try:
    import adlfs  # type: ignore
    ADLFS_AVAILABLE = True
except Exception:
    adlfs = None

# For managed identity we require azure.identity + azure.storage.blob
AZURE_SDK_AVAILABLE = False
try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.storage.blob import BlobServiceClient, ContainerClient  # type: ignore
    AZURE_SDK_AVAILABLE = True
except Exception:
    DefaultAzureCredential = None  # type: ignore
    BlobServiceClient = None  # type: ignore
    ContainerClient = None  # type: ignore

# Optional parsing/token libs
try:
    from markdown_it import MarkdownIt  # type: ignore
except Exception:
    MarkdownIt = None
try:
    import tiktoken
except Exception:
    tiktoken = None

# ---------- storage wiring (dual-mode) ----------
def build_storage_options() -> Dict[str, str]:
    """
    Build fsspec az options for key/SAS/connstr mode.
    Returns empty dict if managed identity (should not be used by fsspec).
    """
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
FS: Optional[object] = None  # will be fsspec filesystem or None
BLOB_CLIENT = None
STORAGE_ROOT = f"az://{AZURE_CONTAINER.rstrip('/')}/"

# Validate prerequisites now (fail fast)
def _prevalidate_auth_envs_or_exit():
    if USE_MANAGED_IDENTITY:
        if not AZURE_SDK_AVAILABLE:
            sys.stderr.write("ERROR: AZURE_USE_MANAGED_IDENTITY=1 but azure.identity / azure-storage-blob not installed.\n")
            sys.stderr.write("Install with: pip install azure-identity azure-storage-blob\n")
            sys.exit(2)
        account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
        if not account_name:
            sys.stderr.write("ERROR: AZURE_STORAGE_ACCOUNT_NAME (or AZURE_ACCOUNT_NAME) required for managed identity mode.\n")
            sys.exit(2)
    else:
        # key/SAS/connstr mode requirement
        if not FSSPEC_AVAILABLE or not ADLFS_AVAILABLE:
            sys.stderr.write("ERROR: key/SAS/connection-string mode requires fsspec + adlfs.\n")
            sys.stderr.write("Install with: pip install fsspec adlfs\n")
            sys.exit(2)
        acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
        sas = os.environ.get("AZURE_SAS_TOKEN")
        if not conn and not (acct and (key or sas or os.environ.get("AZURE_ANON"))):
            sys.stderr.write("ERROR: non-managed identity mode requires AZURE_STORAGE_ACCOUNT_NAME plus one of AZURE_STORAGE_ACCOUNT_KEY or AZURE_SAS_TOKEN or AZURE_STORAGE_CONNECTION_STRING.\n")
            sys.exit(2)

_prevalidate_auth_envs_or_exit()

# Initialize appropriate client(s)
if USE_MANAGED_IDENTITY:
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    account_url = f"https://{account_name}.{os.environ.get('AZURE_ENDPOINT_SUFFIX','core.windows.net')}"
    try:
        CREDENTIAL = DefaultAzureCredential()
        BLOB_CLIENT = BlobServiceClient(account_url=account_url, credential=CREDENTIAL, connection_timeout=60)
        FS = None
        log.info("azure_init", "Initialized BlobServiceClient in managed-identity mode", account=account_name)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to init BlobServiceClient (managed identity): {e}\n")
        sys.exit(2)
else:
    try:
        FS = fsspec.filesystem("az", **FS_OPTS)  # type: ignore
        BLOB_CLIENT = None
        log.info("fsspec_init", "Initialized fsspec az filesystem", opts=list(FS_OPTS.keys()))
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to initialize fsspec az filesystem: {e}\n")
        sys.exit(2)

# ---------- helper functions ----------
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

def sha256_hex_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", ln) for ln in s.split("\n")]
    return "\n".join(lines).strip()

def try_decode_bytes(b: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return b.decode(encoding)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")

# ---------- token encoder & md parser factories ----------
_tiktoken_enc = None
_md_parser = None

def get_encoder():
    global _tiktoken_enc
    if _tiktoken_enc is not None:
        return _tiktoken_enc
    try:
        if tiktoken is None:
            _tiktoken_enc = None
            return None
        try:
            _tiktoken_enc = tiktoken.get_encoding(ENC_NAME)
        except Exception:
            try:
                _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _tiktoken_enc = None
    except Exception:
        _tiktoken_enc = None
    return _tiktoken_enc

def get_md_parser():
    global _md_parser
    if _md_parser is not None:
        return _md_parser
    try:
        if MarkdownIt is None:
            _md_parser = None
        else:
            _md_parser = MarkdownIt()
    except Exception as e:
        log.warning("md_parser_unavailable", "markdown-it-py not available", reason=str(e))
        _md_parser = None
    return _md_parser

def token_count_for(text: str) -> int:
    if not text:
        return 0
    enc = get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())

# ---------- Storage client (dual-mode) ----------
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
            info = self.fs.info(full)  # type: ignore
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
                "ContentLength": getattr(props, "size", 0),
                "ETag": getattr(props, "etag", ""),
                "LastModified": getattr(props, "last_modified", ""),
                "Metadata": getattr(props, "metadata", {}) or {},
            }
            return out

    def get_object(self, Bucket, Key):
        if self.fs is not None:
            full = full_path_from_key(Key)
            with self.fs.open(full, "rb") as f:  # type: ignore
                data = f.read()
            return {"Body": data}
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
            stream = blob_client.download_blob()
            data = stream.readall()
            return {"Body": data}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        if self.fs is not None:
            full = full_path_from_key(Key)
            # normalize Body -> bytes
            data_bytes = None
            if isinstance(Body, (bytes, bytearray)):
                data_bytes = bytes(Body)
            elif isinstance(Body, str):
                data_bytes = Body.encode("utf-8")
            elif hasattr(Body, "read"):
                try:
                    data_bytes = Body.read()
                    if isinstance(data_bytes, str):
                        data_bytes = data_bytes.encode("utf-8")
                except Exception:
                    data_bytes = b""
            else:
                try:
                    data_bytes = bytes(Body)
                except Exception:
                    data_bytes = str(Body).encode("utf-8")
            with self.fs.open(full, "wb") as f:  # type: ignore
                f.write(data_bytes or b"")
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
            if hasattr(self.fs, "put"):  # type: ignore
                self.fs.put(LocalFile, full)  # type: ignore
            else:
                with open(LocalFile, "rb") as lf:
                    d = lf.read()
                with self.fs.open(full, "wb") as f:  # type: ignore
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
            with self.fs.open(full_src, "rb") as rf:  # type: ignore
                data = rf.read()
            with self.fs.open(full_dst, "wb") as wf:  # type: ignore
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
                self.fs.rm(full)  # type: ignore
            except Exception:
                try:
                    self.fs.delete(full)  # type: ignore
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
                            found = self.fs.find(root_path)  # type: ignore
                        else:
                            found = self.fs.glob(root_path + "**", recursive=True)  # type: ignore
                    except Exception:
                        found = []
                    page = {"Contents": []}
                    for f in found:
                        try:
                            info = self.fs.info(f)  # type: ignore
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

# ---------- singleton client ----------
_storage_client: Optional[AzureStorageClient] = None
_storage_lock = threading.Lock()
def get_storage_client_singleton():
    global _storage_client
    if _storage_client is None:
        with _storage_lock:
            if _storage_client is None:
                if USE_MANAGED_IDENTITY:
                    _storage_client = AzureStorageClient(fs_obj=None, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=BLOB_CLIENT)
                else:
                    _storage_client = AzureStorageClient(fs_obj=FS, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=None)
    return _storage_client

def storage_blob_exists(key: str) -> bool:
    """
    Key is the blob path relative to container, e.g. 'data/chunked/x.parquet'
    """
    client = get_storage_client_singleton()
    try:
        client.head_object(Bucket=AZURE_CONTAINER, Key=key)
        return True
    except Exception:
        return False

# ---------- core utilities ----------
def retry(func, retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning("retry_attempt", "attempt=%d error=%s", attempt + 1, str(e))
            time.sleep(delay)
            delay *= backoff

# ---------- chunking / parsing helpers (kept as original) ----------
def _is_rootish(h: Any) -> bool:
    if h is None:
        return True
    try:
        return str(h).strip().lower() in ("", "root")
    except Exception:
        return False

def build_header_sections(raw_text: str) -> List[Dict[str, Any]]:
    lines = raw_text.splitlines(keepends=True)
    mdp = get_md_parser()
    if mdp is None:
        return [{"heading_path": [], "heading": "", "level": 0, "start_line": 0, "end_line": len(lines), "lines": lines}]
    try:
        tokens = mdp.parse(raw_text)
    except Exception:
        return [{"heading_path": [], "heading": "", "level": 0, "start_line": 0, "end_line": len(lines), "lines": lines}]
    stack = [{"heading_path": [], "heading": "", "level": 0, "start_line": None, "end_line": None}]
    sections_out: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        ttype = getattr(tok, "type", "")
        if ttype == "heading_open":
            tag = getattr(tok, "tag", "h1")
            try:
                level = int(tag[1])
            except Exception:
                level = 1
            map_tuple = getattr(tok, "map", None)
            heading_text = ""
            if i + 1 < len(tokens) and getattr(tokens[i + 1], "type", "") == "inline":
                heading_text = getattr(tokens[i + 1], "content", "").strip()
            while stack and stack[-1]["level"] >= level:
                completed = stack.pop()
                if completed.get("start_line") is not None:
                    sections_out.append(completed)
            parent_path = [p for p in (stack[-1]["heading_path"][:] if stack else []) if not _is_rootish(p)]
            new_path = parent_path + ([] if _is_rootish(heading_text) else [heading_text])
            sec = {"heading_path": new_path, "heading": "" if _is_rootish(heading_text) else heading_text, "level": level, "start_line": None, "end_line": None}
            if map_tuple:
                sec["start_line"] = map_tuple[0]
                sec["end_line"] = map_tuple[1]
            stack.append(sec)
            i += 1
            continue
        map_tuple = getattr(tok, "map", None)
        if map_tuple:
            sline, eline = map_tuple[0], map_tuple[1]
            top = stack[-1]
            if top.get("start_line") is None or sline < top["start_line"]:
                top["start_line"] = sline
            if top.get("end_line") is None or eline > top["end_line"]:
                top["end_line"] = eline
        i += 1
    while stack:
        completed = stack.pop()
        if completed.get("start_line") is not None:
            sections_out.append(completed)
    total_lines = len(lines)
    normalized_sections = []
    for sec in sections_out:
        s = sec.get("start_line")
        e = sec.get("end_line")
        if s is None:
            continue
        s = max(0, s)
        e = min(total_lines, e)
        if s >= e and s < total_lines:
            e = s + 1
        heading_path = [h for h in sec.get("heading_path", []) if not _is_rootish(h)]
        heading = "" if _is_rootish(sec.get("heading", "")) else sec.get("heading", "")
        sec_copy = {"heading_path": heading_path, "heading": heading, "level": sec.get("level", 0), "start_line": s, "end_line": e, "lines": lines[s:e]}
        normalized_sections.append(sec_copy)
    normalized_sections_sorted = sorted(normalized_sections, key=lambda x: (x["start_line"], x["end_line"]))
    merged: List[Dict[str, Any]] = []
    last_end = 0
    if normalized_sections_sorted:
        first_start = normalized_sections_sorted[0]["start_line"]
        if first_start > 0:
            merged.append({"heading_path": [], "heading": "", "level": 0, "start_line": 0, "end_line": first_start, "lines": lines[0:first_start]})
    for sec in normalized_sections_sorted:
        if sec["start_line"] > last_end:
            gap_start = last_end
            gap_end = sec["start_line"]
            if gap_end > gap_start:
                merged.append({"heading_path": [], "heading": "", "level": 0, "start_line": gap_start, "end_line": gap_end, "lines": lines[gap_start:gap_end]})
        merged.append(sec)
        last_end = max(last_end, sec["end_line"])
    if last_end < total_lines:
        merged.append({"heading_path": [], "heading": "", "level": 0, "start_line": last_end, "end_line": total_lines, "lines": lines[last_end:total_lines]})
    return merged

def merge_small_sections(sections: List[Dict[str, Any]], merge_threshold: int, max_tokens: int, line_token_cache: Dict[int, int], prevent_merge_across_level: bool = False) -> List[Dict[str, Any]]:
    merged = []
    i = 0
    n = len(sections)
    while i < n:
        sec = sections[i]
        start_line = sec["start_line"]
        end_line = sec["end_line"]
        lines_acc = list(sec.get("lines", []))
        headings_acc = [] if _is_rootish(sec.get("heading", "")) else [sec.get("heading", "")]
        heading_path = [h for h in (sec.get("heading_path", []) or []) if not _is_rootish(h)]
        level = sec.get("level", 0)
        token_sum = 0
        for idx, l in enumerate(lines_acc):
            abs_idx = start_line + idx
            if abs_idx in line_token_cache:
                cnt = line_token_cache[abs_idx]
            else:
                try:
                    cnt = len(get_encoder().encode(l)) if get_encoder() else len(l.split())
                except Exception:
                    cnt = len(l.split())
                line_token_cache[abs_idx] = cnt
            token_sum += cnt
        if token_sum >= merge_threshold:
            merged.append({"heading_path": heading_path, "headings": [h for h in headings_acc if not _is_rootish(h)], "level": level, "start_line": start_line, "end_line": end_line, "lines": lines_acc, "token_count": token_sum})
            i += 1
            continue
        if merged:
            prev = merged[-1]
            if not (prevent_merge_across_level and level <= prev.get("level", 0)):
                if prev.get("token_count", 0) + token_sum <= max_tokens:
                    prev["lines"].extend(lines_acc)
                    prev["end_line"] = end_line
                    if not _is_rootish(sec.get("heading", "")):
                        prev_headings = prev.get("headings", [])
                        prev_headings.append(sec.get("heading", ""))
                        prev["headings"] = [h for h in prev_headings if not _is_rootish(h)]
                    prev["token_count"] = prev.get("token_count", 0) + token_sum
                    i += 1
                    continue
        j = i + 1
        while j < n:
            next_sec = sections[j]
            if prevent_merge_across_level and next_sec.get("level", 0) <= level:
                break
            next_start = next_sec["start_line"]
            next_lines = next_sec.get("lines", [])
            next_tokens = 0
            for idx, l in enumerate(next_lines):
                abs_idx = next_start + idx
                if abs_idx in line_token_cache:
                    cnt = line_token_cache[abs_idx]
                else:
                    try:
                        cnt = len(get_encoder().encode(l)) if get_encoder() else len(l.split())
                    except Exception:
                        cnt = len(l.split())
                    line_token_cache[abs_idx] = cnt
                next_tokens += cnt
            if token_sum + next_tokens > max_tokens:
                break
            token_sum += next_tokens
            lines_acc = lines_acc + next_lines
            nh = next_sec.get("heading", "")
            if not _is_rootish(nh):
                headings_acc.append(nh)
            end_line = next_sec["end_line"]
            j += 1
            if token_sum >= merge_threshold:
                break
        merged.append({"heading_path": heading_path, "headings": [h for h in headings_acc if not _is_rootish(h)], "level": level, "start_line": start_line, "end_line": end_line, "lines": lines_acc, "token_count": token_sum})
        i = max(j, i + 1)
    return merged

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

def split_section_by_tokens_lines(section: Dict[str, Any], overlap_tokens: int, max_tokens: int, line_token_cache: Dict[int, int]) -> List[Dict[str, Any]]:
    lines = section["lines"]
    base_start_line = section["start_line"]
    token_counts = []
    for idx, l in enumerate(lines):
        abs_idx = base_start_line + idx
        if abs_idx in line_token_cache:
            token_counts.append(line_token_cache[abs_idx])
        else:
            try:
                cnt = len(get_encoder().encode(l)) if get_encoder() else len(l.split())
            except Exception:
                cnt = len(l.split())
            line_token_cache[abs_idx] = cnt
            token_counts.append(cnt)
    n = len(lines)
    chunks = []
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
                chunks.append({"text": canonicalize_text(chunk_text), "token_count": token_count_for(chunk_text), "start_line": chunk_start_line, "end_line": chunk_end_line, "subchunk_index": p["subchunk_index"]})
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

# ---------- Parquet writer ----------
class ParquetWriter:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id; self._rows: List[Dict[str, Any]] = []
    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        fields["document_id"] = payload.get("document_id") or ""
        fields["file_name"] = payload.get("file_name") or ""
        fields["chunk_id"] = payload.get("chunk_id") or ""
        fields["chunk_type"] = payload.get("chunk_type") or ""
        fields["text"] = payload.get("text") or ""
        try: fields["token_count"] = int(payload.get("token_count") or 0)
        except Exception: fields["token_count"] = 0
        for k in ("figures", "tags", "layout_tags", "heading_path", "headings"):
            v = payload.get(k, None)
            try:
                if v is None:
                    fields[k] = "[]"
                elif isinstance(v, (list, tuple, dict)):
                    fields[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
                else:
                    fields[k] = json.dumps([v], ensure_ascii=False)
            except Exception:
                fields[k] = "[]"
        fields["file_type"] = payload.get("file_type") or "text/markdown"
        fields["source_url"] = payload.get("source_url") or ""
        lr = payload.get("line_range") or []
        if isinstance(lr, (list, tuple)) and len(lr) >= 2:
            try:
                fields["line_start"] = int(lr[0]); fields["line_end"] = int(lr[1])
            except Exception:
                fields["line_start"] = 1; fields["line_end"] = 1
        else:
            fields["line_start"] = 1; fields["line_end"] = 1
        fields["timestamp"] = payload.get("timestamp") or ""
        fields["parser_version"] = payload.get("parser_version") or PARSER_VERSION
        fields["used_ocr"] = bool(payload.get("used_ocr", False))
        return fields
    def write_payload(self, payload: Dict[str, Any]) -> int:
        self._rows.append(self._normalize(payload)); return 1
    def finalize_and_upload(self, out_basename: str) -> Tuple[int, str, str, int]:
        if not self._rows:
            return 0, "", "", 0
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:
            log.error("pyarrow_missing", "pyarrow required to write parquet", reason=str(e))
            raise
        schema = pa.schema([
            pa.field("document_id", pa.string()), pa.field("file_name", pa.string()), pa.field("chunk_id", pa.string()),
            pa.field("chunk_type", pa.string()), pa.field("text", pa.string()), pa.field("token_count", pa.int64()),
            pa.field("figures", pa.string()), pa.field("tags", pa.string()), pa.field("layout_tags", pa.string()),
            pa.field("heading_path", pa.string()), pa.field("headings", pa.string()), pa.field("file_type", pa.string()),
            pa.field("source_url", pa.string()), pa.field("line_start", pa.int64()), pa.field("line_end", pa.int64()),
            pa.field("timestamp", pa.string()), pa.field("parser_version", pa.string()), pa.field("used_ocr", pa.bool_())
        ])
        cols: Dict[str, List[Any]] = {name: [] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table = pa.Table.from_pydict(cols, schema=schema)
        existing_md = table.schema.metadata or {}
        new_md = dict(existing_md)
        new_md.update({b"schema_version": CHUNKED_SCHEMA_VERSION.encode("utf-8"), b"parser_version": PARSER_VERSION.encode("utf-8"), b"producer": b"md_parser", b"created_at": datetime.utcnow().isoformat().encode("utf-8")})
        table = table.replace_schema_metadata(new_md)
        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".parquet", dir="/tmp"); tmpfile.close()
        pq.write_table(table, tmpfile.name, compression="zstd", flavor="spark")
        local_parquet_path = tmpfile.name
        with open(local_parquet_path, "rb") as fh:
            b = fh.read()
        sha = hashlib.sha256(b).hexdigest()
        size = os.path.getsize(local_parquet_path)
        parquet_key = out_basename + ".parquet"
        # enforce chunked prefix
        storage_upload_file_atomic(local_parquet_path, STORAGE_CHUNKED_PREFIX + parquet_key, content_type="application/octet-stream")
        try: os.unlink(local_parquet_path)
        except Exception: pass
        return len(self._rows), STORAGE_CHUNKED_PREFIX + parquet_key, sha, size

# ---------- storage_upload_file_atomic (dual-mode) ----------
def storage_upload_file_atomic(local_path: str, key: str, content_type: str = "application/octet-stream"):
    """
    Atomic upload: in FS mode we use tmp file then mv; in blob mode we upload directly with overwrite.
    """
    client = get_storage_client_singleton()
    if client.fs is not None:
        full = full_path_from_key(key)
        tmp = f"{full}.tmp.{os.getpid()}.{int(time.time())}"
        for attempt in range(1, PUT_RETRIES + 1):
            try:
                if hasattr(client.fs, "put"):
                    client.fs.put(local_path, tmp)  # type: ignore
                else:
                    with open(local_path, "rb") as lf:
                        d = lf.read()
                    with client.fs.open(tmp, "wb") as f:  # type: ignore
                        f.write(d)
                # atomic move/copy into place
                if hasattr(client.fs, "mv"):
                    client.fs.mv(tmp, full)  # type: ignore
                else:
                    with client.fs.open(tmp, "rb") as rf:  # type: ignore
                        data = rf.read()
                    with client.fs.open(full, "wb") as wf:  # type: ignore
                        wf.write(data)
                    try: client.fs.rm(tmp)  # type: ignore
                    except Exception: pass
                return
            except Exception as e:
                log.warning("upload_retry", "attempt=%d key=%s error=%s", attempt, key, str(e))
                time.sleep(PUT_BACKOFF * attempt)
        raise Exception(f"atomic upload failed for {key} after {PUT_RETRIES} attempts")
    else:
        # blob client mode: upload directly (overwrite).
        container_client = client._container_client()
        blob_client = container_client.get_blob_client(key)
        for attempt in range(1, PUT_RETRIES + 1):
            try:
                with open(local_path, "rb") as lf:
                    blob_client.upload_blob(lf, overwrite=True)
                return
            except Exception as e:
                log.warning("upload_retry_blob", "attempt=%d key=%s error=%s", attempt, key, str(e))
                time.sleep(PUT_BACKOFF * attempt)
        raise Exception(f"blob upload failed for {key} after {PUT_RETRIES} attempts")

def sanitize_payload_for_raw_manifest(doc_id: str, raw_key: str, chunked_key: str, rows: int, sha: str, size: int) -> Dict[str, Any]:
    return {"raw_key": raw_key, "doc_id": doc_id, "chunked_key": chunked_key, "rows": rows, "sha256": sha, "size_bytes": size, "schema_version": CHUNKED_SCHEMA_VERSION, "parser_version": PARSER_VERSION, "created_at": datetime.utcnow().isoformat() + "Z"}

def sanitize_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    for key in ("tags", "layout_tags", "heading_path", "headings", "figures"):
        v = payload.get(key)
        if v is None:
            payload[key] = []
        elif isinstance(v, (list, tuple)):
            payload[key] = list(v)
        else:
            payload[key] = [v]
    payload["file_name"] = str(payload.get("file_name") or "")
    payload["source_url"] = str(payload.get("source_url") or "")
    try:
        payload["token_count"] = int(payload.get("token_count") or 0)
    except Exception:
        payload["token_count"] = 0
    lr = payload.get("line_range")
    if isinstance(lr, (list, tuple)) and len(lr) >= 2:
        try:
            payload["line_range"] = [int(lr[0]), int(lr[1])]
        except Exception:
            payload["line_range"] = [1, 1]
    else:
        payload["line_range"] = [1, 1]
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    payload["parser_version"] = payload.get("parser_version") or PARSER_VERSION
    payload["used_ocr"] = bool(payload.get("used_ocr", False))

# ---------- public API for router compatibility (preserves original parse_file semantics) ----------
def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    client = get_storage_client_singleton()
    try:
        head_obj = client.head_object(Bucket=AZURE_CONTAINER, Key=s3_key)
    except Exception:
        head_obj = {}
    last_modified = head_obj.get("LastModified", "")
    etag = head_obj.get("ETag", "")
    if isinstance(etag, str):
        etag = etag.strip('"')
    content_len = head_obj.get("ContentLength", 0) or 0
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash")
    else:
        if etag:
            doc_id = sha256_hex_str(s3_key + str(etag))
        else:
            doc_id = sha256_hex_str(s3_key + str(last_modified or ""))
    out_basename = f"{doc_id}"
    raw_manifest_key = s3_key + ".manifest.json"
    try:
        if not FORCE_OVERWRITE:
            if storage_blob_exists(raw_manifest_key):
                total_ms = int((time.perf_counter() - start_all) * 1000)
                log.info("skip_manifest_exists", "raw_manifest_exists", key=raw_manifest_key)
                return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
            if storage_blob_exists(STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"):
                total_ms = int((time.perf_counter() - start_all) * 1000)
                log.info("skip_parquet_exists", "parquet_exists", key=out_basename + ".parquet")
                try:
                    if not storage_blob_exists(raw_manifest_key):
                        head = client.head_object(Bucket=AZURE_CONTAINER, Key=STORAGE_CHUNKED_PREFIX + out_basename + ".parquet")
                        etag2 = head.get("ETag", "")
                        if isinstance(etag2, str): etag2 = etag2.strip('"')
                        size = head.get("ContentLength", 0)
                        raw_manifest = sanitize_payload_for_raw_manifest(doc_id, s3_key, STORAGE_CHUNKED_PREFIX + out_basename + ".parquet", 0, etag2, size)
                        client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
                except Exception:
                    pass
                return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    except Exception:
        pass
    if content_len == 0:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("skip_empty_object", "Skipping empty object", key=s3_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    try:
        obj = client.get_object(Bucket=AZURE_CONTAINER, Key=s3_key)
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("read_failed", "Could not read object", key=s3_key, error=str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    raw_body = obj.get("Body", b"")
    if isinstance(raw_body, (bytes, bytearray)):
        raw_text = try_decode_bytes(raw_body)
    else:
        try:
            raw_text = try_decode_bytes(raw_body.read())
        except Exception:
            raw_text = str(raw_body)
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash"); out_basename = f"{doc_id}"
    source_url = f"az://{AZURE_CONTAINER}/{s3_key}"
    if SAVE_SNAPSHOT:
        try:
            key = f"{STORAGE_CHUNKED_PREFIX}{doc_id}.snapshot.md"
            client.put_object(Bucket=AZURE_CONTAINER, Key=key, Body=raw_text.encode("utf-8"), ContentType="text/markdown")
        except Exception:
            pass
    canonical_full = canonicalize_text(raw_text)
    sections = build_header_sections(canonical_full)
    line_token_cache: Dict[int, int] = {}
    merged_sections = merge_small_sections(sections, MIN_TOKENS_PER_CHUNK, MAX_TOKENS_PER_CHUNK, line_token_cache)
    saved = 0; chunk_index = 1
    if not FORCE_OVERWRITE and storage_blob_exists(STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("skip_parquet_post_download", "parquet_exists", key=out_basename + ".parquet")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    writer = ParquetWriter(doc_id=doc_id)
    file_name = os.path.basename(s3_key)
    try:
        for sec in merged_sections:
            sec_lines = sec.get("lines", [])
            if not sec_lines:
                continue
            sec_text = "".join(sec_lines).strip()
            sec_token_count = sec.get("token_count", token_count_for(sec_text))
            heading_path = [h for h in (sec.get("heading_path", []) or []) if not _is_rootish(h)]
            headings_raw = sec.get("headings") or []
            headings = [h for h in headings_raw if not _is_rootish(h)]
            if not headings and heading_path:
                headings = list(heading_path)
            sec_start_line = sec.get("start_line", 0)
            sec_end_line = sec.get("end_line", sec_start_line)
            start_line_1b = sec_start_line + 1
            end_line_1b = sec_end_line
            if sec_token_count <= MAX_TOKENS_PER_CHUNK:
                chunk_id = f"{doc_id}_{chunk_index}"; chunk_index += 1
                payload = {"document_id": doc_id or "", "file_name": file_name, "chunk_id": chunk_id or "", "chunk_type": "md_section", "text": canonicalize_text(sec_text) or "", "token_count": int(sec_token_count or 0), "figures": [], "embedding": None, "file_type": "text/markdown", "source_url": source_url, "timestamp": datetime.utcnow().isoformat() + "Z", "parser_version": PARSER_VERSION or "", "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [], "layout_tags": [], "used_ocr": False, "heading_path": heading_path or [], "headings": headings or [], "line_range": [int(start_line_1b), int(end_line_1b)] if (start_line_1b and end_line_1b is not None) else [1, 1]}
                sanitize_payload(payload)
                writer.write_payload(payload)
                saved += 1
                log.info("buffered_chunk", "Buffered chunk", chunk_id=payload["chunk_id"])
            else:
                subchunks = split_section_by_tokens_lines(sec, OVERLAP_TOKENS, MAX_TOKENS_PER_CHUNK, line_token_cache)
                for sub in subchunks:
                    chunk_text = sub.get("text", ""); token_ct = int(sub.get("token_count", 0)); sline = sub.get("start_line", 0); eline = sub.get("end_line", sline)
                    chunk_id = f"{doc_id}_{chunk_index}"; chunk_index += 1
                    start_line_sub = sline + 1; end_line_sub = eline
                    payload = {"document_id": doc_id or "", "file_name": file_name, "chunk_id": chunk_id or "", "chunk_type": "md_subchunk", "text": canonicalize_text(chunk_text) or "", "token_count": token_ct, "figures": [], "embedding": None, "file_type": "text/markdown", "source_url": source_url, "timestamp": datetime.utcnow().isoformat() + "Z", "parser_version": PARSER_VERSION or "", "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [], "layout_tags": [], "used_ocr": False, "heading_path": heading_path or [], "headings": headings or [], "line_range": [int(start_line_sub), int(end_line_sub)] if (start_line_sub and end_line_sub is not None) else [1, 1]}
                    sanitize_payload(payload)
                    writer.write_payload(payload)
                    saved += 1
                    log.info("buffered_subchunk", "Buffered subchunk", chunk_id=payload["chunk_id"], lines=f"{start_line_sub}-{end_line_sub}")
    except Exception:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.exception("buffering_failed", "Error while buffering chunks for %s", s3_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": "buffering_failed"}
    try:
        if saved == 0:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("no_chunks", "No chunks produced", key=s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": False}
        count, uploaded_key, sha, size = writer.finalize_and_upload(out_basename)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        try:
            raw_manifest = sanitize_payload_for_raw_manifest(doc_id, s3_key, uploaded_key, count, sha, size)
            client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
        except Exception:
            log.warning("manifest_write_failed", "Failed to write raw manifest", key=s3_key)
        log.info("write_complete", "Wrote chunks", count=count, raw=s3_key, chunked=uploaded_key, duration_ms=total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("upload_failed", "Failed to upload chunked file", key=s3_key, error=str(e_up))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

# ---------- CLI mode ----------
def _ensure_cli_env_or_exit():
    missing = []
    if not AZURE_CONTAINER:
        missing.append("AZURE_CONTAINER")
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    log.info("startup", "Starting markdown -> parquet parser (CLI mode)", use_managed_identity=str(USE_MANAGED_IDENTITY).lower())
    _ensure_cli_env_or_exit()
    client = get_storage_client_singleton()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AZURE_CONTAINER, Prefix=STORAGE_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.lower().endswith(".md") or key.lower().endswith(".markdown")):
                continue
            log.info("cli_route", "routing parse_file", key=key)
            manifest_key = key + ".manifest.json"
            try:
                mf_obj = client.get_object(Bucket=AZURE_CONTAINER, Key=manifest_key)
                body = mf_obj.get("Body", b"")
                if isinstance(body, (bytes, bytearray)):
                    manifest = json.loads(body.decode("utf-8"))
                else:
                    try:
                        manifest = json.loads(body.read())
                    except Exception:
                        manifest = {}
            except Exception:
                manifest = {}
            try:
                result = parse_file(key, manifest)
                log.info("cli_result", "Result", key=key, result=result)
            except Exception:
                log.exception("cli_parse_failed", "Failed to parse", key=key)
