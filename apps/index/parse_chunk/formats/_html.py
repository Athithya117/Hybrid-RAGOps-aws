#!/usr/bin/env python3
# html_trafilatura_dualmode.py
# Dual-mode storage: controlled ONLY by USE_MANAGED_IDENTITY (preferred) or AZURE_USE_MANAGED_IDENTITY (compat).
# - USE_MANAGED_IDENTITY=1 -> use DefaultAzureCredential + azure-storage-blob
# - USE_MANAGED_IDENTITY=0 -> use fsspec ("az" / adlfs) using connection string / account key / SAS
#
# Deterministic behavior, fail-fast, pre-validate dependencies and required envs.

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
import io
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator, Tuple

# ----------------------- Structured logger -----------------------
class LoggerShim:
    def __init__(self, name: str):
        self.name = name
    def _emit(self, level: str, event: str, msg: str = "", **extra):
        o = {"ts": datetime.utcnow().isoformat() + "Z", "level": level, "event": event, "msg": msg}
        if extra:
            o.update(extra)
        # print to stdout for info / debug, stderr for error
        if level in ("ERROR","WARN"):
            print(json.dumps(o, ensure_ascii=False), file=sys.stderr, flush=True)
        else:
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
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "info"); self._emit("INFO", event, msg, **kw)
    def warning(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "warn"); self._emit("WARN", event, msg, **kw)
    def warn(self, a, b=None, *fmt_args, **kwargs):
        self.warning(a, b, *fmt_args, **kwargs)
    def error(self, a, b=None, *fmt_args, **kwargs):
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "error"); self._emit("ERROR", event, msg, **kw)
    def exception(self, a, b=None, *fmt_args, **kwargs):
        import traceback
        tb = traceback.format_exc()
        event, msg, kw = self._unpack(a, b, fmt_args, kwargs, "exception"); kw.update({"tb": tb}); self._emit("ERROR", event, msg, **kw)

log = LoggerShim("html_trafilatura")

# ----------------------- ENV / defaults -----------------------
# Deterministic auth switch: prefer USE_MANAGED_IDENTITY; fallback to AZURE_USE_MANAGED_IDENTITY for compatibility.
_MI_RAW = os.getenv("USE_MANAGED_IDENTITY")
if _MI_RAW is None:
    _MI_RAW = os.getenv("AZURE_USE_MANAGED_IDENTITY", "")
USE_MANAGED_IDENTITY = str(_MI_RAW).strip().lower() in ("1", "true", "yes")

AZURE_CONTAINER = os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
if not AZURE_CONTAINER:
    log.error("startup_missing_container", "AZURE_CONTAINER (or STORAGE_CONTAINER / AZ_CONTAINER) must be set")
    sys.exit(2)

STORAGE_RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("S3_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
STORAGE_CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("S3_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"
PARSER_VERSION = os.getenv("PARSER_VERSION_HTML", "trafilatura-only-v2")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "3"))
FETCH_BACKOFF = float(os.getenv("FETCH_BACKOFF", "0.5"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
NUMBER_OF_OVERLAPPING_SENTENCES = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", "2"))
CHUNKED_SCHEMA_VERSION = os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1")
PUT_RETRIES = int(os.getenv("PUT_RETRIES", "3"))
PUT_BACKOFF = float(os.getenv("PUT_BACKOFF", "0.3"))

# ----------------------- lazy deps / init placeholders -----------------------
_requests = None
_trafilatura = None
_tiktoken = None
_ENCODER = None
_ENCODER_ENCODE = None
_ENCODER_DECODE = None
_ENCODER_BACKEND = "whitespace"
_spacy = None
_Sentencizer = None
_NLP_SENTENCIZER = None

# ----------------------- optional libs (fsspec) -----------------------
try:
    import fsspec
    from fsspec.spec import AbstractFileSystem
except Exception:
    fsspec = None
    AbstractFileSystem = object  # type: ignore

# ----------------------- azure SDK imports (conditionally) -----------------------
AZURE_SDK_AVAILABLE = False
DefaultAzureCredential = None
BlobServiceClient = None
ContainerClient = None
try:
    # Try importing but we will only require these if USE_MANAGED_IDENTITY == True
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.storage.blob import BlobServiceClient, ContainerClient  # type: ignore
    AZURE_SDK_AVAILABLE = True
except Exception:
    DefaultAzureCredential = None
    BlobServiceClient = None
    ContainerClient = None
    AZURE_SDK_AVAILABLE = False

# ----------------------- helper: retry -----------------------
def retry_call(fn, retries: int = 3, backoff_base: float = 0.5, allowed_exceptions: tuple = (Exception,)):
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except allowed_exceptions as e:
            if attempt >= retries:
                raise
            sleep = backoff_base * (2 ** (attempt - 1))
            time.sleep(sleep)

# ----------------------- build fsspec options for non-managed mode -----------------------
def build_fs_opts() -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        opts["connection_string"] = conn
        return opts
    acct = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
    key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or os.getenv("AZURE_ACCOUNT_KEY")
    sas = os.getenv("AZURE_SAS_TOKEN")
    eps = os.getenv("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"
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
    if os.getenv("AZURE_ANON"):
        if acct:
            opts["account_name"] = acct
        opts["anon"] = True
        return opts
    return opts

# ----------------------- runtime validation (fail-fast) -----------------------
def validate_runtime_envs():
    # If managed identity requested, require azure sdk + storage account name
    if USE_MANAGED_IDENTITY:
        if not AZURE_SDK_AVAILABLE:
            log.error("azure_sdk_missing", "USE_MANAGED_IDENTITY=1 but azure-identity / azure-storage-blob packages are not available. Install: pip install azure-identity azure-storage-blob")
            sys.exit(2)
        acct = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
        if not acct:
            log.error("env_missing", "AZURE_STORAGE_ACCOUNT_NAME (or AZURE_ACCOUNT_NAME) required when USE_MANAGED_IDENTITY=1")
            sys.exit(2)
    else:
        # non-managed: require fsspec/adlfs and at least one credential (connstring | account_key | sas) OR AZURE_ANON allowed
        if fsspec is None:
            log.error("fsspec_missing", "USE_MANAGED_IDENTITY=0 but fsspec/adlfs not installed. Install: pip install fsspec adlfs")
            sys.exit(2)
        opts = build_fs_opts()
        if not opts:
            log.error("env_missing", "non-managed mode requires AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_KEY or AZURE_SAS_TOKEN (or export AZURE_ANON for anonymous)")
            sys.exit(2)

# run validation immediately to fail fast
validate_runtime_envs()

# ----------------------- STORAGE ROOT -----------------------
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

# ----------------------- Storage client shim (dual-mode) -----------------------
class AzureStorageClient:
    def __init__(self, fs_obj: Optional[AbstractFileSystem], root: str, container: str, blob_service_client=None):
        self.fs = fs_obj
        self.root = root
        self.container = container
        self.blob_service_client = blob_service_client

    def _container_client(self):
        if self.blob_service_client is None:
            raise RuntimeError("blob_service_client not initialized for managed-identity mode")
        return self.blob_service_client.get_container_client(self.container)

    def head_object(self, Bucket, Key):
        # mimic AWS S3 head_object style response keys used by callers
        if self.fs is not None:
            full = full_path_from_key(Key)
            info = self.fs.info(full)
            out = {}
            out["ContentLength"] = int(info.get("size", 0))
            etag = info.get("etag") or info.get("ETag") or ""
            out["ETag"] = etag
            lm = info.get("Last-Modified") or info.get("last_modified") or ""
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
            return {"Body": BytesIO(data)}
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
            stream = retry_call(lambda: blob_client.download_blob(), retries=3, backoff_base=0.5)
            data = stream.readall()
            return {"Body": BytesIO(data)}

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
            retry_call(lambda: blob_client.upload_blob(data, overwrite=True), retries=3, backoff_base=0.5)
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def upload_file(self, LocalFile, Bucket, Key, ExtraArgs=None):
        if self.fs is not None:
            full = full_path_from_key(Key)
            if hasattr(self.fs, "put"):
                # some fsspec implementations provide put()
                self.fs.put(LocalFile, full)
                return
            with open(LocalFile, "rb") as lf:
                d = lf.read()
            with self.fs.open(full, "wb") as f:
                f.write(d)
            return
        else:
            container_client = self._container_client()
            blob_client = container_client.get_blob_client(Key)
            with open(LocalFile, "rb") as lf:
                retry_call(lambda: blob_client.upload_blob(lf, overwrite=True), retries=3, backoff_base=0.5)
            return

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
            retry_call(lambda: dst_blob_client.start_copy_from_url(src_url), retries=3, backoff_base=0.5)
            return

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
                retry_call(lambda: blob_client.delete_blob(), retries=3, backoff_base=0.5)
            except Exception:
                pass

    def exists(self, full_path: str) -> bool:
        # Accepts full "az://container/..." path
        try:
            if self.fs is not None:
                return self.fs.exists(full_path)
            else:
                # if full_path is key or full az path
                key = strip_root_from_path(full_path)
                blob_client = self._container_client().get_blob_client(key)
                try:
                    return blob_client.exists()
                except Exception:
                    # older SDK may not have exists; try get properties
                    try:
                        blob_client.get_blob_properties()
                        return True
                    except Exception:
                        return False
        except Exception:
            return False

    def get_paginator(self, name):
        # returns an object with paginate(Bucket=..., Prefix=...)
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

# ----------------------- singleton storage client -----------------------
_storage_client = None
_storage_lock = threading.Lock()

def get_storage_client_singleton():
    global _storage_client
    if _storage_client is None:
        with _storage_lock:
            if _storage_client is None:
                if USE_MANAGED_IDENTITY:
                    # build blob service client based on connection string OR AAD DefaultAzureCredential
                    # Prefer explicit connection string if provided (makes local debug easier)
                    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                    if conn:
                        try:
                            bsc = BlobServiceClient.from_connection_string(conn)
                        except Exception as e:
                            log.error("blob_init_failed", "BlobServiceClient.from_connection_string failed: %s", str(e))
                            raise
                    else:
                        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
                        endpoint_suffix = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
                        account_url = f"https://{account_name}.{endpoint_suffix}"
                        try:
                            cred = DefaultAzureCredential()
                            bsc = BlobServiceClient(account_url=account_url, credential=cred, connection_timeout=60)
                        except Exception as e:
                            log.error("blob_init_failed", "Failed to init BlobServiceClient with DefaultAzureCredential: %s", str(e))
                            raise
                    _storage_client = AzureStorageClient(None, STORAGE_ROOT, AZURE_CONTAINER, blob_service_client=bsc)
                else:
                    # non-managed -> use fsspec filesystem "az"
                    opts = build_fs_opts()
                    try:
                        fs = fsspec.filesystem("az", **opts)
                    except Exception as e:
                        log.error("fsspec_init_failed", "Failed to init fsspec az filesystem: %s", str(e))
                        raise
                    _storage_client = AzureStorageClient(fs, STORAGE_ROOT, AZURE_CONTAINER, blob_service_client=None)
    return _storage_client

# ----------------------- small helpers -----------------------
def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
def sha256_hex_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def canonicalize_text(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------- optional deps (lazy) -----------------------
def _ensure_optional_deps():
    global _requests, _trafilatura, _tiktoken, _ENCODER, _ENCODER_ENCODE, _ENCODER_DECODE, _ENCODER_BACKEND, _spacy, _Sentencizer, _NLP_SENTENCIZER
    if _requests is None:
        try:
            import requests as _r
            _requests = _r
        except Exception:
            _requests = None
    if _trafilatura is None:
        try:
            import trafilatura as _t
            _trafilatura = _t
        except Exception:
            _trafilatura = None
    if _tiktoken is None:
        try:
            import tiktoken as _tk
            _tiktoken = _tk
            try:
                _ENCODER = _tiktoken.get_encoding(ENC_NAME)
            except Exception:
                try:
                    _ENCODER = _tiktoken.encoding_for_model("gpt2")
                except Exception:
                    _ENCODER = None
        except Exception:
            _tiktoken = None
            _ENCODER = None
    if _ENCODER is not None:
        _ENCODER_ENCODE = lambda txt: _ENCODER.encode(txt)
        _ENCODER_DECODE = lambda toks: _ENCODER.decode(toks)
        _ENCODER_BACKEND = "tiktoken"
        log.info("encoder_init", "tiktoken encoder loaded", backend=_ENCODER_BACKEND)
    else:
        _ENCODER_ENCODE = lambda txt: txt.split()
        _ENCODER_DECODE = lambda toks: " ".join(toks)
        _ENCODER_BACKEND = "whitespace"
    if _spacy is None:
        try:
            import spacy as _s
            from spacy.pipeline import Sentencizer as _S
            _spacy = _s; _Sentencizer = _S
        except Exception:
            _spacy = None; _Sentencizer = None
    return

# ----------------------- fetch helpers -----------------------
def fetch_html_with_retries(url: str, timeout: int = REQUEST_TIMEOUT, retries: int = FETCH_RETRIES, backoff: float = FETCH_BACKOFF) -> str:
    if _requests is None:
        raise RuntimeError("requests is required to fetch remote HTML")
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = _requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last

def upload_snapshot_to_azure(snapshot_html: str, doc_id: str) -> Optional[str]:
    if not SAVE_SNAPSHOT:
        return None
    client = get_storage_client_singleton()
    key = f"{STORAGE_CHUNKED_PREFIX}{doc_id}.snapshot.html"
    try:
        client.put_object(Bucket=AZURE_CONTAINER, Key=key, Body=snapshot_html.encode("utf-8"), ContentType="text/html")
        return f"az://{AZURE_CONTAINER}/{key}"
    except Exception:
        return None

# ----------------------- trafilatura extraction -----------------------
def trafilatura_extract_markdown(html_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    if _trafilatura is None:
        return None, {}
    try:
        md = _trafilatura.extract(html_text, output_format="markdown", with_metadata=True)
    except Exception:
        md = None
    parsed = {}
    try:
        json_doc = _trafilatura.extract(html_text, output_format="json", with_metadata=True)
        if json_doc:
            parsed = json.loads(json_doc)
    except Exception:
        parsed = {}
    return md, parsed

# ----------------------- sentence splitting & token windows -----------------------
def _make_sentencizer():
    global _NLP_SENTENCIZER
    if _NLP_SENTENCIZER is not None:
        return _NLP_SENTENCIZER
    if _spacy is None:
        _NLP_SENTENCIZER = None
        return None
    try:
        nlp = _spacy.blank("en")
        try:
            nlp.add_pipe("sentencizer")
        except Exception:
            if _Sentencizer is not None:
                nlp.add_pipe(_Sentencizer())
            else:
                nlp.add_pipe("sentencizer")
        _NLP_SENTENCIZER = nlp
        return nlp
    except Exception:
        _NLP_SENTENCIZER = None
        return None

def _regex_sentences_with_offsets(text: str):
    spans = []
    pattern = re.compile(r'(.+?[\.\?\!]["\']?\s+)|(.+?$)', re.DOTALL)
    cursor = 0
    for m in pattern.finditer(text):
        s = (m.group(1) or m.group(2) or "").strip()
        if not s:
            continue
        start = text.find(s, cursor)
        if start == -1:
            start = cursor
        end = start + len(s)
        spans.append((s, start, end))
        cursor = end
    return spans

def _sentences_with_offsets(text: str):
    nlp = _make_sentencizer()
    if nlp is not None:
        doc = nlp(text)
        return [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
    return _regex_sentences_with_offsets(text)

def _make_encoder_clients():
    global _ENCODER_ENCODE, _ENCODER_DECODE, _ENCODER_BACKEND
    if _ENCODER_ENCODE is None:
        _ENCODER_ENCODE = lambda txt: txt.split()
        _ENCODER_DECODE = lambda toks: " ".join(toks)
        _ENCODER_BACKEND = "whitespace"
    return _ENCODER_ENCODE, _ENCODER_DECODE, _ENCODER_BACKEND

def split_into_token_windows(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, min_tokens: int = MIN_TOKENS_PER_CHUNK, overlap_sentences: int = NUMBER_OF_OVERLAPPING_SENTENCES) -> Iterator[Dict[str, Any]]:
    if not text:
        yield {"window_index": 0, "text": "", "token_count": 0, "token_start": 0, "token_end": 0}
        return
    text = canonicalize_text(text)
    sentences = _sentences_with_offsets(text)
    enc_encode, enc_decode, enc_backend = _make_encoder_clients()
    sent_items = []
    token_cursor = 0
    for s, sc, ec in sentences:
        toks = enc_encode(s)
        tok_len = len(toks)
        sent_items.append({"text": s, "start_char": sc, "end_char": ec, "token_len": tok_len, "tokens": toks})
    if not sent_items:
        all_toks = enc_encode(text)
        yield {"window_index": 0, "text": text, "token_count": len(all_toks), "token_start": 0, "token_end": len(all_toks)}
        return
    for si in sent_items:
        si["token_start_idx"] = token_cursor
        si["token_end_idx"] = token_cursor + si["token_len"]
        token_cursor = si["token_end_idx"]
    windows = []
    i = 0; window_index = 0
    while i < len(sent_items):
        cur_token_count = 0; chunk_sent_texts = []; chunk_token_start = sent_items[i]["token_start_idx"]; chunk_token_end = chunk_token_start; is_truncated_sentence = False; start_i = i
        while i < len(sent_items):
            sent = sent_items[i]; sent_tok_len = sent["token_len"]
            if cur_token_count + sent_tok_len > max_tokens:
                if not chunk_sent_texts:
                    if sent_tok_len > 0:
                        if enc_backend == "tiktoken":
                            prefix_tok_ids = sent["tokens"][:max_tokens]; prefix_text = enc_decode(prefix_tok_ids); chunk_sent_texts.append(prefix_text); cur_token_count = len(prefix_tok_ids); is_truncated_sentence = True
                            remainder_tok_ids = sent["tokens"][max_tokens:]
                            if remainder_tok_ids:
                                remainder_text = enc_decode(remainder_tok_ids)
                                sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "token_len": len(remainder_tok_ids), "tokens": remainder_tok_ids, "token_start_idx": None, "token_end_idx": None}
                            else:
                                i += 1
                            chunk_token_end = chunk_token_start + cur_token_count
                            break
                        else:
                            tokens = sent["tokens"]; prefix = tokens[:max_tokens]; prefix_text = " ".join(prefix); chunk_sent_texts.append(prefix_text); cur_token_count = len(prefix); is_truncated_sentence = True
                            remainder = tokens[max_tokens:]
                            if remainder:
                                remainder_text = " ".join(remainder)
                                sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "token_len": len(remainder), "tokens": remainder, "token_start_idx": None, "token_end_idx": None}
                            else:
                                i += 1
                            chunk_token_end = chunk_token_start + cur_token_count
                            break
                    else:
                        i += 1; break
                else:
                    break
            else:
                chunk_sent_texts.append(sent["text"]); cur_token_count += sent_tok_len; chunk_token_end = sent.get("token_end_idx", chunk_token_start + cur_token_count); i += 1
        if not chunk_sent_texts:
            i += 1; continue
        chunk_text = " ".join(chunk_sent_texts).strip()
        chunk_meta = {"window_index": window_index, "text": chunk_text, "token_count": cur_token_count, "token_start": chunk_token_start, "token_end": chunk_token_end, "start_sentence_idx": start_i, "end_sentence_idx": i, "is_truncated_sentence": is_truncated_sentence}
        window_index += 1
        new_start = max(start_i + 1, chunk_meta["end_sentence_idx"] - overlap_sentences)
        if windows and chunk_meta["token_count"] < min_tokens:
            prev = windows[-1]
            prev["text"] = prev["text"] + " " + chunk_meta["text"]
            prev["token_count"] = prev["token_count"] + chunk_meta["token_count"]
            prev["token_end"] = chunk_meta["token_end"]
            prev["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
            prev["is_truncated_sentence"] = prev.get("is_truncated_sentence", False) or chunk_meta.get("is_truncated_sentence", False)
        else:
            windows.append(chunk_meta)
        i = new_start
    for w in windows:
        yield w

# ----------------------- storage helpers -----------------------
def storage_object_exists(key: str) -> bool:
    full = full_path_from_key(key)
    client = get_storage_client_singleton()
    try:
        if client.fs is not None:
            return client.fs.exists(full)
        else:
            return client.exists(full)
    except Exception:
        return False

def storage_upload_file_atomic(local_path: str, key: str, content_type: str = "application/octet-stream") -> None:
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
                # managed-identity blob path: upload directly with overwrite (no atomic rename primitive)
                client.upload_file(local_path, AZURE_CONTAINER, key)
                return
        except Exception as e:
            log.warning("upload_retry", "attempt=%d key=%s error=%s", attempt, key, str(e))
            time.sleep(PUT_BACKOFF * attempt)
    raise Exception(f"atomic upload failed for {key} after {PUT_RETRIES} attempts")

# ----------------------- parquet writer -----------------------
class ParquetWriter:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id; self._rows: List[Dict[str, Any]] = []
    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        fields["document_id"] = payload.get("document_id") or ""
        fields["file_name"] = payload.get("file_name") or ""
        fields["raw_key"] = payload.get("raw_key") or ""
        fields["chunk_id"] = payload.get("chunk_id") or ""
        fields["chunk_type"] = payload.get("chunk_type") or ""
        fields["chunk_index"] = payload.get("chunk_index") or 0
        fields["text"] = payload.get("text") or ""
        try:
            fields["token_count"] = int(payload.get("token_count") or 0)
        except Exception:
            fields["token_count"] = 0
        for k in ("figures", "tags", "layout_tags", "heading_path", "headings", "line_range"):
            v = payload.get(k, None)
            try:
                fields[k] = json.dumps(v, ensure_ascii=False, sort_keys=True) if v is not None else "[]"
            except Exception:
                fields[k] = "[]"
        fields["file_type"] = payload.get("file_type") or ""
        fields["source_url"] = payload.get("source_url") or ""
        try:
            tr = payload.get("token_range")
            if isinstance(tr, (list, tuple)) and len(tr) >= 2:
                fields["token_start"] = int(tr[0]); fields["token_end"] = int(tr[1])
            else:
                fields["token_start"] = 0; fields["token_end"] = 0
        except Exception:
            fields["token_start"] = 0; fields["token_end"] = 0
        fields["timestamp"] = payload.get("timestamp") or ""
        fields["parser_version"] = payload.get("parser_version") or PARSER_VERSION
        fields["used_ocr"] = bool(payload.get("used_ocr", False))
        try:
            om = payload.get("original_manifest")
            fields["original_manifest"] = json.dumps(om, ensure_ascii=False, sort_keys=True) if om is not None else ""
        except Exception:
            fields["original_manifest"] = ""
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
            raise RuntimeError("pyarrow required") from e
        schema = pa.schema([
            pa.field("document_id", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("raw_key", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("token_count", pa.int64()),
            pa.field("figures", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("layout_tags", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("headings", pa.string()),
            pa.field("line_range", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("token_start", pa.int64()),
            pa.field("token_end", pa.int64()),
            pa.field("timestamp", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("used_ocr", pa.bool_()),
            pa.field("original_manifest", pa.string())
        ])
        cols = {name: [] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table = pa.Table.from_pydict(cols, schema=schema)
        existing_md = table.schema.metadata or {}
        new_md = dict(existing_md)
        new_md.update({
            b"schema_version": CHUNKED_SCHEMA_VERSION.encode("utf-8"),
            b"parser_version": PARSER_VERSION.encode("utf-8"),
            b"producer": b"html_trafilatura",
            b"created_at": datetime.utcnow().isoformat().encode("utf-8")
        })
        table = table.replace_schema_metadata(new_md)
        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".parquet", dir="/tmp"); tmpfile.close()
        pq.write_table(table, tmpfile.name, compression="zstd", flavor="spark")
        with open(tmpfile.name, "rb") as fh:
            b = fh.read()
        sha = sha256_hex_bytes(b); size = os.path.getsize(tmpfile.name)
        parquet_key = out_basename + ".parquet"
        storage_upload_file_atomic(tmpfile.name, STORAGE_CHUNKED_PREFIX + parquet_key, content_type="application/octet-stream")
        try: os.unlink(tmpfile.name)
        except Exception: pass
        return len(self._rows), STORAGE_CHUNKED_PREFIX + parquet_key, sha, size

# ----------------------- misc helpers -----------------------
def sanitize_payload_for_raw_manifest(doc_id: str, raw_key: str, chunked_key: str, rows: int, sha: str, size: int) -> Dict[str, Any]:
    return {"raw_key": raw_key, "doc_id": doc_id, "chunked_key": chunked_key, "rows": rows, "sha256": sha, "size_bytes": size, "schema_version": CHUNKED_SCHEMA_VERSION, "parser_version": PARSER_VERSION, "created_at": datetime.utcnow().isoformat() + "Z"}

def _derive_file_name_from_source(source: Optional[str], raw_key: str) -> str:
    if source:
        try:
            base = source.split("?")[0].rstrip("/")
            base_name = os.path.basename(base)
            if base_name:
                return base_name
        except Exception:
            pass
    return os.path.basename(raw_key)

# ----------------------- parse_file API (main logic) -----------------------
def parse_file(s3_key: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    start_all = time.perf_counter()
    if not AZURE_CONTAINER:
        raise RuntimeError("AZURE_CONTAINER must be set")
    _ensure_optional_deps()
    client = get_storage_client_singleton()
    try:
        head = client.head_object(Bucket=AZURE_CONTAINER, Key=s3_key)
    except Exception as e:
        log.error("head_failed", "Could not head object %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    last_modified = head.get("LastModified", "")
    doc_id = manifest.get("file_hash") or sha256_hex_str(s3_key + str(last_modified or ""))
    out_basename = f"{doc_id}"
    raw_manifest_key = s3_key + ".manifest.json"
    try:
        if not FORCE_OVERWRITE and storage_object_exists(raw_manifest_key):
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("skip_manifest_exists", "raw_manifest_exists", key=raw_manifest_key); return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
        if not FORCE_OVERWRITE and storage_object_exists(STORAGE_CHUNKED_PREFIX + out_basename + ".parquet"):
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("skip_parquet_exists", "parquet_exists", key=out_basename + ".parquet")
            try:
                if not storage_object_exists(raw_manifest_key):
                    head2 = client.head_object(Bucket=AZURE_CONTAINER, Key=STORAGE_CHUNKED_PREFIX + out_basename + ".parquet")
                    etag = head2.get("ETag", ""); etag = etag.strip('"') if isinstance(etag, str) else etag
                    size = head2.get("ContentLength", 0)
                    raw_manifest = sanitize_payload_for_raw_manifest(doc_id, s3_key, STORAGE_CHUNKED_PREFIX + out_basename + ".parquet", 0, etag, size)
                    client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
            except Exception:
                pass
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    except Exception:
        pass
    try:
        obj = client.get_object(Bucket=AZURE_CONTAINER, Key=s3_key)
    except Exception as e:
        log.error("get_object_failed", "Could not get object %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    raw_body = obj["Body"].read()
    try:
        raw_text = raw_body.decode("utf-8", errors="replace")
    except Exception:
        raw_text = raw_body.decode("latin-1", errors="replace")
    source_path = f"az://{AZURE_CONTAINER}/{s3_key}"
    stripped = raw_text.strip()
    use_remote_fetch = False; remote_url = None
    if stripped.startswith("http://") or stripped.startswith("https://"):
        use_remote_fetch = True; remote_url = stripped.splitlines()[0].strip()
    if use_remote_fetch:
        try:
            html_text = fetch_html_with_retries(remote_url)
            source_url = remote_url
        except Exception:
            html_text = raw_text; source_url = source_path
    else:
        html_text = raw_text; source_url = source_path
    try:
        _ = upload_snapshot_to_azure(html_text, doc_id)
    except Exception:
        pass
    t0 = time.perf_counter()
    md, parsed = trafilatura_extract_markdown(html_text)
    extract_duration_ms = int((time.perf_counter() - t0) * 1000)
    if not md:
        fallback = re.sub(r'(?is)<(script|style).*?>.*?</\1>', '', html_text)
        fallback = re.sub(r'(?is)<.*?>', ' ', fallback)
        md = re.sub(r'\s+', ' ', fallback).strip()
    title = parsed.get("title") if isinstance(parsed, dict) else None
    canonical_full = canonicalize_text(md or "")
    enc_encode = _ENCODER_ENCODE if _ENCODER_ENCODE is not None else (lambda txt: txt.split())
    token_ct = len(enc_encode(canonical_full)) if canonical_full else 0
    saved = 0; writer = ParquetWriter(doc_id=doc_id); file_name = _derive_file_name_from_source(source_url, s3_key)
    try:
        windows = list(split_into_token_windows(canonical_full))
        if len(windows) <= 1:
            chunk_index = 1; chunk_id = f"{doc_id}_{chunk_index}"
            payload = {
                "document_id": doc_id,
                "file_name": file_name,
                "raw_key": s3_key,
                "original_manifest": manifest if isinstance(manifest, dict) else {},
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_type": "page",
                "text": canonical_full,
                "token_count": int(token_ct),
                "figures": [],
                "embedding": None,
                "file_type": "text/html",
                "source_url": source_url,
                "token_range": [0, token_ct],
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "parser_version": PARSER_VERSION,
                "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                "layout_tags": ["page"],
                "used_ocr": False,
                "heading_path": [],
                "headings": [title] if title else [],
                "line_range": None
            }
            writer.write_payload(payload)
            saved += 1
        else:
            for w in windows:
                window_idx = int(w.get("window_index", 0))
                chunk_index = window_idx + 1
                chunk_id = f"{doc_id}_{chunk_index}"
                wtext = w.get("text", "")
                w_token_count = int(w.get("token_count", 0))
                token_range = [int(w.get("token_start", 0)), int(w.get("token_end", 0))]
                payload = {
                    "document_id": doc_id,
                    "file_name": file_name,
                    "raw_key": s3_key,
                    "original_manifest": manifest if isinstance(manifest, dict) else {},
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_type": "token_window",
                    "text": wtext,
                    "token_count": int(w_token_count),
                    "figures": [],
                    "embedding": None,
                    "file_type": "text/html",
                    "source_url": source_url,
                    "token_range": token_range,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION,
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": ["page"],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": [title] if title else [],
                    "line_range": None
                }
                writer.write_payload(payload)
                saved += 1
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.exception("buffering_failed", "Error while buffering chunks", key=s3_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if saved == 0:
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("no_chunks", "No chunks produced", key=s3_key); return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
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
        log.error("upload_failed", "Failed to upload chunked file", key=s3_key, error=str(e_up)); return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

# ----------------------- CLI runner -----------------------
if __name__ == "__main__":
    try:
        _ensure_optional_deps()
        client = get_storage_client_singleton()
    except Exception as e:
        log.error("init_failed", "Runtime initialization failed: %s", str(e)); sys.exit(1)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AZURE_CONTAINER, Prefix=STORAGE_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # accept both .html and .htm
            if not (key.lower().endswith(".html") or key.lower().endswith(".htm")):
                continue
            log.info("cli_route", "routing parse_file", key=key)
            manifest_key = key + ".manifest.json"
            try:
                mf_obj = client.get_object(Bucket=AZURE_CONTAINER, Key=manifest_key)
                try:
                    manifest = json.load(mf_obj["Body"])
                except Exception:
                    manifest = {}
            except Exception:
                manifest = {}
            try:
                res = parse_file(key, manifest)
                log.info("cli_result", "Result", key=key, result=res)
            except Exception as e:
                log.exception("cli_parse_failed", "Failed to parse", key=key)

# End of file
