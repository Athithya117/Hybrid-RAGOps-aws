#!/usr/bin/env python3
from __future__ import annotations
import importlib
import os
import sys
import time
import json
import hashlib
import tempfile
import unicodedata
import re
import threading
import io
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple

try:
    import fsspec
    from fsspec.spec import AbstractFileSystem  # type: ignore
except Exception:
    fsspec = None

_tiktoken = None
try:
    import tiktoken as _tiktoken  # type: ignore
except Exception:
    _tiktoken = None

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None

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

log = LoggerShim("txt_parser")

USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower() in ("1", "true", "yes")

STORAGE_PROTOCOL = "az"
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
if not AZURE_CONTAINER:
    log.error("startup_missing_container", "AZURE_CONTAINER (or STORAGE_CONTAINER) must be set"); sys.exit(1)

STORAGE_RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
STORAGE_CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
PARSER_VERSION = os.getenv("PARSER_VERSION_TXT", "plain-txt-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
PUT_RETRIES = int(os.getenv("PUT_RETRIES", "3"))
PUT_BACKOFF = float(os.getenv("PUT_BACKOFF", "0.3"))
CHUNKED_SCHEMA_VERSION = os.getenv("CHUNKED_SCHEMA_VERSION", "chunked_v1")

def _norm_prefix(p: str) -> str:
    p = (p or "").rstrip("/")
    return p + "/" if p else ""

STORAGE_RAW_PREFIX = _norm_prefix(STORAGE_RAW_PREFIX)
STORAGE_CHUNKED_PREFIX = _norm_prefix(STORAGE_CHUNKED_PREFIX)

def validate_runtime():
    missing = []
    if USE_MANAGED_IDENTITY:
        acct = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
        if not acct:
            missing.append("AZURE_STORAGE_ACCOUNT_NAME")
        if not pa or not pq:
            log.warning("dep_missing", "pyarrow not installed (writing parquet will fail). Install: pip install pyarrow")
        try:
            importlib.import_module("azure.identity")
            importlib.import_module("azure.storage.blob")
        except Exception as e:
            log.error("dep_missing", "azure.identity / azure.storage.blob required for managed identity mode", error=str(e)); sys.exit(2)
    else:
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        acct = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
        key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        sas = os.getenv("AZURE_SAS_TOKEN")
        if conn:
            try:
                importlib.import_module("azure.storage.blob")
            except Exception:
                if fsspec is None and not (acct and (key or sas)):
                    missing.append("azure-storage-blob or fsspec+adlfs and credentials")
        else:
            if not acct:
                missing.append("AZURE_STORAGE_ACCOUNT_NAME")
            if not (key or sas):
                missing.append("AZURE_STORAGE_ACCOUNT_KEY or AZURE_SAS_TOKEN or AZURE_STORAGE_CONNECTION_STRING")
            if fsspec is None:
                missing.append("fsspec/adlfs (pip install fsspec adlfs)")
    if missing:
        log.error("env_missing", "Missing required environment variables or deps for selected auth mode", missing=missing, use_managed_identity=str(USE_MANAGED_IDENTITY)); sys.exit(2)

validate_runtime()

def build_storage_options() -> Dict[str, str]:
    opts: Dict[str, str] = {}
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        opts["connection_string"] = conn; return opts
    acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
    sas = os.environ.get("AZURE_SAS_TOKEN")
    eps = os.environ.get("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"
    if acct and key:
        opts["account_name"] = acct; opts["account_key"] = key; opts["endpoint_suffix"] = eps; return opts
    if acct and sas:
        opts["account_name"] = acct; opts["sas_token"] = sas; opts["endpoint_suffix"] = eps; return opts
    if os.environ.get("AZURE_ANON"):
        if acct:
            opts["account_name"] = acct
        opts["anon"] = True
        return opts
    return opts

FS_OPTS = build_storage_options()
STORAGE_ROOT = f"az://{AZURE_CONTAINER.rstrip('/')}/"

BLOB_CLIENT = None
FS = None

if USE_MANAGED_IDENTITY:
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception as e:
        log.error("azure_import", "Failed to import azure sdk despite earlier validation", error=str(e)); sys.exit(2)
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME")
    endpoint_suffix = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
    account_url = f"https://{account_name}.{endpoint_suffix}"
    try:
        uai_client = os.getenv("UAI_RAG_RW_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
        if uai_client:
            CREDENTIAL = DefaultAzureCredential(managed_identity_client_id=uai_client)
        else:
            CREDENTIAL = DefaultAzureCredential()
        BLOB_CLIENT = BlobServiceClient(account_url=account_url, credential=CREDENTIAL)
        log.info("client_init", "Initialized BlobServiceClient (managed identity)", account=account_name)
    except Exception as e:
        log.error("blobclient_init", "Failed to create BlobServiceClient (managed identity)", error=str(e)); sys.exit(2)
else:
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    azure_sdk_available = False
    try:
        importlib.import_module("azure.storage.blob"); azure_sdk_available = True
    except Exception:
        azure_sdk_available = False
    if conn_str and azure_sdk_available:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
            BLOB_CLIENT = BlobServiceClient.from_connection_string(conn_str)
            log.info("client_init", "Initialized BlobServiceClient (connection string)")
        except Exception as e:
            log.warning("connstr_failed", "BlobServiceClient.from_connection_string failed, will attempt fsspec fallback", error=str(e)); BLOB_CLIENT = None
    if BLOB_CLIENT is None:
        if fsspec is None:
            log.error("fsspec_missing", "fsspec/adlfs required for key/SAS mode fallback (pip install fsspec adlfs)"); sys.exit(2)
        try:
            FS = fsspec.filesystem("az", **FS_OPTS)  # type: ignore
            log.info("fs.init", "Initialized fsspec az filesystem", opts_keys=list(FS_OPTS.keys()))
        except Exception as e:
            log.error("fsspec_init_failed", "failed to init fsspec az filesystem", error=str(e)); sys.exit(2)

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

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

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
            pass
    return b.decode("utf-8", errors="replace")

def token_count_for(text: str) -> int:
    if not text:
        return 0
    try:
        enc = None
        if _tiktoken is not None:
            try:
                enc = _tiktoken.get_encoding(ENC_NAME)
            except Exception:
                try:
                    enc = _tiktoken.encoding_for_model("gpt2")
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
            raise RuntimeError("blob_client not initialized for managed-identity or azure-sdk mode")
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
                            info_obj = self.fs.info(f)
                        except Exception:
                            continue
                        if info_obj.get("type") == "directory":
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

_storage_single = None
_storage_lock2 = threading.Lock()

def get_storage_client_singleton():
    global _storage_single
    if _storage_single is None:
        with _storage_lock2:
            if _storage_single is None:
                if USE_MANAGED_IDENTITY or (BLOB_CLIENT is not None and not FS):
                    _storage_single = AzureStorageClient(fs_obj=None, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=BLOB_CLIENT)
                else:
                    _storage_single = AzureStorageClient(fs_obj=FS, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=None)
    return _storage_single

def storage_exists(key: str) -> bool:
    client = get_storage_client_singleton()
    try:
        client.head_object(Bucket=AZURE_CONTAINER, Key=key)
        return True
    except Exception:
        return False

class TokenEncoder:
    def __init__(self, model_name: str = "gpt2", enc_name: Optional[str] = None):
        self.model_name = model_name; self.enc_name = enc_name or ENC_NAME; self.backend = "whitespace"; self._enc = None
        try:
            if _tiktoken is None:
                raise RuntimeError("tiktoken not installed")
            try:
                self._enc = _tiktoken.get_encoding(self.enc_name)
            except Exception:
                try:
                    self._enc = _tiktoken.encoding_for_model(self.model_name)
                except Exception:
                    self._enc = None
            if self._enc is not None:
                self.encode = lambda txt: self._enc.encode(txt)
                self.decode = lambda toks: self._enc.decode(toks)
                self.backend = "tiktoken"
                log.info("encoder_init", "tiktoken encoder loaded", enc_name=self.enc_name)
            else:
                raise RuntimeError("no encoder")
        except Exception as e:
            log.warning("encoder_fallback", "tiktoken unavailable, falling back to whitespace", reason=str(e))
            self.encode = lambda txt: txt.split()
            self.decode = lambda toks: " ".join(str(x) for x in toks)

class SentenceChunker:
    def __init__(self, max_tokens_per_chunk: Optional[int] = None, overlap_sentences: Optional[int] = None, token_model: str = "gpt2", min_tokens_per_chunk: Optional[int] = None):
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512")) if max_tokens_per_chunk is None else int(max_tokens_per_chunk)
        self.overlap_sentences = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", "2")) if overlap_sentences is None else int(overlap_sentences)
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.min_tokens_per_chunk = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100")) if min_tokens_per_chunk is None else int(min_tokens_per_chunk)
        if self.min_tokens_per_chunk < 0:
            raise ValueError("min_tokens_per_chunk must be >= 0")
        self.encoder = TokenEncoder(model_name=token_model, enc_name=os.getenv("TOKEN_ENCODER", ENC_NAME))
        self.nlp = self._make_sentencizer()
        log.info("chunker_init", "sentence chunker initialized", max_tokens=self.max_tokens_per_chunk, min_tokens=self.min_tokens_per_chunk, overlap=self.overlap_sentences, token_backend=getattr(self.encoder, "backend", "unknown"))
    def _make_sentencizer(self):
        try:
            import spacy  # type: ignore
            from spacy.pipeline import Sentencizer  # type: ignore
            nlp = spacy.blank("en")
            try:
                nlp.add_pipe("sentencizer")
            except Exception:
                try:
                    nlp.add_pipe(Sentencizer())
                except Exception:
                    nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None
    def _sentences_with_offsets_spacy(self, text: str):
        doc = self.nlp(text)
        return [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
    def _sentences_with_offsets_regex(self, text: str):
        pattern = re.compile(r'(?s).*?[\.\!\?]["\']?\s+|.+$')
        items = []
        pos = 0
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
                return self._sentences_with_offsets_spacy(text)
            except Exception:
                pass
        return self._sentences_with_offsets_regex(text)
    def chunk_document(self, text: str) -> Generator[Dict, None, None]:
        sentences = self._sentences_with_offsets(text)
        sent_items = [{"text": s, "start_char": sc, "end_char": ec, "orig_idx": i, "is_remainder": False} for i, (s, sc, ec) in enumerate(sentences)]
        i = 0; n = len(sent_items); prev_chunk = None
        while i < n:
            cur_token_count = 0; chunk_sent_texts = []; chunk_start_idx = i
            chunk_start_char = sent_items[i]["start_char"] if i < n else None
            chunk_end_char = None; is_truncated_sentence = False
            while i < n:
                sent_text = sent_items[i]["text"]
                tok_ids = self.encoder.encode(sent_text)
                sent_tok_len = len(tok_ids)
                if cur_token_count + sent_tok_len > self.max_tokens_per_chunk:
                    if not chunk_sent_texts:
                        prefix_tok_ids = tok_ids[: self.max_tokens_per_chunk]
                        try:
                            prefix_text = self.encoder.decode(prefix_tok_ids)
                        except Exception:
                            prefix_text = " ".join(str(x) for x in prefix_tok_ids)
                        chunk_sent_texts.append(prefix_text); cur_token_count = len(prefix_tok_ids); is_truncated_sentence = True
                        remainder_tok_ids = tok_ids[self.max_tokens_per_chunk:]
                        if remainder_tok_ids:
                            try:
                                remainder_text = self.encoder.decode(remainder_tok_ids)
                            except Exception:
                                remainder_text = " ".join(str(x) for x in remainder_tok_ids)
                            sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True}
                        else:
                            i += 1
                        break
                    else:
                        break
                else:
                    chunk_sent_texts.append(sent_text); cur_token_count += sent_tok_len; chunk_end_char = sent_items[i]["end_char"]; i += 1
            if not chunk_sent_texts:
                i += 1; continue
            chunk_text = " ".join(chunk_sent_texts).strip()
            chunk_meta = {"text": chunk_text, "token_count": cur_token_count, "start_sentence_idx": chunk_start_idx, "end_sentence_idx": i, "start_char": chunk_start_char, "end_char": chunk_end_char, "is_truncated_sentence": is_truncated_sentence}
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

def _charpos_to_line_range(start_char: Optional[int], end_char: Optional[int], lines: List[str]) -> Tuple[int, int]:
    if start_char is None or end_char is None or not lines:
        return 1, max(1, len(lines))
    cum = [0]
    for ln in lines:
        cum.append(cum[-1] + len(ln))
    start_line = 1; end_line = len(lines)
    for i in range(len(cum) - 1):
        if cum[i] <= start_char < cum[i + 1]:
            start_line = i + 1; break
    for j in range(len(cum) - 1, -1, -1):
        if j == 0:
            end_line = 1; break
        if cum[j - 1] <= max(0, end_char - 1) < cum[j]:
            end_line = j; break
    return start_line, end_line

def _derive_doc_id_from_head(blob_key: str, head_obj: dict, manifest: dict) -> str:
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        return manifest.get("file_hash")
    etag = head_obj.get("ETag", "")
    if isinstance(etag, str):
        etag = etag.strip('"')
    if etag:
        return sha256_hex_str(blob_key + str(etag))
    lm = head_obj.get("LastModified", "")
    if lm:
        return sha256_hex_str(blob_key + str(lm))
    base = os.path.basename(blob_key)
    if base:
        return base
    return sha256_hex_str(blob_key)

class ParquetWriter:
    def __init__(self, doc_id: str, source_url: str, content_type: str = "application/octet-stream"):
        self.doc_id = doc_id; self.source_url = source_url; self.content_type = content_type; self._rows: List[Dict[str, Any]] = []
    def _normalize_for_parquet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        fields["audio_range"] = json.dumps(payload.get("audio_range")) if payload.get("audio_range") is not None else ""
        fields["timestamp"] = payload.get("timestamp") or ""
        fields["parser_version"] = payload.get("parser_version") or PARSER_VERSION
        fields["used_ocr"] = bool(payload.get("used_ocr", False))
        lr = payload.get("line_range") or []
        if isinstance(lr, (list, tuple)) and len(lr) >= 2:
            try:
                fields["line_start"] = int(lr[0]); fields["line_end"] = int(lr[1])
            except Exception:
                fields["line_start"] = 1; fields["line_end"] = 1
        else:
            fields["line_start"] = 1; fields["line_end"] = 1
        fields["semantic_region"] = payload.get("semantic_region") or ""
        return fields
    def write_payload(self, payload: Dict[str, Any]) -> int:
        normalized = self._normalize_for_parquet(payload); self._rows.append(normalized); return 1
    def finalize_and_upload(self, out_basename: str) -> Tuple[int, str]:
        if not self._rows:
            return 0, ""
        if pa is None or pq is None:
            raise RuntimeError("pyarrow required to write parquet files (pip install pyarrow)")
        schema = pa.schema(
            [
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
                pa.field("audio_range", pa.string()),
                pa.field("timestamp", pa.string()),
                pa.field("parser_version", pa.string()),
                pa.field("used_ocr", pa.bool_()),
                pa.field("line_start", pa.int64()),
                pa.field("line_end", pa.int64()),
                pa.field("semantic_region", pa.string()),
            ]
        )
        cols: Dict[str, List[Any]] = {name: [] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table = pa.Table.from_pydict(cols, schema=schema)
        existing_md = table.schema.metadata or {}
        new_md = dict(existing_md)
        new_md.update(
            {
                b"schema_version": CHUNKED_SCHEMA_VERSION.encode("utf-8"),
                b"parser_version": PARSER_VERSION.encode("utf-8"),
                b"producer": b"txt_parser",
                b"created_at": datetime.utcnow().isoformat().encode("utf-8"),
            }
        )
        table = table.replace_schema_metadata(new_md)
        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".parquet", dir="/tmp")
        tmpfile.close()
        pq.write_table(table, tmpfile.name, compression="zstd", flavor="spark")
        local_parquet_path = tmpfile.name
        parquet_key = out_basename + ".parquet"
        dest_key = STORAGE_CHUNKED_PREFIX + parquet_key
        storage_upload_file_atomic(local_parquet_path, dest_key, content_type=self.content_type)
        try:
            os.unlink(local_parquet_path)
        except Exception:
            pass
        return len(self._rows), dest_key

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
            log.warning("upload_retry", "attempt=%d key=%s error=%s", attempt, key, str(e)); time.sleep(PUT_BACKOFF * attempt)
    raise Exception(f"atomic upload failed for {key} after {PUT_RETRIES} attempts")

def sanitize_payload_for_raw_manifest(doc_id: str, raw_key: str, chunked_key: str, rows: int, sha: str, size: int) -> Dict[str, Any]:
    return {
        "raw_key": raw_key,
        "doc_id": doc_id,
        "chunked_key": chunked_key,
        "rows": rows,
        "sha256": sha,
        "size_bytes": size,
        "schema_version": CHUNKED_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

def urllib_parse_unquote(x: str) -> str:
    try:
        from urllib.parse import unquote
        return unquote(x.split("?", 1)[0].split("#", 1)[0])
    except Exception:
        return x

def detect_ext_from_key(key: str) -> str:
    k = urllib_parse_unquote(key)
    base, ext = os.path.splitext(k)
    ext = ext.lstrip(".").lower()
    if ext in ("markdown", "mdown"):
        ext = "md"
    if ext:
        return ext
    client = get_storage_client_singleton()
    try:
        head = client.head_object(Bucket=AZURE_CONTAINER, Key=key)
        ctype = (head.get("ContentType") or head.get("content-type") or head.get("content_type") or "").lower()
        metadata = head.get("Metadata") or head.get("metadata") or {}
        meta_fn = metadata.get("filename") or metadata.get("originalname") or ""
        if meta_fn:
            _, mext = os.path.splitext(meta_fn)
            mext = mext.lstrip(".").lower()
            if mext in ("markdown", "mdown"):
                return "md"
            if mext:
                return mext
        if "markdown" in ctype or "text/markdown" in ctype:
            return "md"
        if ctype.startswith("text/"):
            return "txt"
    except Exception:
        pass
    return ""

_storage_single = None
_storage_lock2 = threading.Lock()

def get_storage_client_singleton():
    global _storage_single
    if _storage_single is None:
        with _storage_lock2:
            if _storage_single is None:
                if USE_MANAGED_IDENTITY or (BLOB_CLIENT is not None and not FS):
                    _storage_single = AzureStorageClient(fs_obj=None, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=BLOB_CLIENT)
                else:
                    _storage_single = AzureStorageClient(fs_obj=FS, root=STORAGE_ROOT, container=AZURE_CONTAINER, blob_client=None)
    return _storage_single

def storage_exists(key: str) -> bool:
    client = get_storage_client_singleton()
    try:
        client.head_object(Bucket=AZURE_CONTAINER, Key=key)
        return True
    except Exception:
        return False

def compute_semantic_region(start_line: Optional[int], end_line: Optional[int], total_lines: int) -> str:
    try:
        if total_lines <= 0:
            return "middle"
        if start_line is None or end_line is None:
            return "middle"
        start = max(1, int(start_line)); end = max(1, int(end_line))
        mid = (start + end) // 2
        ratio = mid / float(total_lines)
        if total_lines < 20:
            if ratio <= 0.30:
                return "intro"
            if ratio <= 0.70:
                return "middle"
            return "footer"
        if ratio <= 0.10:
            return "intro"
        if ratio <= 0.30:
            return "early"
        if ratio <= 0.70:
            return "middle"
        if ratio <= 0.90:
            return "late"
        return "footer"
    except Exception:
        return "middle"

def parse_file(blob_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    client = get_storage_client_singleton()
    try:
        head_obj = client.head_object(Bucket=AZURE_CONTAINER, Key=blob_key)
    except Exception:
        head_obj = {}
    doc_id = _derive_doc_id_from_head(blob_key, head_obj or {}, manifest or {})
    out_basename = f"{doc_id}"
    out_parquet_key = f"{doc_id}.parquet"
    raw_manifest_key = blob_key + ".manifest.json"
    if not FORCE_OVERWRITE:
        try:
            if storage_exists(raw_manifest_key):
                total_ms = int((time.perf_counter() - start_all) * 1000); log.info("skip_manifest_exists", "raw_manifest_exists", key=raw_manifest_key); return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
            if storage_exists(STORAGE_CHUNKED_PREFIX + out_parquet_key):
                total_ms = int((time.perf_counter() - start_all) * 1000); log.info("skip_parquet_exists", "parquet_exists", key=out_parquet_key)
                try:
                    if not storage_exists(raw_manifest_key):
                        head = client.head_object(Bucket=AZURE_CONTAINER, Key=STORAGE_CHUNKED_PREFIX + out_parquet_key)
                        etag = head.get("ETag", ""); etag = etag.strip('"') if isinstance(etag, str) else etag
                        size = head.get("ContentLength", 0)
                        raw_manifest = sanitize_payload_for_raw_manifest(doc_id, blob_key, STORAGE_CHUNKED_PREFIX + out_parquet_key, 0, etag, size)
                        client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
                except Exception:
                    pass
                return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
        except Exception:
            pass
    try:
        obj = client.get_object(Bucket=AZURE_CONTAINER, Key=blob_key)
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000); log.error("read_object_failed", "Could not read object %s: %s", blob_key, str(e)); return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    raw_body = obj["Body"].read()
    raw_text = try_decode_bytes(raw_body)
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash"); out_parquet_key = f"{doc_id}.parquet"
    if not FORCE_OVERWRITE:
        if storage_exists(STORAGE_CHUNKED_PREFIX + out_parquet_key):
            total_ms = int((time.perf_counter() - start_all) * 1000); log.info("skip_parquet_post_download", "parquet_exists", key=out_parquet_key)
            try:
                if not storage_exists(raw_manifest_key):
                    head = client.head_object(Bucket=AZURE_CONTAINER, Key=STORAGE_CHUNKED_PREFIX + out_parquet_key)
                    etag = head.get("ETag", ""); etag = etag.strip('"') if isinstance(etag, str) else etag
                    size = head.get("ContentLength", 0)
                    raw_manifest = sanitize_payload_for_raw_manifest(doc_id, blob_key, STORAGE_CHUNKED_PREFIX + out_parquet_key, 0, etag, size)
                    client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
            except Exception:
                pass
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    source_url = f"az://{AZURE_CONTAINER}/{blob_key}"
    canonical_full = canonicalize_text(raw_text)
    lines = [ln + ("\n" if not ln.endswith("\n") else "") for ln in canonical_full.split("\n")]
    total_lines = max(1, len(lines))
    max_tokens = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
    min_tokens = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
    overlap_sentences = int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES", "2"))
    full_token_count = token_count_for(canonical_full)
    saved = 0
    chunk_index = 1
    writer = ParquetWriter(doc_id=doc_id, source_url=source_url)
    file_name = os.path.basename(blob_key)
    try:
        if full_token_count <= max_tokens:
            chunk_id = f"{doc_id}_{chunk_index}"; chunk_index += 1
            start_line = 1; end_line = len(lines)
            semantic_region = compute_semantic_region(start_line, end_line, total_lines)
            payload = {
                "document_id": doc_id or "",
                "file_name": file_name,
                "chunk_id": chunk_id or "",
                "chunk_type": "txt_subchunk",
                "text": canonical_full or "",
                "token_count": int(full_token_count or 0),
                "figures": [],
                "embedding": None,
                "file_type": "text/plain",
                "source_url": source_url,
                "page_number": None,
                "slide_range": None,
                "row_range": None,
                "token_range": None,
                "audio_range": None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "parser_version": PARSER_VERSION,
                "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                "layout_tags": [],
                "used_ocr": False,
                "heading_path": [],
                "headings": [],
                "line_range": [1, len(lines)],
                "semantic_region": semantic_region,
            }
            writer.write_payload(payload)
            log.info("buffer_single_chunk", "buffered single chunk", chunk_id=payload["chunk_id"])
            saved += 1
        else:
            chunker = SentenceChunker(max_tokens_per_chunk=max_tokens, overlap_sentences=overlap_sentences, token_model="gpt2", min_tokens_per_chunk=min_tokens)
            for sub in chunker.chunk_document(canonical_full):
                chunk_text = sub.get("text", "")
                token_ct = sub.get("token_count", 0)
                schar = sub.get("start_char"); echar = sub.get("end_char")
                start_line, end_line = _charpos_to_line_range(schar, echar, lines)
                semantic_region = compute_semantic_region(start_line, end_line, total_lines)
                chunk_id = f"{doc_id}_{chunk_index}"; chunk_index += 1
                payload = {
                    "document_id": doc_id or "",
                    "file_name": file_name,
                    "chunk_id": chunk_id or "",
                    "chunk_type": "txt_subchunk",
                    "text": chunk_text or "",
                    "token_count": int(token_ct or 0),
                    "figures": [],
                    "embedding": None,
                    "file_type": "text/plain",
                    "source_url": source_url,
                    "page_number": None,
                    "slide_range": None,
                    "row_range": None,
                    "token_range": None,
                    "audio_range": None,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION,
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "heading_path": [],
                    "headings": [],
                    "line_range": [int(start_line), int(end_line)],
                    "semantic_region": semantic_region,
                }
                writer.write_payload(payload)
                log.info("buffer_subchunk", "buffered subchunk", chunk_id=payload["chunk_id"], lines=f"{start_line}-{end_line}", semantic_region=semantic_region)
                saved += 1
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.exception("buffering_failed", "Error while buffering chunks", key=blob_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if saved == 0:
            total_ms = int((time.perf_counter() - start_all) * 1000); log.info("no_chunks", "No chunks produced", key=blob_key); return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(doc_id)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        try:
            head = client.head_object(Bucket=AZURE_CONTAINER, Key=uploaded_key)
            etag = head.get("ETag", ""); etag = etag.strip('"') if isinstance(etag, str) else etag
            size = head.get("ContentLength", 0)
            raw_manifest = sanitize_payload_for_raw_manifest(doc_id, blob_key, uploaded_key, count, etag, size)
            client.put_object(Bucket=AZURE_CONTAINER, Key=raw_manifest_key, Body=json.dumps(raw_manifest).encode("utf-8"), ContentType="application/json")
        except Exception:
            log.warning("manifest_write_failed", "Failed to write raw manifest", key=blob_key)
        log.info("write_complete", "Wrote chunks", count=count, raw=blob_key, chunked=uploaded_key, duration_ms=total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("upload_failed", "Failed to upload chunked file", key=blob_key, error=str(e_up))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

if __name__ == "__main__":
    log.info("startup", "txt parser start", use_managed_identity=str(USE_MANAGED_IDENTITY).lower(), token_encoder=os.getenv("TOKEN_ENCODER", ENC_NAME), tiktoken_present="yes" if _tiktoken is not None else "no")
    client = get_storage_client_singleton()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AZURE_CONTAINER, Prefix=STORAGE_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".txt"):
                continue
            log.info("cli_route", "routing parse_file", key=key)
            manifest_key = key + ".manifest.json"
            manifest = {}
            try:
                mf_obj = client.get_object(Bucket=AZURE_CONTAINER, Key=manifest_key)
                data = mf_obj["Body"].read()
                try:
                    manifest = json.loads(data.decode("utf-8"))
                except Exception:
                    manifest = {}
            except Exception:
                manifest = {}
            try:
                res = parse_file(key, manifest)
                log.info("cli_result", "parse_result", key=key, result=res)
            except Exception:
                log.exception("cli_parse_failed", "Failed to parse", key=key)
