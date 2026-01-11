#!/usr/bin/env python3
"""
Indexer (retrieve CSV/parquet chunk files from Azure blob storage, embed, upsert to Qdrant).

Dual-mode storage auth (deterministic):
 - AZURE_USE_MANAGED_IDENTITY=1 (or "true") -> Managed Identity (DefaultAzureCredential).
   Optionally set AZURE_CLIENT_ID (or UAI_RAG_RW_CLIENT_ID) to select a user-assigned identity.
 - AZURE_USE_MANAGED_IDENTITY=0 (or not set / "false") -> key / SAS / connection-string mode.
"""

from __future__ import annotations
import os
import json
import hashlib
import logging
import time
import signal
import sys
import re
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timezone
import random

import numpy as np
import httpx

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("pyarrow required: pip install pyarrow", file=sys.stderr)
    raise SystemExit("pyarrow missing") from e

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_third_party_names = (
    "httpx", "httpcore", "urllib3", "qdrant_client",
    "uvicorn.access", "uvicorn.error", "asyncio",
    "botocore", "botocore.client", "botocore.hooks", "botocore.parsers",
    "boto3", "s3transfer", "aiobotocore", "boto3.resources"
)
for _n in _third_party_names:
    logging.getLogger(_n).handlers.clear()
    logging.getLogger(_n).setLevel(logging.CRITICAL)
    logging.getLogger(_n).propagate = False

_root = logging.getLogger()
_root.handlers.clear()
_root.setLevel(logging.CRITICAL)
_app_level = logging.DEBUG if LOG_LEVEL == "DEBUG" else (logging.INFO if LOG_LEVEL == "INFO" else logging.DEBUG)
logger = logging.getLogger("index")
logger.handlers.clear()
logger.setLevel(_app_level)
_h = logging.StreamHandler(stream=sys.stdout)
_h.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_h)
logger.propagate = False

USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", "").strip().lower() in ("1", "true", "yes")
if not USE_MANAGED_IDENTITY:
    USE_MANAGED_IDENTITY = os.getenv("USE_MANAGED_IDENTITY", "").strip().lower() in ("1", "true", "yes")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_rag_collection1")

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "e2e-rag-system-42")
AZURE_CHUNKED_PREFIX = os.getenv("AZURE_CHUNKED_PREFIX", "data/chunked/")
AZURE_ENDPOINT_SUFFIX = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID") or os.getenv("UAI_RAG_RW_CLIENT_ID") or None

QDRANT_URL = os.getenv("QDRANT_URL", "http://0.0.0.0:6333")
DENSE_URL   = os.getenv("DENSE_URL",   "http://0.0.0.0:8200")
SPARSE_URL  = os.getenv("SPARSE_URL",  "http://0.0.0.0:8201")

DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
UPSERT_CHUNK = int(os.getenv("UPSERT_CHUNK", "500"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10.0"))

DENSE_EMBED_TIMEOUT = float(os.getenv("DENSE_EMBED_TIMEOUT", str(max(HTTP_TIMEOUT, 30.0))))
SPARSE_EMBED_TIMEOUT = float(os.getenv("SPARSE_EMBED_TIMEOUT", str(max(HTTP_TIMEOUT, 30.0))))
EMBED_RETRIES = int(os.getenv("EMBED_RETRIES", "3"))
EMBED_BACKOFF_BASE = float(os.getenv("EMBED_BACKOFF_BASE", "1.0"))

NETWORK_RETRY_COUNT = int(os.getenv("NETWORK_RETRY_COUNT", "5"))
NETWORK_RETRY_BACKOFF_BASE = float(os.getenv("NETWORK_RETRY_BACKOFF_BASE", "1.0"))
NETWORK_RETRY_BACKOFF_MAX = float(os.getenv("NETWORK_RETRY_BACKOFF_MAX", "30.0"))

SPARSE_BATCH_FALLBACK = int(os.getenv("SPARSE_BATCH_FALLBACK", "8"))
QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "128"))
QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "32"))
QDRANT_HNSW_FULL_SCAN_THRESHOLD = int(os.getenv("QDRANT_HNSW_FULL_SCAN_THRESHOLD", "10000"))
QDRANT_ONDISK = os.getenv("QDRANT_ONDISK", "TRUE").upper() in ("1", "TRUE", "YES")
NORMALIZE_DENSE = True
SHUTDOWN = False
INFO_EVENTS = {"index.start", "batch.embedded", "index.prepared", "index.completed", "load.chunks", "collection.created", "collection.exists"}

for p in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(p, None)

def iso_ts():
    return datetime.now(timezone.utc).isoformat()

def _escape_stack(exc: BaseException) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__) if exc else [])
    return tb.replace("\n", "\\n")

def slog(level: str, evt: str, exc: Optional[BaseException] = None, **kw):
    entry = {"ts": iso_ts(), "lvl": level, "evt": evt}
    entry.update(kw)
    if level == "error" and exc is not None:
        entry["error"] = str(exc)
        entry["stack"] = _escape_stack(exc)
    elif level == "warning" and exc is not None:
        entry["error"] = str(exc)
    msg = json.dumps(entry, ensure_ascii=False)
    if level == "info":
        if LOG_LEVEL == "INFO":
            if evt in INFO_EVENTS:
                logger.info(msg)
        else:
            logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)

def handle_sigterm(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    slog("warning", "shutdown.requested", signal=signum)

signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)

def id_from_string(s: str) -> int:
    h = hashlib.md5(s.encode("utf8")).hexdigest()
    return int(h[:16], 16)

def l2_normalize(v: List[float]) -> List[float]:
    a = np.asarray(v, dtype=np.float32); n = np.linalg.norm(a)
    if n > 0: a = a / n
    return a.astype(float).tolist()

class TransientHTTPError(Exception):
    pass

def _sleep_with_jitter(base: float, cap: float, attempt: int) -> None:
    backoff = min(cap, base * (2 ** max(0, attempt - 1)))
    jittered = backoff * (0.5 + random.random() * 0.5)
    time.sleep(jittered)

def retry_call(func: Callable, *args, retries: int = NETWORK_RETRY_COUNT, backoff_base: float = NETWORK_RETRY_BACKOFF_BASE, backoff_cap: float = NETWORK_RETRY_BACKOFF_MAX, retriable: Optional[Callable[[BaseException], bool]] = None, **kwargs):
    attempt = 0
    while True:
        attempt += 1
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            should_retry = True
            if retriable is not None:
                try:
                    should_retry = bool(retriable(e))
                except Exception:
                    should_retry = False
            if SHUTDOWN:
                raise
            if not should_retry or attempt > retries:
                raise
            slog("warning", "transient.retry", error=str(e), attempt=attempt, max_retries=retries)
            _sleep_with_jitter(backoff_base, backoff_cap, attempt)

class DenseClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT, embed_timeout: float = DENSE_EMBED_TIMEOUT):
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.embed_timeout = embed_timeout

    def _get_with_retries(self, path: str, timeout: float = None) -> httpx.Response:
        url = f"{self.url}{path}"
        def call():
            try:
                r = self.client.get(url, timeout=timeout or HTTP_TIMEOUT)
                if 500 <= r.status_code < 600:
                    raise TransientHTTPError(f"server error {r.status_code}")
                return r
            except httpx.HTTPError as e:
                raise e
        return retry_call(call, retriable=lambda e: isinstance(e, (httpx.HTTPError, TransientHTTPError)))

    def _post_with_retries(self, path: str, json: Any, timeout: float = None) -> httpx.Response:
        url = f"{self.url}{path}"
        def call():
            try:
                r = self.client.post(url, json=json, timeout=timeout or self.embed_timeout)
                if 500 <= r.status_code < 600:
                    raise TransientHTTPError(f"server error {r.status_code}")
                return r
            except httpx.HTTPError as e:
                raise e
        return retry_call(call, retriable=lambda e: isinstance(e, (httpx.HTTPError, TransientHTTPError)))

    def health(self) -> bool:
        try:
            r = self._get_with_retries("/health", timeout=HTTP_TIMEOUT)
            ok = r.status_code == 200
            slog("debug", "dense.health.check", url=self.url, status=r.status_code)
            return ok
        except Exception as e:
            slog("warning", "dense.health.error", exc=e)
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        start = time.time()
        r = self._post_with_retries("/embed", json={"texts": texts}, timeout=self.embed_timeout)
        elapsed = round(time.time() - start, 3)
        slog("debug", "dense.call", url=self.url, count=len(texts), status=r.status_code, elapsed=elapsed)
        if r.status_code == 200:
            j = r.json(); vecs = j.get("vectors")
            if not isinstance(vecs, list) or len(vecs) != len(texts):
                raise RuntimeError("dense embed invalid response")
            out = []
            for v in vecs:
                if not isinstance(v, list):
                    raise RuntimeError("dense embed vector invalid")
                vv = [float(x) for x in v]
                if NORMALIZE_DENSE:
                    vv = l2_normalize(vv)
                if len(vv) != DENSE_DIM:
                    raise RuntimeError("dense embed dim mismatch")
                out.append(vv)
            slog("debug", "dense.embedded", count=len(out))
            return out
        raise RuntimeError(f"dense embed failed status={r.status_code} body={r.text}")

class SparseClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT, embed_timeout: float = SPARSE_EMBED_TIMEOUT):
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.embed_timeout = embed_timeout

    def _get_with_retries(self, path: str, timeout: float = None) -> httpx.Response:
        url = f"{self.url}{path}"
        def call():
            try:
                r = self.client.get(url, timeout=timeout or HTTP_TIMEOUT)
                if 500 <= r.status_code < 600:
                    raise TransientHTTPError(f"server error {r.status_code}")
                return r
            except httpx.HTTPError as e:
                raise e
        return retry_call(call, retriable=lambda e: isinstance(e, (httpx.HTTPError, TransientHTTPError)))

    def _post_with_retries(self, path: str, json: Any, timeout: float = None) -> httpx.Response:
        url = f"{self.url}{path}"
        def call():
            try:
                r = self.client.post(url, json=json, timeout=timeout or self.embed_timeout)
                if 500 <= r.status_code < 600:
                    raise TransientHTTPError(f"server error {r.status_code}")
                return r
            except httpx.HTTPError as e:
                raise e
        return retry_call(call, retriable=lambda e: isinstance(e, (httpx.HTTPError, TransientHTTPError)))

    def health(self) -> bool:
        try:
            r = self._get_with_retries("/health", timeout=HTTP_TIMEOUT)
            ok = r.status_code == 200
            slog("debug", "sparse.health.check", url=self.url, status=r.status_code)
            return ok
        except Exception as e:
            slog("warning", "sparse.health.error", exc=e)
            return False

    def embed_chunked(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        r = self._post_with_retries("/embed", json={"texts": texts}, timeout=self.embed_timeout)
        slog("debug", "sparse.call.attempt", url=self.url, count=len(texts), status=r.status_code)
        if r.status_code == 200:
            j = r.json(); vecs = j.get("vectors")
            if not isinstance(vecs, list) or len(vecs) != len(texts):
                raise RuntimeError("sparse embed invalid response")
            out = []
            for s in vecs:
                if not isinstance(s, dict) or "indices" not in s or "values" not in s:
                    raise RuntimeError("sparse embed element invalid")
                inds = [int(x) for x in s.get("indices", [])]; vals = [float(x) for x in s.get("values", [])]
                out.append({"indices": inds, "values": vals})
            slog("debug", "sparse.embedded", count=len(out))
            return out
        if r.status_code == 400:
            try:
                j = r.json(); detail = j.get("detail", ""); m = re.search(r"max=(\d+)", str(detail))
                if m:
                    maxb = int(m.group(1))
                    if maxb <= 0:
                        raise RuntimeError("invalid server max")
                    out = []; i = 0
                    while i < len(texts):
                        chunk = texts[i:i+maxb]
                        out.extend(self.embed_chunked(chunk))
                        i += maxb
                    slog("debug", "sparse.batch.split", original=len(texts), split_to=maxb)
                    return out
            except Exception as e:
                slog("warning", "sparse.batch.split.failed", exc=e)
        if r.status_code == 422:
            maxb = SPARSE_BATCH_FALLBACK; out = []; i = 0
            while i < len(texts):
                out.extend(self.embed_chunked(texts[i:i+maxb])); i += maxb
            slog("debug", "sparse.batch.fallback", original=len(texts), fallback=maxb)
            return out
        raise RuntimeError(f"sparse embed failed status={r.status_code} body={r.text}")

def sparse_to_qdrant_sparsevector(sparse_obj: Any) -> SparseVector:
    if sparse_obj is None:
        return SparseVector(indices=[], values=[])
    if isinstance(sparse_obj, dict):
        inds = list(map(int, sparse_obj.get("indices", []))); vals = list(map(float, sparse_obj.get("values", [])))
        return SparseVector(indices=inds, values=vals)
    raise RuntimeError("unsupported sparse object")

def _parse_list_like(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            inner = s[1:-1].strip()
            if not inner: return []
            parts = [p.strip() for p in inner.split(",")]
            out = []
            for p in parts:
                if p.isdigit():
                    out.append(int(p))
                else:
                    out.append(p.strip('"').strip("'"))
            return out
    if "-" in s and all(part.strip().isdigit() for part in s.split("-", 1)):
        a, b = s.split("-", 1)
        return [int(a), int(b)]
    return [s]

def _to_optional_int(v):
    if v is None: return None
    if isinstance(v, int): return v
    s = str(v).strip()
    if s == "": return None
    try:
        return int(s)
    except Exception:
        return None

def _to_optional_list_of_ints(v):
    parsed = _parse_list_like(v)
    out = []
    for e in parsed:
        try:
            out.append(int(e))
        except Exception:
            pass
    return out if out else None

def _to_optional_list_of_str(v):
    parsed = _parse_list_like(v)
    out = [str(x) for x in parsed]
    return out if out else None

def _to_optional_audio_range(v):
    parsed = _parse_list_like(v)
    out = [str(x) for x in parsed]
    return out if out else None

def normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(chunk)
    c["headings"] = _to_optional_list_of_str(c.get("headings")) or []
    c["heading_path"] = _to_optional_list_of_str(c.get("heading_path")) or []
    c["tags"] = _to_optional_list_of_str(c.get("tags")) or []
    c["layout_tags"] = _to_optional_list_of_str(c.get("layout_tags")) or []
    c["figures"] = _parse_list_like(c.get("figures")) or []
    c["row_range"] = _to_optional_list_of_ints(c.get("row_range"))
    c["line_range"] = _to_optional_list_of_ints(c.get("line_range"))
    c["token_range"] = _to_optional_list_of_ints(c.get("token_range"))
    # -- semantic_region normalization (safe, optional)
    if "semantic_region" in c and c.get("semantic_region") is not None:
        try:
            c["semantic_region"] = str(c.get("semantic_region"))
        except Exception:
            c["semantic_region"] = None
    c["slide_range"] = _parse_list_like(c.get("slide_range")) or None
    c["audio_range"] = _to_optional_audio_range(c.get("audio_range"))
    c["page_number"] = _to_optional_int(c.get("page_number"))
    c["token_count"] = _to_optional_int(c.get("token_count"))
    if "used_ocr" in c:
        if isinstance(c["used_ocr"], bool):
            pass
        else:
            s = str(c["used_ocr"]).lower()
            c["used_ocr"] = True if s in ("1", "true", "yes") else False
    if "timestamp" in c and c.get("timestamp") is not None:
        c["timestamp"] = str(c.get("timestamp"))
    for t in ("text", "file_name", "file_type", "source_url", "parser_version", "chunk_type", "chunk_id", "document_id"):
        if t in c and c.get(t) is not None:
            c[t] = str(c.get(t))
    return c

def create_collection_hybrid(client, name, dense_dim):
    try:
        if client.collection_exists(name):
            slog("info", "collection.exists", name=name)
            return
    except Exception as e:
        slog("warning", "collection.exists.check.failed", exc=e)
    hnsw = {"m": QDRANT_HNSW_M, "ef_construct": QDRANT_HNSW_EF_CONSTRUCT, "full_scan_threshold": QDRANT_HNSW_FULL_SCAN_THRESHOLD, "on_disk": QDRANT_ONDISK}
    try:
        retry_call(lambda: client.create_collection(
            collection_name=name,
            vectors_config={"dense": {"size": dense_dim, "distance": "Cosine", "hnsw_config": hnsw}},
            sparse_vectors_config={"sparse": {}}
        ))
        slog("info", "collection.created", name=name, dense_dim=dense_dim)
    except Exception as e:
        slog("error", "collection.create.failed", exc=e)
        raise

def create_collection_sparse_only(client, name):
    try:
        if client.collection_exists(name):
            slog("info", "collection.exists", name=name)
            return
    except Exception as e:
        slog("warning", "collection.exists.check.failed", exc=e)
    try:
        retry_call(lambda: client.create_collection(
            collection_name=name,
            vectors_config={},
            sparse_vectors_config={"sparse": {}}
        ))
        slog("info", "collection.created.sparse", name=name)
    except Exception as e:
        slog("error", "collection.create.sparse.failed", exc=e)
        raise

FULL_PAYLOAD_KEYS = [
    "document_id", "file_name", "chunk_id", "chunk_type", "text", "token_count", "source_url", "timestamp",
    "parser_version", "page_number", "row_range", "line_range", "token_range", "semantic_region", "audio_range", "slide_range",
    "headings", "heading_path", "tags", "layout_tags", "figures", "file_type", "used_ocr", "layout"
]

def make_pointstruct(pid: int, vectors_payload: Dict[str, Any], payload: Dict[str, Any]):
    try:
        return {"id": pid, "vector": vectors_payload, "payload": payload}
    except Exception as e:
        slog("error", "pointstruct.failed", exc=e)
        raise

def chunk_and_vectors_to_pointstructs(items: List[Dict[str, Any]], hybrid: bool) -> List[Dict[str, Any]]:
    pts = []
    for chunk, dvec, sparse_obj in items:
        cid = chunk.get("chunk_id") or (str(chunk.get("document_id")) + "_0")
        pid = id_from_string(cid)
        q_sv = sparse_to_qdrant_sparsevector(sparse_obj) if sparse_obj is not None else None
        vectors_payload = {}
        if hybrid:
            if dvec is not None:
                vectors_payload["dense"] = dvec
            if q_sv is not None and q_sv.indices:
                vectors_payload["sparse"] = q_sv
            if not vectors_payload:
                slog("warning", "skip.no_vectors", chunk_id=cid)
                continue
        else:
            if q_sv is None or (not q_sv.indices):
                slog("warning", "skip.no_sparse", chunk_id=cid)
                continue
            vectors_payload["sparse"] = q_sv
        payload = {}
        for k in FULL_PAYLOAD_KEYS:
            payload[k] = chunk.get(k) if k in chunk else None
        pts.append(make_pointstruct(pid, vectors_payload, payload))
    return pts

def existing_point_ids(client, collection_name, ids: List[int]) -> set:
    if not ids: return set()
    try:
        res = retry_call(lambda: client.retrieve(collection_name=collection_name, ids=ids), retriable=lambda e: True)
    except Exception as e:
        slog("warning", "retrieve.failed", exc=e)
        return set()
    out = set()
    if isinstance(res, list):
        for r in res:
            try:
                m = r.model_dump() if hasattr(r, "model_dump") else (r.dict() if hasattr(r, "dict") else r)
                pid = m.get("id")
                if pid is not None: out.add(pid)
            except Exception:
                continue
    elif isinstance(res, dict):
        for k in ("result", "points", "data"):
            if k in res and isinstance(res[k], list):
                for p in res[k]:
                    pid = p.get("id")
                    if pid is not None: out.add(pid)
    return out

def _embed_with_retry_and_split_dense(client: DenseClient, texts: List[str]) -> List[Optional[List[float]]]:
    attempts = 0
    while attempts <= EMBED_RETRIES:
        try:
            return client.embed(texts)
        except Exception as e:
            attempts += 1
            backoff = EMBED_BACKOFF_BASE * (2 ** (attempts - 1))
            slog("warning", "dense.embed.attempt.failed", error=str(e), attempt=attempts, backoff=backoff, count=len(texts))
            if attempts <= EMBED_RETRIES:
                time.sleep(backoff)
            else:
                break
    if len(texts) <= 1:
        slog("error", "dense.embed.failed.final", count=len(texts))
        raise RuntimeError("dense embed failed after retries")
    mid = len(texts) // 2
    left = _embed_with_retry_and_split_dense(client, texts[:mid])
    right = _embed_with_retry_and_split_dense(client, texts[mid:])
    return left + right

def _embed_sparse_with_retry_and_split(client: SparseClient, texts: List[str]) -> List[Optional[Dict[str,Any]]]:
    attempts = 0
    while attempts <= EMBED_RETRIES:
        try:
            return client.embed_chunked(texts)
        except Exception as e:
            attempts += 1
            backoff = EMBED_BACKOFF_BASE * (2 ** (attempts - 1))
            slog("warning", "sparse.embed.attempt.failed", error=str(e), attempt=attempts, backoff=backoff, count=len(texts))
            if attempts <= EMBED_RETRIES:
                time.sleep(backoff)
            else:
                break
    if len(texts) <= 1:
        slog("error", "sparse.embed.failed.final", count=len(texts))
        raise RuntimeError("sparse embed failed after retries")
    mid = len(texts) // 2
    left = _embed_sparse_with_retry_and_split(client, texts[:mid])
    right = _embed_sparse_with_retry_and_split(client, texts[mid:])
    return left + right

def safe_embed_and_points(chunks: List[Dict[str, Any]], sparse_client: Optional[SparseClient], dense_client: Optional[DenseClient], hybrid: bool):
    texts = [c.get("text", "") or "" for c in chunks]
    dense_vecs = [None] * len(texts)
    sparse_objs = [None] * len(texts)
    if dense_client is not None:
        try:
            dense_vecs = _embed_with_retry_and_split_dense(dense_client, texts)
            slog("debug", "dense.emb.ok", count=len(dense_vecs))
        except Exception as e:
            slog("warning", "dense.embed.failed", exc=e)
            dense_vecs = [None] * len(texts)
    if sparse_client is not None:
        try:
            sparse_objs = _embed_sparse_with_retry_and_split(sparse_client, texts)
            slog("debug", "sparse.emb.ok", count=len(sparse_objs))
        except Exception as e:
            slog("warning", "sparse.embed.failed", exc=e)
            sparse_objs = [None] * len(texts)
    items = []
    for i in range(len(chunks)):
        items.append((chunks[i], dense_vecs[i] if i < len(dense_vecs) else None, sparse_objs[i] if i < len(sparse_objs) else None))
    return chunk_and_vectors_to_pointstructs(items, hybrid)

def embed_and_upsert(client, collection_name, chunks, sparse_client, dense_client, hybrid):
    total = len(chunks); slog("info", "index.start", total_input_chunks=total, batch=BATCH_SIZE, hybrid=hybrid)
    to_upsert = []; processed = 0
    for i in range(0, total, BATCH_SIZE):
        if SHUTDOWN:
            slog("warning", "shutdown.during_index")
            break
        batch = chunks[i:i+BATCH_SIZE]; ids = [id_from_string(c.get("chunk_id") or (str(c.get("document_id")) + "_0")) for c in batch]
        start = time.time()
        existing = existing_point_ids(client, collection_name, ids)
        elapsed = round(time.time() - start, 3)
        to_process = [c for c, pid in zip(batch, ids) if pid not in existing]; skipped = len(batch) - len(to_process)
        slog("debug", "batch.check", batch_range=f"{i}..{i+len(batch)-1}", retrieve_time=elapsed, skipped=skipped)
        if to_process:
            pts = safe_embed_and_points(to_process, sparse_client, dense_client, hybrid)
            to_upsert.extend(pts); processed += len(pts)
            slog("info", "batch.embedded", embedded=len(pts), processed=processed)
    total_prepared = len(to_upsert); slog("info", "index.prepared", total_points_to_upsert=total_prepared)
    if total_prepared == 0:
        slog("warning", "index.no_new_points")
        return
    for j in range(0, total_prepared, UPSERT_CHUNK):
        if SHUTDOWN:
            slog("warning", "shutdown.before_upsert")
            break
        slice_pts = to_upsert[j:j+UPSERT_CHUNK]
        try:
            retry_call(lambda: client.upsert(collection_name=collection_name, points=slice_pts), retriable=lambda e: True)
            slog("debug", "upsert.chunk", start=j, end=j+len(slice_pts)-1)
        except Exception as e:
            slog("error", "upsert.failed", exc=e)
            raise
    slog("info", "index.completed")

def _safe_json_load(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple, dict)):
        return s
    s = str(s).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            out = []
            for p in parts:
                try:
                    out.append(json.loads(p))
                except Exception:
                    out.append(p.strip('"').strip("'"))
            return out
        return [s]

def _maybe_int(x, default=None):
    if x is None:
        return default
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s == "":
        return default
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return default

def _maybe_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).lower().strip()
    return s in ("1", "true", "yes", "y", "t")

def _paginate_with_retries(paginator, **params):
    def call():
        pages = []
        for p in paginator.paginate(**params):
            pages.append(p)
        return pages
    return retry_call(call, retriable=lambda e: True)

def _build_blob_service_client(account_name: Optional[str], account_key: Optional[str] = None, conn_str: Optional[str] = None, sas_token: Optional[str] = None):
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception as e:
        slog("error", "azure.sdk.missing", exc=e)
        raise SystemExit("azure-storage-blob required: pip install azure-storage-blob") from e

    if USE_MANAGED_IDENTITY:
        try:
            try:
                from azure.identity import DefaultAzureCredential  # type: ignore
            except Exception as e:
                slog("error", "azure.identity.missing", exc=e)
                raise SystemExit("azure-identity required for managed identity mode: pip install azure-identity") from e
            if AZURE_CLIENT_ID:
                try:
                    cred = DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID)
                except TypeError:
                    from azure.identity import ManagedIdentityCredential  # type: ignore
                    cred = ManagedIdentityCredential(client_id=AZURE_CLIENT_ID)
            else:
                cred = DefaultAzureCredential()
            if not account_name:
                raise SystemExit("AZURE_STORAGE_ACCOUNT_NAME required for managed identity mode")
            account_url = f"https://{account_name}.blob.{AZURE_ENDPOINT_SUFFIX}"
            client = BlobServiceClient(account_url=account_url, credential=cred)
            return client
        except SystemExit:
            raise
        except Exception as e:
            slog("error", "blobclient.managed.init.failed", exc=e)
            raise SystemExit(f"Failed to init BlobServiceClient with managed identity: {e}") from e

    if conn_str:
        try:
            client = BlobServiceClient.from_connection_string(conn_str)
            return client
        except Exception as e:
            slog("error", "blobclient.connstr.init.failed", exc=e)
            raise SystemExit(f"Failed to init BlobServiceClient from connection string: {e}") from e
    if account_name and account_key:
        try:
            account_url = f"https://{account_name}.blob.{AZURE_ENDPOINT_SUFFIX}"
            client = BlobServiceClient(account_url=account_url, credential=account_key)
            return client
        except Exception as e:
            slog("error", "blobclient.key.init.failed", exc=e)
            raise SystemExit(f"Failed to init BlobServiceClient with account key: {e}") from e
    if account_name and sas_token:
        try:
            account_url = f"https://{account_name}.blob.{AZURE_ENDPOINT_SUFFIX}"
            token = sas_token if sas_token.startswith("?") else ("?" + sas_token)
            client = BlobServiceClient(account_url=account_url + token)
            return client
        except Exception as e:
            slog("error", "blobclient.sas.init.failed", exc=e)
            raise SystemExit(f"Failed to init BlobServiceClient with SAS token: {e}") from e
    raise SystemExit("No valid Azure storage credential available for initializing BlobServiceClient")

def load_chunks_from_azure(account_name: Optional[str], account_key: Optional[str], container: str, prefix: str) -> List[Dict[str, Any]]:
    conn_str = AZURE_STORAGE_CONNECTION_STRING or None
    sas = AZURE_SAS_TOKEN or None
    client = _build_blob_service_client(account_name or None, account_key=account_key or None, conn_str=conn_str, sas_token=sas)
    try:
        container_client = client.get_container_client(container)
    except Exception as e:
        slog("error", "azure.container.client.failed", exc=e)
        raise SystemExit(f"Unable to get container client: {e}") from e

    try:
        blob_iter = list(container_client.list_blobs(name_starts_with=prefix))
    except Exception as e:
        slog("error", "azure.list.failed", exc=e)
        raise SystemExit(f"azure list failed: {e}") from e

    parquet_keys = []
    json_keys = []
    for b in blob_iter:
        k = getattr(b, "name", None)
        if not k:
            continue
        if k.lower().endswith(".parquet"):
            parquet_keys.append(k)
        elif k.lower().endswith(".json"):
            json_keys.append(k)
    keys = sorted(parquet_keys) if parquet_keys else sorted(json_keys)
    if not keys:
        slog("error", "no_azure_chunks", container=container, prefix=prefix)
        raise SystemExit(f"No chunk files in azure://{account_name}/{container}/{prefix} (looked for .parquet/.json)")

    chunks = []
    for k in keys:
        try:
            blob_client = container_client.get_blob_client(k)
            body = retry_call(lambda: blob_client.download_blob().readall(), retriable=lambda e: True)
        except Exception as e:
            slog("error", "azure.get_blob.failed", key=k, exc=e)
            raise
        if k.lower().endswith(".parquet"):
            try:
                table = pq.read_table(pa.BufferReader(body))
                data = table.to_pydict()
                if not data:
                    continue
                n = len(next(iter(data.values())))
                for i in range(n):
                    try:
                        row = {col: (data.get(col)[i] if col in data and i < len(data.get(col)) else None) for col in data.keys()}
                        chunk = {
                            "document_id": row.get("document_id") or "",
                            "file_name": row.get("file_name") or "",
                            "chunk_id": row.get("chunk_id") or "",
                            "chunk_type": row.get("chunk_type") or "",
                            "text": row.get("text") or "",
                            "token_count": _maybe_int(row.get("token_count"), 0),
                            "figures": _safe_json_load(row.get("figures")),
                            "tags": _safe_json_load(row.get("tags")),
                            "layout_tags": _safe_json_load(row.get("layout_tags")),
                            "heading_path": _safe_json_load(row.get("heading_path")),
                            "headings": _safe_json_load(row.get("headings")),
                            "file_type": row.get("file_type") or "",
                            "source_url": row.get("source_url") or "",
                            "audio_range": _safe_json_load(row.get("audio_range")) if row.get("audio_range") is not None else None,
                            "timestamp": row.get("timestamp") or None,
                            "parser_version": row.get("parser_version") or None,
                            "used_ocr": _maybe_bool(row.get("used_ocr")),
                            "line_range": [ _maybe_int(row.get("line_start"), 1), _maybe_int(row.get("line_end"), 1) ],
                            "page_number": _maybe_int(row.get("page_number")),
                            "row_range": _safe_json_load(row.get("row_range")) or None,
                            "token_range": _safe_json_load(row.get("token_range")) or None,
                            "slide_range": _safe_json_load(row.get("slide_range")) or None,
                            "semantic_region": row.get("semantic_region") if "semantic_region" in row else None
                        }
                        norm = normalize_chunk(chunk)
                        chunks.append(norm)
                    except Exception as e:
                        slog("warning", "chunk.normalize.failed", key=k, exc=e)
            except Exception as e:
                slog("error", "azure.parquet.read.failed", key=k, exc=e)
                raise
        else:
            try:
                body_text = body.decode("utf8") if isinstance(body, (bytes, bytearray)) else str(body)
                data = json.loads(body_text)
                if isinstance(data, list):
                    for raw in data:
                        try:
                            norm = normalize_chunk(raw)
                            chunks.append(norm)
                        except Exception as e:
                            slog("warning", "chunk.normalize.failed", key=k, exc=e)
                else:
                    slog("error", "azure.chunk.format.invalid", key=k, type=str(type(data)))
                    raise SystemExit(f"Expected list in azure://{account_name}/{container}/{k}")
            except Exception as e:
                slog("error", "azure.get_blob.failed", key=k, exc=e)
                raise
    slog("info", "load.chunks", original_chunks=len(chunks))
    return chunks

def validate_envs():
    missing = []
    if not AZURE_CONTAINER:
        missing.append("AZURE_CONTAINER")
    if USE_MANAGED_IDENTITY:
        if not AZURE_STORAGE_ACCOUNT_NAME:
            missing.append("AZURE_STORAGE_ACCOUNT_NAME")
        if missing:
            slog("error", "env.missing", missing=missing)
            raise SystemExit(f"Missing required envs for managed identity mode: {', '.join(missing)}")
        try:
            import importlib
            importlib.import_module('azure.identity')
            importlib.import_module('azure.storage.blob')
        except Exception as e:
            slog("error", "azure.sdk.missing", exc=e)
            raise SystemExit("azure-identity and azure-storage-blob Python packages required for managed identity mode (pip install azure-identity azure-storage-blob)")
    else:
        if not (AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY) or (AZURE_STORAGE_ACCOUNT_NAME and AZURE_SAS_TOKEN)):
            missing.append("AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY) or (AZURE_STORAGE_ACCOUNT_NAME + AZURE_SAS_TOKEN)")
        if missing:
            slog("error", "env.missing", missing=missing)
            raise SystemExit(f"Missing required envs for key/SAS/connstr mode: {', '.join(missing)}")
        try:
            import importlib
            importlib.import_module('azure.storage.blob')
        except Exception as e:
            slog("error", "azure.sdk.missing", exc=e)
            raise SystemExit("azure-storage-blob Python package required (pip install azure-storage-blob)")

def validate_and_build_clients() -> Tuple[Optional[DenseClient], Optional[SparseClient]]:
    dc = DenseClient(DENSE_URL, timeout=HTTP_TIMEOUT, embed_timeout=DENSE_EMBED_TIMEOUT)
    sc = SparseClient(SPARSE_URL, timeout=HTTP_TIMEOUT, embed_timeout=SPARSE_EMBED_TIMEOUT)
    slog("info", "clients.created", dense_url=dc.url, sparse_url=sc.url, qdrant_url=QDRANT_URL)
    dense = None; sparse = None
    try:
        if dc.health():
            dense = dc
            slog("debug", "dense.ready", url=dc.url)
            try:
                resp = dc._post_with_retries("/embed", json={"texts": ["ping"]}, timeout=min(dc.embed_timeout, 5.0))
                slog("debug", "dense.smoke", status=resp.status_code, body=(resp.text[:200] if resp.text else None))
                if resp.status_code != 200:
                    slog("warning", "dense.smoke.badstatus", status=resp.status_code)
                    dense = None
            except Exception as e:
                slog("warning", "dense.smoke.failed", error=str(e), url=dc.url)
                dense = None
        else:
            slog("warning", "dense.unhealthy", url=dc.url)
    except Exception as e:
        slog("warning", "dense.check.error", exc=e)
        dense = None
    try:
        if sc.health():
            sparse = sc
            slog("debug", "sparse.ready", url=sc.url)
            try:
                resp = sc._post_with_retries("/embed", json={"texts": ["ping"]}, timeout=min(sc.embed_timeout, 5.0))
                slog("debug", "sparse.smoke", status=resp.status_code, body=(resp.text[:200] if resp.text else None))
                if resp.status_code != 200:
                    slog("warning", "sparse.smoke.badstatus", status=resp.status_code)
                    sparse = None
            except Exception as e:
                slog("warning", "sparse.smoke.failed", error=str(e), url=sc.url)
                sparse = None
        else:
            slog("warning", "sparse.unhealthy", url=sc.url)
    except Exception as e:
        slog("warning", "sparse.check.error", exc=e)
        sparse = None
    if dense is None and sparse is None:
        slog("error", "no_embed_services")
        raise SystemExit("Neither dense nor sparse service healthy (see logs)")
    return dense, sparse

def retrieve_and_index():
    account_name = AZURE_STORAGE_ACCOUNT_NAME or None
    account_key = AZURE_STORAGE_ACCOUNT_KEY or None
    chunks = load_chunks_from_azure(account_name, account_key, AZURE_CONTAINER, AZURE_CHUNKED_PREFIX)
    dense_client, sparse_client = validate_and_build_clients()
    hybrid_mode = (dense_client is not None and sparse_client is not None)
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)
    except Exception as e:
        slog("error", "qdrant.client.init.failed", exc=e)
        raise SystemExit(f"Unable to contact Qdrant: {e}")
    try:
        retry_call(lambda: client.get_collections(), retriable=lambda e: True)
    except Exception as e:
        slog("error", "qdrant.unreachable", exc=e)
        raise SystemExit(f"Unable to contact Qdrant: {e}")
    if hybrid_mode:
        create_collection_hybrid(client, COLLECTION_NAME, dense_dim=DENSE_DIM)
    else:
        create_collection_sparse_only(client, COLLECTION_NAME)
    embed_and_upsert(client, COLLECTION_NAME, chunks, sparse_client, dense_client, hybrid_mode)


if __name__ == "__main__":
    # Runtime wrapper that captures input/output metrics without changing core logic.
    # Emits a single-line JSON summary as the last stdout line:
    # {"collection":"<name>","indexed_points":<n>,"total_input_chunks":<n>,"skipped_existing":<n>}
    try:
        slog("info", "startup", use_managed_identity=str(USE_MANAGED_IDENTITY).lower(), azure_client_id=str(AZURE_CLIENT_ID or ""))
        validate_envs()
    except SystemExit:
        raise
    except Exception as e:
        slog("error", "startup.failed", exc=e)
        raise

    metrics = {"total_input_chunks": 0, "indexed_points": 0}

    # --- patch load_chunks_from_azure to capture total_input_chunks ---
    try:
        _orig_load_chunks = load_chunks_from_azure  # type: ignore
        def _load_chunks_wrapper(account_name, account_key, container, prefix):
            chunks = _orig_load_chunks(account_name, account_key, container, prefix)
            try:
                metrics["total_input_chunks"] = len(chunks) if chunks is not None else 0
            except Exception:
                metrics["total_input_chunks"] = 0
            return chunks
        globals()["load_chunks_from_azure"] = _load_chunks_wrapper  # type: ignore
    except Exception as e:
        slog("warning", "wrap.load_chunks.failed", exc=e)

    # --- patch QdrantClient.upsert to count points actually upserted (increment only on success) ---
    try:
        _orig_upsert = QdrantClient.upsert  # type: ignore
        def _upsert_wrapper(self, *args, **kwargs):
            # find points list defensively (either kwargs or first list-like positional arg)
            pts = kwargs.get("points", None)
            if pts is None:
                for a in args:
                    if isinstance(a, list):
                        pts = a
                        break
            # call original; increment only if call does not raise
            result = _orig_upsert(self, *args, **kwargs)
            try:
                n = len(pts) if pts is not None else 0
            except Exception:
                n = 0
            try:
                metrics["indexed_points"] = int(metrics.get("indexed_points", 0)) + int(n)
            except Exception:
                metrics["indexed_points"] = metrics.get("indexed_points", 0)
            return result
        QdrantClient.upsert = _upsert_wrapper  # type: ignore
    except Exception as e:
        slog("warning", "wrap.qdrant.upsert.failed", exc=e)

    # Run the original flow and capture exit semantics
    exit_code = 0
    try:
        retrieve_and_index()
        exit_code = 0
    except SystemExit as se:
        exit_code = getattr(se, "code", 1) or 1
    except Exception as e:
        slog("error", "index.unhandled_exception", exc=e)
        exit_code = 1

    # Compose summary. skipped_existing is best-effort: total_input_chunks - indexed_points
    total_input_chunks = int(metrics.get("total_input_chunks", 0) or 0)
    indexed_points = int(metrics.get("indexed_points", 0) or 0)
    skipped = total_input_chunks - indexed_points
    if skipped < 0:
        skipped = 0

    summary = {
        "collection": COLLECTION_NAME,
        "indexed_points": indexed_points,
        "total_input_chunks": total_input_chunks,
        "skipped_existing": skipped,
    }

    # Print single-line JSON as the very last stdout output (deterministic, parseable)
    try:
        print(json.dumps(summary, separators=(",", ":")), flush=True)
    except Exception:
        print(f'{{"collection":"{COLLECTION_NAME}","indexed_points":{indexed_points},"total_input_chunks":{total_input_chunks},"skipped_existing":{skipped}}}', flush=True)

    # Exit with non-zero if indexing had a non-zero/system error
    if exit_code:
        sys.exit(exit_code)
    sys.exit(0)


