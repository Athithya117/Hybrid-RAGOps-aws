#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import logging
import signal
import traceback
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import httpx
import socket
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import query_helpers as helpers

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(stream=sys.stderr, level=getattr(logging, LOG_LEVEL, logging.INFO))
root_logger = logging.getLogger("inference_pipeline.query")
# Ensure library logging doesn't propagate unexpectedly (keeps stderr deterministic)
root_logger.propagate = False

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_rag_collection1")
DENSE_URL = os.getenv("DENSE_URL", "http://dense-svc.models.svc.cluster.local:8200")
SPARSE_URL = os.getenv("SPARSE_URL", "http://sparse-svc.models.svc.cluster.local:8201")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker-svc.models.svc.cluster.local:8202")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10.0"))
SPARSE_BATCH_FALLBACK = int(os.getenv("SPARSE_BATCH_FALLBACK", "8"))
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are an assistant that must base all factual claims ONLY on the provided numbered passages. Each factual sentence MUST end with a citation in the exact format [n].")
LLM_USER_PROMPT_TEMPLATE = os.getenv("LLM_USER_PROMPT_TEMPLATE", "Summarize the following retrieved passages and answer the question in 2-3 sentences.\n\nPASSAGES:\n{passages}\n\nQUESTION: {question}\n\nAnswer:")
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "6000"))
RERANKER_MODE = os.getenv("RERANKER_MODE", helpers.RERANKER_MODE if hasattr(helpers, "RERANKER_MODE") else "AUTO").upper()
RERANK_TOPK = int(os.getenv("RERANK_TOPK", str(getattr(helpers, "RERANK_TOPK", 20))))
RERANKER_TOP_K = RERANK_TOPK
RERANK_AUTO_THRESHOLD = float(os.getenv("RERANK_AUTO_THRESHOLD", str(getattr(helpers, "RERANK_AUTO_THRESHOLD", 0.75))))
RERANK_THRESHOLD = int(os.getenv("RERANK_THRESHOLD", str(getattr(helpers, "RERANK_THRESHOLD", 30))))
RERANK_MARGIN = float(os.getenv("RERANK_MARGIN", str(getattr(helpers, "RERANK_MARGIN", 0.08))))
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", str(getattr(helpers, "RERANK_ALPHA", 0.6))))
MAX_CHUNKS_TO_LLM = int(os.getenv("MAX_CHUNKS_TO_LLM", str(getattr(helpers, "MAX_CHUNKS_TO_LLM", 6))))
QUERY_TOPK_DENSE = int(os.getenv("QUERY_TOPK_DENSE", str(getattr(helpers, "QUERY_TOPK_DENSE", 200))))
QUERY_TOPK_SPARSE = int(os.getenv("QUERY_TOPK_SPARSE", str(getattr(helpers, "QUERY_TOPK_SPARSE", 200))))
RRF_TOP_N = int(os.getenv("RRF_TOP_N", str(getattr(helpers, "RRF_TOP_N", 10))))
ENV = os.getenv("ENV", "STAGING").upper()
SERVICE_NAME = os.getenv("SERVICE_NAME", "retrieval").strip()
if not SERVICE_NAME:
    raise RuntimeError("SERVICE_NAME must be set and non-empty at process start")
LABELS = ["service", "env", "endpoint", "status_code"]
REQUEST_COUNT = Counter("retrieval_requests_total", "Total HTTP requests served by retrieval", LABELS)
REQUEST_LATENCY = Histogram("retrieval_request_duration_seconds", "Request latency (seconds) observed by retrieval", LABELS, buckets=(0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0))
ERROR_COUNT = Counter("retrieval_errors_total", "Retrieval error counts", LABELS)
SERVICE_READY = Gauge("service_ready", "Service readiness (1=ready, 0=not ready)", ["service", "env"])
SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(0)
DENSE_EMBED_COUNT = Counter("dense_embed_requests_total", "Dense embed requests", ["service", "env"])
DENSE_EMBED_LATENCY = Histogram("dense_embed_duration_seconds", "Dense embed latency", ["service", "env"], buckets=(0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0))
SPARSE_EMBED_COUNT = Counter("sparse_embed_requests_total", "Sparse embed requests", ["service", "env"])
SPARSE_EMBED_LATENCY = Histogram("sparse_embed_duration_seconds", "Sparse embed latency", ["service", "env"], buckets=(0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5))
QDRANT_QUERY_COUNT = Counter("qdrant_query_total", "Qdrant queries issued", ["service", "env"])
QDRANT_QUERY_LATENCY = Histogram("qdrant_query_duration_seconds", "Qdrant query latency", ["service", "env"], buckets=(0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0))
LLM_CALL_COUNT = Counter("llm_calls_total", "LLM calls", ["service", "env"])
LLM_CALL_LATENCY = Histogram("llm_call_duration_seconds", "LLM call latency", ["service", "env"], buckets=(0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0))
PRESIGN_COUNT = Counter("presign_requests_total", "Presign requests", ["service", "env"])
PRESIGN_LATENCY = Histogram("presign_duration_seconds", "Presign latency", ["service", "env"])
RETRIEVED_DOCS = Histogram("retrieved_docs_count", "Number of docs retrieved per request", ["service", "env"], buckets=(0,1,2,5,10,20,50))
RERANK_LATENCY = Histogram("rerank_duration_seconds", "Reranker latency", ["service", "env"], buckets=(0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5))
dense_client: Optional["AsyncDenseClient"] = None
sparse_client: Optional["AsyncSparseClient"] = None
reranker_client: Optional["AsyncRerankerClient"] = None
qdrant_client: Optional[QdrantClient] = None
ui_helpers = helpers
SHUTDOWN = False
background_task: Optional[asyncio.Task] = None
health_state = {"qdrant": False, "dense": False, "sparse": False, "reranker": False, "ready": False}

# Logging level numeric mapping for source-side gating
_LEVEL_ORDER = {"debug": 10, "info": 20, "warn": 30, "error": 40}
# Normalize the configured LOG_LEVEL into canonical lower-case (fallback to info)
_configured_log_level = LOG_LEVEL.lower()
if _configured_log_level not in _LEVEL_ORDER:
    _configured_log_level = "info"

def iso_ts():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _escape_stack(exc: Exception) -> str:
    if exc is None:
        return ""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return tb.replace("\n", "\\n")

def _json_log(level: str, evt: str, **kw):
    if not isinstance(evt, str):
        try:
            evt = str(evt)
        except Exception:
            evt = ""
    lvl = (level or "").strip().lower()
    if lvl == "warning":
        lvl = "warn"
    if lvl == "err" or lvl == "fatal" or lvl == "critical":
        lvl = "error"
    if lvl not in ("debug", "info", "warn", "error"):
        lvl = "info"

    # Enforce source-side LOG_LEVEL gating (critical fix for contract)
    try:
        if _LEVEL_ORDER.get(lvl, 20) < _LEVEL_ORDER.get(_configured_log_level, 20):
            # suppressed due to configured LOG_LEVEL
            return
    except Exception:
        # Fail-safe: if mapping fails, allow log to proceed
        pass

    event: Dict[str, Any] = {
        "timestamp": iso_ts(),
        "level": lvl,
        "message": evt,
        "service": SERVICE_NAME,
        "env": ENV,
    }
    if kw:
        for k, v in kw.items():
            if k in ("timestamp", "level", "message", "service"):
                continue
            event[k] = v
    try:
        sys.stdout.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception as e:
        try:
            # Fallback: write a concise error to stderr via python logging
            root_logger.error("failed_to_emit_json_log: %s", str(e))
        except Exception:
            pass

signal.signal(signal.SIGINT, lambda s, f: setattr(sys.modules[__name__], "SHUTDOWN", True))
signal.signal(signal.SIGTERM, lambda s, f: setattr(sys.modules[__name__], "SHUTDOWN", True))

def l2_normalize(v: List[float]) -> List[float]:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    if n > 0:
        a = a / n
    return a.astype(float).tolist()

class AsyncDenseClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT):
        self.url = url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self.timeout = timeout
    async def _client_get(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    async def health(self) -> bool:
        try:
            c = await self._client_get()
            r = await c.get(f"{self.url}/health", timeout=self.timeout)
            _json_log("debug", "dense.health", url=self.url, status=r.status_code)
            return r.status_code == 200
        except Exception as e:
            _json_log("warn", "dense.health.error", error=str(e))
            return False
    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        start = time.time()
        DENSE_EMBED_COUNT.labels(service=SERVICE_NAME, env=ENV).inc()
        try:
            c = await self._client_get()
            r = await c.post(f"{self.url}/embed", json={"texts": texts}, timeout=self.timeout)
            if r.status_code == 200:
                j = r.json()
                vecs = j.get("vectors")
                if not isinstance(vecs, list) or len(vecs) != len(texts):
                    raise RuntimeError("dense embed invalid")
                out = []
                for v in vecs:
                    vv = [float(x) for x in v]
                    if len(vv) != DENSE_DIM:
                        raise RuntimeError("dense dim mismatch")
                    out.append(l2_normalize(vv))
                _json_log("debug", "dense.embedded", count=len(out))
                return out
            raise RuntimeError(f"dense embed failed status={r.status_code} body={r.text}")
        finally:
            elapsed = max(time.time() - start, 1e-6)
            try:
                DENSE_EMBED_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(elapsed)
            except Exception:
                pass

class AsyncSparseClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT):
        self.url = url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self.timeout = timeout
    async def _client_get(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    async def health(self) -> bool:
        try:
            c = await self._client_get()
            r = await c.get(f"{self.url}/health", timeout=self.timeout)
            _json_log("debug", "sparse.health", url=self.url, status=r.status_code)
            return r.status_code == 200
        except Exception as e:
            _json_log("warn", "sparse.health.error", error=str(e))
            return False
    async def embed_chunked(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        start = time.time()
        SPARSE_EMBED_COUNT.labels(service=SERVICE_NAME, env=ENV).inc()
        try:
            c = await self._client_get()
            r = await c.post(f"{self.url}/embed", json={"texts": texts}, timeout=self.timeout)
            if r.status_code == 200:
                j = r.json()
                vecs = j.get("vectors")
                if not isinstance(vecs, list) or len(vecs) != len(texts):
                    raise RuntimeError("sparse embed invalid")
                out = []
                for s in vecs:
                    if not isinstance(s, dict) or "indices" not in s or "values" not in s:
                        raise RuntimeError("sparse element invalid")
                    out.append({"indices": [int(x) for x in s["indices"]], "values": [float(x) for x in s["values"]]})
                _json_log("debug", "sparse.embedded", count=len(out))
                return out
            if r.status_code == 400:
                try:
                    j = r.json()
                    detail = j.get("detail", "")
                    import re
                    m = re.search(r"max=(\d+)", str(detail))
                    if m:
                        maxb = int(m.group(1))
                        out = []
                        i = 0
                        while i < len(texts):
                            out.extend(await self.embed_chunked(texts[i:i + maxb]))
                            i += maxb
                        _json_log("debug", "sparse.batch.split", original=len(texts), split_to=maxb)
                        return out
                except Exception as e:
                    _json_log("warn", "sparse.batch.split.failed", error=str(e))
            if r.status_code == 422:
                maxb = SPARSE_BATCH_FALLBACK
                out = []
                i = 0
                while i < len(texts):
                    out.extend(await self.embed_chunked(texts[i:i + maxb]))
                    i += maxb
                _json_log("debug", "sparse.batch.fallback", original=len(texts), fallback=maxb)
                return out
            raise RuntimeError(f"sparse embed failed status={r.status_code}")
        finally:
            elapsed = max(time.time() - start, 1e-6)
            try:
                SPARSE_EMBED_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(elapsed)
            except Exception:
                pass

class AsyncRerankerClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT):
        self.url = url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self.timeout = timeout
    async def _client_get(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    async def health(self) -> bool:
        try:
            c = await self._client_get()
            r = await c.get(f"{self.url}/health", timeout=self.timeout)
            _json_log("debug", "reranker.health", url=self.url, status=r.status_code)
            return r.status_code == 200
        except Exception as e:
            _json_log("warn", "reranker.health.error", error=str(e))
            return False
    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        start = time.time()
        try:
            c = await self._client_get()
            r = await c.post(f"{self.url}/rerank", json={"query": query, "documents": documents}, timeout=self.timeout)
            if r.status_code == 200:
                j = r.json()
                scores = j.get("scores")
                if not isinstance(scores, list) or len(scores) != len(documents):
                    raise RuntimeError("rerank invalid")
                return [float(x) for x in scores]
            raise RuntimeError("rerank failed")
        finally:
            elapsed = max(time.time() - start, 1e-6)
            try:
                RERANK_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(elapsed)
            except Exception:
                pass

def _sanitize_chunk_for_llm(payload: Dict[str, Any], index: int) -> Dict[str, Any]:
    full_text = ui_helpers._full_text_from_payload(payload)
    heading = None
    fields = ui_helpers.ui_fields_from_payload(payload, prefer_snippet_len=None)
    d = dict(fields)
    if isinstance(d.get("headings"), (list, tuple)) and d.get("headings"):
        heading = d.get("headings")[0]
    content = full_text or ""
    return {"index": index, "heading": heading, "content": content}

def _ordered_meta_items_from_payload(payload: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return ui_helpers.ui_fields_from_payload(payload, prefer_snippet_len=None)

def build_numbered_prompt_and_ui_chunks(results: List[Dict[str, Any]], query: str):
    llm_blocks: List[str] = []
    llm_lines: List[str] = []
    ui_chunks: List[Dict[str, Any]] = []
    for idx, r in enumerate(results, start=1):
        payload = r.get("payload") or {}
        fields = _ordered_meta_items_from_payload(payload)
        full_text = ui_helpers._full_text_from_payload(payload)
        existing_keys = {k for k, _ in fields}
        if full_text and "content" not in existing_keys:
            fields = list(fields) + [("content", full_text)]
        ui_chunk = {k: v for k, v in fields}
        ui_chunk["index"] = idx
        ui_chunk["meta_items"] = fields
        ui_chunks.append(ui_chunk)
        llm_chunk = _sanitize_chunk_for_llm(payload, index=idx)
        block_lines = [f"[{idx}]"]
        if llm_chunk.get("heading"):
            block_lines.append(f"Heading: {llm_chunk['heading']}")
        if llm_chunk.get("content"):
            block_lines.append(f"Content: {llm_chunk['content']}")
        llm_blocks.append("\n".join(block_lines))
        llm_lines.append(json.dumps(llm_chunk, ensure_ascii=False))
    prompt_body = "\n\n".join(llm_blocks) + f"\n\nQ: {query}\nA:"
    return prompt_body, llm_lines, ui_chunks

def query_response_to_items(resp):
    data = None
    if hasattr(resp, "model_dump"):
        try:
            data = resp.model_dump()
        except Exception:
            data = None
    if data is None and hasattr(resp, "dict"):
        try:
            data = resp.dict()
        except Exception:
            data = None
    if data is None:
        try:
            data = resp.__dict__
        except Exception:
            data = resp
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("result", "points", "hits", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    if isinstance(data, dict) and any(k in data for k in ("id", "payload", "score")):
        return [data]
    return []

def extract_point_fields(item):
    pid = item.get("id")
    if pid is None and isinstance(item.get("point"), dict):
        pid = item["point"].get("id")
    score = item.get("score", item.get("payload_score", 0.0))
    payload = item.get("payload") or (item.get("point", {}).get("payload") if isinstance(item.get("point"), dict) else None)
    return {"id": pid, "score": float(score) if score is not None else 0.0, "payload": payload, "raw": item}

def softmax(x):
    a = np.array(x, dtype=float)
    a = a - np.max(a) if a.size else a
    e = np.exp(a)
    return (e / e.sum()).tolist() if e.sum() > 0 else (np.ones_like(a) / max(1, len(a))).tolist()

async def hybrid_query(client, collection_name, query_text, sparse_client, dense_client, reranker_client, hybrid, top_k=10, prefetch_k_dense=200, prefetch_k_sparse=200, rrf_top_n=10):
    if client is None:
        _json_log("warn", "qdrant.missing")
        return []
    if hybrid and dense_client is None:
        hybrid = False
    q_dense = None
    q_sparse = None
    items = []
    if hybrid:
        try:
            if dense_client is not None:
                dense_vecs = await dense_client.embed([query_text])
                q_dense = dense_vecs[0] if dense_vecs else None
            if sparse_client is not None:
                sparse_vecs = await sparse_client.embed_chunked([query_text])
                q_sparse = sparse_vecs[0] if sparse_vecs is not None else None
        except Exception as e:
            _json_log("warn", "embed.failed", error=str(e))
            q_dense = None
            q_sparse = None
        q_sparse_obj = None
        if q_sparse is not None:
            indices = list(map(int, q_sparse.get("indices", [])))
            values = list(map(float, q_sparse.get("values", [])))
            q_sparse_obj = SparseVector(indices=indices, values=values)
        try:
            prefetch_arg = [Prefetch(query=q_dense, using="dense", limit=prefetch_k_dense)] if q_dense is not None else None
            q_start = time.time()
            fused = await asyncio.to_thread(lambda: client.query_points(collection_name=collection_name, prefetch=prefetch_arg, query=FusionQuery(fusion=Fusion.RRF), limit=rrf_top_n, with_payload=True, with_vectors=False))
            q_elapsed = max(time.time() - q_start, 1e-6)
            QDRANT_QUERY_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(q_elapsed)
            QDRANT_QUERY_COUNT.labels(service=SERVICE_NAME, env=ENV).inc()
            items = query_response_to_items(fused)
        except Exception as e:
            _json_log("warn", "qdrant.query.failed", error=str(e))
            items = []
    else:
        q_sparse_vecs = None
        try:
            if sparse_client is not None:
                q_sparse_vecs = await sparse_client.embed_chunked([query_text])
        except Exception as e:
            _json_log("warn", "sparse.embed.failed", error=str(e))
        q_sparse = q_sparse_vecs[0] if q_sparse_vecs is not None else None
        q_sparse_obj = None
        if q_sparse is not None:
            indices = list(map(int, q_sparse.get("indices", [])))
            values = list(map(float, q_sparse.get("values", [])))
            q_sparse_obj = SparseVector(indices=indices, values=values)
        try:
            q_start = time.time()
            resp = await asyncio.to_thread(lambda: client.query_points(collection_name=collection_name, query=q_sparse_obj, using="sparse", limit=prefetch_k_sparse, with_payload=True, with_vectors=False))
            q_elapsed = max(time.time() - q_start, 1e-6)
            QDRANT_QUERY_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(q_elapsed)
            QDRANT_QUERY_COUNT.labels(service=SERVICE_NAME, env=ENV).inc()
            items = query_response_to_items(resp)
        except Exception as e:
            _json_log("warn", "qdrant.query.failed", error=str(e))
            items = []
    results = [extract_point_fields(it) for it in items]
    seen = set()
    dedup = []
    for r in results:
        payload = r.get("payload") or {}
        chunk_id = payload.get("chunk_id") if isinstance(payload, dict) else None
        key = chunk_id or r.get("id")
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
        if len(dedup) >= top_k:
            break
    return dedup

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    enable_tracing: Optional[bool] = False
    top_k: conint(ge=1, le=50) = 5
    return_chunks: Optional[bool] = True
    max_tokens: Optional[conint(ge=16, le=4096)] = LLM_MAX_TOKENS

class GenerateResponse(BaseModel):
    answer: str
    chunks: Optional[List[Dict[str, Any]]] = None

async def _background_health_checker():
    global dense_client, sparse_client, reranker_client, qdrant_client, health_state
    _json_log("info", "background.init", status="starting")
    try:
        dense_client = AsyncDenseClient(DENSE_URL)
        sparse_client = AsyncSparseClient(SPARSE_URL)
        reranker_client = AsyncRerankerClient(RERANKER_URL)
        def build_qdrant():
            try:
                return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)
            except Exception as e:
                _json_log("warn", "qdrant.init.failed", error=str(e))
                return None
        qdrant_client = await asyncio.to_thread(build_qdrant)
        for _ in range(6):
            try:
                ok_dense = await dense_client.health() if dense_client is not None else False
            except Exception:
                ok_dense = False
            try:
                ok_sparse = await sparse_client.health() if sparse_client is not None else False
            except Exception:
                ok_sparse = False
            try:
                ok_rerank = await reranker_client.health() if reranker_client is not None else False
            except Exception:
                ok_rerank = False
            q_ok = bool(qdrant_client)
            health_state["dense"] = bool(ok_dense)
            health_state["sparse"] = bool(ok_sparse)
            health_state["reranker"] = bool(ok_rerank)
            health_state["qdrant"] = bool(q_ok)
            health_state["ready"] = bool(q_ok)
            SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1 if health_state["ready"] else 0)
            _json_log("info", "clients.status", dense_ready=bool(ok_dense), sparse_ready=bool(ok_sparse), reranker_ready=bool(ok_rerank), qdrant_ready=bool(q_ok))
            if health_state["ready"]:
                break
            await asyncio.sleep(5)
        while True:
            try:
                ok_dense = await dense_client.health() if dense_client is not None else False
            except Exception:
                ok_dense = False
            try:
                ok_sparse = await sparse_client.health() if sparse_client is not None else False
            except Exception:
                ok_sparse = False
            try:
                ok_rerank = await reranker_client.health() if reranker_client is not None else False
            except Exception:
                ok_rerank = False
            q_ok = True if qdrant_client else False
            health_state["dense"] = bool(ok_dense)
            health_state["sparse"] = bool(ok_sparse)
            health_state["reranker"] = bool(ok_rerank)
            health_state["qdrant"] = bool(q_ok)
            health_state["ready"] = bool(q_ok)
            SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1 if health_state["ready"] else 0)
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        _json_log("info", "background.init", status="cancelled")
    except Exception as e:
        _json_log("error", "background.init.failed", error=str(e), stack=_escape_stack(e))

app = FastAPI()

@app.on_event("startup")
async def _startup_event():
    global background_task
    if background_task is None:
        background_task = asyncio.create_task(_background_health_checker())
        _json_log("info", "startup.scheduled_background_checker", status="scheduled")

@app.on_event("shutdown")
async def _shutdown_event():
    global background_task, dense_client, sparse_client, reranker_client, qdrant_client
    if background_task is not None:
        background_task.cancel()
        try:
            await background_task
        except Exception:
            pass
        background_task = None
    for c in (dense_client, sparse_client, reranker_client):
        if c and getattr(c, "_client", None) is not None:
            try:
                await c._client.aclose()
            except Exception:
                pass
    _json_log("info", "shutdown.complete", status="stopping")
    SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(0)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    endpoint = getattr(request.url, "path", str(request.url))
    status_code = 422
    try:
        REQUEST_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
        ERROR_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
    except Exception:
        pass
    _json_log("warn", "request.validation_error", endpoint=endpoint, error=str(exc))
    return JSONResponse(status_code=422, content=json.loads(json.dumps({"detail": exc.errors()})))

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    endpoint = getattr(request.url, "path", str(request.url))
    status_code = 500
    try:
        REQUEST_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
        ERROR_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
    except Exception:
        pass
    _json_log("error", "unhandled.exception", endpoint=endpoint, error=str(exc), stack=_escape_stack(exc))
    return JSONResponse(status_code=500, content={"detail": "internal server error"})

async def generate_handler(req: GenerateRequest) -> GenerateResponse:
    endpoint = "/generate"
    start = time.time()
    status_code = 200
    try:
        if not req.query or not req.query.strip():
            status_code = 400
            raise HTTPException(status_code=400, detail="query required")
        if qdrant_client is None:
            msg = "retrieval backend (qdrant) unavailable; check QDRANT_URL/QDRANT_API_KEY"
            _json_log("error", "generate.failed", error=msg)
            status_code = 503
            try:
                ERROR_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
            except Exception:
                pass
            return GenerateResponse(answer=msg)
        try:
            results = await hybrid_query(
                qdrant_client,
                COLLECTION_NAME,
                req.query,
                sparse_client,
                dense_client,
                reranker_client,
                (dense_client is not None and sparse_client is not None),
                top_k=int(req.top_k),
                prefetch_k_dense=QUERY_TOPK_DENSE,
                prefetch_k_sparse=QUERY_TOPK_SPARSE,
                rrf_top_n=RRF_TOP_N,
            )
        except Exception as e:
            _json_log("error", "retrieval.failed", error=str(e))
            status_code = 500
            try:
                ERROR_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
            except Exception:
                pass
            return GenerateResponse(answer=f"retrieval failed: {e}")
        RETRIEVED_DOCS.labels(service=SERVICE_NAME, env=ENV).observe(len(results))
        try:
            do_rerank = False
            if RERANKER_MODE == "ALWAYS":
                do_rerank = True
            elif RERANKER_MODE == "DISABLE":
                do_rerank = False
            else:
                top_score = results[0].get("score", 0.0) if results else 0.0
                second_score = results[1].get("score", 0.0) if len(results) > 1 else 0.0
                if top_score < RERANK_AUTO_THRESHOLD:
                    do_rerank = True
                elif (top_score - second_score) < RERANK_MARGIN:
                    do_rerank = True
            if do_rerank and reranker_client is not None and results:
                candidate_count = min(len(results), RERANK_TOPK)
                candidates = results[:candidate_count]
                docs = [ui_helpers._full_text_from_payload(c.get("payload") or {}) or "" for c in candidates]
                try:
                    rerank_scores = await reranker_client.rerank(req.query, docs)
                    if rerank_scores and len(rerank_scores) == len(candidates):
                        fused_scores = [c.get("score", 0.0) for c in candidates]
                        fused_norm = softmax(fused_scores)
                        rerank_norm = softmax([float(x) for x in rerank_scores])
                        combined = [(RERANK_ALPHA * r) + ((1.0 - RERANK_ALPHA) * f) for r, f in zip(rerank_norm, fused_norm)]
                        order = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
                        reordered = [candidates[i] for i in order]
                        for i, r in enumerate(reordered):
                            r["combined_score"] = combined[order.index(i)] if i < len(combined) else reordered[i].get("score", 0.0)
                        results = reordered + results[candidate_count:]
                except Exception as e:
                    _json_log("warn", "rerank.failed", error=str(e))
        except Exception as e:
            _json_log("warn", "rerank.decision.failed", error=str(e))
        docs_for_llm = results[:min(len(results), max(1, MAX_CHUNKS_TO_LLM))]
        if not docs_for_llm:
            return GenerateResponse(answer="no documents retrieved")
        prompt_body, llm_lines, ui_chunks = [], [], []
        answer_text = ""
        try:
            if req.enable_tracing:
                if not API_KEY:
                    msg = "LLM API key not configured; cannot produce traced answer"
                    _json_log("error", "llm.missing.key")
                    ui_chunks = []
                    for idx, r in enumerate(docs_for_llm, start=1):
                        payload = r.get("payload") or {}
                        fields = _ordered_meta_items_from_payload(payload)
                        full_text = ui_helpers._full_text_from_payload(payload)
                        existing_keys = {k for k, _ in fields}
                        if full_text and "content" not in existing_keys:
                            fields = list(fields) + [("content", full_text)]
                        meta_items = [{"k": k, "v": v} for k, v in fields]
                        src = dict(fields).get("source_url")
                        ui_chunks.append({"index": idx, "meta_items": meta_items, "source_url": src})
                    return GenerateResponse(answer=msg, chunks=[{"index": c["index"], "meta_items": c["meta_items"], "source_url": c["source_url"]} for c in ui_chunks])
                prompt_body, llm_lines, ui_chunks = build_numbered_prompt_and_ui_chunks(docs_for_llm, req.query)
                system_prompt = LLM_SYSTEM_PROMPT
                user_prompt = LLM_USER_PROMPT_TEMPLATE.format(question=req.query, passages=prompt_body)
                if API_KEY:
                    answer_text = await _call_llm_via_http(system_prompt, user_prompt, model=LLM_MODEL, max_tokens=req.max_tokens or LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
                else:
                    answer_text = "tracing requires LLM API key"
            else:
                prompt_body, llm_lines, ui_chunks = build_numbered_prompt_and_ui_chunks(docs_for_llm, req.query)
                system_prompt = LLM_SYSTEM_PROMPT
                user_prompt = LLM_USER_PROMPT_TEMPLATE.format(question=req.query, passages=prompt_body)
                if API_KEY:
                    max_toks = req.max_tokens or max(128, LLM_MAX_TOKENS)
                    answer_text = await _call_llm_via_http(system_prompt, user_prompt, model=LLM_MODEL, max_tokens=max_toks, temperature=LLM_TEMPERATURE)
                    if isinstance(answer_text, str) and len(answer_text.strip()) < 3:
                        _json_log("warn", "llm.too_short", len=len(answer_text))
                        answer_text = deterministic_summarize(llm_lines, req.query)
                else:
                    answer_text = deterministic_summarize(llm_lines, req.query)
        except Exception as e:
            _json_log("error", "llm.call.failed", error=str(e))
            answer_text = deterministic_summarize(llm_lines, req.query) or f"llm call failed: {e}"
        valid_indexes = [c["index"] for c in ui_chunks] if ui_chunks else []
        try:
            answer_text = _validate_and_filter_citations(answer_text, valid_indexes)
        except Exception as e:
            _json_log("warn", "citation.filter.failed", error=str(e))
        out_chunks = None
        if req.return_chunks and req.enable_tracing:
            out_chunks = []
            for c in ui_chunks:
                meta_items = [{"k": k, "v": v} for k, v in c.get("meta_items", [])]
                out_chunks.append({"index": c["index"], "meta_items": meta_items, "source_url": c.get("source_url")})
        return GenerateResponse(answer=answer_text, chunks=out_chunks)
    finally:
        elapsed = max(time.time() - start, 1e-6)
        try:
            REQUEST_COUNT.labels(**{"service": SERVICE_NAME, "env": ENV, "endpoint": endpoint, "status_code": str(status_code)}).inc()
            REQUEST_LATENCY.labels(**{"service": SERVICE_NAME, "env": ENV, "endpoint": endpoint, "status_code": str(status_code)}).observe(elapsed)
            if status_code >= 400:
                try:
                    ERROR_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
                except Exception:
                    pass
        except Exception:
            pass

async def _call_llm_via_http(system: str, user_prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    key = API_KEY or ""
    if not key:
        raise RuntimeError("No LLM API key found")
    base = "https://api.openai.com/v1"
    if key.startswith("gsk_"):
        base = os.getenv("GROQ_API_BASE") or "https://api.groq.com/openai/v1"
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}], "max_tokens": max_tokens, "temperature": temperature}
    LLM_CALL_COUNT.labels(service=SERVICE_NAME, env=ENV).inc()
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(url, json=payload, headers=headers)
        if r.status_code == 200:
            try:
                j = r.json()
                ch = j.get("choices")
                if isinstance(ch, list) and ch:
                    first = ch[0]
                    msg = first.get("message") or first.get("text")
                    c = msg.get("content") or first.get("text")
                    if isinstance(c, str) and c.strip():
                        return c.strip()
            except Exception:
                raise RuntimeError("LLM returned non-json response")
            raise RuntimeError("LLM HTTP response missing content")
        else:
            text = r.text or ""
            raise RuntimeError(f"LLM http error status={r.status_code} body={text}")
    finally:
        elapsed = max(time.time() - start, 1e-6)
        try:
            LLM_CALL_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(elapsed)
        except Exception:
            pass

def deterministic_summarize(lines: List[str], query: str, max_chars: int = 800) -> str:
    texts = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            c = obj.get("content", "")
        except Exception:
            c = str(ln)
        if c:
            texts.append(c)
    joined = " ".join(texts).strip()
    if not joined:
        return ""
    sents = __import__("re").split(r'(?<=[.!?])\s+', joined)
    out = []
    for s in sents:
        s = s.strip()
        if s:
            out.append(s)
            if len(out) >= 2 or sum(len(x) for x in out) >= max_chars:
                break
    if not out:
        return joined[:max_chars]
    return " ".join(out)[:max_chars]

def _validate_and_filter_citations(ans: str, valid_indexes: List[int]) -> str:
    if not ans:
        return ans
    ans = __import__("re").sub(r"\[.*?(source_url|page_number|file_name|row_range|token_range|audio_range|headings|headings_path|chunk_id).*?\]", " ", ans, flags=__import__("re").IGNORECASE)
    def repl(match):
        num = int(match.group(1))
        return f"[{num}]" if num in valid_indexes else ""
    ans = __import__("re").sub(r"\[(\d+)\]", repl, ans)
    ans = __import__("re").sub(r"https?://\S+", "", ans)
    ans = __import__("re").sub(r"\s+", " ", ans).strip()
    return ans

@app.post("/generate", response_model=GenerateResponse)
async def api_generate(req: GenerateRequest):
    return await generate_handler(req)

@app.post("/presign")
async def api_presign(request: Request):
    PRESIGN_COUNT.labels(service=SERVICE_NAME, env=ENV).inc()
    start = time.time()
    try:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("invalid json body")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid json body: {e}")
        path = payload.get("path") or payload.get("s3_path") or payload.get("az_path") or None
        if not path:
            raise HTTPException(status_code=400, detail="missing 'path' or 's3_path' in request body")
        try:
            expires = int(payload.get("expires", 3600))
        except Exception:
            expires = 3600
        inline = bool(payload.get("inline", True))
        try:
            url = await asyncio.to_thread(helpers.presign_azure_blob_blocking, path, int(expires), bool(inline))
            return {"url": url}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            _json_log("error", "presign.runtime.failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            _json_log("error", "presign.failed", error=str(e), stack=_escape_stack(e))
            raise HTTPException(status_code=500, detail=f"presign failed: {e}")
    finally:
        elapsed = max(time.time() - start, 1e-6)
        try:
            PRESIGN_LATENCY.labels(service=SERVICE_NAME, env=ENV).observe(elapsed)
        except Exception:
            pass

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    try:
        ready_val = bool(health_state.get("ready", False))
    except Exception:
        ready_val = False
    return {"status": "ready" if ready_val else "not_ready", "service_ready": ready_val, "qdrant": bool(health_state.get("qdrant", False)), "dense": bool(health_state.get("dense", False)), "sparse": bool(health_state.get("sparse", False)), "reranker": bool(health_state.get("reranker", False))}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")), log_level=os.getenv("LOG_LEVEL", "info").lower())
