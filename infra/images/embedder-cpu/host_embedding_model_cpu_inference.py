#!/usr/bin/env python3
"""
K8s-native embedding server (FastAPI + ONNX + Prometheus)

Usage notes (Kubernetes recommended):
 - Build container image containing this file and model files.
 - Run one Python process per container (recommended). Example container CMD:
     uvicorn host_embedding_model_k8s:app --host 0.0.0.0 --port 8000 --workers 1
   or use gunicorn with a single worker class if desired.

 - If you DO run multiple processes inside a pod (not recommended), set:
     PROMETHEUS_MULTIPROC_DIR=/prom-multiproc
   and mount an emptyDir at /prom-multiproc so prometheus_client multiprocess
   mode will aggregate metrics across the processes.

Prometheus:
 - Scrape the app at /metrics on whatever port you expose (default 8000).
 - Do NOT rely on a fallback metrics port in k8s; the /metrics endpoint is the single source.

This file intentionally constructs `app` at import time so k8s process managers can import it.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time
import uuid

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import Response

# transformers (tokenizer)
from transformers import AutoTokenizer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
try:
    from prometheus_client import multiprocess as prom_multiprocess
    _HAS_MULTIPROC = True
except Exception:
    prom_multiprocess = None
    _HAS_MULTIPROC = False

# ---------- Config ----------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models/onnx/gte-modernbert-base-onnx-int8")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
SERVER_DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "500"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", "0.030"))

# Prometheus multi-process directory (optional; only when you intentionally run multiple processes per pod)
PROM_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", None)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-server-k8s")

# ---------- metrics ----------
REQUESTS = Counter("embed_requests_total", "Total embed requests", ["status"])
REQUEST_DURATION = Histogram(
    "embed_request_duration_seconds",
    "Embed request latency (s)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
TOKENIZATION_H = Histogram("embed_tokenization_seconds", "Tokenization time (s)")
ONNX_INFERENCE_H = Histogram("embed_onnx_inference_seconds", "ONNX inference time (s)")
BATCH_SIZE_H = Histogram("embed_batch_size", "Batch size (number of texts processed at once)", buckets=(1,2,4,8,16,32,64,128,256))
QUEUE_GAUGE = Gauge("embed_queue_length", "Current embed queue length")
ERRORS = Counter("embed_errors_total", "Embed errors", ["error_type"])
EMBED_DIM_GAUGE = Gauge("embed_vector_dim", "Last observed embedding vector dimension")
EMBED_NORM_H = Histogram("embed_vector_norm", "Embedding L2 norm distribution")

# ---------- Tokenizer / model utils ----------
class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    max_length: Optional[int] = None

def choose_providers() -> List[str]:
    prov = ort.get_available_providers()
    log.info("ONNX Runtime providers available: %s", prov)
    if "CUDAExecutionProvider" in prov:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def load_local_tokenizer(local_dir: str):
    if not os.path.isdir(local_dir):
        raise RuntimeError(f"Tokenizer folder not found: {local_dir}")
    try:
        log.info("Loading fast tokenizer from local dir: %s", local_dir)
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True)
        log.info("Loaded fast tokenizer (local). is_fast=%s", getattr(tok, "is_fast", None))
        return tok
    except Exception as e_fast:
        log.warning("Fast tokenizer load failed: %s", e_fast)
    try:
        log.info("Loading slow tokenizer from local dir: %s", local_dir)
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
        log.info("Loaded slow tokenizer (local). is_fast=%s", getattr(tok, "is_fast", None))
        return tok
    except Exception as e_slow:
        log.error("Slow tokenizer load failed: %s", e_slow)

    msg = (
        f"Failed to load tokenizer from local dir: {local_dir}\n"
        "Ensure tokenizer.json and tokenizer_config.json (or tokenizer.model/vocab) exist in that folder.\n"
        "If you see conversion errors mentioning SentencePiece or tiktoken, install them:\n"
        "  pip install sentencepiece tiktoken\n"
    )
    raise RuntimeError(msg)

def get_abs_max_from_tokenizer(tok) -> int:
    try:
        m = int(getattr(tok, "model_max_length", 0) or 0)
        if m <= 0 or m > 1_000_000:
            return 8192
        return m
    except Exception:
        return 8192

def clamp_max_length(requested: Optional[int], server_default: int, abs_max: int) -> int:
    if requested is None:
        v = int(server_default)
    else:
        try:
            v = int(requested)
        except Exception:
            v = int(server_default)
    v = max(1, min(v, abs_max))
    return v

def map_inputs(sess: ort.InferenceSession, tokenized: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    sess_input_names = [i.name for i in sess.get_inputs()]
    lowered = {k.lower(): v for k, v in tokenized.items()}
    feed: Dict[str, np.ndarray] = {}
    for sin in sess_input_names:
        sl = sin.lower()
        if "input" in sl and "id" in sl and "input_ids" in lowered:
            feed[sin] = lowered["input_ids"]; continue
        if "attention" in sl and "mask" in sl and "attention_mask" in lowered:
            feed[sin] = lowered["attention_mask"]; continue
        if ("token_type" in sl or "segment" in sl) and "token_type_ids" in lowered:
            feed[sin] = lowered["token_type_ids"]; continue
        for tk, arr in lowered.items():
            if tk in sl or sl in tk or tk.replace("_", "") in sl.replace("_", ""):
                feed[sin] = arr
                break
    if not feed:
        raise RuntimeError(f"Could not map tokenizer outputs to ONNX inputs: {sess_input_names} vs {list(tokenized.keys())}")
    return feed

def mean_pooling(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[..., :, None]
    masked = hidden_states * mask
    sums = masked.sum(axis=1)
    counts = mask.sum(axis=1)
    counts = np.maximum(counts, 1e-9)
    return sums / counts

# ---------- AsyncBatcher with metrics ----------
class AsyncBatcher:
    def __init__(
        self,
        sess: ort.InferenceSession,
        tokenizer,
        abs_max: int,
        server_max: int,
        max_batch: int = 64,
        batch_wait_s: float = 0.03,
    ):
        self.sess = sess
        self.tokenizer = tokenizer
        self.abs_max = abs_max
        self.server_max = server_max
        self.max_batch = max_batch
        self.batch_wait_s = batch_wait_s

        self._lock = asyncio.Lock()
        self._pending_requests: List[Dict[str, Any]] = []
        self._pending_futures: List[asyncio.Future] = []
        self._flush_task: Optional[asyncio.Task] = None

    async def enqueue(self, item: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._pending_requests.append(item)
            self._pending_futures.append(fut)
            QUEUE_GAUGE.set(len(self._pending_requests))
            pending_len = len(self._pending_requests)
            if self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_worker())
            if pending_len >= self.max_batch:
                if not self._flush_task.done():
                    asyncio.create_task(self._trigger_flush())
        try:
            result = await fut
            return result
        finally:
            async with self._lock:
                if self._flush_task and self._flush_task.done():
                    self._flush_task = None

    async def _trigger_flush(self):
        await asyncio.sleep(0)

    async def _flush_worker(self):
        try:
            await asyncio.sleep(self.batch_wait_s)
            async with self._lock:
                requests = self._pending_requests
                futures = self._pending_futures
                self._pending_requests = []
                self._pending_futures = []
                QUEUE_GAUGE.set(len(self._pending_requests))
            if not requests:
                return
            try:
                results = self._process_batch(requests)
            except Exception as e:
                log.exception("Batch processing failed")
                ERRORS.labels(error_type=type(e).__name__).inc()
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
                return

            if not isinstance(results, list) or len(results) != len(futures):
                err = RuntimeError("Batch processing produced unexpected result length")
                ERRORS.labels(error_type="batch_length_mismatch").inc()
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(err)
                return

            for fut, res in zip(futures, results):
                if not fut.done():
                    fut.set_result(res)
        finally:
            async with self._lock:
                self._flush_task = None

    def _process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_texts: List[str] = []
        per_text_maxes: List[int] = []
        slices: List[Tuple[int,int]] = []

        for req in requests:
            start = len(all_texts)
            if isinstance(req, dict) and req.get("texts") is not None:
                ts = req["texts"]
                if not isinstance(ts, list):
                    ts = [ts]
                requested = req.get("max_length", None)
                effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                all_texts.extend([str(t) for t in ts])
                per_text_maxes.extend([effective_for_req] * len(ts))
                slices.append((start, len(ts)))
            elif isinstance(req, dict) and req.get("text") is not None:
                requested = req.get("max_length", None)
                effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                all_texts.append(str(req["text"]))
                per_text_maxes.append(effective_for_req)
                slices.append((start, 1))
            else:
                requested = None
                effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                all_texts.append(str(req))
                per_text_maxes.append(effective_for_req)
                slices.append((start, 1))

        if len(all_texts) == 0:
            return [{"embeddings": [], "onnx_output": None, "used_max_length": None} for _ in requests]

        batch_max = max(per_text_maxes) if per_text_maxes else self.server_max
        log.info("Batch tokenization: total_texts=%d batch_max=%d", len(all_texts), batch_max)

        t0 = time.perf_counter()
        tok = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=batch_max,
            return_tensors="np",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        token_time = time.perf_counter() - t0
        TOKENIZATION_H.observe(token_time)
        BATCH_SIZE_H.observe(len(all_texts))

        tokenized: Dict[str, np.ndarray] = {}
        for k, v in tok.items():
            arr = v
            if arr.dtype in (np.int32, np.int64):
                arr = arr.astype(np.int64)
            tokenized[k] = arr

        try:
            feed = map_inputs(self.sess, tokenized)
        except Exception as e:
            log.exception("Input mapping failed")
            ERRORS.labels(error_type="input_mapping_failed").inc()
            return [{"error": f"Input mapping failed: {e}"} for _ in requests]

        t1 = time.perf_counter()
        try:
            outputs = self.sess.run(None, feed)
            output_names = [o.name for o in self.sess.get_outputs()]
        except Exception as e:
            log.exception("ONNX runtime failed")
            ERRORS.labels(error_type="onnx_runtime_failed").inc()
            return [{"error": f"ONNX runtime failed: {e}"} for _ in requests]
        inf_time = time.perf_counter() - t1
        ONNX_INFERENCE_H.observe(inf_time)

        # select embedding output
        chosen = None; chosen_name = None
        for n,out in zip(output_names, outputs):
            nl = n.lower()
            if "pooled" in nl or "sentence" in nl or "cls" in nl:
                chosen = out; chosen_name = n; break
        if chosen is None:
            for n,out in zip(output_names, outputs):
                if getattr(out, "ndim", None) == 2:
                    chosen = out; chosen_name = n; break
        if chosen is None:
            chosen = outputs[0]; chosen_name = output_names[0]

        if getattr(chosen, "ndim", None) == 3:
            att = tokenized.get("attention_mask")
            if att is None:
                embeddings = chosen.mean(axis=1)
            else:
                embeddings = mean_pooling(chosen, att)
        elif getattr(chosen, "ndim", None) == 2:
            embeddings = chosen
        else:
            try:
                embeddings = chosen.reshape((chosen.shape[0], -1))
            except Exception:
                ERRORS.labels(error_type="onnx_shape_unexpected").inc()
                return [{"error":"Unexpected ONNX output shape", "onnx_output": chosen_name} for _ in requests]

        embeddings = np.asarray(embeddings).astype(float)

        # record embedding metrics (dims/norms)
        try:
            if embeddings.ndim == 2:
                EMBED_DIM_GAUGE.set(int(embeddings.shape[1]))
                norms = np.linalg.norm(embeddings, axis=1)
                for nrm in norms:
                    EMBED_NORM_H.observe(float(nrm))
            else:
                EMBED_DIM_GAUGE.set(int(np.prod(embeddings.shape[1:])))
        except Exception:
            log.debug("Failed to record embedding metrics", exc_info=True)

        results: List[Dict[str, Any]] = []
        for slice_start, length in slices:
            if length == 0:
                results.append({"embeddings": [], "onnx_output": chosen_name, "used_max_length": None})
            else:
                slice_emb = embeddings[slice_start:slice_start+length]
                used_for_request = int(max(per_text_maxes[slice_start:slice_start+length]))
                results.append({"embeddings": slice_emb.tolist(), "onnx_output": chosen_name, "used_max_length": used_for_request})
        return results

# ---------- FastAPI app and metrics endpoint ----------
def build_fastapi_app():
    app = FastAPI(title="ONNX Embedding (K8s + Prometheus)")

    if not os.path.exists(MODEL_PATH):
        log.error("ONNX model not found at %s", MODEL_PATH)
        raise SystemExit(f"ONNX model not found at {MODEL_PATH}")

    tokenizer = load_local_tokenizer(MODEL_DIR)
    ABS_MAX = get_abs_max_from_tokenizer(tokenizer)
    SERVER_MAX = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), ABS_MAX))
    log.info("Tokenizer ABS_MAX=%s, SERVER_DEFAULT_MAX=%s", ABS_MAX, SERVER_MAX)

    providers = choose_providers()
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    log.info("ONNX model loaded. inputs=%s outputs=%s", [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()])

    batcher = AsyncBatcher(sess=sess, tokenizer=tokenizer, abs_max=ABS_MAX, server_max=SERVER_MAX, max_batch=MAX_BATCH, batch_wait_s=BATCH_WAIT_S)

    @app.get("/metrics")
    async def metrics():
        try:
            if PROM_MULTIPROC_DIR:
                if not _HAS_MULTIPROC:
                    log.error("PROMETHEUS_MULTIPROC_DIR set but prometheus_client.multiprocess unavailable")
                    return Response(content="", media_type=CONTENT_TYPE_LATEST)
                registry = CollectorRegistry()
                prom_multiprocess.MultiProcessCollector(registry)
                data = generate_latest(registry)
            else:
                data = generate_latest()
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)
        except Exception:
            log.exception("Failed to generate metrics")
            return Response(content="", media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        # lightweight health: indicates model loaded and service up
        return {"status": "ok", "abs_max_length": ABS_MAX, "server_default_max_length": SERVER_MAX}

    @app.post("/embed")
    async def embed(payload: EmbedRequest, request: Request):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        if payload.texts:
            normalized = {"texts": payload.texts}
        elif payload.text:
            normalized = {"text": payload.text}
        else:
            REQUESTS.labels(status="400").inc()
            raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'")

        if payload.max_length is not None:
            normalized["max_length"] = int(payload.max_length)

        try:
            with REQUEST_DURATION.time():
                result = await batcher.enqueue(normalized)
            REQUESTS.labels(status="200").inc()
            return result
        except HTTPException as he:
            code = str(getattr(he, "status_code", 500))
            REQUESTS.labels(status=code).inc()
            raise
        except Exception as e:
            log.exception("Worker call failed (request_id=%s)", request_id)
            ERRORS.labels(error_type="worker_exception").inc()
            REQUESTS.labels(status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            total = time.perf_counter() - start
            log.info("request_id=%s path=%s total_s=%.4f", request_id, request.url.path, total)

    return app

# ---------- Create app at import time (k8s-friendly) ----------
# build the app so process managers (uvicorn/gunicorn) can import it directly
app = build_fastapi_app()

# __main__ kept minimal for local dev (optional). In k8s you typically do NOT use this block.
if __name__ == "__main__":
    # Run locally for quick dev (single-process). Do NOT use this in production pods.
    log.info("Running local dev server on %s:%d (single process).", HOST, PORT)
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", workers=1)
