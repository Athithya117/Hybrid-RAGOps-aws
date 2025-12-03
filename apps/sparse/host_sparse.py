# apps/sparse/host_sparse.py
import os
import sys
import time
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import numpy as np

# fastembed sparse
from fastembed import SparseTextEmbedding

# prometheus
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# logging
logging.basicConfig(level=os.getenv("SPARSE_LOGLEVEL", "INFO"))
log = logging.getLogger("host_sparse")

# config
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/minicoil-v1")
SPARSE_HOST = os.getenv("SPARSE_HOST", "0.0.0.0")
SPARSE_PORT = int(os.getenv("SPARSE_PORT", "8201"))
SPARSE_BATCH_SIZE = int(os.getenv("SPARSE_BATCH_SIZE", "8"))
SPARSE_CUDA = os.getenv("SPARSE_CUDA", "0").upper() in ("1", "TRUE", "YES")
ENV = os.getenv("ENV", "dev")

app = FastAPI(title="sparse-embedder")

# models
class SparseOut(BaseModel):
    indices: List[int]
    values: List[float]

class SparseRequest(BaseModel):
    texts: List[str]

class SparseResponse(BaseModel):
    vectors: List[SparseOut]

# metrics (single-process)
LABELS = ["service", "env", "endpoint", "status_code"]
SERVICE_NAME = "sparse"

REQUEST_COUNT = Counter("requests_total", "Total HTTP requests", LABELS)
REQUEST_LATENCY = Histogram(
    "request_duration_seconds",
    "Request latency (seconds)",
    LABELS,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
EMBED_BATCH_SIZE = Histogram("embed_batch_size", "Observed embed batch sizes", ["service", "env"], buckets=(1,2,4,8,16,32,64))
MODEL_READY = Gauge("model_ready", "Model readiness (1=ready,0=not ready)", ["service","env"])
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time to load model (s)", ["service","env"])
MODEL_LOAD_FAILURES = Counter("model_load_failures_total", "Model load failures", ["service","env"])

# bootstrap readiness to 0
MODEL_READY.labels(service=SERVICE_NAME, env=ENV).set(0)

# global model
MODEL = None
MODEL_ERROR = None

def to_sparse(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {"indices": [], "values": []}
    if isinstance(obj, dict):
        inds = list(map(int, obj.get("indices", [])))
        vals = list(map(float, obj.get("values", [])))
        return {"indices": inds, "values": vals}
    if hasattr(obj, "indices") and hasattr(obj, "values"):
        inds = obj.indices.tolist() if hasattr(obj.indices, "tolist") else list(obj.indices)
        vals = obj.values.tolist() if hasattr(obj.values, "tolist") else list(obj.values)
        return {"indices": [int(x) for x in inds], "values": [float(x) for x in vals]}
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        inds, vals = obj
        return {"indices": [int(x) for x in inds], "values": [float(x) for x in vals]}
    raise RuntimeError("unsupported sparse object")

def metric_labels(endpoint: str, status_code: int):
    return {"service": SERVICE_NAME, "env": ENV, "endpoint": endpoint, "status_code": str(status_code)}

# --------------------------
# Deterministic model load (synchronous, fail-fast)
# --------------------------
t0 = time.time()
try:
    log.info("loading sparse model %s cuda=%s", SPARSE_MODEL_NAME, SPARSE_CUDA)
    if SPARSE_CUDA:
        try:
            MODEL = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, providers=["CUDAExecutionProvider"])
        except TypeError:
            MODEL = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
            log.warning("provider arg not supported, falling back to default provider")
    else:
        MODEL = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    # warmup call
    tmp = list(MODEL.embed(["_init_"], batch_size=1))
    if tmp is None or len(tmp) == 0:
        log.error("sparse init produced no output")
        raise RuntimeError("sparse init failed")
    load_time = time.time() - t0
    MODEL_LOAD_TIME.labels(service=SERVICE_NAME, env=ENV).set(load_time)
    MODEL_READY.labels(service=SERVICE_NAME, env=ENV).set(1)
    log.info("sparse model loaded in %.3fs", load_time)
except Exception as e:
    MODEL = None
    MODEL_ERROR = str(e)
    MODEL_LOAD_FAILURES.labels(service=SERVICE_NAME, env=ENV).inc()
    MODEL_READY.labels(service=SERVICE_NAME, env=ENV).set(0)
    log.exception("sparse model init failed: %s", e)
    # Fail loudly so container doesn't run incorrectly
    sys.exit(1)

# --------------------------
# Routes
# --------------------------
@app.post("/embed", response_model=SparseResponse)
def embed(req: SparseRequest):
    endpoint = "/embed"
    start = time.time()
    status_code = 200

    batch_len = len(req.texts) if req.texts is not None else 0
    if batch_len <= 0:
        status_code = 400
        REQUEST_COUNT.labels(**metric_labels(endpoint, status_code)).inc()
        REQUEST_LATENCY.labels(**metric_labels(endpoint, status_code)).observe(max(time.time()-start,1e-6))
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    if batch_len > SPARSE_BATCH_SIZE:
        status_code = 400
        REQUEST_COUNT.labels(**metric_labels(endpoint, status_code)).inc()
        REQUEST_LATENCY.labels(**metric_labels(endpoint, status_code)).observe(max(time.time()-start,1e-6))
        raise HTTPException(status_code=400, detail=f"batch too large max={SPARSE_BATCH_SIZE}")

    try:
        EMBED_BATCH_SIZE.labels(service=SERVICE_NAME, env=ENV).observe(batch_len)
        gen = MODEL.embed(req.texts, batch_size=len(req.texts))
        out = []
        for s in gen:
            out.append(to_sparse(s))
        return {"vectors": out}
    except HTTPException:
        raise
    except Exception as e:
        status_code = 500
        log.exception("sparse embed failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed = max(time.time()-start,1e-6)
        REQUEST_COUNT.labels(**metric_labels(endpoint, status_code)).inc()
        REQUEST_LATENCY.labels(**metric_labels(endpoint, status_code)).observe(elapsed)

@app.get("/health")
def health():
    ready = bool(MODEL_READY.labels(service=SERVICE_NAME, env=ENV)._value.get())
    return {"status": "ok" if ready else "not_ready", "model": SPARSE_MODEL_NAME, "ready": ready, "error": MODEL_ERROR}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SPARSE_HOST, port=SPARSE_PORT)
