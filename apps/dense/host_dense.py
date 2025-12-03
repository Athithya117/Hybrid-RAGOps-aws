import os
import sys
import time
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import numpy as np
from fastembed import TextEmbedding

# Prometheus client (single-process usage)
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Basic logging
logging.basicConfig(level=os.getenv("DENSE_LOGLEVEL", "INFO"))
log = logging.getLogger("host_dense")

# Config via env (port default aligned to Dockerfile: 8200)
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
DENSE_HOST = os.getenv("DENSE_HOST", "0.0.0.0")
DENSE_PORT = int(os.getenv("DENSE_PORT", "8200"))
DENSE_BATCH_SIZE = int(os.getenv("DENSE_BATCH_SIZE", "16"))
DENSE_NORMALIZE = os.getenv("DENSE_NORMALIZE", "TRUE").upper() in ("1", "TRUE", "YES")
DENSE_CUDA = os.getenv("DENSE_CUDA", "0").upper() in ("1", "TRUE", "YES")
ENV = os.getenv("ENV", "dev")

app = FastAPI(title="dense-embedder")

# Request/response models
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    vectors: List[List[float]]

def l2_normalize(v):
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    if n > 0:
        a = a / n
    return a.astype(float).tolist()

# -----------------------
# Prometheus metrics (single-process)
# -----------------------
LABELS = ["service", "env", "endpoint", "status_code"]

REQUEST_COUNT = Counter(
    "requests_total",
    "Total HTTP requests",
    LABELS,
)

REQUEST_LATENCY = Histogram(
    "request_duration_seconds",
    "Request latency (seconds)",
    LABELS,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

EMBED_BATCH_SIZE = Histogram(
    "embed_batch_size",
    "Observed embed batch sizes",
    ["service", "env"],
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

MODEL_READY = Gauge("model_ready", "Model readiness (1=ready, 0=not ready)", ["service", "env"])
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time to load model (s)", ["service", "env"])
MODEL_LOAD_FAILURES = Counter("model_load_failures_total", "Model load failures", ["service", "env"])

SERVICE_NAME = "dense"
MODEL_READY.labels(service=SERVICE_NAME, env=ENV).set(0)

# -----------------------
# Model load (deterministic); record load time/ready
# -----------------------
MODEL = None
MODEL_ERROR = None

t0 = time.time()
try:
    log.info("loading dense model %s cuda=%s", DENSE_MODEL_NAME, DENSE_CUDA)
    if DENSE_CUDA:
        try:
            MODEL = TextEmbedding(model_name=DENSE_MODEL_NAME, providers=["CUDAExecutionProvider"])
        except TypeError:
            MODEL = TextEmbedding(model_name=DENSE_MODEL_NAME)
            log.warning("provider arg not supported, falling back to default provider")
    else:
        MODEL = TextEmbedding(model_name=DENSE_MODEL_NAME)
    tmp = list(MODEL.embed(["_init_"]))
    if not tmp:
        log.error("dense init produced no output")
        raise RuntimeError("dense init failed")
    got_dim = len(tmp[0]) if hasattr(tmp[0], "__len__") else None
    if got_dim is None or got_dim != DENSE_DIM:
        log.error("embedding dimension mismatch expected=%d got=%s", DENSE_DIM, str(got_dim))
        sys.exit(1)
    load_time = time.time() - t0
    MODEL_LOAD_TIME.labels(service=SERVICE_NAME, env=ENV).set(load_time)
    MODEL_READY.labels(service=SERVICE_NAME, env=ENV).set(1)
    log.info("dense model loaded in %.3fs", load_time)
except Exception as e:
    MODEL_ERROR = str(e)
    MODEL_LOAD_FAILURES.labels(service=SERVICE_NAME, env=ENV).inc()
    MODEL_READY.labels(service=SERVICE_NAME, env=ENV).set(0)
    log.exception("dense model init failed: %s", e)
    sys.exit(1)

# -----------------------
# Instrumentation helpers
# -----------------------
def metric_labels(endpoint: str, status_code: int):
    return {"service": SERVICE_NAME, "env": ENV, "endpoint": endpoint, "status_code": str(status_code)}

# -----------------------
# Routes
# -----------------------
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    endpoint = "/embed"
    start = time.time()
    status_code = 200
    batch_len = len(req.texts) if req.texts is not None else 0
    if batch_len <= 0:
        status_code = 400
        REQUEST_COUNT.labels(**metric_labels(endpoint, status_code)).inc()
        REQUEST_LATENCY.labels(**metric_labels(endpoint, status_code)).observe(max(time.time() - start, 1e-6))
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    if batch_len > DENSE_BATCH_SIZE:
        status_code = 400
        REQUEST_COUNT.labels(**metric_labels(endpoint, status_code)).inc()
        REQUEST_LATENCY.labels(**metric_labels(endpoint, status_code)).observe(max(time.time() - start, 1e-6))
        raise HTTPException(status_code=400, detail=f"batch too large max={DENSE_BATCH_SIZE}")
    try:
        EMBED_BATCH_SIZE.labels(service=SERVICE_NAME, env=ENV).observe(batch_len)
        gens = MODEL.embed(req.texts)
        vecs = []
        for a in gens:
            if hasattr(a, "astype"):
                v = a.astype(float).tolist()
            else:
                v = [float(x) for x in a]
            if DENSE_NORMALIZE:
                v = l2_normalize(v)
            if len(v) != DENSE_DIM:
                status_code = 500
                log.exception("embedding dimension mismatch (expected %d got %d)", DENSE_DIM, len(v))
                raise HTTPException(status_code=500, detail="embedding dimension mismatch")
            vecs.append([float(x) for x in v])
        return {"vectors": vecs}
    except HTTPException:
        raise
    except Exception as e:
        status_code = 500
        log.exception("embed failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed = max(time.time() - start, 1e-6)
        REQUEST_COUNT.labels(**metric_labels(endpoint, status_code)).inc()
        REQUEST_LATENCY.labels(**metric_labels(endpoint, status_code)).observe(elapsed)

@app.get("/health")
def health():
    ready = bool(MODEL_READY.labels(service=SERVICE_NAME, env=ENV)._value.get())
    return {
        "status": "ok" if ready else "not_ready",
        "model": DENSE_MODEL_NAME,
        "dim": DENSE_DIM,
        "normalize": DENSE_NORMALIZE,
        "cuda": DENSE_CUDA,
        "model_ready": ready,
        "model_error": MODEL_ERROR,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Run server when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=DENSE_HOST, port=DENSE_PORT)
