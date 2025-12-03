# apps/sparse/Dockerfile
FROM python:3.12.12-slim AS builder

ARG FASTEMBED_GPU=0
ARG MODEL_NAME=Qdrant/minicoil-v1

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    MODEL_NAME=${MODEL_NAME}

RUN mkdir -p /models_cache

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git curl build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install fastembed + numpy + huggingface_hub + prometheus_client in builder
RUN if [ "$FASTEMBED_GPU" = "1" ]; then \
      pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3 huggingface_hub prometheus_client==0.23.1; \
    else \
      pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3 huggingface_hub prometheus_client==0.23.1; \
    fi

WORKDIR /build
COPY host_sparse.py /build/host_sparse.py

# Pre-warm model so the weights are downloaded inside builder
RUN python - <<'PY'
import os
from fastembed import SparseTextEmbedding
m = SparseTextEmbedding(model_name=os.environ.get("MODEL_NAME"))
_ = list(m.embed(["_warmup_"], batch_size=1))
PY

# ---------------------------- FINAL RUNTIME IMAGE ----------------------------
FROM python:3.12.12-slim

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache

RUN mkdir -p /models_cache

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /models_cache /models_cache
COPY --from=builder /build/host_sparse.py /app/host_sparse.py

WORKDIR /app

RUN python -m pip install --upgrade pip

ARG FASTEMBED_GPU=0

# runtime packages
RUN sh -c ' \
    if [ "$FASTEMBED_GPU" = "1" ]; then \
      pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 huggingface_hub prometheus_client==0.23.1; \
    else \
      pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 huggingface_hub prometheus_client==0.23.1; \
    fi'

ENV SPARSE_HOST=0.0.0.0 \
    SPARSE_PORT=8201 \
    SPARSE_MODEL_NAME=Qdrant/minicoil-v1 \
    SPARSE_BATCH_SIZE=8 \
    SPARSE_CUDA=0 \
    ENV=dev \
    FASTEMBED_CACHE=/models_cache

EXPOSE 8201

CMD ["uvicorn", "host_sparse:app", "--host", "0.0.0.0", "--port", "8201", "--lifespan", "on"]
