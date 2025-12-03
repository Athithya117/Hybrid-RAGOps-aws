FROM python:3.12.12-slim AS builder

ARG FASTEMBED_GPU=0
ARG MODEL_NAME=BAAI/bge-small-en-v1.5

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    MODEL_NAME=${MODEL_NAME}

RUN mkdir -p /models_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install fastembed + numpy + prometheus_client (builder stage)
RUN if [ "$FASTEMBED_GPU" = "1" ]; then \
      pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3 prometheus_client==0.23.1; \
    else \
      pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3 prometheus_client==0.23.1; \
    fi

WORKDIR /build
COPY host_dense.py /build/host_dense.py

# Pre-warm model so the weights are downloaded inside builder
RUN python - <<'PY'
import os
from fastembed import TextEmbedding
m = TextEmbedding(model_name=os.environ.get("MODEL_NAME"))
_ = list(m.embed(["_warmup_"]))
PY

# ---------------------------- FINAL RUNTIME IMAGE ----------------------------

FROM python:3.12.12-slim

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache

RUN mkdir -p /models_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /models_cache /models_cache
COPY --from=builder /build/host_dense.py /app/host_dense.py

WORKDIR /app

RUN python -m pip install --upgrade pip

ARG FASTEMBED_GPU=0

# Valid conditional install (final image)
RUN sh -c ' \
    if [ "$FASTEMBED_GPU" = "1" ]; then \
        pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 prometheus_client==0.23.1; \
    else \
        pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 prometheus_client==0.23.1; \
    fi \
'

ENV DENSE_HOST=0.0.0.0 \
    DENSE_PORT=8200 \
    DENSE_MODEL_NAME=BAAI/bge-small-en-v1.5 \
    DENSE_DIM=384 \
    DENSE_BATCH_SIZE=16 \
    DENSE_NORMALIZE=TRUE \
    DENSE_CUDA=0 \
    FASTEMBED_CACHE=/models_cache

EXPOSE 8200

CMD ["uvicorn", "host_dense:app", "--host", "0.0.0.0", "--port", "8200", "--lifespan", "on"]
