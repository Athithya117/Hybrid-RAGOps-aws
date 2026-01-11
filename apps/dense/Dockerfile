# Builder stage: install fastembed and pre-warm the model
FROM python:3.12.12-slim AS builder

ARG FASTEMBED_GPU=0
ARG DENSE_MODEL_NAME
ARG DENSE_DIM

# fail-fast: require build args
RUN if [ -z "${DENSE_MODEL_NAME:-}" ]; then echo "DENSE_MODEL_NAME build-arg is required" >&2 && exit 1; fi
RUN if [ -z "${DENSE_DIM:-}" ]; then echo "DENSE_DIM build-arg is required" >&2 && exit 1; fi

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    DENSE_MODEL_NAME=${DENSE_MODEL_NAME} \
    DENSE_DIM=${DENSE_DIM}

RUN mkdir -p /models_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install fastembed + numpy into builder so we can pre-warm
RUN if [ "$FASTEMBED_GPU" = "1" ]; then \
      pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3; \
    else \
      pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3; \
    fi

WORKDIR /build
COPY host_dense.py /build/host_dense.py

# Pre-warm model (best effort, ensures weights cached inside builder)
RUN python - <<'PY'
import os, sys
try:
    from fastembed import TextEmbedding
    m = TextEmbedding(model_name=os.environ.get("DENSE_MODEL_NAME"))
    _ = list(m.embed(["_warmup_"]))
except Exception as e:
    print("WARNING: model pre-warm failed:", e, file=sys.stderr)
    # do not fail the build here: pre-warm is best-effort (you can remove this fallback if you want strict fail)
PY

# ---------------------------- FINAL RUNTIME IMAGE ----------------------------
FROM python:3.12.12-slim

ARG FASTEMBED_GPU=0
ARG DENSE_MODEL_NAME
ARG DENSE_DIM

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    DENSE_MODEL_NAME=${DENSE_MODEL_NAME} \
    DENSE_DIM=${DENSE_DIM} \
    DENSE_BATCH_SIZE=16 \
    DENSE_NORMALIZE=TRUE \
    DENSE_CUDA=0 \
    DENSE_HOST=0.0.0.0 \
    DENSE_PORT=8200 \
    FASTEMBED_CACHE=/models_cache

RUN mkdir -p /models_cache /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /models_cache /models_cache
COPY --from=builder /build/host_dense.py /app/host_dense.py

WORKDIR /app

RUN python -m pip install --upgrade pip

# Install runtime python deps deterministically based on FASTEMBED_GPU
RUN sh -c ' \
    if [ "$FASTEMBED_GPU" = "1" ]; then \
        pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 prometheus-client==0.17.0; \
    else \
        pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 prometheus-client==0.17.0; \
    fi \
'

# Bake the authoritative model+dim values into files so entrypoint can assert at runtime
RUN printf '%s' "${DENSE_MODEL_NAME}" > /app/.baked_dense_model \
 && printf '%s' "${DENSE_DIM}" > /app/.baked_dense_dim \
 && chmod 644 /app/.baked_dense_model /app/.baked_dense_dim

# Entrypoint script enforces that runtime env does not diverge from baked values
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8200

ENTRYPOINT ["sh", "/app/entrypoint.sh"]
