FROM python:3.10-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      wget \
      ca-certificates \
      libsndfile1 \
      libgomp1 \
      libssl-dev \
      pkg-config \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels (wheels cache layer is portable and speeds future image builds)
COPY requirements.txt /build/requirements.txt

# Use pip wheel to pre-build all dependencies into /wheels
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip wheel --no-cache-dir --wheel-dir=/wheels -r /build/requirements.txt

# ---- runtime: slim runtime image with only runtime libs + prebuilt wheels ----
FROM python:3.10-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONWARNINGS="ignore" \
    MODEL_DIR=/workspace/models/onnx/gte-modernbert-base-onnx-int8 \
    HOST=0.0.0.0 \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MAX_BATCH=64 \
    BATCH_WAIT_S=0.03

WORKDIR /app

# Install minimal runtime system libs required by onnxruntime/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libsndfile1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt wheels from builder and install them (fast, deterministic)
COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir /wheels/*

# Copy application code
# (Only copy what you need to keep image small; keep .dockerignore to exclude tests, .git, etc.)
COPY host_embedding_model_cpu_inference.py /app/host_embedding_model_cpu_inference.py
COPY requirements.txt /app/requirements.txt

# Create non-root user and runtime directories
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE ${PORT}

# Healthcheck against the local HTTP health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Final: run the single-file server script (it will start Ray/Serve or fallback to uvicorn)
ENTRYPOINT ["python", "-u", "host_embedding_model_cpu_inference.py"]
