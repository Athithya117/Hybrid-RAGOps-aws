# arm64 sha256:c3c290f22318586071efa3d615d4e142cbf173a66b04f183fcd4e19808914026
# amd64 sha256:61441535a6436f92a5d41832b5683716b87bf9323ea5defb1142db8bb5297340


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
COPY requirements.txt /build/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip wheel --no-cache-dir --wheel-dir=/wheels -r /build/requirements.txt

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

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libsndfile1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir /wheels/*

COPY host_embedding_model_cpu_inference.py /app/host_embedding_model_cpu_inference.py
COPY requirements.txt /app/requirements.txt

RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1
  
ENTRYPOINT ["python", "-u", "host_embedding_model_cpu_inference.py"]
