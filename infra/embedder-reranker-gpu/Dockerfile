ARG CUDA_TAG=12.2.0
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu22.04 AS builder

ENV PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONWARNINGS="ignore"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential ca-certificates curl git \
      libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip wheel --no-cache-dir -r /app/requirements.txt -w /wheels && \
    rm -rf /root/.cache/pip

FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONWARNINGS="ignore" \
    EMBEDDER_MODEL_DIR=/opt/models/gte-modernbert-base \
    EMBEDDER_MODEL_PATH=/opt/models/gte-modernbert-base/model_fp16.onnx \
    RERANKER_MODEL_DIR=/opt/models/gte-reranker-modernbert-base \
    RERANKER_MODEL_PATH=/opt/models/gte-reranker-modernbert-base/model_fp16.onnx \
    HOST=0.0.0.0 \
    PORT=8001 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MAX_BATCH=64 \
    BATCH_WAIT_S=0.03 \
    MAX_LENGTH=512 \
    ORT_INTRA_THREADS=1 \
    ORT_INTER_THREADS=1 \
    CUDA_DEVICE=0 \
    FORCE_CPU=0 \
    LOG_LEVEL=INFO

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ca-certificates libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install --no-index --find-links /wheels -r /app/requirements.txt && \
    rm -rf /wheels /root/.cache/pip

COPY host_onnx_models.py /app/host_onnx_models.py
COPY models /opt/models

RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser && \
    chown -R appuser:appuser /app /opt/models

USER appuser

EXPOSE ${PORT}
ENTRYPOINT ["python3","-u","host_onnx_models.py"]
