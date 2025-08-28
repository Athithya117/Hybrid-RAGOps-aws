# Builder
FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates gcc curl && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY requirements-cpu.txt .
RUN python -m pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements-cpu.txt

# Runtime
FROM python:3.10-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_PYTORCH=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
COPY rayserve_embedder_reranker.py rayserve_entrypoint.py test_and_push.sh requirements-cpu.txt /app/
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    DEPENDENCY_FILE=/src/requirements-cpu.txt \
    RAY_ADDRESS=local \
    HF_HOME=/workspace/models/hf \
    MODEL_ROOT=/workspace/models/onnx \
    MODEL_DIR=/workspace/models/onnx \
    EMBEDDER_PACKAGE_DIR=/workspace/models/onnx/gte-modernbert-base-onnx-int8 \
    EMBEDDER_ONNX_PATH=/workspace/models/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    EMBEDDER_CONFIG_PATH=/workspace/models/onnx/gte-modernbert-base-onnx-int8/config.json \
    EMBEDDER_TOKENIZER_JSON=/workspace/models/onnx/gte-modernbert-base-onnx-int8/tokenizer.json \
    EMBEDDER_TOKENIZER_CONFIG=/workspace/models/onnx/gte-modernbert-base-onnx-int8/tokenizer_config.json \
    EMBEDDER_SPECIAL_TOKENS=/workspace/models/onnx/gte-modernbert-base-onnx-int8/special_tokens_map.json \
    RERANKER_PACKAGE_DIR=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8 \
    RERANKER_ONNX_PATH=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    RERANKER_CONFIG_PATH=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/config.json \
    RERANKER_TOKENIZER_JSON=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/tokenizer.json \
    RERANKER_TOKENIZER_CONFIG=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/tokenizer_config.json \
    RERANKER_SPECIAL_TOKENS=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/special_tokens_map.json \
    HTTP_PORT=8000 \
    PROMETHEUS_DISABLED=1 \
    ENABLE_OTEL_COLLECTOR=false \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
    LOG_LEVEL=INFO \
    WARMUP_SAMPLES=0 \
    EMBEDDER_OMP_NUM_THREADS=1 \
    RERANKER_OMP_NUM_THREADS=1 \
    EMBEDDER_BATCH_MAX_SIZE=8 \
    EMBEDDER_BATCH_WAIT_TIMEOUT_S=0.05 \
    RERANKER_BATCH_MAX_SIZE=4 \
    RERANKER_BATCH_WAIT_TIMEOUT_S=0.1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s CMD python -c "import sys,os,urllib.request; p=os.getenv('HTTP_PORT','8000'); url=f'http://127.0.0.1:{p}/healthz'; try: r=urllib.request.urlopen(url, timeout=5); sys.exit(0 if getattr(r,'status',200)==200 else 1) except Exception: sys.exit(1)"

CMD ["python","rayserve_entrypoint.py"]
