FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml poetry.lock* ./
RUN python -m pip install --upgrade pip setuptools wheel && python -m pip install "poetry==1.6.1" --root-user-action=ignore && poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi
COPY grpc.proto rayserve_embedder_reranker.py rayserve_entrypoint.py download_hf.py ./
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc.proto

FROM python:3.10-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
COPY --from=builder /src/*.py /app/
COPY --from=builder /src/*_pb2.py /app/
COPY --from=builder /src/*_pb2_grpc.py /app/
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 \
    RAY_ADDRESS=local \
    HF_HOME=/workspace/models/hf \
    MODEL_DIR=/workspace/models/onnx \
    MODEL_EMBEDDER_NAME=RAG8s/gte-modernbert-base-onnx-int8 \
    EMBEDDER_ONNX_PATH=/workspace/models/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    MODEL_RERANKER_NAME=RAG8s/gte-reranker-modernbert-base-onnx-int8 \
    RERANKER_ONNX_PATH=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    EMBEDDER_OMP_NUM_THREADS=1 RERANKER_OMP_NUM_THREADS=1 \
    EMBEDDER_BATCH_MAX_SIZE=8 RERANKER_BATCH_MAX_SIZE=4 \
    EMBEDDER_BATCH_WAIT_TIMEOUT_S=0.05 RERANKER_BATCH_WAIT_TIMEOUT_S=0.1 \
    EMBEDDER_NUM_CPUS=1 RERANKER_NUM_CPUS=1 \
    LOG_LEVEL=INFO HTTP_PORT=8000 GRPC_PORT=9000 \
    PROMETHEUS_DISABLED=1
EXPOSE 8000 9000 10001
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s CMD python -c "import sys,os,urllib.request; port=os.getenv('HTTP_PORT','8000');\
try: resp=urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=5); sys.exit(0 if resp.status==200 else 1)\
except Exception: sys.exit(1)"
CMD ["python","rayserve_entrypoint.py"]
