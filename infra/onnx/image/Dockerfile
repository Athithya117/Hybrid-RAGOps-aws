FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt
COPY grpc.proto rayserve_embedder_reranker.py rayserve_entrypoint.py .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc.proto
FROM python:3.10-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY requirements-cpu.txt /src/requirements-cpu.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /src/requirements-cpu.txt
COPY --from=builder /src/*.py /app/
COPY --from=builder /src/*_pb2.py /app/
COPY --from=builder /src/*_pb2_grpc.py /app/
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 RAY_ADDRESS=local HF_HOME=/workspace/models/hf MODEL_DIR=/workspace/models/onnx MODEL_EMBEDDER_NAME=RAG8s/gte-modernbert-base-onnx-int8 EMBEDDER_ONNX_PATH=/workspace/models/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx MODEL_RERANKER_NAME=RAG8s/gte-reranker-modernbert-base-onnx-int8 RERANKER_ONNX_PATH=/workspace/models/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EMBEDDER_OMP_NUM_THREADS=1 RERANKER_OMP_NUM_THREADS=1 EMBEDDER_BATCH_MAX_SIZE=8 RERANKER_BATCH_MAX_SIZE=4 EMBEDDER_BATCH_WAIT_TIMEOUT_S=0.05 RERANKER_BATCH_WAIT_TIMEOUT_S=0.1 EMBEDDER_NUM_CPUS=1 RERANKER_NUM_CPUS=1 LOG_LEVEL=INFO HTTP_PORT=8000 GRPC_PORT=9000 PROMETHEUS_METRICS_PORT=8080 PROMETHEUS_DISABLED=0
EXPOSE 8000 9000 8080 10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import sys,os,urllib.request; port=os.getenv('HTTP_PORT','8000'); resp=urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz'); sys.exit(0 if resp.status==200 else 1)"
CMD ["python","rayserve_entrypoint.py"]
