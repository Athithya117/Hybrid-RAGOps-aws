FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY requirements-cpu.txt .
RUN pip install --upgrade pip && pip --prefer-binary install --no-cache-dir -r requirements-cpu.txt
COPY grpc.proto rayserve_embedder_reranker.py rayserve_entrypoint.py .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc.proto
COPY models/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx models/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx
COPY models/onnx/gte-modernbert-base-onnx-int8/tokenizer.json models/onnx/gte-modernbert-base-onnx-int8/tokenizer.json
COPY models/onnx/gte-modernbert-base-onnx-int8/tokenizer_config.json models/onnx/gte-modernbert-base-onnx-int8/tokenizer_config.json
COPY models/onnx/gte-modernbert-base-onnx-int8/config.json models/onnx/gte-modernbert-base-onnx-int8/config.json
COPY models/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx models/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx
COPY models/onnx/gte-reranker-modernbert-base-onnx-int8/tokenizer.json models/onnx/gte-reranker-modernbert-base-onnx-int8/tokenizer.json
COPY models/onnx/gte-reranker-modernbert-base-onnx-int8/tokenizer_config.json models/onnx/gte-reranker-modernbert-base-onnx-int8/tokenizer_config.json
COPY models/onnx/gte-reranker-modernbert-base-onnx-int8/config.json models/onnx/gte-reranker-modernbert-base-onnx-int8/config.json
FROM python:3.10-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /src/*.py /app/
COPY --from=builder /src/*_pb2.py /app/
COPY --from=builder /src/*_pb2_grpc.py /app/
COPY --from=builder /src/models/onnx /app/models/onnx
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/models/hf MODEL_DIR=/app/models/onnx MODEL_EMBEDDER_NAME=RAG8s/gte-modernbert-base-onnx-int8 MODEL_RERANKER_NAME=RAG8s/gte-reranker-modernbert-base-onnx-int8 EMBEDDER_ONNX_PATH=/app/models/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx RERANKER_ONNX_PATH=/app/models/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EMBEDDER_OMP_NUM_THREADS=1 RERANKER_OMP_NUM_THREADS=1 EMBEDDER_BATCH_MAX_SIZE=8 RERANKER_BATCH_MAX_SIZE=4 EMBEDDER_BATCH_WAIT_TIMEOUT_S=0.05 RERANKER_BATCH_WAIT_TIMEOUT_S=0.1 EMBEDDER_NUM_CPUS=1 RERANKER_NUM_CPUS=1 LOG_LEVEL=INFO HTTP_PORT=8000 GRPC_PORT=9000
EXPOSE 8000 9000 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import sys,urllib.request; resp=urllib.request.urlopen('http://127.0.0.1:8000/healthz'); sys.exit(0 if resp.status==200 else 1)"
CMD ["python","rayserve_entrypoint.py"]
