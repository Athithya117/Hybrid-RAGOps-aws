FROM python:3.10-slim AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt && rm -rf /root/.cache

RUN pip install --no-cache-dir huggingface-hub && rm -rf /root/.cache

COPY grpc.proto .
COPY rayserve-embedder-reranker.py .

RUN mkdir -p /app/models/hf
RUN huggingface-cli download RAG8s/gte-modernbert-base-onnx-int8 onnx/model_int8.onnx \
    --local-dir /app/models/hf/gte-modernbert-base-onnx-int8 --local-dir-use-symlinks False
RUN huggingface-cli download RAG8s/gte-reranker-modernbert-base-onnx-int8 onnx/model_int8.onnx \
    --local-dir /app/models/hf/gte-reranker-modernbert-base-onnx-int8 --local-dir-use-symlinks False

FROM python:3.10-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY requirements-cpu.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r requirements-cpu.txt && rm -rf /root/.cache

COPY --from=build /app/rayserve-embedder-reranker.py /app/
COPY --from=build /app/grpc.proto /app/
COPY --from=build /app/models /app/models

EXPOSE 8000 9000 8080

ENV HF_HOME=/app/models/hf \
    MODEL_EMBEDDER_NAME=RAG8s/gte-modernbert-base-onnx-int8 \
    MODEL_RERANKER_NAME=RAG8s/gte-reranker-modernbert-base-onnx-int8 \
    EMBEDDER_ONNX_PATH=/app/models/hf/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    RERANKER_ONNX_PATH=/app/models/hf/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    EMBEDDER_OMP_NUM_THREADS=1 \
    EMBEDDER_BATCH_MAX_SIZE=8 \
    RAYSERVE_EMBEDDER_NUM_REPLICAS=1 \
    RAYSERVE_EMBEDDER_MIN_REPLICAS=1 \
    RAYSERVE_EMBEDDER_MAX_REPLICAS=4 \
    RAYSERVE_EMBEDDER_TARGET_ONGOING_REQUESTS_PER_REPLICA=10 \
    RAYSERVE_EMBEDDER_BATCH_WAIT_TIMEOUT_S=0.05 \
    RERANKER_OMP_NUM_THREADS=1 \
    RERANKER_BATCH_MAX_SIZE=4 \
    RAYSERVE_RERANKER_NUM_REPLICAS=1 \
    RAYSERVE_RERANKER_MIN_REPLICAS=1 \
    RAYSERVE_RERANKER_MAX_REPLICAS=2 \
    RAYSERVE_RERANKER_TARGET_ONGOING_REQUESTS_PER_REPLICA=5 \
    RAYSERVE_RERANKER_BATCH_WAIT_TIMEOUT_S=0.1 \
    GRPC_PORT=9000 \
    GRPC_SERVICER_FUNCTIONS=embed_service_pb2_grpc.add_EmbedServiceServicer_to_server,rerank_service_pb2_grpc.add_RerankServiceServicer_to_server \
    GRPC_REQUEST_TIMEOUT_S=30 \
    EMBEDDER_NUM_CPUS=1 \
    RERANKER_NUM_CPUS=1 \
    LOG_LEVEL=INFO

RUN chmod -R a+rX /app/models

CMD ["python", "rayserve-embedder-reranker.py"]
