# ===== Build Stage =====
FROM python:3.10-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (make sure numpy<2.0 is pinned in requirements-cpu.txt)
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Compile the protobuf definitions
COPY grpc.proto .
RUN python -m grpc_tools.protoc \
      -I. \
      --python_out=. \
      --grpc_python_out=. \
      grpc.proto

# Copy your Ray Serve entrypoint
COPY rayserve-embedder-reranker.py .

# ===== Runtime Stage =====
FROM python:3.10-slim AS runtime

# Bring over installed Python packages
COPY --from=build /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

# Copy app code and generated gRPC stubs
WORKDIR /app
COPY --from=build /app/rayserve-embedder-reranker.py /app/
COPY --from=build /app/grpc_pb2.py               /app/
COPY --from=build /app/grpc_pb2_grpc.py          /app/

EXPOSE 8000 9000

# Environment configuration
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MODEL_EMBEDDER_NAME=RAG8s/gte-modernbert-base-onnx-int8 \
    EMBEDDER_ONNX_PATH=/opt/models/hf/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx \
    EMBEDDER_OMP_NUM_THREADS=1 \
    EMBEDDER_BATCH_MAX_SIZE=8 \
    RAYSERVE_EMBEDDER_NUM_REPLICAS=1 \
    RAYSERVE_EMBEDDER_MIN_REPLICAS=1 \
    RAYSERVE_EMBEDDER_MAX_REPLICAS=4 \
    RAYSERVE_EMBEDDER_TARGET_ONGOING_REQUESTS_PER_REPLICA=10 \
    RAYSERVE_EMBEDDER_BATCH_WAIT_TIMEOUT_S=0.05 \
    MODEL_RERANKER_NAME=RAG8s/gte-reranker-modernbert-base-onnx-int8 \
    RERANKER_ONNX_PATH=/opt/models/hf/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx \
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
    RERANKER_NUM_CPUS=1

# Prepare mount point for dynamic models
RUN mkdir -p /opt/models && chmod -R 777 /opt/models
VOLUME ["/opt/models"]

CMD ["python", "rayserve-embedder-reranker.py"]



