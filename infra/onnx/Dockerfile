# ===== Build Stage =====
FROM python:3.10-slim AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG HF_TOKEN=""            # optional build-arg for private Hugging Face repos (pass via --build-arg HF_TOKEN=...)
ENV HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install build-time deps
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy sources needed for any imports/compilation
COPY grpc.proto .
COPY rayserve-embedder-reranker.py .

# Download model ONNX files from Hugging Face into /app/models/hf
# We use huggingface_hub.hf_hub_download to fetch a specific file (handles LFS)
RUN python - <<'PY'
import os
from huggingface_hub import hf_hub_download, RepositoryNotFoundError, EntryNotFoundError

models = {
    "RAG8s/gte-modernbert-base-onnx-int8": "onnx/model_int8.onnx",
    "RAG8s/gte-reranker-modernbert-base-onnx-int8": "onnx/model_int8.onnx",
}

target_root = "/app/models/hf"
os.makedirs(target_root, exist_ok=True)

hf_token = os.environ.get("HF_TOKEN") or None

for repo_id, filename in models.items():
    slug = repo_id.split("/")[-1]
    dest_dir = os.path.join(target_root, slug, os.path.dirname(filename))
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[build] Downloading {repo_id}/{filename} -> {dest_dir}")
    try:
        local_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="/tmp/hf_cache", token=hf_token)
    except (RepositoryNotFoundError, EntryNotFoundError) as e:
        print(f"[build][warning] Could not download {repo_id}/{filename}: {e}")
        raise
    # copy to deterministic location inside image
    import shutil
    shutil.copy(local_file, os.path.join(dest_dir, os.path.basename(filename)))
    print(f"[build] Saved to {os.path.join(dest_dir, os.path.basename(filename))}")
print("[build] model download complete")
PY

# ===== Runtime Stage =====
FROM python:3.10-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Runtime deps (ray CLI and runtime libs)
COPY requirements-cpu.txt /app/requirements-cpu.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r /app/requirements-cpu.txt

# Copy application code
COPY --from=build /app/rayserve-embedder-reranker.py /app/
COPY --from=build /app/grpc_pb2.py               /app/ || true
COPY --from=build /app/grpc_pb2_grpc.py          /app/ || true

# Copy downloaded models into image
COPY --from=build /app/models /app/models

# Expose expected ports
EXPOSE 8000 9000 8080

# Environment and model paths
ENV HF_HOME=/app/models/hf \
    MODEL_EMBEDDER_NAME=RAG8s/gte-modernbert-base-onnx-int8 \
    MODEL_RERANKER_NAME=RAG8s/gte-reranker-modernbert-base-onnx-int8 \
    # explicit ONNX paths (match the files downloaded during build)
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

# Ensure model files are readable
RUN chmod -R a+rX /app/models || true

# Entrypoint: start the Python Serve app (KubeRay will run the Serve config)
CMD ["python", "rayserve-embedder-reranker.py"]
