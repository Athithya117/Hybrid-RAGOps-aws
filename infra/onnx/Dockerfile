# Multi-stage build: builder -> runtime
# Builder stage: install build deps, generate gRPC stubs, pre-download models optionally
FROM python:3.10-slim AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies for generating stubs and for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy requirements and install build-time deps (keep build-time packages here)
COPY requirements-cpu.txt .
# Install runtime deps in builder so we can copy site-packages to runtime stage
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy source files
COPY grpc.proto .
COPY rayserve-embedder-reranker.py .
COPY rayserve_entrypoint.py .

# Generate python gRPC code (builder stage)
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc.proto

# Optionally, download models into /src/models if you decide to bake them into image.
# IMPORTANT: do not pass HF tokens in plain text in public CI. Use build args carefully in private CI.
# Example (commented): RUN huggingface-cli download ... --local-dir /src/models/hf/...
# If you prefer not to bake models, comment out model download and use HF token secret at runtime.

# Runtime stage: smaller image
FROM python:3.10-slim AS runtime
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy installed packages from builder to runtime
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and generated stubs
COPY --from=builder /src/*.py /app/
COPY --from=builder /src/*_pb2.py /app/ || true
COPY --from=builder /src/*_pb2_grpc.py /app/ || true
# (If you baked models in builder: copy them too)
COPY --from=builder /src/models /app/models || true

# Set non-root user
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 8000 9000 8080

# Healthcheck (local dev)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import sys,urllib.request; \
  resp = urllib.request.urlopen('http://127.0.0.1:8000/healthz'); \
  sys.exit(0 if resp.status==200 else 1)" || exit 1

# Default command: run the entrypoint module. In production KubeRay manages Serve lifecycle,
# but this cmd allows local dev runs. KubeRay will import the module via serveConfigV2.
CMD ["python", "rayserve_entrypoint.py"]
