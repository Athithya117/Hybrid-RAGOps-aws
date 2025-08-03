# infra/ctransformers-chart/docker/Dockerfile
# syntax=docker/dockerfile:1.4

#
# Stage 1: Securely download the GGUF model
#
FROM python:3.10-slim AS downloader

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir huggingface-hub==0.17.2

RUN mkdir -p /models

RUN --mount=type=secret,id=hf_token \
    python3 - <<EOF
from huggingface_hub import hf_hub_download
token = open('/run/secrets/hf_token').read().strip()
hf_hub_download(
    repo_id="unsloth/Qwen3-4B-GGUF",
    filename="Qwen3-4B-Q2_K.gguf",
    cache_dir="/models",
    token=token
)
EOF

#
# Stage 2: Build the minimal runtime image
#
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libopenblas-dev netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Cache pip downloads between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
      ctransformers==0.2.27 \
      ray[serve]==2.23.0 \
      fastapi==0.111.0 \
      "uvicorn[standard]"==0.29.0

# Copy the downloaded model
COPY --from=downloader /models /models

# Copy your Ray Serve app
WORKDIR /app
COPY rayserve_app.py .

# Healthcheck for Kubernetes
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Combined entrypoint: start Ray head, wait for GCS, then launch Serve
ENTRYPOINT ["/bin/bash", "-eo", "pipefail", "-c", "\
  ray start --head --dashboard-host=0.0.0.0 --block & \
  echo 'Waiting for Ray GCS on port 6379...' && \
  until nc -z localhost 6379; do sleep 1; done && \
  echo 'Launching Ray Serve...' && \
  serve run rayserve_app:entrypoint \
"]
