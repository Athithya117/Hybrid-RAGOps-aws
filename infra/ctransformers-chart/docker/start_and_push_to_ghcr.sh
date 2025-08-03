#!/usr/bin/env bash
# infra/ctransformers-chart/docker/start_and_push_to_ghcr.sh
set -euo pipefail

# Variables (override via env)
GHCR_USER=$(echo "${GITHUB_USER:-myuser}" | tr '[:upper:]' '[:lower:]')
IMAGE_NAME="ctransformers-qwen3-cpu"
TAG="${TAG:-latest}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${TAG}"

# Validate inputs
: "${GHCR_USER:?GHCR_USER must be set}"
: "${GHCR_PAT:?GHCR_PAT (GitHub PAT) must be set}"
: "${HF_TOKEN:?HF_TOKEN (Hugging Face token) must be set}"

echo "[1/3] Logging into GHCR..."
echo "$GHCR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

echo "[2/3] Building Docker image $IMAGE..."
export DOCKER_BUILDKIT=1
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --cache-to=type=inline \
  --cache-from=type=registry,ref=$IMAGE \
  -t "$IMAGE" \
  -f infra/ctransformers-chart/docker/Dockerfile \
  infra/ctransformers-chart/docker/

echo "[3/3] Pushing $IMAGE..."
docker push "$IMAGE"

echo "[INFO] Image pushed: $IMAGE"
