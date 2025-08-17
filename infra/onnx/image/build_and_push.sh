#!/usr/bin/env bash
set -euo pipefail

DOCKER_USERNAME="${DOCKER_USERNAME:-rag8s}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
DOCKER_REPO="${DOCKER_REPO:-${DOCKER_USERNAME}/rag8s-onnx-embedder-reranker-cpu-amd64}"
DOCKER_TAG="${DOCKER_TAG:-gte-modernbert}"
HOST_MODELS_DIR="${HOST_MODELS_DIR:-/opt/models}"   # location on your machine containing hf/ and onnx/
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$WORKDIR"
TEMP_MODELS_DIR="$IMAGE_DIR/models"

if [ ! -d "$HOST_MODELS_DIR" ]; then
  echo "ERROR: host models dir not found: $HOST_MODELS_DIR"
  exit 1
fi

if [ -d "$TEMP_MODELS_DIR" ]; then
  echo "Removing existing $TEMP_MODELS_DIR"
  rm -rf "$TEMP_MODELS_DIR"
fi

echo "[*] Copying models from $HOST_MODELS_DIR -> $TEMP_MODELS_DIR (this may take time)"
cp -a "$HOST_MODELS_DIR" "$TEMP_MODELS_DIR"

echo "[*] Building image: $DOCKER_REPO:$DOCKER_TAG"
docker build --file "$IMAGE_DIR/Dockerfile" --tag "$DOCKER_REPO:$DOCKER_TAG" "$IMAGE_DIR"

if [ -n "$DOCKER_PASSWORD" ]; then
  echo "[*] Logging in to Docker registry as $DOCKER_USERNAME"
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
fi

if command -v trivy >/dev/null 2>&1; then
  echo "[*] Scanning image with trivy"
  trivy image --exit-code 1 --severity HIGH,CRITICAL "$DOCKER_REPO:$DOCKER_TAG" || {
    echo "Security scan failed. Aborting push."
    rm -rf "$TEMP_MODELS_DIR"
    exit 1
  }
else
  echo "Trivy not found; skipping image scan."
fi

echo "[*] Pushing image: $DOCKER_REPO:$DOCKER_TAG"
docker push "$DOCKER_REPO:$DOCKER_TAG"

echo "[*] Running smoke test container (background)..."
docker run --rm -d --name tmp_rag8s_test -e HF_TOKEN="${HF_TOKEN:-}" -p 8000:8000 "$DOCKER_REPO:$DOCKER_TAG"
sleep 8
if curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
  echo "Smoke test passed"
  docker stop tmp_rag8s_test >/dev/null 2>&1 || true
else
  echo "Smoke test failed; stopping container and aborting"
  docker stop tmp_rag8s_test >/dev/null 2>&1 || true
  rm -rf "$TEMP_MODELS_DIR"
  exit 1
fi

echo "[*] Cleaning up temporary copied models"
rm -rf "$TEMP_MODELS_DIR"

if [ -n "$DOCKER_PASSWORD" ]; then
  docker logout
fi

echo "[*] Done: pushed $DOCKER_REPO:$DOCKER_TAG"
