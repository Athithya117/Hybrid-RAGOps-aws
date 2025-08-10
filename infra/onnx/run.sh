#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
: "${DOCKER_USERNAME:=rag8s}"                                # Set default DockerHub username
: "${DOCKER_PASSWORD:?Docker password/token not provided}"   # Fail if password not set
DOCKER_REPO_NAME="${DOCKER_USERNAME}/rag8s-onnx-embedder-reranker-cpu"
DOCKER_IMAGE_TAG="v1"

# === BUILD IMAGE ===
echo "[*] Building image: $DOCKER_REPO_NAME:$DOCKER_IMAGE_TAG"
docker build -f infra/onnx/Dockerfile infra/onnx --no-cache -t "$DOCKER_REPO_NAME:$DOCKER_IMAGE_TAG"
# docker build -f infra/onnx/Dockerfile infra/onnx -t "$DOCKER_REPO_NAME:$DOCKER_IMAGE_TAG"

# === LOGIN TO DOCKER HUB ===
echo "[*] Logging in to Docker Hub as $DOCKER_USERNAME"
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

# === PUSH IMAGE ===
echo "[*] Pushing image to Docker Hub"
docker push "$DOCKER_REPO_NAME:$DOCKER_IMAGE_TAG"

# === CLEANUP ==
docker logout
echo "[*] Docker logout complete."
