#!/usr/bin/env bash
set -euo pipefail

kubectl create secret generic hf-secret \
  -n rag8s \
  --from-literal=HF_TOKEN=$HF_TOKEN

kubectl create secret docker-registry regcred \
  -n rag8s \
  --docker-server=<registry> \
  --docker-username=$DOCKER_USER \
  --docker-password=$DOCKER_PASS \
  --docker-email=$EMAIL



# Local helper for quick builds. Prefer CI for production builds.
IMAGE_REPO="${IMAGE_REPO:-rag8s/rag8s-onnx-embedder-reranker-cpu-amd64}"
TAG="${TAG:-gte-modernbert}"

echo "[*] Building image ${IMAGE_REPO}:${TAG}"
docker build -t "${IMAGE_REPO}:${TAG}" -f Dockerfile .

echo "[*] Please push image with your registry credentials, e.g.:"
echo "    docker push ${IMAGE_REPO}:${TAG}"
