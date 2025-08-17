#!/usr/bin/env bash
set -euo pipefail

# Kubernetes secrets (idempotent: delete+create to avoid duplicate errors)
kubectl delete secret hf-secret -n rag8s --ignore-not-found
kubectl create secret generic hf-secret \
  -n rag8s \
  --from-literal=HF_TOKEN="${HF_TOKEN:-}"

kubectl delete secret regcred -n rag8s --ignore-not-found
kubectl create secret docker-registry regcred \
  -n rag8s \
  --docker-server="${DOCKER_SERVER:-docker.io}" \
  --docker-username="${DOCKER_USER:-}" \
  --docker-password="${DOCKER_PASS:-}" \
  --docker-email="${EMAIL:-}"

# Image repo/tag with sane defaults
IMAGE_REPO="${IMAGE_REPO:-rag8s/rag8s-onnx-embedder-reranker-cpu-amd64}"
TAG="${TAG:-gte-modernbert}"

echo "[*] Building image ${IMAGE_REPO}:${TAG}"
docker build -t "${IMAGE_REPO}:${TAG}" -f ./infra/onnx/Dockerfile ./infra/onnx

echo "[*] Pushing image ${IMAGE_REPO}:${TAG}"
docker push "${IMAGE_REPO}:${TAG}"

echo "[*] Also tagging as :latest for convenience"
docker tag "${IMAGE_REPO}:${TAG}" "${IMAGE_REPO}:latest"
docker push "${IMAGE_REPO}:latest"

echo "[✔] Done. Images pushed to ${IMAGE_REPO}:${TAG} and :latest"
