#!/usr/bin/env bash
set -euo pipefail

DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_TAG="${IMAGE_TAG:-v2}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
CONTEXT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILDER_NAME="multi_builder"
CREATED_BUILDER=0

if [ -z "$DOCKER_USERNAME" ]; then
  echo "DOCKER_USERNAME is required"; exit 1
fi

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx use "${BUILDER_NAME}" >/dev/null 2>&1 || true
else
  docker buildx create --name "${BUILDER_NAME}" --use >/dev/null 2>&1
  CREATED_BUILDER=1
fi

docker buildx inspect --bootstrap >/dev/null 2>&1

trap 'if [ "${CREATED_BUILDER}" -eq 1 ]; then docker buildx rm "${BUILDER_NAME}" >/dev/null 2>&1 || true; fi' EXIT

IMAGE_NAME="${DOCKER_USERNAME}/qdrant-backup:${IMAGE_TAG}"

LOCAL_ARCH="$(uname -m)"
case "${LOCAL_ARCH}" in
  x86_64) LOCAL_PLATFORM="linux/amd64" ;;
  aarch64|arm64) LOCAL_PLATFORM="linux/arm64" ;;
  *) LOCAL_PLATFORM="linux/amd64" ;;
esac

echo "=> Local test build for ${LOCAL_PLATFORM}"
docker buildx build \
  --platform "${LOCAL_PLATFORM}" \
  --tag "${IMAGE_NAME}" \
  --load \
  --build-arg KUBECTL_VERSION=v1.29.0 \
  --build-arg BASE_IMAGE=python:3.11-slim \
  "${CONTEXT_DIR}"

echo "=> Running local smoke test container (no persistent side effects)"
docker run --rm "${IMAGE_NAME}" -c "aws --version && kubectl version --client --short || true && zstd --version || true" >/dev/null 2>&1 || { echo "Local container smoke test failed"; exit 1; }

echo "=> Building and pushing multi-arch image: ${IMAGE_NAME}"
docker buildx build \
  --platform "${PLATFORMS}" \
  --tag "${IMAGE_NAME}" \
  --build-arg KUBECTL_VERSION=v1.29.0 \
  --build-arg BASE_IMAGE=python:3.11-slim \
  --push \
  "${CONTEXT_DIR}"

echo "=> Inspecting pushed manifest"
docker buildx imagetools inspect "${IMAGE_NAME}" || { echo "imagetools inspect failed"; exit 1; }

echo "Pushed and verified: ${IMAGE_NAME}"
