#!/usr/bin/env bash
set -euo pipefail

# Always cd to the script's own directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-indexing_pipeline_cpu}"
IMAGE_TAG="${IMAGE_TAG:-amd64-arm64-v7}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ]; then
  echo "Missing DOCKER_USERNAME or DOCKER_PASSWORD"
  exit 1
fi

echo "=> Logging into Docker Hub"
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

BUILDER_NAME="builder_indexing"
CREATED=0

if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx use "$BUILDER_NAME" >/dev/null 2>&1
else
  docker buildx create --name "$BUILDER_NAME" --use >/dev/null 2>&1
  CREATED=1
fi

docker buildx inspect --bootstrap >/dev/null 2>&1

trap 'if [ "$CREATED" -eq 1 ]; then docker buildx rm "$BUILDER_NAME" >/dev/null 2>&1 || true; fi' EXIT

LOCAL_ARCH="$(uname -m)"
case "$LOCAL_ARCH" in
  x86_64) LOCAL_PLATFORM="linux/amd64" ;;
  aarch64|arm64) LOCAL_PLATFORM="linux/arm64" ;;
  *) LOCAL_PLATFORM="linux/amd64" ;;
esac

CONTEXT_DIR="$SCRIPT_DIR"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=> Local test build: ${FULL_IMAGE} for ${LOCAL_PLATFORM}"
docker buildx build \
  --platform "$LOCAL_PLATFORM" \
  --tag "$FULL_IMAGE" \
  --load \
  "$CONTEXT_DIR"

echo "=> Smoke test container"
docker run --rm "$FULL_IMAGE" --version >/dev/null 2>&1 || true

echo "=> Multi-arch build + push"
docker buildx build \
  --platform "$PLATFORMS" \
  --tag "$FULL_IMAGE" \
  --push \
  "$CONTEXT_DIR"

echo "=> Inspect manifest"
docker buildx imagetools inspect "$FULL_IMAGE"

echo "=> Pushed: $FULL_IMAGE"
