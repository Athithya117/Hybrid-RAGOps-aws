#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

: "${DOCKER_USERNAME:?missing DOCKER_USERNAME}"
: "${DOCKER_PASSWORD:?missing DOCKER_PASSWORD}"

IMAGE_NAME="${IMAGE_NAME:-indexing_pipeline_cpu}"
IMAGE_TAG="${IMAGE_TAG:-base}"
IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Logging into Docker Hub"
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

echo "Building image"
docker build -t "$IMAGE" .

echo "Pushing image"
docker push "$IMAGE"

echo "Done: $IMAGE"
