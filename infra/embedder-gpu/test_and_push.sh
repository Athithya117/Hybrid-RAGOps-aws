#!/usr/bin/env bash
set -euo pipefail
DOCKER_USERNAME="${DOCKER_USERNAME:-your-docker-hub-username}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}" # optional
IMAGE_NAME="${IMAGE_NAME:-$DOCKER_USERNAME/embedder-gpu:latest}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/workspace/models}"
CONTAINER_NAME="${CONTAINER_NAME:-embedder-gpu-test-run}"
APP_PORT="${APP_PORT:-8000}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
CUDA_TAG="${CUDA_TAG:-12.2.0}"

if [ ! -f "${MODEL_HOST_PATH}/onnx/gte-modernbert-base-onnx-int8/model.onnx" ]; then
  echo "WARNING: model.onnx not found at expected path: ${MODEL_HOST_PATH}/onnx/gte-modernbert-base-onnx-int8/model.onnx"
fi

if [ -n "${DOCKER_PASSWORD:-}" ]; then
  echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
fi

echo "Building image: ${IMAGE_NAME} (CUDA_TAG=${CUDA_TAG})"
docker build --build-arg CUDA_TAG="${CUDA_TAG}" -t "${IMAGE_NAME}" .

echo "Pushing image: ${IMAGE_NAME}"
docker push "${IMAGE_NAME}"

# smoke-run locally (requires host with NVIDIA driver + nvidia-container-toolkit)
trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true' EXIT

CID=$(docker run --rm -d --name "${CONTAINER_NAME}" --gpus all -p "${APP_PORT}:${APP_PORT}" --shm-size=1.8g -v "${MODEL_HOST_PATH}:/workspace/models:ro" -e PORT="${APP_PORT}" "${IMAGE_NAME}")
if [ -z "${CID}" ]; then
  echo "ERROR: docker run failed"; docker images | head -n 50; exit 1
fi

start_ts=$(date +%s)
while true; do
  status=$(docker inspect -f '{{.State.Status}}' "${CID}" 2>/dev/null || echo "missing")
  if [ "${status}" = "exited" ] || [ "${status}" = "dead" ] || [ "${status}" = "missing" ]; then
    docker logs "${CID}" --tail 200 || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${APP_PORT}${HEALTH_PATH}" >/dev/null 2>&1; then
    break
  fi
  now_ts=$(date +%s)
  if [ $((now_ts - start_ts)) -ge "${WAIT_TIMEOUT}" ]; then
    docker logs "${CID}" --tail 200 || true
    exit 1
  fi
  sleep 1
done

# smoke embed
curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" -H "Content-Type: application/json" -d '{"text":"hello gpu","max_length":50}' | head -n 60

if [ -n "${DOCKER_PASSWORD:-}" ]; then docker logout || true; fi
