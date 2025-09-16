#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-gpu}"
MODE_LOWER="$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MODE_LOWER" != "gpu" ] && [ "$MODE_LOWER" != "cpu" ]; then
  echo "Usage: $0 <cpu|gpu>"
  exit 2
fi

DOCKER_USERNAME="${DOCKER_USERNAME:-athithya324}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-$DOCKER_USERNAME/indexing-embedder-gpu:v1}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/workspace/models}"
CONTAINER_NAME_BASE="${CONTAINER_NAME:-embedder-gpu-test-run}"
CONTAINER_NAME="${CONTAINER_NAME_BASE}-${MODE_LOWER}"
APP_PORT="${APP_PORT:-8001}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
CUDA_TAG="${CUDA_TAG:-12.2.0}"
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-10}"

if [ ! -f "${MODEL_HOST_PATH}/onnx/gte-modernbert-base-onnx-int8/model.onnx" ]; then
  echo "WARNING: model.onnx not found at expected path: ${MODEL_HOST_PATH}/onnx/gte-modernbert-base-onnx-int8/model.onnx"
fi

if [ -n "${DOCKER_PASSWORD:-}" ]; then
  echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
fi

echo "Building image: ${IMAGE_NAME} (CUDA_TAG=${CUDA_TAG})"
docker build --build-arg CUDA_TAG="${CUDA_TAG}" -t "${IMAGE_NAME}" .

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

RUN_FLAGS=(--name "${CONTAINER_NAME}" -d -p "${APP_PORT}:${APP_PORT}" --shm-size=1.8g -v "${MODEL_HOST_PATH}:/workspace/models:ro" -e "PORT=${APP_PORT}")

if [ "$MODE_LOWER" = "gpu" ]; then
  RUN_FLAGS+=(--gpus all -e "FORCE_CPU=0" -e "REPLICA_GPUS=1.0")
  SAMPLE_TEXT='hello gpu'
else
  RUN_FLAGS+=(-e "FORCE_CPU=1" -e "REPLICA_GPUS=0" -e "NVIDIA_VISIBLE_DEVICES=" -e "CUDA_VISIBLE_DEVICES=")
  SAMPLE_TEXT='hello cpu'
fi

echo "Starting container ${CONTAINER_NAME} for smoke test..."
CID="$(docker run "${RUN_FLAGS[@]}" "${IMAGE_NAME}")" || {
  echo "ERROR: docker run failed to start container"
  docker images | head -n 50
  exit 1
}

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap 'cleanup' EXIT

start_ts=$(date +%s)
echo "Waiting for health at http://127.0.0.1:${APP_PORT}${HEALTH_PATH} (timeout ${WAIT_TIMEOUT}s)..."
while true; do
  status=$(docker inspect -f '{{.State.Status}}' "${CID}" 2>/dev/null || echo "missing")
  if [ "${status}" = "exited" ] || [ "${status}" = "dead" ] || [ "${status}" = "missing" ]; then
    echo "Container is not running (status=${status}). Showing last 400 lines of logs:"
    docker logs --tail 400 "${CID}" || true
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${APP_PORT}${HEALTH_PATH}" >/dev/null 2>&1; then
    echo "Health check OK"
    break
  fi

  now_ts=$(date +%s)
  if [ $((now_ts - start_ts)) -ge "${WAIT_TIMEOUT}" ]; then
    echo "Timed out waiting for health. Showing last 400 lines of logs:"
    docker logs --tail 400 "${CID}" || true
    exit 1
  fi
  sleep 1
done

echo "Running sample embed request..."
if ! curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" -H "Content-Type: application/json" \
    -d "{\"text\":\"${SAMPLE_TEXT}\",\"max_length\":50}" | head -n 60; then
  echo "Sample embed failed; container logs (last 400 lines):"
  docker logs --tail 400 "${CID}" || true
  exit 1
fi

echo
echo "Smoke test passed; pushing image ${IMAGE_NAME}"
docker push "${IMAGE_NAME}"

if [ -n "${DOCKER_PASSWORD:-}" ]; then
  docker logout || true
fi

echo "Done."
