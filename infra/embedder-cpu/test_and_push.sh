#!/usr/bin/env bash
set -euo pipefail

# ---- Configuration ----
DOCKER_USERNAME="${DOCKER_USERNAME:-athithya324}"
IMAGE_NAME="${IMAGE_NAME:-$DOCKER_USERNAME/embedder-cpu-inference:v2}"
PUSH_REGISTRY="${PUSH_REGISTRY:-$IMAGE_NAME}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/workspace/models}"   # host path to mount into the container
CONTAINER_NAME="${CONTAINER_NAME:-embedder-test-run}"
APP_PORT="${APP_PORT:-8000}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
SMOKE_POSTS="${SMOKE_POSTS:-5}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"

# Controls
NO_CACHE="${NO_CACHE:-0}"   # set to 1 to rebuild without cache

# Optional runtime envs for the container (can export before running script to override)
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
NUM_REPLICAS="${NUM_REPLICAS:-1}"
REPLICA_CPUS="${REPLICA_CPUS:-1.0}"
CUSTOM_RESOURCE="${CUSTOM_RESOURCE:-app1}"
CUSTOM_RESOURCE_UNITS="${CUSTOM_RESOURCE_UNITS:-1}"
MAX_BATCH="${MAX_BATCH:-64}"
BATCH_WAIT_S="${BATCH_WAIT_S:-0.03}"
ORT_INTRA_THREADS="${ORT_INTRA_THREADS:-1}"
ORT_INTER_THREADS="${ORT_INTER_THREADS:-1}"

# Path the python code expects: MODEL_DIR=/workspace/models/onnx/gte-modernbert-base-onnx-int8
EXPECTED_MODEL_RELPATH="onnx/gte-modernbert-base-onnx-int8/model.onnx"
EXPECTED_MODEL_HOSTPATH="${MODEL_HOST_PATH}/${EXPECTED_MODEL_RELPATH}"

echo "Using model host path: ${MODEL_HOST_PATH}"
echo "Expecting model at: ${EXPECTED_MODEL_HOSTPATH}"

if [ ! -f "${EXPECTED_MODEL_HOSTPATH}" ]; then
  echo "WARNING: model.onnx not found at expected path:"
  echo "  ${EXPECTED_MODEL_HOSTPATH}"
  echo "If you want to run the smoke test, mount the host model directory so the container sees the ONNX model."
  echo "Example:"
  echo "  MODEL_HOST_PATH=/full/path/to/models ./test_and_push.sh"
  echo "Continuing: the build will still run, but the container may exit if the app cannot find the model."
fi

# ---- Phase 1: build image (optionally --no-cache) ----
echo "Building local image: $IMAGE_NAME"
BUILD_CMD=(docker build -t "$IMAGE_NAME" .)
if [ "${NO_CACHE:-0}" -eq 1 ]; then
  BUILD_CMD+=(--no-cache)
fi
"${BUILD_CMD[@]}"

# remove any previous container with same name
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# ---- Phase 2: run container (detached) ----
echo "Starting container ($CONTAINER_NAME) for smoke tests..."
# increase shm for onnxruntime performance inside container
CID=$(docker run --rm -d --name "$CONTAINER_NAME" -p "${APP_PORT}:${APP_PORT}" \
  --shm-size=1.8g \
  -v "${MODEL_HOST_PATH}:/workspace/models:ro" \
  -e PORT="${APP_PORT}" \
  -e RAY_ADDRESS="${RAY_ADDRESS}" \
  -e NUM_REPLICAS="${NUM_REPLICAS}" \
  -e REPLICA_CPUS="${REPLICA_CPUS}" \
  -e CUSTOM_RESOURCE="${CUSTOM_RESOURCE}" \
  -e CUSTOM_RESOURCE_UNITS="${CUSTOM_RESOURCE_UNITS}" \
  -e MAX_BATCH="${MAX_BATCH}" \
  -e BATCH_WAIT_S="${BATCH_WAIT_S}" \
  -e ORT_INTRA_THREADS="${ORT_INTRA_THREADS}" \
  -e ORT_INTER_THREADS="${ORT_INTER_THREADS}" \
  "$IMAGE_NAME")

if [ -z "${CID}" ]; then
  echo "ERROR: docker run failed (no container id). Docker images for debugging:"
  docker images | head -n 50
  exit 1
fi

cleanup() {
  echo "Cleaning up container..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- Phase 3: wait for health, bail if container dies ----
echo "Waiting for container health at ${HEALTH_PATH} (port ${APP_PORT})..."
start_ts=$(date +%s)
while true; do
  # check container status
  status=$(docker inspect -f '{{.State.Status}}' "$CID" 2>/dev/null || echo "missing")
  if [ "$status" = "exited" ] || [ "$status" = "dead" ] || [ "$status" = "created" ] || [ "$status" = "missing" ]; then
    echo "Container is not running (status=${status}). Showing last 200 lines of logs:"
    docker logs "$CID" --tail 200 || true
    echo "Aborting smoke tests."
    exit 1
  fi

  # try health endpoint
  if curl -fsS "http://127.0.0.1:${APP_PORT}${HEALTH_PATH}" >/dev/null 2>&1; then
    echo "Health OK."
    break
  fi

  now_ts=$(date +%s)
  if [ $((now_ts - start_ts)) -ge "$WAIT_TIMEOUT" ]; then
    echo "ERROR: Health timeout after ${WAIT_TIMEOUT} seconds. Container status:"
    docker inspect "$CID" --format '{{json .State}}' || true
    echo
    echo "Container logs (last 200 lines):"
    docker logs "$CID" --tail 200 || true
    exit 1
  fi

  sleep 1
done

# ---- Phase 4: smoke requests ----
echo "Sending $SMOKE_POSTS smoke requests..."
for i in $(seq 1 $SMOKE_POSTS); do
  curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"smoke-$i\"}" >/dev/null &
done
wait

echo "Sample embedding response:"
curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world","max_length":500}' | head -n 60

# ---- Phase 5: push image (optional) ----
echo "Logging in and pushing image: $PUSH_REGISTRY"
if [ -z "${DOCKER_PASSWORD:-}" ]; then
  echo "NOTE: DOCKER_PASSWORD not set. Skipping push step."
else
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
  docker tag "$IMAGE_NAME" "$PUSH_REGISTRY"
  docker push "$PUSH_REGISTRY"
  echo "Push complete: $PUSH_REGISTRY"
fi

echo "All done."


