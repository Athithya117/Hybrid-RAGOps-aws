#!/usr/bin/env bash
# cd infra/images/embedder-cpu && bash test_and_push.sh
# stable version: sha256:fa20af66c26d8e38fb0bfac947cb213f017ef41a008d773ed9ffa5821ac2b767
set -euo pipefail

# defaults
IMAGE_NAME="${IMAGE_NAME:-embedder-cpu-inference:v1}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/workspace/models}"
CONTAINER_NAME="${CONTAINER_NAME:-embedder-test-run}"
APP_PORT="${APP_PORT:-8000}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
METRICS_PATH="${METRICS_PATH:-/metrics}"
SMOKE_POSTS="${SMOKE_POSTS:-5}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
NO_CACHE=0
PUSH_REGISTRY=""
PLATFORMS="linux/amd64,linux/arm64"   # multi-arch build

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache) NO_CACHE=1; shift;;
    --push) PUSH_REGISTRY="$2"; shift 2;;
    --model) MODEL_HOST_PATH="$2"; shift 2;;
    --image) IMAGE_NAME="$2"; shift 2;;
    --port) APP_PORT="$2"; shift 2;;
    --help|-h) 
      echo "Usage: $0 [--no-cache] [--push registry/repo:tag] [--model /host/models] [--image name:tag] [--port 8000]"
      exit 0;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

echo "IMAGE=$IMAGE_NAME MODEL=$MODEL_HOST_PATH PORT=$APP_PORT PUSH=${PUSH_REGISTRY:-<auto>} PLATFORMS=$PLATFORMS"

# sanity checks
command -v docker >/dev/null 2>&1 || { echo "docker CLI not found"; exit 1; }
[ -d "$MODEL_HOST_PATH" ] || { echo "Model path not found: $MODEL_HOST_PATH"; exit 1; }

# ensure buildx builder exists
if ! docker buildx inspect multiarch-builder >/dev/null 2>&1; then
  docker buildx create --name multiarch-builder --use
  docker buildx inspect --bootstrap
else
  docker buildx use multiarch-builder
fi

# build multi-arch image
BUILD_CMD=(docker buildx build --platform "$PLATFORMS" -t "$IMAGE_NAME" . --load)
[ "$NO_CACHE" -eq 1 ] && BUILD_CMD+=(--no-cache)
echo "Building image for $PLATFORMS..."
"${BUILD_CMD[@]}"

# remove any previous container
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# run test container detached (uses amd64 locally unless you're on ARM machine)
docker run --rm -d --name "$CONTAINER_NAME" -p "${APP_PORT}:${APP_PORT}" -v "${MODEL_HOST_PATH}:/workspace/models:ro" "$IMAGE_NAME" >/dev/null

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# wait for health
echo "Waiting for ${HEALTH_PATH} ..."
start=$(date +%s)
while ! curl -fsS "http://127.0.0.1:${APP_PORT}${HEALTH_PATH}" >/dev/null 2>&1; do
  sleep 1
  now=$(date +%s)
  if [ $((now - start)) -ge "$WAIT_TIMEOUT" ]; then
    echo "ERROR: Health timeout. Container logs:"
    docker logs "$CONTAINER_NAME" --tail 200 || true
    exit 1
  fi
done
echo "Health OK."

# baseline metric
MET_URL="http://127.0.0.1:${APP_PORT}${METRICS_PATH}"
BASE_LINE=$(curl -s "$MET_URL" | grep '^embed_requests_total' | head -n1 || true)
BASE_VAL=$(awk '{print $2}' <<<"$BASE_LINE" 2>/dev/null || echo "")

# send smoke requests
echo "Sending $SMOKE_POSTS requests..."
for i in $(seq 1 $SMOKE_POSTS); do
  curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" -H "Content-Type: application/json" -d "{\"text\":\"smoke-$i\"}" >/dev/null &
done
wait
sleep 1

AFTER_LINE=$(curl -s "$MET_URL" | grep '^embed_requests_total' | head -n1 || true)
AFTER_VAL=$(awk '{print $2}' <<<"$AFTER_LINE" 2>/dev/null || echo "")

echo "Metrics before: ${BASE_VAL:-<none>}"
echo "Metrics after: ${AFTER_VAL:-<none>}"

if [ -z "$AFTER_VAL" ]; then
  echo "ERROR: embed_requests_total not found in metrics; dumping metrics:"
  curl -s "$MET_URL"
  exit 2
fi

if [ -z "$BASE_VAL" ]; then
  delta="$AFTER_VAL"
else
  delta=$(awk -v a="$BASE_VAL" -v b="$AFTER_VAL" 'BEGIN{printf "%.0f", b - a}')
fi

echo "Requests counted (delta): $delta"

echo "last 8 lines of embedding response: "
curl -s -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world","max_length":500}' | jq . | tail -n 8

# push if target decided
if [ -n "${PUSH_REGISTRY:-}" ]; then
  echo "Pushing multi-arch image to $PUSH_REGISTRY"

  if [ -z "${DOCKER_USERNAME:-}" ] || [ -z "${DOCKER_PASSWORD:-}" ]; then
    echo "ERROR: DOCKER_USERNAME and DOCKER_PASSWORD must be exported to push"
    exit 3
  fi

  echo "Logging in to Docker Hub as $DOCKER_USERNAME"
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

  echo "Building and pushing multi-arch image..."
  docker buildx build \
    --platform "$PLATFORMS" \
    -t "$PUSH_REGISTRY" \
    . \
    --push

  echo "Push complete: $PUSH_REGISTRY"
fi

echo "All done."
