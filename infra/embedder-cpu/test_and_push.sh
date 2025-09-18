#!/usr/bin/env bash
set -euo pipefail
DOCKER_USERNAME="${DOCKER_USERNAME:-athithya324}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-$DOCKER_USERNAME/embedder-cpu-inference:linux-amd64-arm64}"
PUSH_REGISTRY="${PUSH_REGISTRY:-$IMAGE_NAME}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/workspace/models}"
CONTAINER_NAME="${CONTAINER_NAME:-embedder-test-run}"
APP_PORT="${APP_PORT:-8000}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
SMOKE_POSTS="${SMOKE_POSTS:-5}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
NO_CACHE="${NO_CACHE:-0}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
EXPECTED_MODEL_RELPATH="onnx/gte-modernbert-base-onnx-int8/model.onnx"
EXPECTED_MODEL_HOSTPATH="${MODEL_HOST_PATH}/${EXPECTED_MODEL_RELPATH}"
echo "Using model host path: ${MODEL_HOST_PATH}"
echo "Expecting model at: ${EXPECTED_MODEL_HOSTPATH}"
if [ ! -f "${EXPECTED_MODEL_HOSTPATH}" ]; then
  echo "WARNING: model.onnx not found at expected path: ${EXPECTED_MODEL_HOSTPATH}"
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI not found in PATH"; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot connect to Docker daemon"; exit 1
fi
if [ -n "${DOCKER_PASSWORD:-}" ]; then
  echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
  LOGIN_OK=$?
  if [ $LOGIN_OK -ne 0 ]; then
    echo "Docker login failed"; exit 1
  fi
fi
docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || true
if ! docker buildx version >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y docker-buildx-plugin || apt-get install -y docker-buildx || true
  fi
  if ! docker buildx version >/dev/null 2>&1; then
    mkdir -p ~/.docker/cli-plugins
    TAG=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep -E '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || true)
    ARCH=$(uname -m)
    case "${ARCH}" in
      x86_64) ASSET="linux-amd64" ;;
      aarch64|arm64) ASSET="linux-arm64" ;;
      *) echo "Unsupported arch: ${ARCH}"; exit 1 ;;
    esac
    if [ -n "${TAG}" ]; then
      URL="https://github.com/docker/buildx/releases/download/${TAG}/buildx-${TAG}.${ASSET}"
      curl -fsSL "${URL}" -o ~/.docker/cli-plugins/docker-buildx
      chmod +x ~/.docker/cli-plugins/docker-buildx
    else
      echo "Failed to discover buildx release tag"; exit 1
    fi
  fi
fi
set +e
docker buildx inspect crossbuilder >/dev/null 2>&1
BX_INSPECT_RC=$?
set -e
if [ $BX_INSPECT_RC -ne 0 ]; then
  set +e
  docker buildx create --name crossbuilder --driver docker-container --use >/dev/null 2>&1
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then
    set +e
    docker buildx create crossbuilder --driver docker-container --use >/dev/null 2>&1
    RC2=$?
    set -e
    if [ $RC2 -ne 0 ]; then
      docker buildx create --use >/dev/null
    fi
  fi
fi
docker buildx inspect --bootstrap >/dev/null
BUILDX_NO_CACHE=""
if [ "${NO_CACHE}" = "1" ]; then BUILDX_NO_CACHE="--no-cache"; fi
docker buildx build --builder crossbuilder --platform "${PLATFORMS}" -t "${PUSH_REGISTRY}" ${BUILDX_NO_CACHE} --push .
BUILD_RC=$?
if [ $BUILD_RC -ne 0 ]; then
  echo "buildx push failed"; exit 1
fi
set +e
docker pull "${PUSH_REGISTRY}"
PULL_RC=$?
set -e
if [ $PULL_RC -ne 0 ]; then
  echo "Warning: docker pull ${PUSH_REGISTRY} failed; continuing but registry image may not be available"
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
CID=$(docker run --rm -d --name "${CONTAINER_NAME}" -p "${APP_PORT}:${APP_PORT}" --shm-size=1.8g -v "${MODEL_HOST_PATH}:/workspace/models:ro" -e PORT="${APP_PORT}" "${PUSH_REGISTRY}")
if [ -z "${CID}" ]; then
  echo "ERROR: docker run failed"; docker images | head -n 50; exit 1
fi
cleanup(){ docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
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
for i in $(seq 1 "${SMOKE_POSTS}"); do
  curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" -H "Content-Type: application/json" -d "{\"text\":\"smoke-${i}\"}" >/dev/null &
done
wait
curl -s -X POST "http://127.0.0.1:${APP_PORT}/embed" -H "Content-Type: application/json" -d '{"text":"hello world","max_length":50}' | head -n 60
if [ -n "${DOCKER_PASSWORD:-}" ]; then docker logout || true; fi

# curl -s -X POST "http://127.0.0.1:8000/embed" -H "Content-Type: application/json" -d '{"text":"hello world","max_length":8000}' | head -n 60
