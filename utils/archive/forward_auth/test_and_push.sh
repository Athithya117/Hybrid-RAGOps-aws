#!/usr/bin/env bash
# Usage:
#   bash apps/inference/forward_auth/test_and_push.sh

set -euo pipefail

SCRIPT_DIR="apps/inference/forward_auth/"
BUILD_CONTEXT_DIR="apps/inference/forward_auth/"

IMAGE_TAG="${IMAGE_TAG:-v3}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}forward-auth:${IMAGE_TAG}}"

CONTAINER_NAME="${CONTAINER_NAME:-test-forward-auth}"
HOST_PORT="${HOST_PORT:-8010}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64}"
BUILDX_BUILDER="${BUILDX_BUILDER:-buildx-temp-forward-auth}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
SLEEP_BETWEEN_TRIES=1

log() { printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container() {
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    log "Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}

cleanup_builder() {
  set +e
  if docker buildx inspect "${BUILDX_BUILDER}" >/dev/null 2>&1; then
    docker buildx rm "${BUILDX_BUILDER}" >/dev/null 2>&1 || true
  fi
  set -e
}

trap 'cleanup_container; cleanup_builder' EXIT

wait_for_http() {
  local url="$1"; local timeout="$2"
  local start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then return 1; fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

log "Building local image ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 4; }

cleanup_container
log "Starting container ${CONTAINER_NAME}"
docker run --name "${CONTAINER_NAME}" \
  -d -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e AUTH_ALLOW_MISSING_SECRETS=1 \
  "${IMAGE_NAME}" >/dev/null

log "Waiting for /health on http://127.0.0.1:${HOST_PORT}/health"
if ! wait_for_http "http://127.0.0.1:${HOST_PORT}/health" "${WAIT_TIMEOUT}"; then
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "Container did not become healthy"
  exit 5
fi

log "GET /health"
curl -fsS "http://127.0.0.1:${HOST_PORT}/health" || {
  err "Health check failed"
  exit 6
}

docker rm -f "${CONTAINER_NAME}" >/dev/null || true
log "Local smoke test passed."

if [ -n "${DOCKER_USERNAME}" ]; then
  log "DOCKER_USERNAME provided — pushing image"

  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into registry"
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin \
      || { err "docker login failed"; exit 11; }
  else
    warn "DOCKER_PASSWORD not provided — push may fail"
  fi

  if ! docker buildx inspect "${BUILDX_BUILDER}" >/dev/null 2>&1; then
    log "Creating buildx builder ${BUILDX_BUILDER}"
    docker buildx create --name "${BUILDX_BUILDER}" --driver docker-container --use >/dev/null 2>&1 \
      || { warn "Failed to create docker-container builder; fallback to default"; docker buildx use default >/dev/null 2>&1 || true; }
  else
    docker buildx use "${BUILDX_BUILDER}" >/dev/null 2>&1 || true
  fi

  docker buildx inspect --bootstrap >/dev/null 2>&1 || warn "buildx bootstrap failed"

  log "Building and pushing multi-arch image ${IMAGE_NAME}"
  docker buildx build \
    --platform "${DOCKER_IMAGES_PLATFORM}" \
    --tag "${IMAGE_NAME}" \
    --push \
    "${BUILD_CONTEXT_DIR}" \
    || { err "buildx push failed"; exit 12; }

  log "Multi-arch push done: ${IMAGE_NAME}"
else
  log "DOCKER_USERNAME not set — skipping push."
fi

log "Done OK"
exit 0
