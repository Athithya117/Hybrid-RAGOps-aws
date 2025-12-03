#!/usr/bin/env bash
# apps/inference/frontend/test_and_push_frontend.sh
set -euo pipefail

MODE="${1:-cpu}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_TAG="${IMAGE_TAG:-amd64-arm64-v1}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}frontend:${IMAGE_TAG}}"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-.}"
CONTAINER_NAME="${CONTAINER_NAME:-test-frontend}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64,linux/arm64}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
BUILDX_BUILDER="${BUILDX_BUILDER:-buildx-temp-frontend}"

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

log "Building local image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 4; }

cleanup_container
log "Running container ${CONTAINER_NAME} (SKIP_PRE_CHECKS=true)"
docker run --name "${CONTAINER_NAME}" -d -p "${HOST_PORT}:${CONTAINER_PORT}" -e SKIP_PRE_CHECKS=true "${IMAGE_NAME}" >/dev/null

log "Waiting for health"
start=$(date +%s)
while true; do
  if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
    log "Health OK"
    break
  fi
  if [ $(( $(date +%s) - start )) -ge "${WAIT_TIMEOUT}" ]; then
    docker logs --tail 200 "${CONTAINER_NAME}" || true
    err "Container did not become healthy"
    exit 5
  fi
  sleep 1
done

log "GET /readyz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/readyz" || true
log "GET /metrics (head)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" | sed -n '1,120p' || true

log "POST /run (form-encoded)"
PAYLOAD="query=smoke test from push script&top_k=3"
# accept 2xx or 502 (backend unreachable) but fail on other unexpected codes
HTTP_CODE=$(curl -s -o /tmp/frontend_generate_resp.json -w "%{http_code}" -X POST "http://127.0.0.1:${HOST_PORT}/run" \
  -H "Content-Type: application/x-www-form-urlencoded" --data "${PAYLOAD}" || true)
log "POST /run returned ${HTTP_CODE}; response head:"
sed -n '1,200p' /tmp/frontend_generate_resp.json || true
if [ "${HTTP_CODE}" != "200" ] && [ "${HTTP_CODE}" != "502" ]; then
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "/run unexpected status ${HTTP_CODE}"
  exit 6
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null || true
log "Local tests passed."

if [ -n "${DOCKER_USERNAME}" ]; then
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into registry ${DOCKER_USERNAME}"
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "docker login failed"; exit 11; }
  else
    warn "DOCKER_PASSWORD not provided; attempting unauthenticated push"
  fi

  if ! docker buildx inspect "${BUILDX_BUILDER}" >/dev/null 2>&1; then
    log "Creating buildx builder ${BUILDX_BUILDER}"
    docker buildx create --name "${BUILDX_BUILDER}" --driver docker-container --use >/dev/null 2>&1 || {
      warn "Failed to create docker-container builder; trying default buildx"
      docker buildx use default >/dev/null 2>&1 || true
    }
  fi
  docker buildx inspect --bootstrap >/dev/null 2>&1 || warn "buildx bootstrap failed - continuing"

  log "Building & pushing multi-arch image for ${DOCKER_IMAGES_PLATFORM}"
  docker buildx build --platform "${DOCKER_IMAGES_PLATFORM}" --tag "${IMAGE_NAME}" --push "${BUILD_CONTEXT_DIR}" || { err "buildx push failed"; exit 12; }
  log "Multi-arch image pushed: ${IMAGE_NAME}"
else
  log "DOCKER_USERNAME not set — skipping multi-arch push. Local image: ${IMAGE_NAME}"
fi

log "Done OK"
exit 0
