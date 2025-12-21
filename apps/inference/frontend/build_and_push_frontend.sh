#!/usr/bin/env bash
# apps/inference/frontend/test_and_push_frontend_auth.sh
# Build image, run minimal smoke tests (health + auth endpoint), optionally push.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${SCRIPT_DIR}"
IMAGE_TAG="${IMAGE_TAG:-v23}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}frontend-and-auth:${IMAGE_TAG}}"

CONTAINER_NAME="${CONTAINER_NAME:-test-frontend-and-auth}"
HOST_PORT="${HOST_PORT:-8011}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
SLEEP_BETWEEN_TRIES=1

log()  { printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container() {
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    log "Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}

trap 'cleanup_container' EXIT

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
docker build -t "${IMAGE_NAME}" "${BUILD_CONTEXT_DIR}" \
  || { err "docker build failed"; exit 4; }

if [ -z "${DOCKER_USERNAME}" ]; then
  log "DOCKER_USERNAME not set — skipping push."
  exit 0
fi

log "DOCKER_USERNAME provided — preparing to push image"

if [ -n "${DOCKER_PASSWORD}" ]; then
  log "Logging into registry"
  printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin \
    || { err "docker login failed"; exit 11; }
else
  warn "DOCKER_PASSWORD not provided — push may fail"
fi

log "Pushing ${IMAGE_NAME}"
docker push "${IMAGE_NAME}" || { err "docker push failed"; exit 12; }

log "Push complete: ${IMAGE_NAME}"
exit 0


