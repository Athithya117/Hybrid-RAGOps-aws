#!/usr/bin/env bash
# Usage:
#   bash apps/inference/retrieval/test_and_push_retriever.sh

set -euo pipefail

# ---------------------------
# Paths / Context
# ---------------------------
SCRIPT_DIR="apps/inference/retrieval/"
BUILD_CONTEXT_DIR="apps/inference/retrieval/"

# ---------------------------
# Image / Registry
# ---------------------------
IMAGE_TAG="${IMAGE_TAG:-v4}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}retrieval:${IMAGE_TAG}}"

# ---------------------------
# Runtime
# ---------------------------
CONTAINER_NAME="${CONTAINER_NAME:-test-retrieval}"
HOST_PORT="${HOST_PORT:-8001}"
CONTAINER_PORT="${CONTAINER_PORT:-8001}"

# ---------------------------
# Buildx
# ---------------------------
DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64,linux/arm64}"
BUILDX_BUILDER="${BUILDX_BUILDER:-buildx-temp-retrieval}"

WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
SLEEP_BETWEEN_TRIES=1

log()  { printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container() {
  set +e
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  set -e
}

cleanup_builder() {
  set +e
  docker buildx rm "${BUILDX_BUILDER}" >/dev/null 2>&1 || true
  set -e
}

trap 'cleanup_container; cleanup_builder' EXIT

wait_for_http() {
  local url="$1" timeout="$2" start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then return 1; fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

metric_value() {
  local metric="$1" file="$2"
  grep -E "^${metric}(\\{|\\s)" "$file" | tail -n1 | awk '{print $NF}' || true
}

has_buildx() {
  docker buildx version >/dev/null 2>&1
}

# ---------------------------
# 1) Local single-arch build
# ---------------------------
log "Building local image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${BUILD_CONTEXT_DIR}"

# ---------------------------
# 2) Run container
# ---------------------------
cleanup_container
log "Starting container ${CONTAINER_NAME}"
docker run -d --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "${IMAGE_NAME}" >/dev/null

log "Waiting for /healthz"
wait_for_http "http://127.0.0.1:${HOST_PORT}/healthz" "${WAIT_TIMEOUT}" || {
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "Health check failed"
  exit 5
}

# ---------------------------
# 3) Smoke tests
# ---------------------------
log "GET /healthz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/healthz"

log "GET /readyz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/readyz"

log "GET /metrics (before)"
METRICS_BEFORE="$(mktemp)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" -o "${METRICS_BEFORE}" || true

log "POST /generate"
PAYLOAD='{"query":"smoke test","top_k":3,"return_chunks":false}'
RESP="$(mktemp)"
curl -fsS -X POST \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  "http://127.0.0.1:${HOST_PORT}/generate" \
  -o "${RESP}"

sed -n '1,200p' "${RESP}"

log "GET /metrics (after)"
METRICS_AFTER="$(mktemp)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" -o "${METRICS_AFTER}" || true

before=$(metric_value retrieval_requests_total "${METRICS_BEFORE}")
after=$(metric_value retrieval_requests_total "${METRICS_AFTER}")
log "retrieval_requests_total before='${before}' after='${after}'"

cleanup_container
log "Local smoke tests passed."

# ---------------------------
# 4) Optional multi-arch push
# ---------------------------
if [ -z "${DOCKER_USERNAME}" ]; then
  log "DOCKER_USERNAME not set — skipping push"
  exit 0
fi

if ! has_buildx; then
  warn "docker buildx not available — pushing single-arch image"
  if [ -n "${DOCKER_PASSWORD}" ]; then
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
  fi
  docker push "${IMAGE_NAME}"
  exit 0
fi

log "docker buildx detected"

if [ -n "${DOCKER_PASSWORD}" ]; then
  printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
fi

docker buildx create --name "${BUILDX_BUILDER}" --use >/dev/null 2>&1 || docker buildx use default
docker buildx inspect --bootstrap >/dev/null 2>&1 || true

log "Building & pushing multi-arch image"
docker buildx build \
  --platform "${DOCKER_IMAGES_PLATFORM}" \
  -t "${IMAGE_NAME}" \
  --push \
  "${BUILD_CONTEXT_DIR}"

log "Multi-arch push complete: ${IMAGE_NAME}"
log "Done OK"
exit 0
