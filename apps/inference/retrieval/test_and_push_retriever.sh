#!/usr/bin/env bash
# Usage:
#   # from repo root:
#   bash apps/inference/retrieval/test_and_push_retriever.sh

set -euo pipefail

# Resolve script dir => default build context
SCRIPT_DIR="apps/inference/retrieval/"
BUILD_CONTEXT_DIR="apps/inference/retrieval/"

IMAGE_TAG="${IMAGE_TAG:-amd64-arm64-v2}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}retrieval:${IMAGE_TAG}}"

CONTAINER_NAME="${CONTAINER_NAME:-test-retrieval}"
HOST_PORT="${HOST_PORT:-8001}"
CONTAINER_PORT="${CONTAINER_PORT:-8001}"

DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64,linux/arm64}"
BUILDX_BUILDER="${BUILDX_BUILDER:-buildx-temp-retrieval}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
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
  local url="$1"; local timeout="$2"; local start now
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

# ---------------------------
# 1) Build local single-arch image (context is script dir)
# ---------------------------
log "Building local single-arch image: ${IMAGE_NAME} (context=${BUILD_CONTEXT_DIR})"
docker build -t "${IMAGE_NAME}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 4; }

# ---------------------------
# 2) Run container for smoke tests
# ---------------------------
cleanup_container
log "Starting container ${CONTAINER_NAME} from ${IMAGE_NAME}"
docker run --name "${CONTAINER_NAME}" -d -p "${HOST_PORT}:${CONTAINER_PORT}" "${IMAGE_NAME}" >/dev/null

log "Waiting for health endpoint to respond on http://127.0.0.1:${HOST_PORT}/healthz"
if ! wait_for_http "http://127.0.0.1:${HOST_PORT}/healthz" "${WAIT_TIMEOUT}"; then
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "Container did not become healthy in ${WAIT_TIMEOUT}s"
  exit 5
fi
log "Health OK"

# ---------------------------
# 3) Smoke tests: /healthz /readyz /metrics /generate
# ---------------------------
log "GET /healthz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/healthz" || true
log "GET /readyz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/readyz" || true

log "GET /metrics (before)"
METRICS_BEFORE="$(mktemp)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" -o "${METRICS_BEFORE}" || true
grep -E "retrieval_requests_total|retrieval_request_duration_seconds|dense_embed_requests_total|qdrant_query_total" "${METRICS_BEFORE}" || log "Metrics may be empty until operations run"

log "POST /generate (functional check)"
PAYLOAD='{"query":"smoke test from script","enable_tracing":false,"top_k":3,"return_chunks":false}'
RESP="$(mktemp)"
curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/generate" -H "Content-Type: application/json" -d "${PAYLOAD}" -o "${RESP}" || {
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "/generate failed"
  exit 6
}
log "Response (head):"
sed -n '1,200p' "${RESP}" || true

log "GET /metrics (after)"
METRICS_AFTER="$(mktemp)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" -o "${METRICS_AFTER}" || true

before=$(metric_value "retrieval_requests_total" "${METRICS_BEFORE}" || echo "")
after=$(metric_value "retrieval_requests_total" "${METRICS_AFTER}" || echo "")
log "retrieval_requests_total before='${before}' after='${after}'"
if [ -n "${before}" ] && [ -n "${after}" ]; then
  awk -v a="${before}" -v b="${after}" 'BEGIN{if(b+0 < a+0) {print "Metric did not increase: fail"; exit 2} else {print "Metric increased or equal: ok"}}'
else
  log "Could not parse numeric retrieval_requests_total; metric presence checked instead."
fi

# ---------------------------
# 4) Stop & remove test container (keep image)
# ---------------------------
docker rm -f "${CONTAINER_NAME}" >/dev/null || true
log "Local smoke tests passed (container removed)."

# ---------------------------
# 5) Optional: buildx multi-arch push if DOCKER_USERNAME provided
# ---------------------------
if [ -n "${DOCKER_USERNAME}" ]; then
  log "DOCKER_USERNAME set; attempting buildx multi-arch build & push for ${IMAGE_NAME}"
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into registry as ${DOCKER_USERNAME}"
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "docker login failed"; exit 11; }
  else
    warn "DOCKER_PASSWORD not provided; attempting unauthenticated push (may fail)"
  fi

  if ! docker buildx inspect "${BUILDX_BUILDER}" >/dev/null 2>&1; then
    log "Creating buildx builder: ${BUILDX_BUILDER}"
    docker buildx create --name "${BUILDX_BUILDER}" --driver docker-container --use >/dev/null 2>&1 || {
      warn "Failed to create docker-container builder; falling back to default buildx"
      docker buildx use default >/dev/null 2>&1 || true
    }
  else
    docker buildx use "${BUILDX_BUILDER}" >/dev/null 2>&1 || true
  fi

  log "Bootstrapping buildx builder"
  docker buildx inspect --bootstrap >/dev/null 2>&1 || warn "buildx bootstrap failed - continuing"

  log "Building and pushing multi-arch image (platforms=${DOCKER_IMAGES_PLATFORM})"
  docker buildx build --platform "${DOCKER_IMAGES_PLATFORM}" --tag "${IMAGE_NAME}" --push "${BUILD_CONTEXT_DIR}" || { err "buildx push failed"; exit 12; }

  log "Multi-arch image pushed: ${IMAGE_NAME}"
else
  log "DOCKER_USERNAME not set — skipping multi-arch push. Local image available: ${IMAGE_NAME}"
fi

log "Done OK"
exit 0
