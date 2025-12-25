#!/usr/bin/env bash
# test_and_push_reranker.sh
# Robust buildx multi-arch build + local smoke tests + optional push.
set -euo pipefail

# Configuration (env overrides)
MODE="${1:-cpu}"                               # cpu | gpu
DOCKER_USERNAME="${DOCKER_USERNAME:-}"         # if empty, skip push
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"         # optional; if provided, will login
IMAGE_TAG="${IMAGE_TAG:-amd64-arm64-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}reranker:${IMAGE_TAG}}"
CONTAINER_NAME="${CONTAINER_NAME:-test-reranker-${MODE}}"
HOST_PORT="${HOST_PORT:-9023}"                 # host -> container mapping
CONTAINER_PORT="${CONTAINER_PORT:-8202}"       # must match runtime port
FASTEMBED_GPU_ARG="0"
if [ "${MODE,,}" = "gpu" ]; then FASTEMBED_GPU_ARG="1"; fi

DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64,linux/arm64}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
SLEEP_BETWEEN_TRIES=1

BUILDER_NAME="${BUILDER_NAME:-buildx-multi-builder}"
CREATED_BUILDER=0
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

log() { printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container() {
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}

cleanup_builder() {
  if [ "${CREATED_BUILDER}" -eq 1 ]; then
    log "Removing buildx builder ${BUILDER_NAME}"
    docker buildx rm "${BUILDER_NAME}" >/dev/null 2>&1 || true
  fi
}
trap 'cleanup_container; cleanup_builder' EXIT

pretty_json() {
  if command -v jq >/dev/null 2>&1; then
    jq . 2>/dev/null || cat
  else
    python3 - <<'PY' 2>/dev/null || true
import sys,json
print(json.dumps(json.load(sys.stdin), indent=2))
PY
  fi
}

wait_for_health() {
  local host_port="$1"; local path="$2"; local timeout="$3"; local require_ready="${4:-0}"
  local start now health
  start=$(date +%s)
  while true; do
    health=$(curl -fsS "http://127.0.0.1:${host_port}${path}" 2>/dev/null || true)
    if [ -n "${health}" ]; then
      if printf '%s' "${health}" | grep -q '"status"' && printf '%s' "${health}" | grep -q '"ok"'; then
        if [ "${require_ready}" -eq 1 ]; then
          if printf '%s' "${health}" | grep -q '"ready"\s*:\s*true'; then
            log "Health OK and model_ready=true"
            printf '%s\n' "${health}" | pretty_json
            return 0
          fi
        else
          log "Health OK (status ok)"
          printf '%s\n' "${health}" | pretty_json
          return 0
        fi
      fi
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "${timeout}" ]; then
      err "Timed out waiting for health at http://127.0.0.1:${host_port}${path}"
      printf '%s\n' "${health}" | sed -n '1,200p' || true
      return 1
    fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

# Normalize platforms
PLATFORMS_RAW="$(printf '%s' "${DOCKER_IMAGES_PLATFORM}" | tr -d '[:space:]')"
if [ -z "${PLATFORMS_RAW}" ]; then
  PLATFORMS_RAW="linux/amd64,linux/arm64"
fi
IFS=',' read -r -a _arr <<< "${PLATFORMS_RAW}"
PLATFORMS=""
for p in "${_arr[@]}"; do
  case "${p}" in
    linux/amd64|linux/arm64)
      if printf '%s' "${PLATFORMS}" | grep -q "${p}"; then :; else
        PLATFORMS="${PLATFORMS:+${PLATFORMS},}${p}"
      fi
      ;;
    *)
      warn "Unsupported platform '${p}' — ignoring"
      ;;
  esac
done
if [ -z "${PLATFORMS}" ]; then
  PLATFORMS="linux/amd64,linux/arm64"
  warn "No valid platforms found; defaulting to ${PLATFORMS}"
fi
log "Target platforms: ${PLATFORMS}"

LOCAL_ARCH="$(uname -m || true)"
case "${LOCAL_ARCH}" in
  x86_64|amd64) LOCAL_PLATFORM="linux/amd64" ;;
  aarch64|arm64) LOCAL_PLATFORM="linux/arm64" ;;
  *) LOCAL_PLATFORM="linux/amd64"; warn "Unknown local arch ${LOCAL_ARCH}, assuming linux/amd64" ;;
esac
log "Local platform: ${LOCAL_PLATFORM}"

# Ensure docker & buildx present
if ! command -v docker >/dev/null 2>&1; then
  err "docker CLI not found; install docker"
  exit 2
fi
if ! docker buildx version >/dev/null 2>&1; then
  err "docker buildx not available"
  exit 2
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT}"
export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"

# binfmt/qemu best-effort install
BINFMT_DIR="/proc/sys/fs/binfmt_misc"
QEMU_PRESENT=0
if [ -d "${BINFMT_DIR}" ] && ls "${BINFMT_DIR}" 2>/dev/null | grep -q 'qemu'; then
  QEMU_PRESENT=1
  log "binfmt_misc has qemu handlers"
fi
if [ "${QEMU_PRESENT}" -eq 0 ]; then
  warn "No qemu handlers detected. Attempting install via tonistiigi/binfmt (may require privileged)"
  if docker run --rm --privileged tonistiigi/binfmt --install all >/dev/null 2>&1; then
    log "Installed binfmt/qemu handlers"
    QEMU_PRESENT=1
  else
    warn "Failed to install tonistiigi/binfmt; multi-arch build may fail for non-local platforms"
  fi
fi

# Create or reuse buildx builder
if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  log "Using existing builder ${BUILDER_NAME}"
  docker buildx use "${BUILDER_NAME}" >/dev/null 2>&1 || true
else
  log "Creating builder ${BUILDER_NAME}"
  if docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use >/dev/null 2>&1; then
    CREATED_BUILDER=1
    log "Created builder ${BUILDER_NAME}"
  else
    warn "Create builder failed; trying default"
    if docker buildx create --name "${BUILDER_NAME}" --use >/dev/null 2>&1; then
      CREATED_BUILDER=1
      log "Created builder ${BUILDER_NAME} (fallback)"
    else
      err "Failed to create buildx builder"
      exit 3
    fi
  fi
fi

docker buildx inspect --bootstrap >/dev/null 2>&1 || warn "buildx bootstrap failed (continuing)"

# Local single-arch build (load into daemon)
log "Local single-arch build for ${LOCAL_PLATFORM}"
docker buildx build \
  --platform "${LOCAL_PLATFORM}" \
  --tag "${IMAGE_NAME}" \
  --load \
  --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" \
  "${BUILD_CONTEXT_DIR}" || { err "Local build failed"; exit 4; }

# Run local container for smoke test
cleanup_container
log "Starting container ${CONTAINER_NAME} mapping ${HOST_PORT}->${CONTAINER_PORT}"
docker run --name "${CONTAINER_NAME}" -d -p "${HOST_PORT}:${CONTAINER_PORT}" --shm-size=1.8g "${IMAGE_NAME}" >/dev/null

if ! wait_for_health "${HOST_PORT}" "/health" "${WAIT_TIMEOUT}" 1; then
  err "Service did not become healthy"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 5
fi

# POST test: request rerank and validate response length equals document count
PAYLOAD='{"query":"what is retrieval?","documents":["doc one","doc two","doc three"]}'
log "POST /rerank smoke test"
resp=$(printf '%s' "${PAYLOAD}" | curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/rerank" -H "Content-Type: application/json" -d @- ) || {
  err "POST /rerank failed"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 6
}
printf '%s\n' "${resp}" | pretty_json

# validate number of scores == number of documents
DOCS_COUNT=3
if command -v jq >/dev/null 2>&1; then
  SCORES_COUNT=$(printf '%s' "${resp}" | jq -r '.scores | length' 2>/dev/null || true)
else
  SCORES_COUNT=$(python3 - <<PY 2>/dev/null || true
import sys,json
j=json.load(sys.stdin)
print(len(j.get("scores",[])))
PY
  <<<"${resp}")
fi
if [ -z "${SCORES_COUNT}" ]; then
  err "Failed to parse rerank response"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 7
fi
log "Received ${SCORES_COUNT} scores (expected ${DOCS_COUNT})"
if [ "${SCORES_COUNT}" -ne "${DOCS_COUNT}" ]; then
  err "Unexpected scores count"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 8
fi

# stop local test container before multi-arch build
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Multi-arch build & push (only if DOCKER_USERNAME set)
if [ -n "${DOCKER_USERNAME}" ]; then
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into Docker registry as ${DOCKER_USERNAME}"
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "Docker login failed"; exit 11; }
  else
    log "DOCKER_PASSWORD not provided; attempting push (may fail if auth required)"
  fi

  log "Building & pushing multi-arch image for ${PLATFORMS}"
  docker buildx build \
    --platform "${PLATFORMS}" \
    --tag "${IMAGE_NAME}" \
    --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" \
    --push \
    "${BUILD_CONTEXT_DIR}" || { err "Multi-arch buildx push failed"; exit 12; }

  log "Inspecting pushed manifest"
  docker buildx imagetools inspect "${IMAGE_NAME}" || warn "imagetools inspect failed"
  log "Multi-arch image pushed: ${IMAGE_NAME}"
else
  warn "DOCKER_USERNAME not set — skipping multi-arch push. Local image: ${IMAGE_NAME}"
fi

log "Done OK"
exit 0
