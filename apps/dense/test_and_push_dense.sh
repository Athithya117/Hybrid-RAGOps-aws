#!/usr/bin/env bash
# test_and_push_dense.sh
# Enhanced: BuildKit + buildx multi-arch (linux/amd64,linux/arm64) support, binfmt bootstrap,
# local single-arch smoke test, then multi-arch push (if DOCKER_USERNAME provided).
#
#
# Usage for cpu image: cd apps/dense && bash test_and_push_dense.sh --cpu 
# Usage for gpu image: cd apps/dense && bash test_and_push_dense.sh --gpu
#
# Environment:
#   DOCKER_USERNAME (optional) - if set, script will attempt docker login and push
#   DOCKER_PASSWORD (optional) - used for docker login
#   DOCKER_IMAGES_PLATFORM (optional) - CSV of platforms, e.g. "linux/amd64,linux/arm64"
#   IMAGE_TAG, HOST_PORT, BUILD_CONTEXT_DIR, etc can be overridden

set -euo pipefail

# Config (override via env)
# ---------------------------
MODE="${1:-cpu}"                               # cpu | gpu
DOCKER_USERNAME="${DOCKER_USERNAME:-}"         # if empty, skip push
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"         # optional; if set script will attempt login
IMAGE_TAG="${IMAGE_TAG:-amd64-arm64-v3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}dense:${IMAGE_TAG}}"
CONTAINER_NAME="${CONTAINER_NAME:-test-dense-${MODE}}"
HOST_PORT="${HOST_PORT:-9021}"                 # host -> container mapping
CONTAINER_PORT="${CONTAINER_PORT:-8200}"       # must match Dockerfile runtime port
FASTEMBED_GPU_ARG="0"
if [ "${MODE,,}" = "gpu" ]; then FASTEMBED_GPU_ARG="1"; fi

# platforms: accept env var DOCKER_IMAGES_PLATFORM (comma separated, spaces allowed)
# default to both amd64 and arm64 for AWS compatibility
DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64,linux/arm64}"

WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
SLEEP_BETWEEN_TRIES=1

BUILDER_NAME="${BUILDER_NAME:-buildx-multi-builder}"
CREATED_BUILDER=0
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

# ---------------------------
# Helpers
# ---------------------------
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
  if [ "${CREATED_BUILDER}" -eq 1 ]; then
    log "Removing temporary buildx builder ${BUILDER_NAME}"
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
  local host_port="$1"
  local path="$2"
  local timeout="$3"
  local require_ready="${4:-0}"
  local start now health
  start=$(date +%s)
  while true; do
    health=$(curl -fsS "http://127.0.0.1:${host_port}${path}" 2>/dev/null || true)
    if [ -n "${health}" ]; then
      if printf '%s' "${health}" | grep -q '"status"' && printf '%s' "${health}" | grep -q '"ok"'; then
        if [ "${require_ready}" -eq 1 ]; then
          if printf '%s' "${health}" | grep -q '"model_ready"\s*:\s*true'; then
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

# ---------------------------
# Normalize platforms
# ---------------------------
# remove spaces, ensure comma-separated, and dedupe
PLATFORMS_RAW="$(printf '%s' "${DOCKER_IMAGES_PLATFORM}" | tr -d '[:space:]')"
# fallback if empty
if [ -z "${PLATFORMS_RAW}" ]; then
  PLATFORMS_RAW="linux/amd64,linux/arm64"
fi
# dedupe simple
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
      warn "Unsupported/unknown platform '${p}' — ignoring"
      ;;
  esac
done
if [ -z "${PLATFORMS}" ]; then
  PLATFORMS="linux/amd64,linux/arm64"
  warn "No valid platforms found in DOCKER_IMAGES_PLATFORM, defaulting to ${PLATFORMS}"
fi

log "Target multi-arch platforms: ${PLATFORMS}"

# ---------------------------
# Detect local platform mapping
# ---------------------------
LOCAL_ARCH="$(uname -m || true)"
case "${LOCAL_ARCH}" in
  x86_64|amd64) LOCAL_PLATFORM="linux/amd64" ;;
  aarch64|arm64) LOCAL_PLATFORM="linux/arm64" ;;
  *) LOCAL_PLATFORM="linux/amd64"; warn "Unknown local arch ${LOCAL_ARCH}, assuming linux/amd64" ;;
esac
log "Local host arch: ${LOCAL_ARCH} -> local platform: ${LOCAL_PLATFORM}"

# ---------------------------
# Ensure docker buildx available
# ---------------------------
if ! command -v docker >/dev/null 2>&1; then
  err "docker CLI not found; install Docker before running this script"
  exit 2
fi
if ! docker buildx version >/dev/null 2>&1; then
  err "docker buildx not available. Ensure Docker CLI has buildx plugin."
  exit 2
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT}"
export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"

# ---------------------------
# Ensure binfmt/QEMU present (best-effort)
# ---------------------------
# If /proc/sys/fs/binfmt_misc contains nothing, try installing QEMU handlers with tonistiigi/binfmt
BINFMT_DIR="/proc/sys/fs/binfmt_misc"
QEMU_PRESENT=0
if [ -d "${BINFMT_DIR}" ] && ls "${BINFMT_DIR}" 2>/dev/null | grep -q 'qemu'; then
  QEMU_PRESENT=1
  log "binfmt_misc appears to have qemu handlers"
fi

if [ "${QEMU_PRESENT}" -eq 0 ]; then
  warn "No qemu binfmt handlers detected. Attempting to install via tonistiigi/binfmt (may require --privileged)"
  if docker run --rm --privileged tonistiigi/binfmt --install all >/dev/null 2>&1; then
    log "Installed binfmt/qemu handlers"
    QEMU_PRESENT=1
  else
    warn "Failed to install tonistiigi/binfmt (privileged may be required). Multi-arch build may fail for non-local platforms."
  fi
fi

# ---------------------------
# Create or reuse buildx builder (docker-container recommended)
# ---------------------------
# prefer docker-container driver for portability; fall back to existing default if creation fails
if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  log "Using existing buildx builder: ${BUILDER_NAME}"
  docker buildx use "${BUILDER_NAME}" >/dev/null 2>&1 || true
else
  log "Creating buildx builder '${BUILDER_NAME}' (driver=docker-container)"
  # try to create with docker-container driver; if it fails, fall back to default
  if docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use >/dev/null 2>&1; then
    CREATED_BUILDER=1
    log "Created builder ${BUILDER_NAME}"
  else
    warn "Failed to create docker-container driver builder; trying docker (host) builder"
    if docker buildx create --name "${BUILDER_NAME}" --use >/dev/null 2>&1; then
      CREATED_BUILDER=1
      log "Created builder ${BUILDER_NAME} (host driver)"
    else
      err "Failed to create or use a buildx builder"
      exit 3
    fi
  fi
fi

# bootstrap the builder (ensures QEMU registration for emulation)
log "Bootstrapping buildx builder"
docker buildx inspect --bootstrap >/dev/null 2>&1 || {
  warn "docker buildx inspect --bootstrap failed; continuing but builds may fail"
}

# ---------------------------
# Local single-arch build (load into local daemon) + smoke test
# ---------------------------
log "Performing local single-arch build for ${LOCAL_PLATFORM} (loads into local docker daemon)"
docker buildx build \
  --platform "${LOCAL_PLATFORM}" \
  --tag "${IMAGE_NAME}" \
  --load \
  --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" \
  "${BUILD_CONTEXT_DIR}" || {
    err "Local single-arch build failed"
    exit 4
  }

log "Running local container from ${IMAGE_NAME} for smoke tests"
cleanup_container
docker run --name "${CONTAINER_NAME}" -d -p "${HOST_PORT}:${CONTAINER_PORT}" --shm-size=1.8g "${IMAGE_NAME}" >/dev/null

if ! wait_for_health "${HOST_PORT}" "/health" "${WAIT_TIMEOUT}" 1; then
  err "Local container did not become healthy"
  log "=== Container logs (tail 200) ==="
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 5
fi

# Run embed test + metrics check (same as previous)
EMBED_PAYLOAD='{"texts":["hello from robust_test_and_push_dense"]}'
log "POST /embed smoke test"
resp=$(printf '%s' "${EMBED_PAYLOAD}" | curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/embed" -H "Content-Type: application/json" -d @- ) || {
  err "Embed POST failed on local container"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 6
}
printf '%s\n' "${resp}" | pretty_json

# discover expected dim from /health
DENSE_DIM_DETECTED=$(curl -fsS "http://127.0.0.1:${HOST_PORT}/health" 2>/dev/null | (jq -r '.dim // empty' 2>/dev/null || python3 -c "import sys,json;print(json.load(sys.stdin).get('dim',''))") ) || true
: "${DENSE_DIM_DETECTED:=384}"

# extract vector length
if command -v jq >/dev/null 2>&1; then
  VEC_LEN=$(printf '%s' "${resp}" | jq -r '.vectors[0] | length' 2>/dev/null || true)
else
  VEC_LEN=$(python3 - <<PY 2>/dev/null || true
import sys,json
j = json.load(sys.stdin)
v = j.get("vectors", [[]])
print(len(v[0]) if v and v[0] else "")
PY
)
fi

if [ -z "${VEC_LEN}" ]; then
  err "Failed to parse vector length from embed response"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 7
fi

log "Embed returned vector length = ${VEC_LEN} (expected ${DENSE_DIM_DETECTED})"
if [ "${VEC_LEN}" -ne "${DENSE_DIM_DETECTED}" ]; then
  err "Vector length mismatch: got ${VEC_LEN} expected ${DENSE_DIM_DETECTED}"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 8
fi

# Stop & remove local test container before multi-arch build
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# ---------------------------
# Multi-arch build & push (only if DOCKER_USERNAME set)
# ---------------------------
if [ -n "${DOCKER_USERNAME}" ]; then
  # ensure docker login if password provided
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into Docker registry ${DOCKER_USERNAME}"
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin || {
      err "Docker login failed"
      exit 11
    }
  else
    log "DOCKER_PASSWORD not provided; attempting push (may fail if registry requires auth)"
  fi

  log "Building and pushing multi-arch image for ${PLATFORMS}"
  docker buildx build \
    --platform "${PLATFORMS}" \
    --tag "${IMAGE_NAME}" \
    --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" \
    --push \
    "${BUILD_CONTEXT_DIR}" || {
      err "Multi-arch buildx push failed"
      exit 12
    }

  log "Inspecting pushed manifest"
  docker buildx imagetools inspect "${IMAGE_NAME}" || {
    warn "imagetools inspect failed - but image might still be pushed"
  }
  log "Multi-arch image pushed: ${IMAGE_NAME}"

else
  warn "DOCKER_USERNAME not set — skipping multi-arch push. You have a locally loaded single-arch image tagged ${IMAGE_NAME}"
fi

log "Done OK"
exit 0
