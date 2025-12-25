#!/usr/bin/env bash
# bash bash apps/sparse/test_and_push_sparse.sh
# Robust build -> local smoke test -> optional multi-arch push
set -euo pipefail

# ---------------------------
# Config (override via env)
# ---------------------------
MODE="${1:-cpu}"                               # cpu|gpu
DOCKER_USERNAME="${DOCKER_USERNAME:-}"         # optional; if set script will try to push
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"         # optional; used only if DOCKER_USERNAME set
IMAGE_TAG="${IMAGE_TAG:-amd64-arm64-v2}"
# SCRIPT_DIR is computed from the script file location so script is runnable from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
# if DOCKER_USERNAME not set, image will be tagged locally with username omitted
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}sparse:${IMAGE_TAG}}"
CONTAINER_NAME="${CONTAINER_NAME:-test-sparse-${MODE}}"
HOST_PORT="${HOST_PORT:-9022}"
CONTAINER_PORT="${CONTAINER_PORT:-8201}"
FASTEMBED_GPU_ARG="0"
if [ "${MODE,,}" = "gpu" ]; then FASTEMBED_GPU_ARG="1"; fi

DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64,linux/arm64}"

WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
SLEEP_BETWEEN_TRIES=1

# buildx builder defaults
BUILDER_NAME="${BUILDER_NAME:-buildx-multi-builder}"
CREATED_BUILDER=0
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

# ---------------------------
log(){ printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn(){ printf '\033[0;33m[WARN]\033[0m %s\n' "$*"; }
err(){ printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container(){
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    log "Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}

cleanup_builder(){
  if [ "${CREATED_BUILDER:-0}" -eq 1 ]; then
    log "Removing temporary buildx builder ${BUILDER_NAME}"
    docker buildx rm "${BUILDER_NAME}" >/dev/null 2>&1 || true
  fi
}

trap 'cleanup_container; cleanup_builder' EXIT

pretty_json(){
  if command -v jq >/dev/null 2>&1; then jq . 2>/dev/null || cat; else python3 - <<'PY' 2>/dev/null || true
import sys,json
print(json.dumps(json.load(sys.stdin), indent=2))
PY
  fi
}

wait_for_health(){
  local host_port="$1"; local path="$2"; local timeout="$3"
  local start now health
  start=$(date +%s)
  while true; do
    health=$(curl -fsS "http://127.0.0.1:${host_port}${path}" 2>/dev/null || true)
    if [ -n "${health}" ]; then
      if printf '%s' "${health}" | grep -q '"status"' && printf '%s' "${health}" | grep -q '"ok"'; then
        if printf '%s' "${health}" | grep -q '"ready"\s*:\s*true'; then
          log "Health OK and ready=true"
          printf '%s\n' "${health}" | pretty_json
          return 0
        fi
        # fallthrough if not model-ready yet
        log "Health returned status ok but not ready=true yet"
        printf '%s\n' "${health}" | pretty_json
      fi
    fi
    now=$(date +%s)
    if [ $((now-start)) -ge "${timeout}" ]; then
      err "Timed out waiting for health at http://127.0.0.1:${host_port}${path}"
      printf '%s\n' "${health}" | sed -n '1,200p' || true
      return 1
    fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

# ---------------------------
# Basic environment sanity
# ---------------------------
if ! command -v docker >/dev/null 2>&1; then
  err "docker CLI not found; please install docker"
  exit 2
fi
if ! docker buildx version >/dev/null 2>&1; then
  warn "docker buildx plugin not found or not functional; multi-arch push may fail"
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT}"
export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"

# ---------------------------
# Normalize platforms (dedupe canonical set)
PLAT_RAW="$(printf '%s' "${DOCKER_IMAGES_PLATFORM}" | tr -d '[:space:]')"
if [ -z "${PLAT_RAW}" ]; then PLAT_RAW="linux/amd64,linux/arm64"; fi
IFS=',' read -r -a __p <<< "${PLAT_RAW}"
PLATFORMS=""
for p in "${__p[@]}"; do
  case "${p}" in
    linux/amd64|linux/arm64)
      if printf '%s' "${PLATFORMS}" | grep -q "${p}"; then :; else PLATFORMS="${PLATFORMS:+${PLATFORMS},}${p}"; fi
      ;;
    *)
      warn "Unsupported platform '${p}' ignored"
      ;;
  esac
done
if [ -z "${PLATFORMS}" ]; then PLATFORMS="linux/amd64,linux/arm64"; fi
log "Target platforms: ${PLATFORMS}"

# ---------------------------
# Create or reuse buildx builder (prefer docker-container driver)
if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  log "Using existing buildx builder: ${BUILDER_NAME}"
  docker buildx use "${BUILDER_NAME}" >/dev/null 2>&1 || true
else
  log "Creating buildx builder ${BUILDER_NAME} (docker-container driver preferred)"
  if docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use >/dev/null 2>&1; then
    CREATED_BUILDER=1
  else
    warn "docker-container driver builder create failed; trying default driver"
    if docker buildx create --name "${BUILDER_NAME}" --use >/dev/null 2>&1; then CREATED_BUILDER=1; else err "Failed to create buildx builder"; exit 3; fi
  fi
fi
docker buildx inspect --bootstrap >/dev/null 2>&1 || warn "buildx bootstrap failed (continuing)"

# ---------------------------
# Local single-arch build (load into local daemon) + smoke test
log "Building local single-arch image (loads into local docker daemon) for testing"
LOCAL_ARCH="$(uname -m || true)"
case "${LOCAL_ARCH}" in
  x86_64|amd64) LOCAL_PLATFORM="linux/amd64" ;;
  aarch64|arm64) LOCAL_PLATFORM="linux/arm64" ;;
  *) LOCAL_PLATFORM="linux/amd64"; warn "unknown local arch ${LOCAL_ARCH}, assuming amd64" ;;
esac

docker buildx build --platform "${LOCAL_PLATFORM}" --tag "${IMAGE_NAME}" --load --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" "${BUILD_CONTEXT_DIR}" || { err "Local build failed"; exit 4; }

# run smoke container
cleanup_container
log "Starting container ${CONTAINER_NAME} mapping ${HOST_PORT}->${CONTAINER_PORT}"
docker run --name "${CONTAINER_NAME}" -d -p "${HOST_PORT}:${CONTAINER_PORT}" --shm-size=1.8g "${IMAGE_NAME}" >/dev/null

if ! wait_for_health "${HOST_PORT}" "/health" "${WAIT_TIMEOUT}"; then
  err "Service did not become healthy"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 5
fi

# embed smoke test
EMBED_PAYLOAD='{"texts":["hello from test_and_push_sparse"]}'
log "POST /embed smoke test"
resp=$(printf '%s' "${EMBED_PAYLOAD}" | curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/embed" -H "Content-Type: application/json" -d @- ) || {
  err "Embed POST failed"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 6
}
printf '%s\n' "${resp}" | pretty_json



# stop local container before multi-arch build/push
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# ---------------------------
# Multi-arch build & push (if DOCKER_USERNAME set)
if [ -n "${DOCKER_USERNAME}" ]; then
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging to registry ${DOCKER_USERNAME}"
    printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "Docker login failed"; exit 8; }
  else
    log "DOCKER_PASSWORD not supplied; attempting push (may fail if registry requires auth)"
  fi

  log "Building & pushing multi-arch image ${IMAGE_NAME} for ${PLATFORORMS:-$PLATFORMS}"
  docker buildx build --platform "${PLATFORMS}" --tag "${IMAGE_NAME}" --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" --push "${BUILD_CONTEXT_DIR}" || { err "Multi-arch build/push failed"; exit 9; }

  log "Inspecting remote manifest"
  docker buildx imagetools inspect "${IMAGE_NAME}" || warn "imagetools inspect failed"
  log "Multi-arch image pushed: ${IMAGE_NAME}"
else
  warn "DOCKER_USERNAME not set; skipping multi-arch push. Local image: ${IMAGE_NAME}"
fi

log "Done OK"
exit 0
