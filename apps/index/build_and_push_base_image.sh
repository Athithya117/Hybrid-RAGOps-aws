#!/usr/bin/env bash
# apps/index/build_and_push_base_image.sh
# Build & push base image (supports Docker Hub or Azure ACR).
# Mirrors behavior/style of apps/reranker/test_and_push_reranker.sh
#
# Required env:
#   DOCKER_USERNAME (optional for Docker Hub push; if absent, build-only)
#   DOCKER_PASSWORD (optional; if present used for docker login)
# Optional env:
#   IMAGE_NAME    (default indexing-base)
#   IMAGE_TAG     (default 2026-01-10)
#   DOCKERFILE    (default Dockerfile.base)
#   BUILD_CONTEXT_DIR (default directory of script)
#   AZURE_REGISTRY (true/1 to push to ACR)
#   ACR_NAME      (required when AZURE_REGISTRY=true)
#   ACR_REPO_PREFIX (optional prefix inside ACR; default rag)
#   RETRY_ATTEMPTS (default 3)
#   RETRY_BACKOFF  (default 2)
#
# Example (Docker Hub):
#   DOCKER_USERNAME=myuser DOCKER_PASSWORD=secret ./build_and_push_base_image.sh
#
# Example (ACR):
# AZURE_REGISTRY=true ACR_NAME=myacr ./build_and_push_base_image.sh

set -euo pipefail

MODE="${1:-}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-indexing-base}"
IMAGE_TAG="${IMAGE_TAG:-2026-01-10}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.base}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
AZURE_REGISTRY="${AZURE_REGISTRY:-false}"
ACR_NAME="${ACR_NAME:-}"
ACR_REPO_PREFIX="${ACR_REPO_PREFIX:-rag}"
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2}"

log(){ printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn(){ printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

retry_cmd(){
  local attempts=0 rc=0
  while :; do
    attempts=$((attempts+1))
    "$@" && { rc=0; break; } || rc=$?
    if [ "$attempts" -ge "$RETRY_ATTEMPTS" ]; then break; fi
    sleep $((RETRY_BACKOFF ** (attempts - 1)))
  done
  return $rc
}

if ! command -v docker >/dev/null 2>&1; then err "docker CLI not found"; exit 2; fi

# Ensure Docker daemon reachable
if ! docker info >/dev/null 2>&1; then err "cannot connect to docker daemon. Is docker running?"; exit 3; fi

# Normalize inputs: docker requires lower-case repo names
lc(){ printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

IMAGE_NAME_SAFE="$(lc "${IMAGE_NAME}")"
IMAGE_TAG_SAFE="$(lc "${IMAGE_TAG}")"
DOCKERFILE_PATH="${BUILD_CONTEXT_DIR%/}/${DOCKERFILE}"

if [ ! -f "${DOCKERFILE_PATH}" ]; then err "Dockerfile not found: ${DOCKERFILE_PATH}"; exit 4; fi

# Local image tag used for the build
IMAGE_LOCAL="${IMAGE_NAME_SAFE}:${IMAGE_TAG_SAFE}"

log "Building local image ${IMAGE_LOCAL} using Dockerfile ${DOCKERFILE_PATH} (context ${BUILD_CONTEXT_DIR})"

# Ensure BuildKit is disabled for this process (and do not set it)
unset DOCKER_BUILDKIT DOCKER_CLI_EXPERIMENTAL

# Build using legacy builder (no BuildKit)
docker build -f "${DOCKERFILE_PATH}" -t "${IMAGE_LOCAL}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 5; }

# Determine push target: ACR if AZURE_REGISTRY true, else Docker Hub if DOCKER_USERNAME set, else skip push
AZURE_REGISTRY_LC="$(printf '%s' "${AZURE_REGISTRY}" | tr '[:upper:]' '[:lower:]')"
if [ "${AZURE_REGISTRY_LC}" = "true" ] || [ "${AZURE_REGISTRY_LC}" = "1" ]; then
  # ACR path
  if ! command -v az >/dev/null 2>&1; then err "az CLI required for ACR push"; exit 6; fi
  if [ -z "${ACR_NAME}" ]; then err "AZURE_REGISTRY=true requires ACR_NAME"; exit 7; fi
  if ! printf '%s' "${ACR_NAME}" | grep -Eq '^[a-z0-9]{5,50}$'; then err "ACR_NAME must be 5-50 lowercase letters/numbers"; exit 8; fi

  log "Resolving ACR login server for ${ACR_NAME}"
  ACR_LOGIN_SERVER="$(az acr show -n "${ACR_NAME}" --query loginServer -o tsv 2>/dev/null || true)"
  if [ -z "${ACR_LOGIN_SERVER}" ]; then err "Unable to resolve ACR '${ACR_NAME}' in current subscription/permissions"; exit 9; fi

  log "Logging into ACR ${ACR_NAME}"
  retry_cmd az acr login --name "${ACR_NAME}" || { err "az acr login failed"; exit 10; }

  TARGET_IMAGE="${ACR_LOGIN_SERVER}/${ACR_REPO_PREFIX}/${IMAGE_NAME_SAFE}:${IMAGE_TAG_SAFE}"
  log "Tagging ${IMAGE_LOCAL} -> ${TARGET_IMAGE}"
  docker tag "${IMAGE_LOCAL}" "${TARGET_IMAGE}"

  log "Pushing ${TARGET_IMAGE}"
  if retry_cmd docker push "${TARGET_IMAGE}"; then
    log "Push complete: ${TARGET_IMAGE}"
    exit 0
  else
    err "docker push to ACR failed"
    exit 11
  fi
else
  # Docker Hub (or user-provided) path
  if [ -z "${DOCKER_USERNAME}" ]; then
    log "DOCKER_USERNAME not set — build-only mode, skipping push"
    exit 0
  fi

  DOCKER_USERNAME_LC="$(lc "${DOCKER_USERNAME}")"
  TARGET_IMAGE="docker.io/${DOCKER_USERNAME_LC}/${IMAGE_NAME_SAFE}:${IMAGE_TAG_SAFE}"

  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into Docker registry"
    # avoid leaking password to logs; temporarily disable xtrace if set
    set +x 2>/dev/null || true
    printf '%s\n' "${DOCKER_PASSWORD}" | retry_cmd docker login -u "${DOCKER_USERNAME_LC}" --password-stdin || { err "Docker login failed"; exit 12; }
    set -x 2>/dev/null || true
  else
    warn "DOCKER_PASSWORD not provided; attempting push may fail"
  fi

  log "Tagging ${IMAGE_LOCAL} -> ${TARGET_IMAGE}"
  docker tag "${IMAGE_LOCAL}" "${TARGET_IMAGE}"

  log "Pushing ${TARGET_IMAGE}"
  if retry_cmd docker push "${TARGET_IMAGE}"; then
    log "Push complete: ${TARGET_IMAGE}"
    exit 0
  else
    err "docker push failed"
    exit 13
  fi
fi
