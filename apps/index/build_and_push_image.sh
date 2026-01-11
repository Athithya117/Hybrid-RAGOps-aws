#!/usr/bin/env bash
# apps/index/build_and_push_image.sh
# Build and push the final app image (supports Docker Hub or ACR).
# - Disables BuildKit (uses legacy docker build)
# - Optionally applies host ulimit increase using apps/index/ulimit_increase.sh
# - Dual-nature push: Docker Hub (default) or ACR when AZURE_REGISTRY=true
#
# Environment:
#   DOCKER_USERNAME (optional for Docker Hub push)
#   DOCKER_PASSWORD (optional for Docker Hub push)
#   IMAGE_NAME      (default indexing_pipeline_cpu)
#   IMAGE_TAG       (default latest)
#   BASE_IMAGE      (optional; full image ref used as base, e.g. docker.io/you/indexing-base:2026-01-10)
#   BUILD_CONTEXT_DIR (default: directory of this script)
#   AZURE_REGISTRY  (true|1 to push to ACR instead of Docker Hub)
#   ACR_NAME        (required when AZURE_REGISTRY=true)
#   ACR_REPO_PREFIX (default rag)
#   APPLY_ULIMIT    (default true) -- run ulimit_increase.sh on host before build (uses sudo if required)
#   RETRY_ATTEMPTS  (default 3)
#   RETRY_BACKOFF   (default 2)
#
# Example Docker Hub push:
#   DOCKER_USERNAME=me DOCKER_PASSWORD=secret ./build_and_push_image.sh
#
# Example ACR push:
#   AZURE_REGISTRY=true ACR_NAME=myacr ./build_and_push_image.sh
set -euo pipefail

# --- config / defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-indexing_pipeline_cpu}"
IMAGE_TAG="${IMAGE_TAG:-v17}"
BASE_IMAGE_ARG="${BASE_IMAGE:-indexing-base:2026-01-10}"
AZURE_REGISTRY="${AZURE_REGISTRY:-true}"
ACR_NAME="${ACR_NAME:-}"
ACR_REPO_PREFIX="${ACR_REPO_PREFIX:-rag}"
APPLY_ULIMIT="${APPLY_ULIMIT:-true}"
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

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

# --- prechecks ---
if ! command -v docker >/dev/null 2>&1; then err "docker CLI not found"; exit 2; fi
if ! docker info >/dev/null 2>&1; then err "cannot connect to docker daemon; is docker running?"; exit 3; fi
if [ ! -f "${BUILD_CONTEXT_DIR%/}/${DOCKERFILE}" ]; then err "Dockerfile not found at ${BUILD_CONTEXT_DIR%/}/${DOCKERFILE}"; exit 4; fi

# lowercase helpers
lc(){ printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

IMAGE_NAME_SAFE="$(lc "${IMAGE_NAME}")"
IMAGE_TAG_SAFE="$(lc "${IMAGE_TAG}")"
DOCKER_USERNAME_SAFE="$(lc "${DOCKER_USERNAME:-}")"
BASE_IMAGE_SAFE="${BASE_IMAGE_ARG}"

# --- optional ulimit increase (run on host) ---
# We attempt to run apps/index/ulimit_increase.sh on the host prior to building if APPLY_ULIMIT is true.
ULIMIT_SCRIPT_PATH="${BUILD_CONTEXT_DIR%/}/ulimit_increase.sh"
if [ "${APPLY_ULIMIT}" = "1" ] || [ "${APPLY_ULIMIT}" = "true" ] || [ "${APPLY_ULIMIT}" = "yes" ]; then
  if [ -f "${ULIMIT_SCRIPT_PATH}" ]; then
    log "Applying host ulimit increase via ${ULIMIT_SCRIPT_PATH}"
    # If running as root, run directly, else attempt sudo
    if [ "$(id -u)" -eq 0 ]; then
      bash "${ULIMIT_SCRIPT_PATH}"
    else
      if command -v sudo >/dev/null 2>&1; then
        sudo bash "${ULIMIT_SCRIPT_PATH}"
      else
        warn "sudo not present; cannot apply ulimit changes. Continuing without ulimit increase."
      fi
    fi
  else
    warn "ulimit_increase.sh not present at ${ULIMIT_SCRIPT_PATH}; skipping ulimit step"
  fi
else
  log "APPLY_ULIMIT is false; skipping ulimit increase"
fi

# --- build (disable BuildKit) ---
unset DOCKER_BUILDKIT DOCKER_CLI_EXPERIMENTAL

IMAGE_LOCAL="${IMAGE_NAME_SAFE}:${IMAGE_TAG_SAFE}"

log "Building image ${IMAGE_LOCAL} with base ${BASE_IMAGE_SAFE}"
docker build --no-cache -f "${BUILD_CONTEXT_DIR%/}/${DOCKERFILE}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_SAFE}" \
  -t "${IMAGE_LOCAL}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 5; }

# --- push logic (ACR or Docker Hub) ---
AZURE_REGISTRY_LC="$(printf '%s' "${AZURE_REGISTRY}" | tr '[:upper:]' '[:lower:]')"
if [ "${AZURE_REGISTRY_LC}" = "true" ] || [ "${AZURE_REGISTRY_LC}" = "1" ]; then
  # Push to ACR
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
  # Docker Hub path (build-only allowed if DOCKER_USERNAME unset)
  if [ -z "${DOCKER_USERNAME}" ]; then
    log "DOCKER_USERNAME not set — build-only mode; skipping push"
    exit 0
  fi

  TARGET_IMAGE="docker.io/${DOCKER_USERNAME_SAFE}/${IMAGE_NAME_SAFE}:${IMAGE_TAG_SAFE}"

  if [ -n "${DOCKER_PASSWORD:-}" ]; then
    log "Logging into Docker registry"
    set +x 2>/dev/null || true
    printf '%s\n' "${DOCKER_PASSWORD}" | retry_cmd docker login -u "${DOCKER_USERNAME_SAFE}" --password-stdin || { err "Docker login failed"; exit 12; }
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
