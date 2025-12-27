#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
IMAGE_REPO="${IMAGE_REPO:-frontend-and-auth}"
IMAGE_TAG="${IMAGE_TAG:-v8}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
ACR_NAME="${ACR_NAME:-}"
ACR_REPO_PREFIX="${ACR_REPO_PREFIX:-rag}"
AZURE_REGISTRY="${AZURE_REGISTRY:-true}"
IMAGE_LOCAL_NAME="local-${IMAGE_REPO}:${IMAGE_TAG}"
CONTAINER_NAME="${CONTAINER_NAME:-test-frontend-and-auth}"
HOST_PORT="${HOST_PORT:-8011}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
SLEEP_BETWEEN_TRIES=1
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2}"

log(){ printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn(){ printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container(){
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}

trap 'cleanup_container' EXIT

wait_for_http(){
  local url="$1"; local timeout="$2"; local start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then return 1; fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

retry_cmd(){
  local attempts=0
  local cmd_rc=0
  while :; do
    attempts=$((attempts+1))
    "$@" && { cmd_rc=0; break; } || cmd_rc=$?
    if [ "$attempts" -ge "$RETRY_ATTEMPTS" ]; then break; fi
    sleep $((RETRY_BACKOFF ** (attempts - 1)))
  done
  return $cmd_rc
}

log "Building local image ${IMAGE_LOCAL_NAME}"
docker build -t "${IMAGE_LOCAL_NAME}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 4; }

log "Starting container for minimal smoke test"
cleanup_container
docker run -d --name "${CONTAINER_NAME}" -p "${HOST_PORT}:${CONTAINER_PORT}" "${IMAGE_LOCAL_NAME}" >/dev/null
TEST_URL="http://127.0.0.1:${HOST_PORT}/"
if wait_for_http "${TEST_URL}" "${WAIT_TIMEOUT}"; then
  log "Smoke test OK at ${TEST_URL}"
else
  warn "Smoke test failed on ${TEST_URL}; capturing logs"
  docker logs "${CONTAINER_NAME}" || true
  err "Smoke test timeout"
  exit 5
fi
cleanup_container

AZURE_REGISTRY_LC="$(printf '%s' "${AZURE_REGISTRY}" | tr '[:upper:]' '[:lower:]')"
if [ "${AZURE_REGISTRY_LC}" = "true" ] || [ "${AZURE_REGISTRY_LC}" = "1" ]; then
  if [ -z "${ACR_NAME}" ]; then
    err "AZURE_REGISTRY=true requires ACR_NAME to be set"
    exit 6
  fi
  if ! printf '%s' "${ACR_NAME}" | grep -Eq '^[a-z0-9]{5,50}$'; then
    err "ACR_NAME must be 5-50 characters lowercase letters and numbers"
    exit 7
  fi
  log "Resolving ACR login server for ${ACR_NAME}"
  ACR_LOGIN_SERVER="$(az acr show -n "${ACR_NAME}" --query loginServer -o tsv 2>/dev/null || true)"
  if [ -z "${ACR_LOGIN_SERVER}" ]; then
    err "Unable to resolve ACR '${ACR_NAME}'. Ensure it exists and you are logged in to az."
    exit 8
  fi
  log "Logging into ACR ${ACR_NAME}"
  retry_cmd az acr login --name "${ACR_NAME}" || { err "az acr login failed"; exit 9; }
  TARGET_IMAGE="${ACR_LOGIN_SERVER}/${ACR_REPO_PREFIX}/${IMAGE_REPO}:${IMAGE_TAG}"
  log "Tagging ${IMAGE_LOCAL_NAME} -> ${TARGET_IMAGE}"
  docker tag "${IMAGE_LOCAL_NAME}" "${TARGET_IMAGE}"
  log "Pushing ${TARGET_IMAGE}"
  if retry_cmd docker push "${TARGET_IMAGE}"; then
    log "Push complete: ${TARGET_IMAGE}"
    exit 0
  else
    err "docker push to ACR failed"
    exit 10
  fi
else
  if [ -z "${DOCKER_USERNAME}" ]; then
    log "DOCKER_USERNAME not set — skipping push."
    exit 0
  fi
  TARGET_IMAGE="${DOCKER_USERNAME}/${IMAGE_REPO}:${IMAGE_TAG}"
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into Docker registry as ${DOCKER_USERNAME}"
    printf '%s\n' "${DOCKER_PASSWORD}" | retry_cmd docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "docker login failed"; exit 11; }
  else
    warn "DOCKER_PASSWORD not provided — push may fail"
  fi
  log "Tagging ${IMAGE_LOCAL_NAME} -> ${TARGET_IMAGE}"
  docker tag "${IMAGE_LOCAL_NAME}" "${TARGET_IMAGE}"
  log "Pushing ${TARGET_IMAGE}"
  if retry_cmd docker push "${TARGET_IMAGE}"; then
    log "Push complete: ${TARGET_IMAGE}"
    exit 0
  else
    err "docker push failed"
    exit 12
  fi
fi
