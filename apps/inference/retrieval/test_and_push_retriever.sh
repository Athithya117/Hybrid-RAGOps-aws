#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="apps/inference/retrieval"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
IMAGE_REPO="${IMAGE_REPO:-retrieval}"
IMAGE_TAG="${IMAGE_TAG:-v10}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
ACR_NAME="${ACR_NAME:-}"
ACR_REPO_PREFIX="${ACR_REPO_PREFIX:-rag}"
AZURE_REGISTRY="${AZURE_REGISTRY:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-test-retrieval}"
HOST_PORT="${HOST_PORT:-8001}"
CONTAINER_PORT="${CONTAINER_PORT:-8001}"
DOCKER_IMAGES_PLATFORM="${DOCKER_IMAGES_PLATFORM:-linux/amd64}"
BUILDX_BUILDER="${BUILDX_BUILDER:-buildx-temp-retrieval}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
SLEEP_BETWEEN_TRIES=1
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2}"

log(){ printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn(){ printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container(){
  set +e
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  set -e
}

cleanup_builder(){
  set +e
  docker buildx rm "${BUILDX_BUILDER}" >/dev/null 2>&1 || true
  set -e
}

trap 'cleanup_container; cleanup_builder' EXIT

wait_for_http(){
  local url="$1" timeout="$2" start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then return 1; fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

metric_value(){
  local metric="$1" file="$2"
  grep -E "^${metric}(\{|\\s)" "$file" | tail -n1 | awk '{print $NF}' || true
}

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

has_buildx(){
  docker buildx version >/dev/null 2>&1
}

AZURE_REGISTRY_LC="$(printf '%s' "${AZURE_REGISTRY}" | tr '[:upper:]' '[:lower:]')"
USE_ACR=false
if [ "${AZURE_REGISTRY_LC}" = "true" ] || [ "${AZURE_REGISTRY_LC}" = "1" ] || [ "${AZURE_REGISTRY_LC}" = "yes" ]; then
  USE_ACR=true
fi

if [ "$USE_ACR" = true ]; then
  if [ -z "${ACR_NAME}" ]; then
    err "AZURE_REGISTRY=true requires ACR_NAME to be set"
    exit 2
  fi
  if ! printf '%s' "${ACR_NAME}" | grep -Eq '^[a-z0-9]{5,50}$'; then
    err "ACR_NAME must be 5-50 characters, lowercase letters and numbers"
    exit 3
  fi
  if ! command -v az >/dev/null 2>&1; then
    err "az CLI not found; required for ACR operations"
    exit 4
  fi
  ACR_LOGIN_SERVER="$(az acr show -n "${ACR_NAME}" --query loginServer -o tsv 2>/dev/null || true)"
  if [ -z "${ACR_LOGIN_SERVER}" ]; then
    err "Unable to resolve ACR '${ACR_NAME}'. Ensure it exists and you are logged in to az."
    exit 5
  fi
  TARGET_IMAGE="${ACR_LOGIN_SERVER}/${ACR_REPO_PREFIX}/${IMAGE_REPO}:${IMAGE_TAG}"
else
  if [ -z "${DOCKER_USERNAME}" ]; then
    log "DOCKER_USERNAME not set — building and running local smoke tests only; skipping push"
    TARGET_IMAGE=""
  else
    TARGET_IMAGE="${DOCKER_USERNAME}/${IMAGE_REPO}:${IMAGE_TAG}"
  fi
fi

LOCAL_IMAGE="${IMAGE_REPO}:local-${IMAGE_TAG}"

log "Building local image ${LOCAL_IMAGE}"
docker build -t "${LOCAL_IMAGE}" "${BUILD_CONTEXT_DIR}"

cleanup_container
log "Starting container ${CONTAINER_NAME}"
docker run -d --name "${CONTAINER_NAME}" -p "${HOST_PORT}:${CONTAINER_PORT}" "${LOCAL_IMAGE}" >/dev/null

log "Waiting for /healthz"
if ! wait_for_http "http://127.0.0.1:${HOST_PORT}/healthz" "${WAIT_TIMEOUT}"; then
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "Health check failed"
  exit 6
fi

log "GET /healthz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/healthz" >/dev/null

log "GET /readyz"
curl -fsS "http://127.0.0.1:${HOST_PORT}/readyz" >/dev/null

METRICS_BEFORE="$(mktemp)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" -o "${METRICS_BEFORE}" || true

PAYLOAD='{"query":"smoke test","top_k":3,"return_chunks":false}'
RESP="$(mktemp)"
curl -fsS -X POST -H "Content-Type: application/json" -d "${PAYLOAD}" "http://127.0.0.1:${HOST_PORT}/generate" -o "${RESP}"

sed -n '1,200p' "${RESP}" || true

METRICS_AFTER="$(mktemp)"
curl -fsS "http://127.0.0.1:${HOST_PORT}/metrics" -o "${METRICS_AFTER}" || true

before=$(metric_value retrieval_requests_total "${METRICS_BEFORE}")
after=$(metric_value retrieval_requests_total "${METRICS_AFTER}")
log "retrieval_requests_total before='${before}' after='${after}'"

cleanup_container
log "Local smoke tests passed."

if [ -z "${TARGET_IMAGE}" ]; then
  log "No remote target configured; exiting after local verification"
  exit 0
fi

if [ "$USE_ACR" = true ]; then
  log "Logging into ACR ${ACR_NAME}"
  if ! retry_cmd az acr login --name "${ACR_NAME}"; then
    err "az acr login failed"
    exit 7
  fi
  if has_buildx; then
    log "Using docker buildx to build & push multi-arch image to ACR"
    docker buildx create --name "${BUILDX_BUILDER}" --use >/dev/null 2>&1 || docker buildx use default
    docker buildx inspect --bootstrap >/dev/null 2>&1 || true
    if retry_cmd docker buildx build --platform "${DOCKER_IMAGES_PLATFORM}" -t "${TARGET_IMAGE}" --push "${BUILD_CONTEXT_DIR}"; then
      log "Multi-arch push complete: ${TARGET_IMAGE}"
      exit 0
    else
      err "docker buildx push to ACR failed"
      exit 8
    fi
  else
    log "buildx not available; pushing single-arch image to ACR"
    docker tag "${LOCAL_IMAGE}" "${TARGET_IMAGE}"
    if retry_cmd docker push "${TARGET_IMAGE}"; then
      log "Push complete: ${TARGET_IMAGE}"
      exit 0
    else
      err "docker push to ACR failed"
      exit 9
    fi
  fi
else
  if [ -n "${DOCKER_PASSWORD}" ]; then
    log "Logging into Docker registry as ${DOCKER_USERNAME}"
    if ! printf '%s\n' "${DOCKER_PASSWORD}" | retry_cmd docker login -u "${DOCKER_USERNAME}" --password-stdin; then
      err "docker login failed"
      exit 10
    fi
  else
    warn "DOCKER_PASSWORD not provided — attempting unauthenticated push (may fail)"
  fi
  if has_buildx; then
    log "Using docker buildx to build & push multi-arch image to Docker Hub"
    docker buildx create --name "${BUILDX_BUILDER}" --use >/dev/null 2>&1 || docker buildx use default
    docker buildx inspect --bootstrap >/dev/null 2>&1 || true
    if retry_cmd docker buildx build --platform "${DOCKER_IMAGES_PLATFORM}" -t "${TARGET_IMAGE}" --push "${BUILD_CONTEXT_DIR}"; then
      log "Multi-arch push complete: ${TARGET_IMAGE}"
      exit 0
    else
      err "docker buildx push failed"
      exit 11
    fi
  else
    log "buildx not available; pushing single-arch image to Docker Hub"
    docker tag "${LOCAL_IMAGE}" "${TARGET_IMAGE}"
    if retry_cmd docker push "${TARGET_IMAGE}"; then
      log "Push complete: ${TARGET_IMAGE}"
      exit 0
    else
      err "docker push failed"
      exit 12
    fi
  fi
fi
