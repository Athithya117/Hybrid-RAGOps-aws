#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cpu}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_REPO="${IMAGE_REPO:-reranker}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-${SCRIPT_DIR}}"
ACR_NAME="${ACR_NAME:-}"
ACR_REPO_PREFIX="${ACR_REPO_PREFIX:-rag}"
AZURE_REGISTRY="${AZURE_REGISTRY:-true}"
IMAGE_NAME_LOCAL="${IMAGE_REPO}:${IMAGE_TAG}"
CONTAINER_NAME="${CONTAINER_NAME:-test-reranker-${MODE}}"
HOST_PORT="${HOST_PORT:-9023}"
CONTAINER_PORT="${CONTAINER_PORT:-8202}"
FASTEMBED_GPU_ARG="0"
if [ "${MODE,,}" = "gpu" ]; then FASTEMBED_GPU_ARG="1"; fi
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
SLEEP_BETWEEN_TRIES=1
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2}"
DOCS_COUNT_EXPECT="${DOCS_COUNT_EXPECT:-3}"

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

wait_for_health(){
  local host_port="$1" path="$2" timeout="$3" start now body
  start=$(date +%s)
  while true; do
    body=$(curl -fsS --max-time 2 "http://127.0.0.1:${host_port}${path}" 2>/dev/null || true)
    if [ -n "${body}" ]; then
      if printf '%s' "${body}" | grep -q '"status"' && printf '%s' "${body}" | grep -q '"ok"'; then
        if printf '%s' "${body}" | grep -q '"ready"\s*:\s*true'; then
          command -v jq >/dev/null 2>&1 && printf '%s\n' "${body}" | jq . || printf '%s\n' "${body}"
          return 0
        fi
      fi
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "${timeout}" ]; then
      printf '%s\n' "${body:-<no-body>}" | sed -n '1,200p' || true
      return 1
    fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
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

if ! command -v docker >/dev/null 2>&1; then err "docker CLI not found"; exit 2; fi

LOCAL_ARCH="$(uname -m || true)"
case "${LOCAL_ARCH}" in
  x86_64|amd64) LOCAL_PLATFORM="linux/amd64" ;;
  aarch64|arm64) LOCAL_PLATFORM="linux/arm64" ;;
  *) LOCAL_PLATFORM="linux/amd64"; warn "unknown local arch ${LOCAL_ARCH}, assuming amd64" ;;
esac

log "Building image ${IMAGE_NAME_LOCAL} (platform ${LOCAL_PLATFORM})"
docker build --build-arg "FASTEMBED_GPU=${FASTEMBED_GPU_ARG}" -t "${IMAGE_NAME_LOCAL}" "${BUILD_CONTEXT_DIR}" || { err "docker build failed"; exit 4; }

cleanup_container
log "Starting container ${CONTAINER_NAME} mapping ${HOST_PORT}->${CONTAINER_PORT}"
docker run --name "${CONTAINER_NAME}" -d -p "${HOST_PORT}:${CONTAINER_PORT}" --shm-size=1.8g "${IMAGE_NAME_LOCAL}" >/dev/null

if ! wait_for_health "${HOST_PORT}" "/health" "${WAIT_TIMEOUT}"; then
  err "Health check failed"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 5
fi

PAYLOAD='{"query":"what is retrieval?","documents":["doc one","doc two","doc three"]}'
log "POST /rerank smoke test"
resp=$(printf '%s' "${PAYLOAD}" | curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/rerank" -H "Content-Type: application/json" -d @- ) || {
  err "POST /rerank failed"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 6
}
command -v jq >/dev/null 2>&1 && printf '%s\n' "${resp}" | jq . || printf '%s\n' "${resp}"

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
if [ -z "${SCORES_COUNT}" ]; then err "Failed to parse rerank response"; docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; exit 7; fi
log "Received ${SCORES_COUNT} scores (expected ${DOCS_COUNT_EXPECT})"
if [ "${SCORES_COUNT}" -ne "${DOCS_COUNT_EXPECT}" ]; then err "Unexpected scores count"; docker logs --tail 200 "${CONTAINER_NAME}" || true; docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; exit 8; fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

AZURE_REGISTRY_LC="$(printf '%s' "${AZURE_REGISTRY}" | tr '[:upper:]' '[:lower:]')"
if [ "${AZURE_REGISTRY_LC}" = "true" ] || [ "${AZURE_REGISTRY_LC}" = "1" ]; then
  if ! command -v az >/dev/null 2>&1; then err "az CLI required for ACR push"; exit 9; fi
  if [ -z "${ACR_NAME}" ]; then err "AZURE_REGISTRY=true requires ACR_NAME"; exit 10; fi
  if ! printf '%s' "${ACR_NAME}" | grep -Eq '^[a-z0-9]{5,50}$'; then err "ACR_NAME must be 5-50 lowercase letters/numbers"; exit 11; fi
  log "Resolving ACR login server for ${ACR_NAME}"
  ACR_LOGIN_SERVER="$(az acr show -n "${ACR_NAME}" --query loginServer -o tsv 2>/dev/null || true)"
  if [ -z "${ACR_LOGIN_SERVER}" ]; then err "Unable to resolve ACR '${ACR_NAME}' in current subscription/permissions"; exit 12; fi
  log "Logging into ACR ${ACR_NAME}"
  retry_cmd az acr login --name "${ACR_NAME}" || { err "az acr login failed"; exit 13; }
  TARGET_IMAGE="${ACR_LOGIN_SERVER}/${ACR_REPO_PREFIX}/${IMAGE_REPO}:${IMAGE_TAG}"
  log "Tagging ${IMAGE_NAME_LOCAL} -> ${TARGET_IMAGE}"
  docker tag "${IMAGE_NAME_LOCAL}" "${TARGET_IMAGE}"
  log "Pushing ${TARGET_IMAGE}"
  if retry_cmd docker push "${TARGET_IMAGE}"; then log "Push complete: ${TARGET_IMAGE}"; exit 0; else err "docker push to ACR failed"; exit 14; fi
else
  if [ -z "${DOCKER_USERNAME}" ]; then log "DOCKER_USERNAME not set — skipping push"; exit 0; fi
  TARGET_IMAGE="${DOCKER_USERNAME}/${IMAGE_REPO}:${IMAGE_TAG}"
  if [ -n "${DOCKER_PASSWORD}" ]; then log "Logging into Docker registry"; printf '%s\n' "${DOCKER_PASSWORD}" | retry_cmd docker login -u "${DOCKER_USERNAME}" --password-stdin || { err "Docker login failed"; exit 15; }; else warn "DOCKER_PASSWORD not provided; attempting push may fail"; fi
  log "Tagging ${IMAGE_NAME_LOCAL} -> ${TARGET_IMAGE}"
  docker tag "${IMAGE_NAME_LOCAL}" "${TARGET_IMAGE}"
  log "Pushing ${TARGET_IMAGE}"
  if retry_cmd docker push "${TARGET_IMAGE}"; then log "Push complete: ${TARGET_IMAGE}"; exit 0; else err "docker push failed"; exit 16; fi
fi
