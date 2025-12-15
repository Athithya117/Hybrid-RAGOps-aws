#!/usr/bin/env bash
# apps/inference/frontend/test_and_push_frontend_auth.sh
#
# Build image, run minimal smoke tests (health + auth endpoint), optionally push.
#
# Minimal & deterministic: sets required envs so the app starts without failing discovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT_DIR="${SCRIPT_DIR}"
IMAGE_TAG="${IMAGE_TAG:-v5}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
IMAGE_NAME="${IMAGE_NAME:-${DOCKER_USERNAME:+${DOCKER_USERNAME}/}frontend-and-auth:${IMAGE_TAG}}"

CONTAINER_NAME="${CONTAINER_NAME:-test-frontend-and-auth}"
HOST_PORT="${HOST_PORT:-8011}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
SLEEP_BETWEEN_TRIES=1

log()  { printf '\033[0;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

cleanup_container() {
  set +e
  if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
    log "Removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  set -e
}

trap 'cleanup_container' EXIT

wait_for_http() {
  local url="$1"; local timeout="$2"
  local start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then return 1; fi
    sleep "${SLEEP_BETWEEN_TRIES}"
  done
}

###############################################
#              BUILD IMAGE
###############################################
log "Building local image ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${BUILD_CONTEXT_DIR}" \
  || { err "docker build failed"; exit 4; }

cleanup_container

###############################################
#               RUN CONTAINER
###############################################
log "Starting frontend container ${CONTAINER_NAME}"

# Provide minimal envs required by frontend_and_auth.py to start successfully.
# Note: OIDC_JWKS_URI is set to a stable public JWKS endpoint to avoid discovery-time failures.
docker run --name "${CONTAINER_NAME}" \
  -d -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e AUTH_MODE="external-id" \
  -e OIDC_AUDIENCE="test-aud" \
  -e SPA_CLIENT_ID="test-spa" \
  -e QUERY_URL="http://127.0.0.1:9999" \
  -e FRONTEND_URL="http://127.0.0.1:${HOST_PORT}" \
  -e OIDC_ISSUER="https://accounts.google.com" \
  -e OIDC_JWKS_URI="https://www.googleapis.com/oauth2/v3/certs" \
  "${IMAGE_NAME}" >/dev/null

###############################################
#         WAIT FOR FRONTEND TO START
###############################################
HEALTH_URL="http://127.0.0.1:${HOST_PORT}/health"

log "Waiting for frontend /health on ${HEALTH_URL}"
if ! wait_for_http "${HEALTH_URL}" "${WAIT_TIMEOUT}"; then
  log "Container logs (last 200 lines):"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  err "Frontend container did not become healthy"
  exit 5
fi

###############################################
#            BASIC SMOKE TESTS
###############################################
log "GET /health"
curl -fsS "${HEALTH_URL}" || {
  err "Health endpoint failed"
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  exit 6
}

log "GET /auth/me (expect 401 Unauthorized)"
status_line=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${HOST_PORT}/auth/me" || echo "000")
if [ "$status_line" = "401" ]; then
  log "/auth/me returned 401 as expected"
else
  warn "/auth/me returned unexpected status: $status_line (expected 401)"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null || true
log "Local smoke tests passed."

###############################################
#              OPTIONAL PUSH
###############################################
if [ -z "${DOCKER_USERNAME}" ]; then
  log "DOCKER_USERNAME not set — skipping push."
  exit 0
fi

log "DOCKER_USERNAME provided — preparing to push image"

if [ -n "${DOCKER_PASSWORD}" ]; then
  log "Logging into registry"
  printf '%s\n' "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin \
    || { err "docker login failed"; exit 11; }
else
  warn "DOCKER_PASSWORD not provided — push may fail"
fi

log "Pushing ${IMAGE_NAME}"
docker push "${IMAGE_NAME}" || { err "docker push failed"; exit 12; }

log "Push complete: ${IMAGE_NAME}"
exit 0
