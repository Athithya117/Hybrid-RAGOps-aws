#!/usr/bin/env bash
set -euo pipefail
docker rm -f tmp_rag8s_test >/dev/null 2>&1 || true
export DOCKER_USERNAME="${DOCKER_USERNAME:-rag8s}"
export DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
export DOCKER_REPO="${DOCKER_REPO:-${DOCKER_USERNAME}/rag8s-onnx-embedder-reranker-cpu-amd64}"
export DOCKER_TAG="${DOCKER_TAG:-gte-modernbert}"
export HOST_MODELS_DIR="${HOST_MODELS_DIR:-/workspace/models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$SCRIPT_DIR"
log(){ echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $*"; }
error(){ echo "$(date +'%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2; }
cleanup(){ docker stop tmp_rag8s_test >/dev/null 2>&1 || true; docker rm -f tmp_rag8s_test >/dev/null 2>&1 || true; [ -n "${DOCKER_PASSWORD:-}" ] && docker logout >/dev/null 2>&1 || true; }
trap cleanup EXIT
required_files_check(){ local base="$1"; local name="$2"; for f in "onnx/model_int8.onnx" "config.json" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json"; do local c1="$base/onnx/$name/$f"; local c2="$base/onnx/$name/$(basename "$f")"; if [ ! -f "$c1" ] && [ ! -f "$c2" ]; then error "Missing $f for $name under $base/onnx/$name"; return 1; fi; done; return 0; }
log "SYNC_STATUS: checking host models dir $HOST_MODELS_DIR"
if [ ! -d "$HOST_MODELS_DIR" ]; then error "Host models directory not found: $HOST_MODELS_DIR"; exit 1; fi
required_files_check "$HOST_MODELS_DIR" "gte-modernbert-base-onnx-int8"
required_files_check "$HOST_MODELS_DIR" "gte-reranker-modernbert-base-onnx-int8"
log "Building Docker image $DOCKER_REPO:$DOCKER_TAG"
BUILD_OK=0
USE_BUILDX=0
if docker buildx version >/dev/null 2>&1; then USE_BUILDX=1; fi
MAX_BUILD_ATTEMPTS=5
for i in $(seq 1 $MAX_BUILD_ATTEMPTS); do
  if [ "$USE_BUILDX" -eq 1 ]; then
    if docker buildx build --platform linux/amd64 --load -f "$IMAGE_DIR/Dockerfile" -t "$DOCKER_REPO:$DOCKER_TAG" "$IMAGE_DIR"; then BUILD_OK=1; break; fi
  else
    if docker build --pull --file "$IMAGE_DIR/Dockerfile" --tag "$DOCKER_REPO:$DOCKER_TAG" "$IMAGE_DIR"; then BUILD_OK=1; break; fi
  fi
  log "Build attempt $i failed; retrying after backoff"
  sleep $((i*i))
done
if [ "$BUILD_OK" -ne 1 ]; then
  log "Build failed after $MAX_BUILD_ATTEMPTS attempts, attempting docker pull fallback"
  if docker pull "$DOCKER_REPO:$DOCKER_TAG"; then
    log "Pulled remote image $DOCKER_REPO:$DOCKER_TAG"
  else
    error "No local build and remote pull failed"
    exit 2
  fi
else
  if [ -n "${DOCKER_PASSWORD:-}" ]; then
    log "Logging into Docker"
    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
  fi
  if command -v trivy >/dev/null 2>&1; then
    log "Running Trivy security scan"
    trivy image --exit-code 1 --severity HIGH,CRITICAL "$DOCKER_REPO:$DOCKER_TAG" || { error "Trivy scan failed"; exit 3; }
  fi
  log "Pushing Docker image $DOCKER_REPO:$DOCKER_TAG"
  docker push "$DOCKER_REPO:$DOCKER_TAG"
fi
log "Starting smoke test container with mounted models"
CID=$(docker run -d --name tmp_rag8s_test --shm-size=2g --cpus=2 --memory=8g -v "$HOST_MODELS_DIR":/workspace/models:ro -p 8000:8000 "$DOCKER_REPO:$DOCKER_TAG")
ATTEMPTS=0; MAX=60; SLEEP=2; OK=0
while [ $ATTEMPTS -lt $MAX ]; do
  if curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1; then OK=1; break; fi
  ATTEMPTS=$((ATTEMPTS+1)); sleep $SLEEP
done
if [ $OK -ne 1 ]; then
  error "Smoke test failed. Dumping container inspect and logs"
  docker inspect "$CID" || true
  docker logs "$CID" || true
  docker ps -a --no-trunc | head -n 50 || true
  exit 4
fi
docker stop "$CID" >/dev/null 2>&1 || true
docker rm -f "$CID" >/dev/null 2>&1 || true
log "Docker image $DOCKER_REPO:$DOCKER_TAG pushed and smoke test passed"
log "SYNC_STATUS: completed successfully."
