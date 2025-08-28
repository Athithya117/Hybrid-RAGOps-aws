#!/usr/bin/env bash
set -euo pipefail
docker rm -f tmp_rag8s_test >/dev/null 2>&1 || true
DOCKER_USERNAME="${DOCKER_USERNAME:-rag8s}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
DOCKER_REPO="${DOCKER_REPO:-${DOCKER_USERNAME}/rag8s-onnx-embedder-reranker-cpu-amd64}"
DOCKER_TAG="${DOCKER_TAG:-gte-modernbert}"
HOST_MODELS_DIR="${HOST_MODELS_DIR:-/workspace/models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$SCRIPT_DIR"
HTTP_PORT="${HTTP_PORT:-8000}"
log(){ echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $*"; }
error(){ echo "$(date +'%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2; }
cleanup(){ docker stop tmp_rag8s_test >/dev/null 2>&1 || true; docker rm -f tmp_rag8s_test >/dev/null 2>&1 || true; [ -n "${TMP_PAYLOAD:-}" ] && rm -f "${TMP_PAYLOAD}" "${TMP_EMB_RESP}" "${TMP_RER_PAYLOAD}" "${TMP_RER_RESP}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
required_files_check(){ local base="$1"; local name="$2"; for f in "onnx/model_int8.onnx" "config.json" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json"; do local c1="$base/onnx/$name/$f"; local c2="$base/onnx/$name/$(basename "$f")"; if [ ! -f "$c1" ] && [ ! -f "$c2" ]; then error "Missing $f for $name under $base/onnx/$name"; return 1; fi; done; return 0; }
log "checking host models dir $HOST_MODELS_DIR"
if [ ! -d "$HOST_MODELS_DIR" ]; then error "Host models directory not found: $HOST_MODELS_DIR"; exit 1; fi
required_files_check "$HOST_MODELS_DIR" "gte-modernbert-base-onnx-int8"
required_files_check "$HOST_MODELS_DIR" "gte-reranker-modernbert-base-onnx-int8"
log "building $DOCKER_REPO:$DOCKER_TAG"
BUILD_OK=0
USE_BUILDX=0
if docker buildx version >/dev/null 2>&1; then USE_BUILDX=1; fi
MAX_BUILD_ATTEMPTS=4
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
fi
log "starting smoke test container with mounted models"
CID=$(docker run -d --name tmp_rag8s_test --shm-size=2g --cpus=2 --memory=8g -v "$HOST_MODELS_DIR":/workspace/models:ro -p "${HTTP_PORT}:8000" "$DOCKER_REPO:$DOCKER_TAG")
ATTEMPTS=0; MAX=60; SLEEP=2; OK=0
while [ $ATTEMPTS -lt $MAX ]; do
  if curl -fsS "http://127.0.0.1:${HTTP_PORT}/healthz" >/dev/null 2>&1; then OK=1; break; fi
  ATTEMPTS=$((ATTEMPTS+1)); sleep $SLEEP
done
if [ $OK -ne 1 ]; then
  error "Smoke test failed. Dumping container inspect and logs"
  docker inspect "$CID" || true
  docker logs "$CID" || true
  docker ps -a --no-trunc | head -n 50 || true
  exit 4
fi
log "container healthy on port ${HTTP_PORT}"
TEXT="${1:-Test embedding $(date +%s)-$RANDOM}"
TMP_PAYLOAD=$(mktemp)
TMP_EMB_RESP=$(mktemp)
TMP_RER_PAYLOAD=$(mktemp)
TMP_RER_RESP=$(mktemp)
python - <<PY > "$TMP_PAYLOAD"
import json,os
print(json.dumps({'texts':[os.environ.get('TEXT'), 'additional sample text']}))
PY
curl -sS -X POST "http://127.0.0.1:${HTTP_PORT}/v1/embed" -H 'Content-Type: application/json' --data-binary @"$TMP_PAYLOAD" -o "$TMP_EMB_RESP"
python - <<PY
import json,sys
j=json.load(open("$TMP_EMB_RESP"))
e=j.get("embeddings",[])
if not isinstance(e,list):
 print("embed-response-invalid",j); sys.exit(2)
for i,vec in enumerate(e):
 if not isinstance(vec,(list,tuple)): print("embed-item-invalid",i); sys.exit(2)
 print("embed",i,"len",len(vec),"sample",vec[:3])
PY
python - <<PY > "$TMP_RER_PAYLOAD"
import json,os
print(json.dumps({'query':os.environ.get('TEXT'),'documents':['Document about cats','Document about dogs','Document about wine']}))
PY
curl -sS -X POST "http://127.0.0.1:${HTTP_PORT}/v1/rerank" -H 'Content-Type: application/json' --data-binary @"$TMP_RER_PAYLOAD" -o "$TMP_RER_RESP"
python - <<PY
import json,sys
j=json.load(open("$TMP_RER_RESP"))
s=j.get("scores",[])
if not isinstance(s,list):
 print("rerank-response-invalid",j); sys.exit(2)
for i,grp in enumerate(s):
 print("rerank",i,grp)
PY
docker stop "$CID" >/dev/null 2>&1 || true
docker rm -f "$CID" >/dev/null 2>&1 || true
if [ -n "${DOCKER_PASSWORD:-}" ]; then
  log "Logging into Docker and pushing $DOCKER_REPO:$DOCKER_TAG"
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
  docker push "$DOCKER_REPO:$DOCKER_TAG"
else
  log "DOCKER_PASSWORD not set; attempting docker push (may fail if not logged in)"
  if ! docker push "$DOCKER_REPO:$DOCKER_TAG"; then error "Push failed and DOCKER_PASSWORD not provided"; exit 5; fi
fi
log "Docker image $DOCKER_REPO:$DOCKER_TAG tested and pushed successfully"
log "completed successfully"
