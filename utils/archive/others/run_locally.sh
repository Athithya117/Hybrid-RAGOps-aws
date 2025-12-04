#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/workspace}"
PROXY_DIR="${PROXY_DIR:-$ROOT/inference_pipeline/others}"
MAIN_DIR="${MAIN_DIR:-$ROOT/inference_pipeline}"
VALKEY_CONTAINER="${VALKEY_CONTAINER:-valkey-local}"
VALKEY_IMAGE="${VALKEY_IMAGE:-valkey/valkey:latest}"
EMBEDDER_CONTAINER="${EMBEDDER_CONTAINER:-embedder-cpu}"
EMBEDDER_IMAGE="${EMBEDDER_IMAGE:-athithya324/embedder-reranker-gpu:linux-x86_64}"
WEAVIATE_POD="${WEAVIATE_POD:-weaviate-0}"
WEAVIATE_NAMESPACE="${WEAVIATE_NAMESPACE:-inference}"
PROXY_PORT="${PROXY_PORT:-9000}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
EMBEDDER_PORT="${EMBEDDER_PORT:-8001}"
REDIS_PORT="${REDIS_PORT:-6379}"
SESSION_TTL_SECONDS="${SESSION_TTL_SECONDS:-3600}"
ENABLE_RERANKER="${ENABLE_RERANKER:-true}"
MISTRAL_API_KEY="${MISTRAL_API_KEY:-}"
NET_NAME="${NET_NAME:-rag-net}"
IMAGE_NAME="${IMAGE_NAME:-inference_pipeline_cpu}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
CONTAINER_NAME="${CONTAINER_NAME:-inference_pipeline_cpu}"
AUTH_FILE="${MAIN_DIR}/auth_and_rate_limits.py"
PROXY_PIDFILE="/tmp/proxy.pid"
WEAVIATE_PF_LOG="/tmp/weaviate-portforward.log"
PF_PIDFILE="/tmp/weaviate-pf.pid"
info(){ printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err(){ printf "\033[1;31m[ERROR]\033[0m %s\n" "$*" >&2; exit 1; }
for c in docker curl jq python3 kubectl ss perl sed awk uuidgen; do command -v $c >/dev/null 2>&1 || err "required command missing: $c"; done
info "Ensure docker network ${NET_NAME} exists"
docker network inspect "${NET_NAME}" >/dev/null 2>&1 || docker network create "${NET_NAME}"
if [ -f "${AUTH_FILE}" ]; then
  info "Patching ${AUTH_FILE}"
  cp "${AUTH_FILE}" "${AUTH_FILE}.bak" 2>/dev/null || true
  perl -0777 -pe 's/redis_client\.eval\(\s*_sliding_lua\s*,\s*keys=\[\s*\$?key\s*\]\s*,\s*args=\[\s*\$?now\s*,\s*\$?window\*1000\s*,\s*\$?limit\s*,\s*\$?member\s*\]\s*\)/redis_client.eval(_sliding_lua,1,key,now,window*1000,limit,member)/gs' -i "${AUTH_FILE}" || true
  perl -0777 -pe 's/redis_client\.eval\(\s*_token_bucket_lua\s*,\s*keys=\[\s*\$?key\s*\]\s*,\s*args=\[\s*\$?now\s*,\s*\$?capacity\s*,\s*\$?refill_per_ms\s*,\s*\$?requested\s*\]\s*\)/redis_client.eval(_token_bucket_lua,1,key,now,capacity,refill_per_ms,requested)/gs' -i "${AUTH_FILE}" || true
  sed -i -E "s/eval\(\s*_sliding_lua\s*,\s*keys=\[key\]\s*,\s*args=\[now,window\*1000,limit,member\]\s*\)/eval(_sliding_lua,1,key,now,window*1000,limit,member)/g" "${AUTH_FILE}" || true
  sed -i -E "s/eval\(\s*_token_bucket_lua\s*,\s*keys=\[key\]\s*,\s*args=\[now,capacity,refill_per_ms,requested\]\s*\)/eval(_token_bucket_lua,1,key,now,capacity,refill_per_ms,requested)/g" "${AUTH_FILE}" || true
fi
info "Clean previous containers"
docker rm -f "${VALKEY_CONTAINER}" >/dev/null 2>&1 || true
docker rm -f "${EMBEDDER_CONTAINER}" >/dev/null 2>&1 || true
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
info "Start Redis (valkey) on network ${NET_NAME}"
docker run -d --name "${VALKEY_CONTAINER}" --network "${NET_NAME}" -p "${REDIS_PORT}:6379" "${VALKEY_IMAGE}" >/dev/null
info "Waiting for redis"
for i in $(seq 1 30); do
  if docker exec -i "${VALKEY_CONTAINER}" redis-cli PING >/dev/null 2>&1; then info "redis ready"; break; fi
  sleep 1
  [ $i -eq 30 ] && err "redis did not respond"
done
info "Start embedder on network ${NET_NAME}"
docker run -d --name "${EMBEDDER_CONTAINER}" --network "${NET_NAME}" -p "${EMBEDDER_PORT}:8001" -p 8002:8002 -e PORT=8001 -e METRICS_PORT=8002 -e FORCE_CPU=1 -e CUDA_VISIBLE_DEVICES="" -e NVIDIA_VISIBLE_DEVICES="" -e ENABLE_RERANKER="${ENABLE_RERANKER}" "${EMBEDDER_IMAGE}" >/dev/null 2>&1 || warn "embedder start warnings"
info "Waiting for embedder /health..."
for i in $(seq 1 40); do
  if curl -fsS "http://localhost:${EMBEDDER_PORT}/health" >/dev/null 2>&1; then info "embedder ready"; break; fi
  sleep 1
  [ $i -eq 40 ] && warn "embedder /health timeout"
done
info "Port-forwarding Kubernetes weaviate pod ${WEAVIATE_POD} (namespace ${WEAVIATE_NAMESPACE}) -> localhost:8080"
kubectl -n "${WEAVIATE_NAMESPACE}" port-forward "pod/${WEAVIATE_POD}" 8080:8080 >"${WEAVIATE_PF_LOG}" 2>&1 &
PF_PID=$!
echo "${PF_PID}" > "${PF_PIDFILE}"
trap 'info "Stopping port-forward (pid ${PF_PID})"; kill ${PF_PID} >/dev/null 2>&1 || true; docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; exit' INT TERM EXIT
info "Waiting for weaviate readiness on localhost:8080"
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:8080/v1/.well-known/ready" >/dev/null 2>&1; then info "weaviate ready (port-forward)"; break; fi
  sleep 1
  if ! kill -0 "${PF_PID}" >/dev/null 2>&1; then err "port-forward process exited; see ${WEAVIATE_PF_LOG}"; fi
  [ $i -eq 60 ] && err "weaviate readiness timed out; check ${WEAVIATE_PF_LOG}"
done
info "Start proxy (real if MISTRAL_API_KEY defined else mock)"
if [ -n "${MISTRAL_API_KEY}" ]; then
  (cd "${PROXY_DIR}"; nohup env MISTRAL_API_KEY="${MISTRAL_API_KEY}" PYTHONPATH="${PROXY_DIR}:$PYTHONPATH" uvicorn proxy:app --host 0.0.0.0 --port "${PROXY_PORT}" > "${PROXY_DIR}/proxy.log" 2>&1 & echo $! > "${PROXY_PIDFILE}")
  info "Started real proxy on ${PROXY_PORT}"
else
  MOCK_FILE="${PROXY_DIR}/mock_llm.py"
  cat > "${MOCK_FILE}" <<'PY'
import json,uvicorn,sys
from fastapi import FastAPI,Request
app=FastAPI()
@app.get("/health")
async def health(): return {"ok":True}
@app.post("/generate")
async def gen(req:Request):
    b=await req.json()
    model=b.get("model","mock-model")
    prompt=b.get("prompt") or b.get("messages") or b.get("input") or ""
    text=str(prompt)[:2000]
    return {"text":f"[MOCK {model}] {text}"}
if __name__=="__main__": uvicorn.run("mock_llm:app",host="0.0.0.0",port=int(sys.argv[1]) if len(sys.argv)>1 else 9000)
PY
  nohup python3 "${MOCK_FILE}" "${PROXY_PORT}" > "${PROXY_DIR}/proxy.log" 2>&1 & echo $! > "${PROXY_PIDFILE}"
  info "Started mock LLM proxy on ${PROXY_PORT}"
fi
for i in $(seq 1 20); do
  if curl -fsS "http://localhost:${PROXY_PORT}/health" >/dev/null 2>&1; then info "proxy ready"; break; fi
  sleep 1
  [ $i -eq 20 ] && warn "proxy health failed; check ${PROXY_DIR}/proxy.log"
done
info "Build inference image ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${MAIN_DIR}"
info "Run inference container ${CONTAINER_NAME} using forwarded weaviate (WEAVIATE_URL=http://host.docker.internal:8080)"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER_NAME}" --network "${NET_NAME}" -p "${INFERENCE_PORT}:8000" --add-host=host.docker.internal:host-gateway \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL="INFO" \
  -e ENV="dev" \
  -e APP_HOST="0.0.0.0" \
  -e APP_PORT="8000" \
  -e START_UI="false" \
  -e STREAMLIT_PORT="8501" \
  -e INFERENCE_API="/retrieve" \
  -e VALKEY_URL="redis://${VALKEY_CONTAINER}:6379/0" \
  -e VALKEY_IS_CLUSTER="false" \
  -e SESSION_TTL_SECONDS="${SESSION_TTL_SECONDS}" \
  -e RATE_WINDOW_SECONDS="60" \
  -e RATE_LIMIT="60" \
  -e ANON_RATE_LIMIT="20" \
  -e VALKEY_MAX_CONN="50" \
  -e RATE_ALG="token_bucket" \
  -e WEAVIATE_URL="http://host.docker.internal:8080" \
  -e WEAVIATE_CLASS="${WEAVIATE_POD}" \
  -e WEAVIATE_API_KEY="" \
  -e EMBEDDING_URL="http://localhost:${EMBEDDER_PORT}" \
  -e CROSS_ENCODER_URL="http://localhost:${EMBEDDER_PORT}" \
  -e EMBEDDING_SERVICE_URL="" \
  -e EMBED_MODEL="" \
  -e RERANK_MODEL="" \
  -e LLM_SERVICE_URL="http://localhost:${PROXY_PORT}" \
  -e LLM_MODEL="" \
  -e BM25_TOP="100" \
  -e DENSE_TOP="256" \
  -e UNION_CAP="800" \
  -e STAGE_A_TOP_M="50" \
  -e FUSED_TOP_N="200" \
  -e RRF_K="60" \
  -e DEDUPE_SIM_THRESHOLD="0.88" \
  -e ENABLE_RERANKER="${ENABLE_RERANKER}" \
  -e HEAVY_TOP_K="16" \
  -e HEAVY_BATCH_SIZE="8" \
  -e MAX_CHUNKS_TO_LLM="6" \
  -e GLOBAL_REQUEST_TIMEOUT="30" \
  -e EMBEDDING_BATCH_SIZE="32" \
  -e BIOENCODER_BATCH_SIZE="16" \
  -e CROSSENCODER_BATCH_SIZE="8" \
  -e HTTP_RETRY_ATTEMPTS="3" \
  -e HTTP_BACKOFF_BASE="0.4" \
  -e ASYNC_CONCURRENCY="16" \
  -e HTTP2_FOR_HTTPX="false" \
  -e HTTPX_MAX_KEEPALIVE="20" \
  -e HTTPX_MAX_CONNECTIONS="100" \
  -e S3_PRESIGN_EXPIRES="300" \
  -e S3_ENDPOINT_URL="https://s3.amazonaws.com" \
  -e AWS_ACCESS_KEY_ID="" \
  -e AWS_SECRET_ACCESS_KEY="" \
  -e AWS_SESSION_TOKEN="" \
  -e AWS_REGION="" \
  -e LLM_SYSTEM_PROMPT="You are a helpful knowledge assistant who answers user queries concisely using only the provided context chunks.\nAnnotate every fact or statement inline in brackets with all available non-null fields from the chunk: (source_url, file_name, page_number, row_range, token_range, audio_range, headings/headings_path, chunk_id).\nMerge information from multiple chunks naturally, placing inline annotations after each relevant sentence or fact.\nDo not hallucinate any information or sources.\nKeep the answer concise, readable, and factual.\nAlways end your response with a confidence percentage." \
  -e BATCH_FETCH_SIZE="64" \
  -e ALPHA_WEIGHT="0.5" \
  "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null
CONTAINER_ID=$(docker ps -q -f name="^/${CONTAINER_NAME}$" || true)
if [ -n "${CONTAINER_ID}" ]; then info "Container started: ${CONTAINER_ID}"; else { docker logs "${CONTAINER_NAME}" --tail 200 || true; err "failed to start container"; } fi
info "Waiting for inference /health..."
for i in $(seq 1 30); do
  if curl -fsS "http://localhost:${INFERENCE_PORT}/health" >/dev/null 2>&1; then info "inference ready"; break; fi
  sleep 1
  [ $i -eq 30 ] && { docker logs "${CONTAINER_NAME}" --tail 200 || true; err "inference health check timed out"; }
done
info "Create test session in Redis (sess:local-dev)"
docker exec -i "${VALKEY_CONTAINER}" redis-cli -x SET "sess:local-dev" <<<'{"sub":"dev-user","email":"dev@local"}' >/dev/null
docker exec -i "${VALKEY_CONTAINER}" redis-cli EXPIRE "sess:local-dev" "${SESSION_TTL_SECONDS}" >/dev/null
info "Smoke test"
set +e
curl --cookie "session_id=local-dev" -sS -X POST "http://localhost:${INFERENCE_PORT}/retrieve" -H "Content-Type: application/json" -d '{"query":"sample","do_presign":false}' | jq . || true
set -e
info "All done. To stop port-forward: kill $(cat ${PF_PIDFILE} 2>/dev/null || echo 'N/A') and remove container: docker rm -f ${CONTAINER_NAME}"
