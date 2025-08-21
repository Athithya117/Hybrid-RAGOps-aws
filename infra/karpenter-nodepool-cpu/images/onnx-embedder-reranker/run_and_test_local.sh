#!/usr/bin/env bash

# set -euo pipefail
# pip install -r infra/onnx/image/requirements-cpu.txt >/dev/null

IMAGE="${IMAGE:-rag8s/rag8s-onnx-embedder-reranker-cpu-amd64:gte-modernbert}"
HOST_MODELS_DIR="${HOST_MODELS_DIR:-/workspace/models}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPUS="${CPUS:-2}"
MEM="${MEM:-8g}"
SHM="${SHM:-4g}"
HTTP_PORT_CONTAINER="${HTTP_PORT_CONTAINER:-8000}"
HOST_PORT="${HOST_PORT:-8001}"
GRPC_PORT="${GRPC_PORT:-9000}"
EMBEDDER_NAME="${EMBEDDER_NAME:-gte-modernbert-base-onnx-int8}"
RERANK_NAME="${RERANK_NAME:-gte-reranker-modernbert-base-onnx-int8}"
DO_BUILD="${DO_BUILD:-0}"

req=(
  "$HOST_MODELS_DIR/onnx/$EMBEDDER_NAME/onnx/model_int8.onnx"
  "$HOST_MODELS_DIR/onnx/$EMBEDDER_NAME/tokenizer.json"
  "$HOST_MODELS_DIR/onnx/$EMBEDDER_NAME/tokenizer_config.json"
  "$HOST_MODELS_DIR/onnx/$EMBEDDER_NAME/config.json"
  "$HOST_MODELS_DIR/onnx/$RERANK_NAME/onnx/model_int8.onnx"
  "$HOST_MODELS_DIR/onnx/$RERANK_NAME/tokenizer.json"
  "$HOST_MODELS_DIR/onnx/$RERANK_NAME/tokenizer_config.json"
  "$HOST_MODELS_DIR/onnx/$RERANK_NAME/config.json"
)
for f in "${req[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

if [ "$DO_BUILD" = "1" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" "$WORKDIR"
fi

docker rm -f rag8s_local_test >/dev/null 2>&1 || true
CID=$(docker run -d --name rag8s_local_test \
  --cpus="$CPUS" --memory="$MEM" --shm-size="$SHM" \
  -v "$HOST_MODELS_DIR":/workspace/models:ro \
  -e HF_HOME=/workspace/models/hf \
  -e MODEL_DIR=/workspace/models/onnx \
  -e MODEL_EMBEDDER_NAME="RAG8s/$EMBEDDER_NAME" \
  -e MODEL_RERANKER_NAME="RAG8s/$RERANK_NAME" \
  -e EMBEDDER_ONNX_PATH="/workspace/models/onnx/$EMBEDDER_NAME/onnx/model_int8.onnx" \
  -e RERANKER_ONNX_PATH="/workspace/models/onnx/$RERANK_NAME/onnx/model_int8.onnx" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e EMBEDDER_OMP_NUM_THREADS=1 -e RERANKER_OMP_NUM_THREADS=1 \
  -e EMBEDDER_NUM_CPUS=1 -e RERANKER_NUM_CPUS=1 \
  -e EMBEDDER_BATCH_MAX_SIZE=4 -e RERANKER_BATCH_MAX_SIZE=2 \
  --health-interval=30s --health-timeout=10s --health-retries=5 --health-start-period=120s \
  -p "$HOST_PORT":"$HTTP_PORT_CONTAINER" -p "$GRPC_PORT":"$GRPC_PORT" \
  "$IMAGE")
echo "$CID"

# wait health
TIMEOUT=300; elapsed=0
until curl -fsS "http://127.0.0.1:$HOST_PORT/healthz" >/dev/null 2>&1; do
  sleep 1; elapsed=$((elapsed+1))
  if [ $elapsed -ge $TIMEOUT ]; then
    echo "ERROR: health timeout"; docker logs rag8s_local_test --tail 200; exit 2
  fi
done
echo "health ok"

# 1) Internal Ray handle test (inside container — bypasses ports)
echo "--- INTERNAL RAY HANDLE TEST ---"
docker exec -i rag8s_local_test python - <<'PY'
import sys,traceback
import ray
from ray import serve
import grpc_pb2
try:
    ray.init(address="auto", ignore_reinit_error=True)
    e = serve.get_app_handle("rag8s_embedder_app")
    r = serve.get_app_handle("rag8s_reranker_app")
    er = e.Embed.remote(grpc_pb2.EmbedRequest(texts=["hello from internal test"])).result()
    print("INTERNAL_EMBED_LEN", len(er.embeddings))
    rr = r.Rerank.remote(grpc_pb2.RerankRequest(pairs=[grpc_pb2.Pair(query="q", doc="d")])).result()
    print("INTERNAL_RERANK_SCORES", list(rr.scores))
except Exception:
    traceback.print_exc(); sys.exit(3)
sys.exit(0)
PY

# 2) HTTP endpoint test (external: uses mapped host port)
echo "--- HTTP ENDPOINT TEST (external) ---"
set +e
curl -sS -X POST "http://127.0.0.1:$HOST_PORT/rag8s_embedder_app/Embed" \
  -H "Content-Type: application/json" \
  -d '{"texts":["hello via http"]}' -o /tmp/http_embed.out -w "\nHTTP_STATUS:%{http_code}\n"
echo "embed body:"
cat /tmp/http_embed.out || true

curl -sS -X POST "http://127.0.0.1:$HOST_PORT/rag8s_reranker_app/Rerank" \
  -H "Content-Type: application/json" \
  -d '{"pairs":[{"query":"hello","doc":"doc1"},{"query":"hello","doc":"doc2"}]}' -o /tmp/http_rerank.out -w "\nHTTP_STATUS:%{http_code}\n"
echo "rerank body:"
cat /tmp/http_rerank.out || true
set -e

# 3) gRPC test (attempt inside container; will report if gRPC not served)
echo "--- gRPC TEST (inside container) ---"
docker exec -i rag8s_local_test python - <<'PY' || true
import sys,traceback,grpc
import grpc_pb2,grpc_pb2_grpc
try:
    chan = grpc.insecure_channel("127.0.0.1:9000")
    grpc.channel_ready_future(chan).result(timeout=2)
    stub_e = grpc_pb2_grpc.EmbedServiceStub(chan)
    res = stub_e.Embed(grpc_pb2.EmbedRequest(texts=["hello grpc"]), timeout=5)
    print("GRPC_EMBED_LEN", len(res.embeddings))
    stub_r = grpc_pb2_grpc.RerankServiceStub(chan)
    rr = stub_r.Rerank(grpc_pb2.RerankRequest(pairs=[grpc_pb2.Pair(query="q",doc="d")] ), timeout=5)
    print("GRPC_RERANK_SCORES", list(rr.scores))
except Exception as e:
    print("gRPC test failed:", type(e).__name__, e)
    # not fatal
PY

echo "ALL TESTS COMPLETE"
