#!/usr/bin/env bash
# python3 apps/dense/host_dense.py
# python3 apps/reranker/host_reranker.py
# python3 apps/sparse/host_sparse.py
# kubectl -n qdrant port-forward svc/qdrant 6333:6333
# python3 apps/inference/retrieval/query.py

# bash apps/inference/retrieval/test_retrieval_locally.sh
# Usage: RETRIEVAL_URL=http://127.0.0.1:8001 ./test_retrieval_locally.sh

set -euo pipefail

RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8001}"
TIMEOUT="${TIMEOUT:-10}"
TMPDIR="${TMPDIR:-/tmp}"
USE_JQ=0

command -v curl >/dev/null 2>&1 || { echo "curl required"; exit 2; }
if command -v jq >/dev/null 2>&1; then USE_JQ=1; fi
if ! command -v python3 >/dev/null 2>&1 && [ "${USE_JQ}" -eq 0 ]; then
  echo "Either jq or python3 required"; exit 2
fi

echo "=== RETRIEVAL SMOKE TEST ==="
echo "Retrieval URL: $RETRIEVAL_URL"
echo

echo "Waiting for retrieval to accept TCP connections on $RETRIEVAL_URL ..."
# parse host:port
u=${RETRIEVAL_URL#http://}; u=${u#https://}
host=${u%%/*}; host=${host%%:*}
port=${u##*:}
if [[ "$port" == "$host" ]]; then
  # default port
  if [[ "$RETRIEVAL_URL" == http*://*:* ]]; then
    port=80
  else
    port=8001
  fi
fi

# try up to TIMEOUT seconds to connect to HTTP
i=0
while true; do
  if curl -fsS --max-time 2 "$RETRIEVAL_URL/healthz" >/dev/null 2>&1; then
    echo "retrieval HTTP OK"
    break
  fi
  i=$((i+1))
  if [ $i -ge $TIMEOUT ]; then
    echo "Timed out waiting for retrieval HTTP at $RETRIEVAL_URL/healthz"; exit 3
  fi
  sleep 1
done

echo "1) GET /healthz and /readyz"
curl -fsS --max-time "$TIMEOUT" "$RETRIEVAL_URL/healthz" -o "$TMPDIR/retrieval_healthz.json"
curl -fsS --max-time "$TIMEOUT" "$RETRIEVAL_URL/readyz" -o "$TMPDIR/retrieval_readyz.json"

if [ "$USE_JQ" -eq 1 ]; then
  echo "/healthz:"; jq . "$TMPDIR/retrieval_healthz.json"
  echo "/readyz:"; jq . "$TMPDIR/retrieval_readyz.json"
else
  python3 - <<PY
import json
print("/healthz:", json.load(open("$TMPDIR/retrieval_healthz.json")))
print("/readyz:", json.load(open("$TMPDIR/retrieval_readyz.json")))
PY
fi
echo

echo "2) GET /metrics (presence check)"
METRICS_BEFORE="$TMPDIR/retrieval_metrics_before.txt"
curl -fsS --max-time "$TIMEOUT" "$RETRIEVAL_URL/metrics" -o "$METRICS_BEFORE"
grep -E "retrieval_requests_total|retrieval_request_duration_seconds|qdrant_query_total|dense_embed_requests_total|llm_calls_total" "$METRICS_BEFORE" || echo "Note: some metrics may be present only after requests."

metric_value() {
  local metric="$1" file="$2"
  grep -E "^${metric}(\\{|\\s)" "$file" | tail -n1 | awk '{print $NF}'
}

echo "3) POST /generate (functional check)"
PAYLOAD='{"query":"smoke test hello","enable_tracing":false,"top_k":3,"return_chunks":false}'
RESP="$TMPDIR/generate_resp.json"
curl -fsS --max-time 30 -X POST "$RETRIEVAL_URL/generate" -H "Content-Type: application/json" -d "$PAYLOAD" -o "$RESP"

echo "Raw /generate response:"
if [ "$USE_JQ" -eq 1 ]; then jq . "$RESP"; else python3 - <<PY
import json
print(json.load(open("$RESP")))
PY
fi

# Validate answer existence (len may be short if no docs or LLM key missing)
if [ "$USE_JQ" -eq 1 ]; then
  ans=$(jq -r '.answer // ""' "$RESP")
else
  ans=$(python3 - <<PY
import json
j=json.load(open("$RESP"))
print(j.get("answer",""))
PY
)
fi
if [ -z "${ans// }" ]; then
  echo "Warning: /generate returned empty or whitespace answer"
else
  echo "/generate produced an answer (length ${#ans})"
fi
echo

echo "4) Validate metrics increment after /generate"
METRICS_AFTER="$TMPDIR/retrieval_metrics_after.txt"
curl -fsS --max-time "$TIMEOUT" "$RETRIEVAL_URL/metrics" -o "$METRICS_AFTER"

before=$(metric_value "retrieval_requests_total" "$METRICS_BEFORE" || echo "")
after=$(metric_value "retrieval_requests_total" "$METRICS_AFTER" || echo "")
echo "retrieval_requests_total before='$before' after='$after'"
if [ -n "$before" ] && [ -n "$after" ]; then
  awk -v a="$before" -v b="$after" 'BEGIN{if(b+0 < a+0) {print "Metric did not increase: fail"; exit 2} else {print "Metric increased or equal: ok"}}'
else
  echo "Could not parse numeric retrieval_requests_total; metric presence checked instead."
fi
echo

echo "5) Dependency metrics quick presence check"
grep -E "qdrant_query_total|dense_embed_requests_total|llm_calls_total" "$METRICS_AFTER" || echo "Dependency metrics not found; ensure embedders/Qdrant/LLM were exercised."

echo
echo "6) Optional: /presign test (only if AWS creds present)"
if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "AWS creds: testing presign (note: bucket/key may not exist, but presign may still be produced)"
  PRESIGN_PAYLOAD='{"s3_path":"s3://example-bucket/example.txt","expires":3600,"inline":true}'
  curl -fsS --max-time 15 -X POST "$RETRIEVAL_URL/presign" -H "Content-Type: application/json" -d "$PRESIGN_PAYLOAD" -o "$TMPDIR/presign_resp.json"
  if [ "$USE_JQ" -eq 1 ]; then jq . "$TMPDIR/presign_resp.json"; else python3 - <<PY
import json
print(json.load(open("$TMPDIR/presign_resp.json")))
PY
  fi
else
  echo "Skipping presign test (no AWS creds in env)"
fi

echo
echo "RETRIEVAL SMOKE TEST: OK"
exit 0
