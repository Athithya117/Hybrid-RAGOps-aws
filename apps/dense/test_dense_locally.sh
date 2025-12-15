#!/usr/bin/env bash
# bash apps/dense/test_dense_locally.sh

set -euo pipefail
cd apps/dense
GREEN="\033[0;32m"
NC="\033[0m"

echo -e "${GREEN}=== Local dense embedder smoke test ===${NC}"

echo "[1/5] Install CPU dependencies (quiet)"
pip install -q --upgrade pip
pip install -q numpy==2.2.6 fastembed==0.7.3 fastapi==0.122.0 uvicorn==0.38.0 prometheus_client==0.23.1

echo "[2/5] Start server in background (logs -> dense.log)"
pkill -f "host_dense.py" >/dev/null 2>&1 || true
nohup python3 host_dense.py > dense.log 2>&1 &
echo "sleep 60 seconds.."
sleep 60
echo "Server log tail:"

echo "[3/5] Healthcheck"
curl -s http://localhost:8200/health | jq

echo "[4/5] Embed test"
curl -s -X POST http://localhost:8200/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["hello metrics world"]}' | jq

echo "[5/5] Metrics snapshot"
curl -s http://localhost:8200/metrics | grep -E "requests_total|request_duration_seconds_count|embed_batch_size_count" || true

echo -e "${GREEN}Local tests complete${NC}"
