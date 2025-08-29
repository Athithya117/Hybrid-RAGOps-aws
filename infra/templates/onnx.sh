#!/usr/bin/env bash
set -euo pipefail

NETWORK=ragnet
DATA_DIR="$(pwd)/data"

# Ensure network exists
docker network inspect $NETWORK >/dev/null 2>&1 || \
  docker network create $NETWORK

# Clean up old TEI container if exists
if docker ps -a --format '{{.Names}}' | grep -q '^tei$'; then
  docker rm -f tei >/dev/null
fi

docker run -d \
  --name tei \
  --network $NETWORK \
  -p 8080:80 \
  -v "$DATA_DIR":/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id Alibaba-NLP/gte-modernbert-base
