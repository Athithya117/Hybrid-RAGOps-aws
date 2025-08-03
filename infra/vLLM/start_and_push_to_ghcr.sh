#!/bin/bash

set -e

# Model selection
MODEL_ENV=${MODEL_ENV:-"Qwen/Qwen3-4B-AWQ"}
MODEL_PATH="/models/${MODEL_ENV##*/}"

# Dynamic options
TP_SIZE=${TP_SIZE:-1}
GPU_UTIL=${GPU_UTIL:-0.9}
MAX_TOKENS=${MAX_TOKENS:-32768}
ENABLE_YARN=${ENABLE_YARN:-false}

# Optional long context support
YARN_ARGS=""
if [ "$ENABLE_YARN" = "true" ]; then
  echo "Using YaRN to extend context window beyond 32K"
  YARN_ARGS="--rope-scaling '{\"rope_type\": \"yarn\", \"factor\": 2.0, \"original_max_position_embeddings\": 32768}' --max-model-len $MAX_TOKENS"
fi

# Launch vLLM server
echo "Starting vLLM server with model: $MODEL_PATH"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --tokenizer $MODEL_PATH \
  --trust-remote-code \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size $TP_SIZE \
  --gpu-memory-utilization $GPU_UTIL \
  $YARN_ARGS


docker build -t vllm-qwen3-4b-awq \
  --build-arg MODEL_ID=Qwen/Qwen3-4B-AWQ \
  -f infra/vLLM/Dockerfile infra/vLLM/


docker run --gpus all -p 8000:8000 \
  -e MODEL_ENV=Qwen/Qwen3-4B-AWQ \
  -e ENABLE_YARN=true \
  -e MAX_TOKENS=65536 \
  vllm-qwen


docker run --gpus all -p 8000:8000 \
  -e MODEL_ENV=Qwen/Qwen3-4B-AWQ \
  -e TP_SIZE=1 \
  -e GPU_UTIL=0.9 \
  -e ENABLE_YARN=true \
  -e MAX_TOKENS=65536 \
  vllm-qwen3-4b-awq

