# ---------- Base image with CUDA 12.4 and cuDNN 8 ----------
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl unzip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python & pip setup ----------
RUN pip install --upgrade pip setuptools wheel

# ---------- Core dependencies (version pinned as of mid-2025) ----------
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html \
    vllm[serve]==0.10.0 \
    transformers==4.54.1 \
    autoawq==0.2.9 \
    accelerate==1.9.0 \
    sentencepiece==0.2.0 \
    fastapi==0.111.0 uvicorn==0.30.0

# ---------- Model downloader stage ----------
FROM base AS model_downloader

ARG MODEL_ID=Qwen/Qwen3-4B-AWQ  # Can be overridden at build time

RUN mkdir -p /models

RUN python3 - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("${MODEL_ID}", trust_remote_code=True, cache_dir=f"/models")
AutoModelForCausalLM.from_pretrained("${MODEL_ID}", trust_remote_code=True, cache_dir=f"/models")
EOF

# ---------- Final runtime image ----------
FROM base

ARG MODEL_ID=Qwen/Qwen3-4B-AWQ
ENV MODEL_ID=${MODEL_ID}

COPY --from=model_downloader /models /models

# Copy start script and templates if needed
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Optional: Add chat template
COPY templates /templates

# Expose vLLM standard ports
EXPOSE 8000 8001

ENTRYPOINT ["/start.sh"]
