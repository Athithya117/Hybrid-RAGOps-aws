FROM python:3.12.12-slim AS builder
ARG FASTEMBED_GPU=0
ARG MODEL_NAME=Xenova/ms-marco-MiniLM-L-6-v2
ENV FASTEMBED_CACHE=/models_cache HF_HOME=/models_cache HUGGINGFACE_HUB_CACHE=/models_cache TRANSFORMERS_CACHE=/models_cache MODEL_NAME=${MODEL_NAME}
RUN mkdir -p /models_cache
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip
RUN if [ "$FASTEMBED_GPU" = "1" ]; then pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3; else pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3; fi
WORKDIR /build
COPY host_reranker.py /build/host_reranker.py
RUN python - <<'PY'
import os
from fastembed.rerank.cross_encoder import TextCrossEncoder
m=TextCrossEncoder(model_name=os.environ.get("MODEL_NAME"))
_ = m.rerank("warmup",["a","b"])
PY
FROM python:3.12.12-slim
ENV FASTEMBED_CACHE=/models_cache HF_HOME=/models_cache HUGGINGFACE_HUB_CACHE=/models_cache TRANSFORMERS_CACHE=/models_cache
RUN mkdir -p /models_cache
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /models_cache /models_cache
COPY --from=builder /build/host_reranker.py /app/host_reranker.py
WORKDIR /app
RUN python -m pip install --upgrade pip
ARG FASTEMBED_GPU=0
RUN if [ "$FASTEMBED_GPU" = "1" ]; then pip install --no-cache-dir numpy==2.2.6 fastembed-gpu==0.7.3 fastapi==0.122.0 uvicorn==0.38.0; else pip install --no-cache-dir numpy==2.2.6 fastembed==0.7.3 fastapi==0.122.0 uvicorn==0.38.0; fi
ENV RERANKER_HOST=0.0.0.0 RERANKER_PORT=8202 RERANKER_MODEL_NAME=Xenova/ms-marco-MiniLM-L-6-v2 RERANKER_MAX_DOCS=50 RERANKER_CUDA=0 FASTEMBED_CACHE=/models_cache
EXPOSE 8202
CMD ["uvicorn","host_reranker:app","--host","0.0.0.0","--port","8202","--lifespan","on"]
