FROM python:3.10.12-slim-bullseye AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv
ENV MODEL_HOME=/opt/models
ENV RAPIDOCR_MODEL_DIR=/opt/models/rapidocr
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential gcc g++ clang libclang-dev git curl wget ca-certificates pkg-config \
  libjpeg-dev zlib1g-dev libxml2-dev libxslt-dev libpng-dev libfreetype6-dev \
  libharfbuzz-dev libxcb1-dev libx11-dev libxext-dev libxrender-dev libfontconfig1-dev \
  libopenjp2-7-dev poppler-utils ffmpeg libmagic1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
COPY . /indexing_pipeline
RUN python -m venv ${VENV_PATH}
RUN ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel
RUN mkdir -p /wheels
RUN ${VENV_PATH}/bin/pip wheel -r /workspace/requirements.txt -w /wheels
RUN ${VENV_PATH}/bin/pip install --no-index --find-links=/wheels -r /workspace/requirements.txt

FROM python:3.10.12-slim-bullseye AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/opt/venv
ENV MODEL_HOST_PATH=/app/models
ENV PORT=8000
ENV HEALTH_PATH=/health
ENV S3_RAW_PREFIX=data/raw/
ENV S3_CHUNKED_PREFIX=data/chunked/
ENV CHUNK_FORMAT=json
ENV MAX_TOKENS_PER_CHUNK=512
ENV MIN_TOKENS_PER_CHUNK=100
ENV NUMBER_OF_OVERLAPPING_SENTENCES=2
ENV PDF_DISABLE_OCR=false
ENV PDF_OCR_ENGINE=tesseract
ENV PDF_TESSERACT_LANG=eng
ENV PDF_FORCE_OCR=false
ENV PDF_OCR_RENDER_DPI=400
ENV PDF_MIN_IMG_SIZE_BYTES=3072
ENV IMAGE_OCR_ENGINE=tesseract
ENV IMAGE_TESSERACT_LANG=eng
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV TESSERACT_CONFIG="--oem 1 --psm 6"
ENV IMAGE_MIN_IMG_SIZE_BYTES=3072
ENV IMAGE_RENDER_DPI=600
ENV IMAGE_UPSCALE_FACTOR=2.0
ENV IMAGE_ENABLE_WORDSEGMENT=false
ENV TOKEN_ENCODER=cl100k_base
ENV TOKEN_ENCODER_MODEL=cl100k_base
ENV FORCE_PROCESS=false
ENV FORCE_OVERWRITE=false
ENV MODEL_HOME=/opt/models
ENV HF_HOME=/opt/models/hf
ENV HF_HUB_CACHE=/opt/models/hf/hub
ENV HF_ASSETS_CACHE=/opt/models/hf/assets
ENV RAPIDOCR_MODEL_DIR=/opt/models/rapidocr
ENV WORKSPACE_MODELS=/indexing_pipeline/models
ENV FW_MODEL_PATH=/indexing_pipeline/models/faster_whisper/faster-whisper-base
ENV FW_MODEL_BIN=/indexing_pipeline/models/faster_whisper/faster-whisper-base/model.bin
RUN apt-get update && apt-get install -y --no-install-recommends \
  libmagic1 poppler-utils ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
  libjpeg62-turbo tesseract-ocr libleptonica-dev libtesseract-dev wget curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /wheels /wheels
COPY --from=builder /indexing_pipeline /indexing_pipeline
ENV PATH=${VENV_PATH}/bin:$PATH
ENV PYTHONPATH=/indexing_pipeline
WORKDIR /indexing_pipeline
EXPOSE 8000
ENV S3_BUCKET=""
ENV LOCAL_DATA_DIR="./data"
ENV AWS_REGION="us-east-1"
ENV EMBEDDING_URL="http://localhost:8001/embed"
ENV EMBED_BATCH=32
ENV EMBED_CONCURRENCY=2
ENV EMBED_TIMEOUT=60
ENV INDEX_BATCH=32
ENV CONCURRENCY=2
ENV INDEXING_EMBEDDER_MAX_LENGTH=2048
ENV METRICS_PORT=8000
ENV HEALTH_PORT=8082
ENV WEAVIATE_URL="http://localhost:8080"
ENV WEAVIATE_API_KEY=""
ENV WEAVIATE_GRPC_HOST=""
ENV WEAVIATE_GRPC_PORT=50051
ENV WEAVIATE_GRPC_SECURE=false
ENV WEAVIATE_FORCE_REST=0
ENV WEAVIATE_COLLECTION="Chunk"
ENV SELECTIVE_UPSERT=1
ENV FORCE_REHASH=0
ENV MAX_HASH_FETCH=5000
ENV BACKUP_BACKEND="s3"
ENV BACKUP_S3_BUCKET=""
ENV BACKUP_S3_PATH_PREFIX="weaviate-backups/"
ENV LATEST_BACKUP_MANIFEST_PATH="weaviate-backups/latest_weaviate_backup.manifest.json"
ENV BACKUP_POLL_INTERVAL=5
ENV BACKUP_TIMEOUT_SECONDS=1800
ENV HTTP_RETRY_MAX=5
ENV HTTP_RETRY_BACKOFF=2
ENV EF_CONSTRUCTION=200
ENV MAX_CONNECTIONS=64
ENV EF_DEFAULT=128
ENV DISTANCE_METRIC="cosine"
ENV QUANTIZATION_TYPE="float16"
ENV QUANTIZATION_BITS=16
ENV QUANTIZATION_SEGMENTS=128
ENV QUANTIZATION_ROTATION=false
ENV VECTOR_INDEX_TYPE="hnsw"
ENV NAMED_VECTOR_NAME="default"
CMD ["/opt/venv/bin/python","indexing_pipeline.py"]
