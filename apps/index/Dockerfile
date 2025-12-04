FROM python:3.10.12-slim-bullseye AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN python -m venv ${VENV_PATH}
RUN ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel
RUN mkdir -p /wheels
RUN ${VENV_PATH}/bin/pip wheel -r /workspace/requirements.txt -w /wheels
COPY . /indexing_pipeline
RUN ${VENV_PATH}/bin/pip install --no-index --find-links=/wheels -r /workspace/requirements.txt
RUN rm -rf /wheels
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
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1=1:5.41-3ubuntu0.1 poppler-utils=22.02.0-2ubuntu0.12 ffmpeg=7:4.4.2-0ubuntu0.22.04.1 libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 libglib2.0-0=2.72.4-0ubuntu2.6 libsm6=2:1.2.3-1build2 libxrender1=1:0.9.10-1build4 libxext6=2:1.3.4-1build1 tesseract-ocr=4.1.1-2.1build1 libleptonica-dev=1.82.0-3build1 libtesseract-dev=4.1.1-2.1build1 wget=1.21.2-2ubuntu1.1 curl=7.81.0-1ubuntu1.21 ca-certificates=20240203~22.04.1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /indexing_pipeline /indexing_pipeline
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /opt/venv /indexing_pipeline
ENV PATH=${VENV_PATH}/bin:$PATH
ENV PYTHONPATH=/indexing_pipeline
WORKDIR /indexing_pipeline
EXPOSE 8000
USER appuser
CMD ["/opt/venv/bin/python","indexing_pipeline.py"]
