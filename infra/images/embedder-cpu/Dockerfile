# =========================
# Stage 1 – Build wheels
# =========================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      libsndfile1 \
      libgomp1 \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements early (better cache)
COPY requirements.txt .

# Build wheels for dependencies
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt


# =========================
# Stage 2 – Final runtime
# =========================
FROM python:3.10-slim AS runtime

# Environment
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/workspace/models/onnx/gte-modernbert-base-onnx-int8 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install only minimal OS deps for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY host_embedding_model_cpu_inference.py /app/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Use uvicorn directly
CMD ["python", "-m", "uvicorn", "host_embedding_model_cpu_inference:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
