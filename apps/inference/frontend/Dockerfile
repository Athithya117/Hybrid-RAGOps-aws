# apps/inference/frontend/Dockerfile
# Multi-stage build: wheel cache in builder, minimal runtime image.
# Base python version aligned with repo.
FROM python:3.12.12-slim AS builder

ARG PIP_EXTRA_INDEX_URL=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install build deps for building wheels (kept minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy requirements early for deterministic wheel build
COPY requirements.txt ./requirements.txt

# Build wheels into /wheels (deterministic)
RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip wheel --wheel-dir /wheels -r requirements.txt

# ---------------- runtime image ----------------
FROM python:3.12.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUERY_URL="http://localhost:8001" \
    PORT=8000 \
    ENV="dev"

# Create non-root user for runtime
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Minimal runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheel cache from builder and install from it (deterministic)
COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-index --find-links=/wheels -r /app/requirements.txt

# Copy app code (assumes Docker build context is the frontend directory)
COPY . /app

# Ensure non-root ownership
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# Use import string "frontend_ui:app" to run inside the directory as requested.
CMD ["uvicorn", "frontend_ui:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]
