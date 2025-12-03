# Dockerfile — multi-stage, wheel cache, non-root runtime
# Assumes this Dockerfile lives in the retrieval directory and the build context is "."

# ----------------------------
# Builder stage
# ----------------------------
FROM python:3.12.12-slim AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Install build tools needed to build wheels (only in builder)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential gcc curl git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements only (caching)
COPY requirements.txt ./requirements.txt

# Prepare pip, build wheels
RUN python -m pip install --upgrade pip wheel setuptools \
 && python -m pip wheel --wheel-dir /wheels -r requirements.txt

# ----------------------------
# Runtime stage
# ----------------------------
FROM python:3.12.12-slim AS runtime
ARG APP_USER=appuser
ARG APP_UID=1000
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001 \
    PATH="/home/${APP_USER}/.local/bin:${PATH}"

# Create non-root user
RUN groupadd -g ${APP_UID} ${APP_USER} || true \
 && useradd -m -u ${APP_UID} -g ${APP_USER} -s /bin/bash ${APP_USER} || true

# Minimal runtime packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheel cache from builder and application sources from the build context (.)
COPY --from=builder /wheels /wheels
COPY requirements.txt ./requirements.txt
COPY . .

# Install from wheel cache deterministically
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Ensure /app owned by non-root
RUN chown -R ${APP_USER}:${APP_USER} /app

USER ${APP_USER}

EXPOSE ${PORT}

# Docker-level healthcheck (lightweight). Kubernetes should use /readyz for readiness.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

# Run uvicorn with module path pointing to query.py in this directory.
# Using "query:app" so the file can be self-contained in the build context.
CMD ["uvicorn", "query:app", "--host", "0.0.0.0", "--port", "8001", "--lifespan", "on"]
