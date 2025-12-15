# apps/inference/frontend/Dockerfile
FROM python:3.11-slim

# keep Python output unbuffered for logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal system deps (ca-certificates for HTTPS calls)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (requirements.txt should exist next to frontend_and_auth.py)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the app file (minimizes build context & avoids stale scripts)
COPY frontend_and_auth.py /app/frontend_and_auth.py

# Create an unprivileged user and own the app folder
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Run uvicorn directly (no external entrypoint required).
# Use workers=1 to keep container simple for smoke tests; override at runtime if needed.
CMD ["uvicorn", "frontend_and_auth:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
