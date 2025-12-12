# apps/inference/forward_auth/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libssl-dev ca-certificates && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*

COPY stateless_openid_auth.py /app/stateless_openid_auth.py
COPY container_entrypoint.sh /app/container_entrypoint.sh
RUN chmod +x /app/container_entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/container_entrypoint.sh"]
