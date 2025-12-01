FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libssl-dev libffi-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir awscli==1.29.31
FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libreoffice-core libreoffice-writer libreoffice-calc ffmpeg jq ca-certificates fonts-dejavu-core unzip default-jre-headless tini && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
COPY ./convert_all.sh /usr/local/bin/convert_all.sh
RUN chmod +x /usr/local/bin/convert_all.sh && chown -R appuser:appuser /app /usr/local/bin/convert_all.sh
USER appuser
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/convert_all.sh"]

