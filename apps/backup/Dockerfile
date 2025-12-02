ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

ARG TARGETARCH
ARG KUBECTL_VERSION=v1.29.0
ARG AWSCLI_V2_ZIP_AMD64=awscli-exe-linux-x86_64.zip
ARG AWSCLI_V2_ZIP_ARM64=awscli-exe-linux-aarch64.zip

ENV PATH=/usr/local/bin:$PATH
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl unzip jq zstd tar gzip gnupg dirmngr ca-certificates apt-transport-https; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH}" in \
      "arm64") AWSZIP="${AWSCLI_V2_ZIP_ARM64}" ;; \
      *) AWSZIP="${AWSCLI_V2_ZIP_AMD64}" ;; \
    esac; \
    curl -fsSL "https://awscli.amazonaws.com/${AWSZIP}" -o /tmp/awscliv2.zip; \
    unzip -q /tmp/awscliv2.zip -d /tmp/awscliv2; \
    /tmp/awscliv2/aws/install --update; \
    rm -rf /tmp/awscliv2 /tmp/awscliv2.zip

RUN set -eux; \
    case "${TARGETARCH}" in \
      "arm64") KARCH="arm64" ;; \
      "amd64") KARCH="amd64" ;; \
      *) KARCH="${TARGETARCH}" ;; \
    esac; \
    curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${KARCH}/kubectl" -o /usr/local/bin/kubectl; \
    chmod +x /usr/local/bin/kubectl; \
    kubectl version --client --short || true

RUN set -eux; pip --no-cache-dir install --upgrade pip

RUN adduser --disabled-password --gecos "" --home /home/backupuser backupuser && mkdir -p /workspace && chown backupuser:backupuser /workspace
USER backupuser
WORKDIR /workspace
LABEL org.opencontainers.image.source="https://your-repo"
LABEL org.opencontainers.image.version="${IMAGE_TAG:-latest}"
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 CMD which aws >/dev/null 2>&1 && aws --version >/dev/null 2>&1 || exit 1
ENTRYPOINT ["/bin/sh"]
