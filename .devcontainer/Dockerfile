FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DOCKER_GID=999
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal tooling + docker CLI (no dockerd), socat, ssh client, jq
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl gnupg lsb-release sudo openssh-client socat gh jq apt-transport-https gnupg2 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create docker group with host-mapped GID (best-effort)
RUN groupadd -g ${DOCKER_GID} docker 2>/dev/null || true

# Ensure vscode user can sudo and is in docker group
RUN usermod -aG docker vscode 2>/dev/null || true \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99_vscode || true

# Copy helper scripts (executed in post-create)
RUN chown -R vscode:docker /workspace/.devcontainer || true

USER vscode
WORKDIR /workspace
