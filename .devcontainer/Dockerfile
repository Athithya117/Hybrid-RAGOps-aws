# ===================================================================
# Devcontainer with Docker CLI + buildx + QEMU (multi-arch ready)
# ===================================================================
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DOCKER_GID=999
ENV DEBIAN_FRONTEND=noninteractive

# Install Docker CLI + common tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl gnupg lsb-release sudo openssh-client socat gh jq apt-transport-https gnupg2 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli qemu-user-static \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Docker group (to match host socket GID)
RUN groupadd -g ${DOCKER_GID} docker 2>/dev/null || true

# Give vscode user docker access
RUN usermod -aG docker vscode 2>/dev/null || true \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99_vscode || true

RUN chown -R vscode:docker /workspace/.devcontainer || true

USER vscode
WORKDIR /workspace

# ===================================================================
# Buildx + QEMU setup (auto-enabled inside devcontainer)
# ===================================================================
# Note: This ensures that every time the container is rebuilt,
# buildx + qemu emulation are preconfigured for multi-arch builds.
RUN docker buildx create --name multiarch-builder --use || true \
    && docker run --rm --privileged multiarch/qemu-user-static --reset -p yes \
    && docker buildx inspect --bootstrap || true
