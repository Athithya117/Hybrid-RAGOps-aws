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


RUN apt-get update; \
    apt-get install -y ca-certificates curl gnupg lsb-release; \
    rm -f /etc/apt/sources.list.d/azure-cli.*; \
    rm -rf /etc/apt/keyrings/microsoft.gpg /usr/share/keyrings/microsoft.gpg ~/.azure; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor > /etc/apt/keyrings/microsoft.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/microsoft.gpg] \
https://packages.microsoft.com/repos/azure-cli/ jammy main" \
      > /etc/apt/sources.list.d/azure-cli.list; \
    apt-get update; \
    apt-get install -y azure-cli=2.61.0-1~jammy --allow-downgrades; \
    apt-mark hold azure-cli; \
    az version

RUN sudo bash -c 'cat <<EOF >/etc/sysctl.d/99-rag-inotify.conf
fs.inotify.max_user_watches=1048576
fs.inotify.max_user_instances=4096
fs.file-max=2097152
EOF' && sudo bash -c 'cat <<EOF >/etc/sysctl.d/99-rag-inotify.conf
fs.inotify.max_user_watches=1048576
fs.inotify.max_user_instances=4096
fs.file-max=2097152
EOF' && sudo sysctl --system





