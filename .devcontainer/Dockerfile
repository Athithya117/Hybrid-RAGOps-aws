FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ARG DOCKER_GID=999
RUN groupadd -g ${DOCKER_GID} docker \
    && usermod -aG docker vscode

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    sudo \
    tree \
    make \
    gh \
    vim \
    docker.io \
    ca-certificates \
    gnupg \
    lsb-release \
    tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
