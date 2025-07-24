FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

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
