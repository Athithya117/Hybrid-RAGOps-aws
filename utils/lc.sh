#!/usr/bin/env bash

install_k3d() {
  echo "[*] Installing k3d..."
  if ! command -v k3d &>/dev/null; then
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
  fi
}

create_k3d_cluster() {
  echo "[*] Creating k3d cluster 'rag-dev'..."

  if k3d cluster list | grep -q '^rag-dev'; then
    echo "[*] Deleting existing 'rag-dev' cluster..."
    k3d cluster delete rag-dev
  fi

  k3d cluster create rag-dev \
    --servers 1 \
    --agents 2 \
    --port "8080:80@loadbalancer" \
    --port "6443:6443@server:0" \
    --k3s-arg "--disable=traefik@server:0" \
    --wait

  export KUBECONFIG="$(k3d kubeconfig write rag-dev)"
  kubectl config use-context k3d-rag-dev

  echo " k3d cluster 'rag-dev' created and kubeconfig set."
}

install_k3d
create_k3d_cluster
