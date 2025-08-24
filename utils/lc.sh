#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="rag8s-local"
NAMESPACE="rag8s-dev"
KIND_VERSION="v0.29.0"
KUBERAY_VERSION="1.4.2"
LOCAL_BIN="$HOME/.local/bin"

mkdir -p "$LOCAL_BIN"
export PATH="$LOCAL_BIN:$PATH"

install_kind() {
  if ! command -v kind &>/dev/null; then
    echo "Installing kind ${KIND_VERSION}..."
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | sed \
      -e 's/x86_64/amd64/' \
      -e 's/aarch64/arm64/' \
      -e 's/arm64/arm64/')
    URL="https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}"
    curl -sLo "$LOCAL_BIN/kind" "$URL"
    chmod +x "$LOCAL_BIN/kind"
  else
    echo "kind already installed."
  fi
}

create_cluster() {
  if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "Deleting existing cluster: ${CLUSTER_NAME}"
    kind delete cluster --name "${CLUSTER_NAME}"
  fi

  echo "Creating new kind cluster: ${CLUSTER_NAME}"
  cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
EOF
}

install_kuberay_crds() {
  if ! kubectl get crds 2>/dev/null | grep -q "rayclusters.ray.io"; then
    echo "Installing KubeRay CRDs v${KUBERAY_VERSION}..."
    kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v${KUBERAY_VERSION}"
  else
    echo "KubeRay CRDs already installed."
  fi
}

main() {
  install_kind
  create_cluster
  install_kuberay_crds

  CONTEXT="kind-${CLUSTER_NAME}"
  echo "Switching kubectl context to ${CONTEXT}..."
  kubectl cluster-info --context "${CONTEXT}"
  echo "Waiting for cluster to settle..."
  sleep 20
  kubectl get nodes --context "${CONTEXT}"
}

main
kubectl config use-context "kind-${CLUSTER_NAME}"
