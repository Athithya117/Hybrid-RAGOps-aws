#!/usr/bin/env bash
set -euo pipefail
CLUSTER_NAME="rag8s-local"
LOCAL_BIN="$HOME/.local/bin"
KIND_VERSION="v0.29.0"
KUBECTL_VERSION="$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)"
export PATH="$LOCAL_BIN:$PATH"
mkdir -p "$LOCAL_BIN"

install_kind() {
  if ! command -v kind &>/dev/null; then
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/' -e 's/arm64/arm64/')
    URL="https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}"
    curl -fsSL -o "$LOCAL_BIN/kind" "$URL"
    chmod +x "$LOCAL_BIN/kind"
  fi
}

install_kubectl() {
  if ! command -v kubectl &>/dev/null; then
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/' -e 's/arm64/arm64/')
    URL="https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl"
    curl -fsSL -o "$LOCAL_BIN/kubectl" "$URL"
    chmod +x "$LOCAL_BIN/kubectl"
  fi
}

check_docker() {
  if ! command -v docker &>/dev/null; then
    echo "docker not found. Install and ensure the daemon is running." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker daemon not running or current user cannot access it." >&2
    exit 1
  fi
}

create_cluster() {
  if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    kind delete cluster --name "${CLUSTER_NAME}"
  fi
  cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 30080
EOF

  sleep 5
  CONTAINER_NAME="$(docker ps --filter "name=kind-${CLUSTER_NAME}-control-plane" --format '{{.Names}}' | head -n1 || true)"
  if [ -n "$CONTAINER_NAME" ]; then
    docker update --memory 11g --memory-swap 8g --cpus 10 "$CONTAINER_NAME" || true
  fi
}


main() {
  install_kind
  install_kubectl
  check_docker
  create_cluster
  CONTEXT="kind-${CLUSTER_NAME}"
  kubectl cluster-info --context "${CONTEXT}"
  kubectl --context "${CONTEXT}" wait --for=condition=Ready nodes --all --timeout=180s
  kubectl config use-context "${CONTEXT}"
  echo "Switched context to ${CONTEXT}"
}

main