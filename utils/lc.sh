#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="rag8s-local"
LOCAL_BIN="$HOME/.local/bin"
KIND_VERSION="v0.29.0"

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

  echo "Creating new kind cluster: ${CLUSTER_NAME} with extended memory"
  cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 30080
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            # Allow more pods and memory
            max-pods: "110"
    # Override default Docker resource limits (8 CPUs, 8GiB RAM for node)
    # Adjust these numbers to your machine's capacity
    extraMounts:
      - hostPath: /var/run/docker.sock
        containerPath: /var/run/docker.sock
EOF

  # Update container runtime to have more resources
  docker update --memory 10g --cpus 10 "rag8s-local-control-plane" || true
}

main() {
  install_kind
  create_cluster

  CONTEXT="kind-${CLUSTER_NAME}"
  echo "Switching kubectl context to ${CONTEXT}..."
  kubectl cluster-info --context "${CONTEXT}"
  echo "Waiting for cluster to settle..."
  sleep 20
  kubectl get nodes --context "${CONTEXT}"
}

main
kubectl config use-context "kind-${CLUSTER_NAME}"
