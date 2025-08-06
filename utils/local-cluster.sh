#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="rag8s-dev"
NAMESPACE="ray-system"

install_kind() {
  if ! command -v kind &>/dev/null; then
    echo "[*] Installing latest kind..."
    VER=$(curl -s https://api.github.com/repos/kubernetes-sigs/kind/releases/latest \
      | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/' -e 's/arm64/arm64/')
    URL="https://kind.sigs.k8s.io/dl/${VER}/kind-${OS}-${ARCH}"
    curl -sLo kind "$URL"
    chmod +x kind
    sudo mv kind /usr/local/bin/
    echo "[*] ✔ kind ${VER} installed."
  else
    echo "[*] kind already installed: $(kind version)"
  fi
}

create_cluster() {
  echo "[*] (Re)creating kind cluster '${CLUSTER_NAME}'..."
  if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    kind delete cluster --name "${CLUSTER_NAME}"
  fi

  cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF

  echo "[*] ✔ Kind cluster '${CLUSTER_NAME}' ready."
}

install_kuberay_operator() {
  echo "[*] Installing KubeRay operator into '${NAMESPACE}'..."
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/ &>/dev/null || true
  helm repo update &>/dev/null

  helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --create-namespace --namespace "${NAMESPACE}"

  echo "[*] Waiting for KubeRay operator to be ready..."
  kubectl rollout status deployment/kuberay-operator \
    -n "${NAMESPACE}" --timeout=180s

  echo "[*] ✔ KubeRay operator is ready."
}

main() {
  install_kind
  create_cluster
  install_kuberay_operator
}

main

