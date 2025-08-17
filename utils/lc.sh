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
    echo "[*] Installing kind ${KIND_VERSION}..."
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/' -e 's/arm64/arm64/')
    URL="https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}"
    curl -sLo "$LOCAL_BIN/kind" "$URL"
    chmod +x "$LOCAL_BIN/kind"
    echo "[*] ✔ kind ${KIND_VERSION} installed in $LOCAL_BIN."
  else
    echo "[*] kind already present: $(kind version)"
  fi
}

create_cluster() {
  if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "[*] Cluster '${CLUSTER_NAME}' already exists, skipping creation."
  else
    echo "[*] Creating kind cluster '${CLUSTER_NAME}'..."
    cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF
    echo "[*] ✔ Kind cluster '${CLUSTER_NAME}' is ready."
  fi
}

install_kuberay_crds() {
  if ! kubectl get crds | grep -q "rayclusters.ray.io"; then
    echo "[*] Installing KubeRay CRDs (v${KUBERAY_VERSION})..."
    kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v${KUBERAY_VERSION}&timeout=90s"
    echo "[*] ✔ KubeRay CRDs installed."
  else
    echo "[*] KubeRay CRDs already installed, skipping."
  fi
}

install_kuberay_operator() {
  echo "[*] Installing/updating KubeRay operator v${KUBERAY_VERSION}..."
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/ &>/dev/null || true
  helm repo update &>/dev/null

  helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --version "${KUBERAY_VERSION}" \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    --skip-crds \
    --set metrics.serviceMonitor.enabled=false

  echo "[*] Waiting for KubeRay operator deployment..."
  kubectl rollout status deployment/kuberay-operator \
    -n "${NAMESPACE}" --timeout=180s
  echo "[*] ✔ KubeRay operator (${KUBERAY_VERSION}) is ready."
}

main() {
  install_kind
  create_cluster
  install_kuberay_crds
  install_kuberay_operator

  CONTEXT=$(kind get clusters | grep "^${CLUSTER_NAME}$")
  echo "[*] Using cluster context: kind-${CONTEXT}"

  kubectl cluster-info --context "kind-${CONTEXT}"
  kubectl get nodes --context "kind-${CONTEXT}"
  kubectl get pods -n ray-system --context "kind-${CONTEXT}" || echo "[*] No pods yet in ray-system."
}

main



