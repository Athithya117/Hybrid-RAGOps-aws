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
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/' -e 's/arm64/arm64/')
    URL="https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}"
    curl -sLo "$LOCAL_BIN/kind" "$URL"
    chmod +x "$LOCAL_BIN/kind"
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
EOF
}

install_kuberay_crds() {
  if ! kubectl get crds | grep -q "rayclusters.ray.io"; then
    kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v${KUBERAY_VERSION}&timeout=90s"
  fi
}

main() {
  install_kind
  create_cluster
  install_kuberay_crds

  CONTEXT=$(kind get clusters | grep "^${CLUSTER_NAME}$")
  kubectl cluster-info --context "kind-${CONTEXT}"
  kubectl get nodes --context "kind-${CONTEXT}"
}

main


export RELEASE_NAME=release-name
export NAMESPACE=onnx-serving
export VALUES=values.yaml
set -euo pipefail
if [ -f charts/kube-prometheus-stack-76.4.0.tgz ]; then mkdir -p /tmp/kp && tar -xzf charts/kube-prometheus-stack-76.4.0.tgz -C /tmp/kp || true; if [ -d /tmp/kp/kube-prometheus-stack/crds ]; then kubectl apply -f /tmp/kp/kube-prometheus-stack/crds; fi; else helm pull prometheus-community/kube-prometheus-stack --version 76.4.0 --untar && kubectl apply -f kube-prometheus-stack/crds; fi
if [ -f charts/kuberay-operator-1.4.2.tgz ]; then mkdir -p /tmp/kr && tar -xzf charts/kuberay-operator-1.4.2.tgz -C /tmp/kr || true; if [ -d /tmp/kr/kuberay-operator/crds ]; then kubectl apply -f /tmp/kr/kuberay-operator/crds; fi; fi
helm template $RELEASE_NAME . -f $VALUES | kubectl apply -n $NAMESPACE -f -
