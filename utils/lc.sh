#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Idempotent local-kind cluster creator (3-node: 1 control-plane + 2 workers)
# Usage: CLUSTER_NAME=rag8s-local ./utils/lc.sh
CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
KIND_VERSION="${KIND_VERSION:-v0.29.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)}"
CONTROLPLANE_CONTAINER_MEMORY="${CONTROLPLANE_CONTAINER_MEMORY:-2g}"
CONTROLPLANE_CONTAINER_CPUS="${CONTROLPLANE_CONTAINER_CPUS:-3}"
WORKER_CONTAINER_MEMORY="${WORKER_CONTAINER_MEMORY:-2g}"
WORKER_CONTAINER_CPUS="${WORKER_CONTAINER_CPUS:-3}"
INSTALL_MONITORING="${INSTALL_MONITORING:-true}"
PROM_HELM_REPO="${PROM_HELM_REPO:-prometheus-community}"
PROM_HELM_CHART="${PROM_HELM_CHART:-kube-prometheus-stack}"
PROM_HELM_CHART_VERSION="${PROM_HELM_CHART_VERSION:-79.5.0}"

mkdir -p "${LOCAL_BIN}"
export PATH="${LOCAL_BIN}:$PATH"

command_exists(){ command -v "$1" >/dev/null 2>&1; }

# Basic host prechecks
if ! command_exists curl; then echo "curl required" >&2; exit 1; fi
if ! command_exists docker; then echo "docker required" >&2; exit 1; fi
if ! docker info >/dev/null 2>&1; then echo "docker daemon not running or inaccessible" >&2; exit 1; fi

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "${ARCH}" in
  x86_64|amd64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
  armv7l|armhf) ARCH="armv7" ;;
  *) ARCH="amd64" ;;
esac

# Install kind if missing
KIND_PATH="${LOCAL_BIN}/kind"
if ! command_exists kind; then
  echo "[INFO] Installing kind to ${KIND_PATH}"
  curl -fsSL "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}" -o "${KIND_PATH}.tmp"
  chmod +x "${KIND_PATH}.tmp"
  mv "${KIND_PATH}.tmp" "${KIND_PATH}"
fi
export PATH="$(dirname "${KIND_PATH}"):$PATH"

# Install kubectl if missing
if ! command_exists kubectl; then
  echo "[INFO] Installing kubectl to ${LOCAL_BIN}/kubectl"
  curl -fsSL -o "${LOCAL_BIN}/kubectl" "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl"
  chmod +x "${LOCAL_BIN}/kubectl"
fi

# Optionally install helm for monitoring
if [ "${INSTALL_MONITORING}" = "true" ]; then
  if ! command_exists helm; then
    echo "[INFO] Installing helm"
    TMP_HELM_TGZ="$(mktemp)"
    curl -fsSL "https://get.helm.sh/helm-v3.12.0-${OS}-${ARCH}.tar.gz" -o "$TMP_HELM_TGZ"
    mkdir -p /tmp/helm-extract
    tar -xzf "$TMP_HELM_TGZ" -C /tmp/helm-extract || true
    if [ -f "/tmp/helm-extract/${OS}-${ARCH}/helm" ]; then
      mv "/tmp/helm-extract/${OS}-${ARCH}/helm" "${LOCAL_BIN}/helm"
      chmod +x "${LOCAL_BIN}/helm"
    fi
    rm -rf /tmp/helm-extract "$TMP_HELM_TGZ"
  fi
fi

# Delete existing cluster (idempotent)
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "[INFO] Deleting existing kind cluster '${CLUSTER_NAME}'"
  kind delete cluster --name "${CLUSTER_NAME}" || true
  # wait for deletion
  for i in $(seq 1 30); do
    if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then break; fi
    sleep 1
  done
fi

# Create cluster (3 nodes)
echo "[INFO] Creating cluster '${CLUSTER_NAME}'"
cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 30080
  - role: worker
  - role: worker
EOF

# Tune docker container resources (best-effort)
for container in $(docker ps --filter "name=kind-${CLUSTER_NAME}-" --format '{{.Names}}'); do
  if [[ "${container}" == *"-control-plane" ]]; then
    docker update --memory "${CONTROLPLANE_CONTAINER_MEMORY}" --cpus "${CONTROLPLANE_CONTAINER_CPUS}" "${container}" >/dev/null 2>&1 || true
  else
    docker update --memory "${WORKER_CONTAINER_MEMORY}" --cpus "${WORKER_CONTAINER_CPUS}" "${container}" >/dev/null 2>&1 || true
  fi
done

CONTEXT="kind-${CLUSTER_NAME}"

# Wait for nodes ready
echo "[INFO] Waiting for nodes to be Ready (context=${CONTEXT})"
kubectl --context "${CONTEXT}" wait --for=condition=Ready nodes --all --timeout=180s || true

# Ensure context is set and usable
kubectl config use-context "${CONTEXT}" >/dev/null 2>&1 || true

# Create helpful namespaces
kubectl create namespace inference --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
kubectl create namespace indexing --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
kubectl create namespace kubeblocks --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true

# Optionally install monitoring (idempotent)
if [ "${INSTALL_MONITORING}" = "true" ]; then
  if ! helm repo list | grep -q "^${PROM_HELM_REPO}"; then
    helm repo add "${PROM_HELM_REPO}" https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
    helm repo update >/dev/null 2>&1 || true
  fi
  helm upgrade --install monitoring "${PROM_HELM_REPO}/${PROM_HELM_CHART}" \
    --namespace monitoring --create-namespace --wait --version "${PROM_HELM_CHART_VERSION}" >/dev/null 2>&1 || true
fi


docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/qdrant/qdrant:v1.16.0
docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/athithya5354/dense:amd64-arm64-v1
docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/athithya5354/sparse:amd64-arm64-v2
docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/athithya5354/reranker:amd64-arm64-v1
docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/athithya5354/retrieval:amd64-arm64-v1
docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/athithya5354/qdrant-backup:v2
docker exec rag8s-local-control-plane ctr -n k8s.io images pull docker.io/athithya5354/frontend:amd64-arm64-v1

echo "kind cluster ${CLUSTER_NAME} created (1 control-plane + 2 workers). Context: ${CONTEXT}"
kubectl get nodes

exit 0
