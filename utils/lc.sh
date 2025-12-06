#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
KIND_VERSION="${KIND_VERSION:-v0.29.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)}"
# increased control-plane mem/cpu to reduce CoreDNS/API pressure
CONTROLPLANE_CONTAINER_MEMORY="${CONTROLPLANE_CONTAINER_MEMORY:-3g}"
CONTROLPLANE_CONTAINER_CPUS="${CONTROLPLANE_CONTAINER_CPUS:-3}"
WORKER_CONTAINER_MEMORY="${WORKER_CONTAINER_MEMORY:-2.5g}"
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

# Create helpful namespaces (idempotent)
for ns in inference indexing kubeblocks models qdrant monitoring; do
  kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
done

# ---- Networking hardening for dev: ensure DNS & egress won't get accidentally blocked ----
# 1) Delete any restrictive NetworkPolicy in inference/models/qdrant namespaces (dev convenience)
for ns in inference models qdrant; do
  kubectl -n "${ns}" delete networkpolicy --all --ignore-not-found >/dev/null 2>&1 || true
done

# 2) Create permissive egress policy for inference + models + qdrant (dev only)
cat <<'EOF' | kubectl -n inference apply -f - >/dev/null 2>&1 || true
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-all-egress-dev
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
EOF

for ns in models qdrant; do
  cat <<'EOF' | kubectl -n "${ns}" apply -f - >/dev/null 2>&1 || true
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-all-egress-dev
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
EOF
done

# --- make CoreDNS resilient in dev ---
echo "[INFO] CoreDNS: ensure 2 replicas + resource requests/limits + fallback forwarders"

# wait briefly for kube-system components to exist
kubectl -n kube-system wait --for=condition=Available deployment/coredns --timeout=90s >/dev/null 2>&1 || true

# scale to 2 replicas (idempotent)
kubectl -n kube-system get deployment coredns >/dev/null 2>&1 && \
  kubectl -n kube-system scale deployment coredns --replicas=2 --timeout=60s >/dev/null 2>&1 || true

# set sane resource requests/limits so containerd/kubelet doesn't OOM/CPU-throttle it
kubectl -n kube-system get deployment coredns >/dev/null 2>&1 && \
  kubectl -n kube-system set resources deployment coredns \
    --requests=cpu=200m,memory=256Mi \
    --limits=cpu=500m,memory=512Mi >/dev/null 2>&1 || true

# patch CoreDNS Corefile to add public forwarders as fallback (idempotent)
if kubectl -n kube-system get configmap coredns >/dev/null 2>&1; then
  CURRENT_COREFILE=$(kubectl -n kube-system get configmap coredns -o jsonpath='{.data.Corefile}' 2>/dev/null || true)
  if [ -n "${CURRENT_COREFILE}" ]; then
    if ! printf '%s' "${CURRENT_COREFILE}" | grep -q '8.8.8.8'; then
      # prefer replacing 'forward . /etc/resolv.conf' if present, otherwise append a fallback forward stanza
      if printf '%s' "${CURRENT_COREFILE}" | grep -q 'forward[[:space:]]\.[[:space:]]*/etc/resolv.conf'; then
        NEW_COREFILE=$(printf '%s' "${CURRENT_COREFILE}" | sed '0,/forward \. \/etc\/resolv.conf/ s/forward \. \/etc\/resolv.conf/forward . 8.8.8.8 8.8.4.4 {policy sequential}/')
      else
        # append fallback forwarders at end
        NEW_COREFILE="$(printf '%s\n\n# fallback forwarders added by lc.sh\nforward . 8.8.8.8 8.8.4.4 {policy sequential}\n' "${CURRENT_COREFILE}")"
      fi
      # apply updated ConfigMap (preserve name/namespace; this will overwrite data.Corefile only)
      cat <<EOF | kubectl -n kube-system apply -f - >/dev/null 2>&1 || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
$(printf '%s\n' "${NEW_COREFILE}" | sed 's/^/    /')
EOF
      kubectl -n kube-system rollout restart deployment coredns >/dev/null 2>&1 || true
      kubectl -n kube-system rollout status deployment coredns --timeout=120s >/dev/null 2>&1 || true
    else
      echo "[INFO] CoreDNS already configured with public forwarders; skipping patch."
    fi
  else
    echo "[WARN] CoreDNS ConfigMap present but Corefile empty -- skipping Corefile patch."
  fi
else
  echo "[WARN] CoreDNS ConfigMap not found yet; skipping Corefile patch."
fi

# Wait for critical networking pods (coredns, kindnet, kube-proxy) to be ready
echo "[INFO] Waiting for critical networking/system pods to be Ready..."
kubectl -n kube-system wait --for=condition=Available deployment/coredns --timeout=120s >/dev/null 2>&1 || true
kubectl -n kube-system get pods -l k8s-app=kube-dns -o wide || true
# wait for kindnet/kube-proxy (CNI) to be ready
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kindnet --timeout=120s 2>/dev/null || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kube-proxy --timeout=120s 2>/dev/null || true

# Optionally install monitoring (idempotent)
if [ "${INSTALL_MONITORING}" = "true" ]; then
  if ! helm repo list | grep -q "^${PROM_HELM_REPO}"; then
    helm repo add "${PROM_HELM_REPO}" https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
    helm repo update >/dev/null 2>&1 || true
  fi
  helm upgrade --install monitoring "${PROM_HELM_REPO}/${PROM_HELM_CHART}" \
    --namespace monitoring --create-namespace --wait --version "${PROM_HELM_CHART_VERSION}" >/dev/null 2>&1 || true
fi

# Host sysctls helpful for inotify-heavy workloads and file handles
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=1024
sudo sysctl -w fs.file-max=2097152
# persist across reboot (optional)
if ! grep -q '^fs.inotify.max_user_watches=524288' /etc/sysctl.conf 2>/dev/null; then
  echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf >/dev/null
fi
if ! grep -q '^fs.inotify.max_user_instances=1024' /etc/sysctl.conf 2>/dev/null; then
  echo "fs.inotify.max_user_instances=1024" | sudo tee -a /etc/sysctl.conf >/dev/null
fi
if ! grep -q '^fs.file-max=2097152' /etc/sysctl.conf 2>/dev/null; then
  echo "fs.file-max=2097152" | sudo tee -a /etc/sysctl.conf >/dev/null
fi

echo "[+] Updating sysctls inside kind nodes..."
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  echo "  - patching $node"
  docker exec "$node" sysctl -w fs.inotify.max_user_watches=524288 || true
  docker exec "$node" sysctl -w fs.inotify.max_user_instances=1024 || true
  docker exec "$node" sysctl -w fs.file-max=2097152 || true
done
echo "[+] Sysctl patching complete."

# pull required images into all kind node containers (preload)
CLUSTER_NAME=rag8s-local
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  echo "[INFO] Preloading images into ${node}"
  docker exec "${node}" ctr -n k8s.io images pull docker.io/qdrant/qdrant:v1.16.0
  docker exec "${node}" ctr -n k8s.io images pull docker.io/athithya5354/dense:amd64-arm64-v1
  docker exec "${node}" ctr -n k8s.io images pull docker.io/athithya5354/sparse:amd64-arm64-v2 
  docker exec "${node}" ctr -n k8s.io images pull docker.io/athithya5354/reranker:amd64-arm64-v1 
  docker exec "${node}" ctr -n k8s.io images pull docker.io/athithya5354/retrieval:amd64-arm64-v1 
  docker exec "${node}" ctr -n k8s.io images pull docker.io/athithya5354/frontend:amd64-arm64-v1 
  docker exec "${node}" ctr -n k8s.io images pull docker.io/athithya5354/indexing_pipeline_cpu:amd64-arm64-v7
done

echo "kind cluster ${CLUSTER_NAME} created (1 control-plane + 2 workers). Context: ${CONTEXT}"
kubectl get nodes -o wide

exit 0


python3 - <<'PY'
import os, sys, subprocess, textwrap, re
from pathlib import Path

REPO = os.getenv("REPO_URL", "https://github.com/Athithya-Sakthivel/RAG8s.git")
BR = os.getenv("BRANCH", "main")
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", "infra/manifests"))
EXCLUDE = os.getenv("EXCLUDE_DIR", "jobs")
FLUX_NS = os.getenv("FLUX_NS", "flux-system")
GITNAME = "rag8s"

def sanitize(n):
    return re.sub(r'[^a-z0-9-]', '-', n.lower()).strip('-')[:63]

if not MANIFEST_PATH.is_dir():
    print(f"Manifest path not found: {MANIFEST_PATH}", file=sys.stderr)
    sys.exit(1)

gitrepo = textwrap.dedent(f"""\
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: {GITNAME}
  namespace: {FLUX_NS}
spec:
  interval: 1m0s
  url: {REPO}
  ref:
    branch: {BR}
""")

r = subprocess.run("kubectl apply -f -", shell=True, input=gitrepo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if r.returncode != 0:
    sys.exit(1)

subprocess.run(f"kubectl wait gitrepository/{GITNAME} -n {FLUX_NS} --for=condition=Ready --timeout=60s", shell=True)
subprocess.run(f"flux reconcile source git {GITNAME} -n {FLUX_NS}", shell=True)

kustomizations = []
dirs = [d for d in sorted(MANIFEST_PATH.iterdir()) if d.is_dir() and d.name != EXCLUDE]
for d in dirs:
    name = sanitize(d.name)
    subprocess.run(f"kubectl create ns {name} --dry-run=client -o yaml | kubectl apply -f -", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ky = textwrap.dedent(f"""\
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: {GITNAME}-{name}
  namespace: {FLUX_NS}
spec:
  interval: 1m0s
  prune: true
  sourceRef:
    kind: GitRepository
    name: {GITNAME}
  path: ./{MANIFEST_PATH.as_posix()}/{d.name}
  targetNamespace: {name}
""")
    kustomizations.append(ky)

if kustomizations:
    allk = "\n---\n".join(kustomizations)
    r = subprocess.run("kubectl apply -f -", shell=True, input=allk, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.exit(1)
    for d in dirs:
        name = sanitize(d.name)
        subprocess.run(f"flux reconcile kustomization {GITNAME}-{name} -n {FLUX_NS}", shell=True)
        subprocess.run(f"kubectl wait kustomization/{GITNAME}-{name} -n {FLUX_NS} --for=condition=Ready --timeout=60s", shell=True)
PY
