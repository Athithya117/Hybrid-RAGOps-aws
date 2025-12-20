#!/usr/bin/env bash
IFS=$'\n\t'

CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
KIND_VERSION="${KIND_VERSION:-v0.29.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)}"
# increased control-plane mem/cpu to reduce CoreDNS/API pressure
CONTROLPLANE_CONTAINER_MEMORY="${CONTROLPLANE_CONTAINER_MEMORY:-4g}"
CONTROLPLANE_CONTAINER_CPUS="${CONTROLPLANE_CONTAINER_CPUS:-3}"
WORKER_CONTAINER_MEMORY="${WORKER_CONTAINER_MEMORY:-2.5g}"
WORKER_CONTAINER_CPUS="${WORKER_CONTAINER_CPUS:-3}"
INSTALL_MONITORING="${INSTALL_MONITORING:-false}"
PROM_HELM_REPO="${PROM_HELM_REPO:-prometheus-community}"
PROM_HELM_CHART="${PROM_HELM_CHART:-kube-prometheus-stack}"
PROM_HELM_CHART_VERSION="${PROM_HELM_CHART_VERSION:-79.5.0}"

# SYSCTL TARGET VALUES (tweak here if needed)
INOTIFY_WATCHES="${INOTIFY_WATCHES:-524288}"
INOTIFY_INSTANCES="${INOTIFY_INSTANCES:-1024}"
FILE_MAX="${FILE_MAX:-2097152}"

mkdir -p "${LOCAL_BIN}"
export PATH="${LOCAL_BIN}:$PATH"

command_exists(){ command -v "$1" >/dev/null 2>&1; }

# Basic host prechecks
if ! command_exists curl; then echo "curl required" >&2; exit 1; fi
if ! command_exists docker; then echo "docker required" >&2; exit 1; fi
if ! docker info >/dev/null 2>&1; then echo "docker daemon not running or inaccessible" >&2; exit 1; fi
if ! command_exists sudo; then echo "sudo required" >&2; exit 1; fi

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

# --------------------------
# Host sysctls (persistent)
# --------------------------
echo "[INFO] Setting host kernel limits persistently under /etc/sysctl.d/99-k8s-fd.conf (requires sudo)"
SYSCTL_FILE="/etc/sysctl.d/99-k8s-fd.conf"
TMP="$(mktemp)"
cat > "${TMP}" <<EOF
# increased inotify + file handle limits for indexing workloads (managed by create_kind_cluster.sh)
fs.inotify.max_user_watches=${INOTIFY_WATCHES}
fs.inotify.max_user_instances=${INOTIFY_INSTANCES}
fs.file-max=${FILE_MAX}
EOF

# Only overwrite if different (idempotent)
if ! sudo sh -c "cmp -s ${TMP} ${SYSCTL_FILE} >/dev/null 2>&1"; then
  sudo install -m 0644 "${TMP}" "${SYSCTL_FILE}"
  echo "[INFO] wrote ${SYSCTL_FILE}"
else
  echo "[INFO] ${SYSCTL_FILE} already up-to-date"
fi
rm -f "${TMP}"

# apply immediately
echo "[INFO] Applying kernel settings (sudo sysctl --system)"
sudo sysctl --system >/dev/null 2>&1 || {
  echo "[WARN] 'sysctl --system' failed or produced warnings; attempting direct sysctl -w"
  sudo sysctl -w fs.inotify.max_user_watches="${INOTIFY_WATCHES}" || true
  sudo sysctl -w fs.inotify.max_user_instances="${INOTIFY_INSTANCES}" || true
  sudo sysctl -w fs.file-max="${FILE_MAX}" || true
}

echo "[+] Updating sysctls inside kind nodes (best-effort)"
for node in $(kind get nodes --name "${CLUSTER_NAME}" 2>/dev/null || true); do
  echo "  - patching ${node}"
  # write a sysctl.d fragment inside the node (idempotent overwrite)
  docker exec "${node}" sh -c "cat > /etc/sysctl.d/99-kind-fd.conf <<'EOFD'
# increased inotify + file handle limits for indexing workloads (added by host script)
fs.inotify.max_user_watches=${INOTIFY_WATCHES}
fs.inotify.max_user_instances=${INOTIFY_INSTANCES}
fs.file-max=${FILE_MAX}
EOFD" || echo "    [WARN] could not write /etc/sysctl.d/99-kind-fd.conf in ${node}"

  # apply via sysctl -w (works even if sysctl --system not available in container)
  docker exec "${node}" sysctl -w fs.inotify.max_user_watches="${INOTIFY_WATCHES}" >/dev/null 2>&1 || true
  docker exec "${node}" sysctl -w fs.inotify.max_user_instances="${INOTIFY_INSTANCES}" >/dev/null 2>&1 || true
  docker exec "${node}" sysctl -w fs.file-max="${FILE_MAX}" >/dev/null 2>&1 || true

  # also try running sysctl --system inside node (if present)
  docker exec "${node}" sh -c "if command -v sysctl >/dev/null 2>&1 && command -v run-parts >/dev/null 2>&1; then sysctl --system >/dev/null 2>&1 || true; fi" || true
done
echo "[+] Sysctl patching complete."

echo "[INFO] Waiting for cluster to become stable before preloading images..."
kubectl -n kube-system wait --for=condition=Available deployment/coredns --timeout=120s || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kube-proxy --timeout=120s || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kindnet --timeout=120s || true


echo "[INFO] Safe preload complete."

echo "kind cluster ${CLUSTER_NAME} created (1 control-plane + 2 workers). Context: ${CONTEXT}"

kubectl get nodes -o wide

exit 0
