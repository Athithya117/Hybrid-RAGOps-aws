#!/usr/bin/env bash
IFS=$'\n\t'

CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
KIND_VERSION="${KIND_VERSION:-v0.29.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)}"

# increased control-plane mem/cpu to reduce CoreDNS/API pressure
CONTROLPLANE_CONTAINER_MEMORY="${CONTROLPLANE_CONTAINER_MEMORY:-4g}"
CONTROLPLANE_CONTAINER_CPUS="${CONTROLPLANE_CONTAINER_CPUS:-3}"

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

# Delete existing cluster (idempotent)
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "[INFO] Deleting existing kind cluster '${CLUSTER_NAME}'"
  kind delete cluster --name "${CLUSTER_NAME}" || true
  for i in $(seq 1 30); do
    if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then break; fi
    sleep 1
  done
fi

echo "conntrack: current/ max (if available)"
sudo sysctl net.netfilter.nf_conntrack_count net.netfilter.nf_conntrack_max || true
echo "setting nf_conntrack_max=131072 (best-effort, requires sudo)"
sudo sysctl -w net.netfilter.nf_conntrack_max=131072 || true


# Create cluster (single control-plane only)
echo "[INFO] Creating cluster '${CLUSTER_NAME}' (single control-plane)"
cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 30080
EOF

# Tune docker container resources (best-effort) - only control-plane will exist
for container in $(docker ps --filter "name=kind-${CLUSTER_NAME}-" --format '{{.Names}}'); do
  if [[ "${container}" == *"-control-plane" ]]; then
    docker update --memory "${CONTROLPLANE_CONTAINER_MEMORY}" --cpus "${CONTROLPLANE_CONTAINER_CPUS}" "${container}" >/dev/null 2>&1 || true
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

# ---- Networking convenience for dev: ensure DNS & egress won't get accidentally blocked ----
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
      if printf '%s' "${CURRENT_COREFILE}" | grep -q 'forward[[:space:]]\.[[:space:]]*/etc/resolv.conf'; then
        NEW_COREFILE=$(printf '%s' "${CURRENT_COREFILE}" | sed '0,/forward \. \/etc\/resolv.conf/ s/forward \. \/etc\/resolv.conf/forward . 8.8.8.8 8.8.4.4 {policy sequential}/')
      else
        NEW_COREFILE="$(printf '%s\n\n# fallback forwarders added by lc.sh\nforward . 8.8.8.8 8.8.4.4 {policy sequential}\n' "${CURRENT_COREFILE}")"
      fi
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
    fi
  fi
fi

# Wait for critical networking pods (coredns, kindnet, kube-proxy) to be ready
echo "[INFO] Waiting for critical networking/system pods to be Ready..."
kubectl -n kube-system wait --for=condition=Available deployment/coredns --timeout=120s >/dev/null 2>&1 || true
kubectl -n kube-system get pods -l k8s-app=kube-dns -o wide || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kindnet --timeout=120s 2>/dev/null || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kube-proxy --timeout=120s 2>/dev/null || true

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

echo "[+] Updating sysctls inside kind node (best-effort)"
for node in $(kind get nodes --name "${CLUSTER_NAME}" 2>/dev/null || true); do
  echo "  - patching ${node}"
  docker exec "${node}" sh -c "cat > /etc/sysctl.d/99-kind-fd.conf <<'EOFD'
# increased inotify + file handle limits for indexing workloads (added by host script)
fs.inotify.max_user_watches=${INOTIFY_WATCHES}
fs.inotify.max_user_instances=${INOTIFY_INSTANCES}
fs.file-max=${FILE_MAX}
EOFD" || echo "    [WARN] could not write /etc/sysctl.d/99-kind-fd.conf in ${node}"

  docker exec "${node}" sysctl -w fs.inotify.max_user_watches="${INOTIFY_WATCHES}" >/dev/null 2>&1 || true
  docker exec "${node}" sysctl -w fs.inotify.max_user_instances="${INOTIFY_INSTANCES}" >/dev/null 2>&1 || true
  docker exec "${node}" sysctl -w fs.file-max="${FILE_MAX}" >/dev/null 2>&1 || true
done
echo "[+] Sysctl patching complete."

echo "[INFO] Waiting for cluster to become stable before preloading images..."
kubectl -n kube-system wait --for=condition=Available deployment/coredns --timeout=120s || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kube-proxy --timeout=120s || true
kubectl -n kube-system wait --for=condition=Ready pods -l k8s-app=kindnet --timeout=120s || true



CLUSTER_NAME="${CLUSTER_NAME:-rag8s-local}"
echo "[INFO] Preloading images safely (post-bootstrap)..."
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  echo "  → Loading into ${node}"
  docker exec "$node" bash -c 'ctr version >/dev/null 2>&1' || { echo "    [WARN] containerd not ready on $node — skipping"; continue; }
  for IMAGE in \
    docker.io/qdrant/qdrant:v1.16.0
  do
    echo "    checking $IMAGE on $node..."
    docker exec "$node" sh -c "ctr -n k8s.io images ls | grep -q -- '$IMAGE'" >/dev/null 2>&1 || {
      echo "    pulling $IMAGE..."
      docker exec "$node" ctr -n k8s.io images pull "$IMAGE" || echo "    [WARN] failed pulling $IMAGE on $node"
    }
  done
done


echo "[INFO] Safe preload complete."
echo "kind cluster ${CLUSTER_NAME} created (1 control-plane). Context: ${CONTEXT}"
kubectl get nodes -o wide

exit 0
