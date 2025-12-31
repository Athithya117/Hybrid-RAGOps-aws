#!/usr/bin/env bash
K3D_VER="v5.8.3"
K3S_IMAGE="rancher/k3s:v1.31.5-k3s1"
CLUSTER="mycluster"
curl -sL "https://github.com/k3d-io/k3d/releases/download/${K3D_VER}/k3d-linux-amd64" -o /usr/local/bin/k3d
chmod +x /usr/local/bin/k3d
k3d --version
k3d cluster delete "${CLUSTER}" >/dev/null 2>&1 || true
k3d cluster create "${CLUSTER}" --servers 1 --agents 0 --image "${K3S_IMAGE}" --wait
mkdir -p ~/.kube
k3d kubeconfig get "${CLUSTER}" > ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG="${HOME}/.kube/config"
for i in $(seq 1 30); do
  if kubectl get nodes --no-headers 2>/dev/null | awk '{print $2}' | grep -q '^Ready$'; then break; fi
  sleep 2
done
kubectl get nodes

