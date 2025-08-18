#!/usr/bin/env bash
# set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export KUBECONFIG=$HOME/.kube/config

echo "[1/5] Updating system and installing dependencies..."
sudo apt update && sudo apt install -y curl git docker.io docker-compose apt-transport-https ca-certificates gnupg lsb-release

echo "[2/5] Configuring Docker..."
sudo systemctl enable docker || true
sudo usermod -aG docker $USER || true

echo "[3/5] Installing kubectl and Helm..."
curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

echo "[4/5] Cloning vLLM Production Stack..."
if [ ! -d production-stack ]; then
  git clone https://github.com/vllm-project/production-stack.git
fi
cd production-stack

echo "[5/5] Deploying vLLM Production Stack on kind (CPU-only tiny model)..."
helm repo add vllm https://vllm-project.github.io/production-stack
cat <<EOF > cpu-values.yaml
servingEngineSpec:
  runtimeClassName: ""
  modelSpec:
  - name: "opt125m"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "facebook/opt-125m"
    replicaCount: 1
    requestCPU: 6
    requestMemory: "8Gi"
    requestGPU: 0
    pvcStorage: "10Gi"
EOF

helm install vllm vllm/vllm-stack -f cpu-values.yaml

echo ">>> Deployment submitted. Checking pods..."
kubectl get pods -n default

echo ">>> When pods are READY, run:"
echo "kubectl port-forward svc/vllm-router-service 30080:80"
echo "Then test with:"
echo 'curl -X POST http://localhost:30080/completions -H "Content-Type: application/json" -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"Hello\",\"max_tokens\":10}"'
