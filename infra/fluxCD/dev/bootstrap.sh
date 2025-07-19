#!/bin/bash
set -euo pipefail

echo "=== Prerequisites check and installation ==="

# Install sops if not installed
if ! command -v sops &> /dev/null; then
  echo "Installing sops..."
  curl -LO https://github.com/getsops/sops/releases/download/v3.8.1/sops-v3.8.1.linux.amd64
  chmod +x sops-v3.8.1.linux.amd64
  sudo mv sops-v3.8.1.linux.amd64 /usr/local/bin/sops
  echo "sops installed"
else
  echo "sops already installed"
fi

# Check flux CLI
if ! command -v flux &> /dev/null; then
  echo "Error: flux CLI not found. Please install flux: https://fluxcd.io/docs/installation/"
  exit 1
fi

# Check kubectl CLI
if ! command -v kubectl &> /dev/null; then
  echo "Error: kubectl CLI not found. Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
  exit 1
fi

# Check vault CLI
if ! command -v vault &> /dev/null; then
  echo "Error: vault CLI not found. Please install vault: https://www.vaultproject.io/downloads"
  exit 1
fi

echo "All prerequisites satisfied."

# === User Inputs - fill these before running ===
export GITHUB_TOKEN="your_github_token"
export GITHUB_USER="your_github_username"
export REPO="RAG8s"
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN="your_vault_token"             # Vault token with transit encrypt/decrypt rights
export VAULT_TRANSIT_KEY="flux-sops-key"          # Vault transit key name
export FLUX_NAMESPACE="flux-system"
export FLUX_PATH="infra/fluxCD/dev"



# Export Vault token env var for sops and vault CLI usage
export VAULT_TOKEN

echo "=== Creating .sops.yaml for Vault transit KMS integration ==="

cat > .sops.yaml <<EOF
creation_rules:
  - path_regex: '${FLUX_PATH}/02-secrets.yaml'
    vault_transit:
      address: "${VAULT_ADDR}"
      mount: "transit"
      key: "${VAULT_TRANSIT_KEY}"
      token_env: "VAULT_TOKEN"
    encrypted_regex: '^(data|stringData)$'
EOF

echo ".sops.yaml created."

# Encrypt the secrets file (must exist and be unencrypted!)
if [ ! -f "${FLUX_PATH}/02-secrets.yaml" ]; then
  echo "Error: ${FLUX_PATH}/02-secrets.yaml not found!"
  exit 1
fi

echo "Encrypting secrets file ${FLUX_PATH}/02-secrets.yaml using sops + Vault transit..."
sops -e --output "${FLUX_PATH}/02-secrets.yaml" "${FLUX_PATH}/02-secrets.yaml"
echo "Secrets encrypted."

echo "=== Bootstrapping Flux with GitHub repo ==="

flux bootstrap github \
  --owner="$GITHUB_USER" \
  --repository="$REPO" \
  --branch=main \
  --path="$FLUX_PATH" \
  --personal \
  --token-auth

echo "Flux bootstrap completed."

echo "=== Creating Kubernetes secret for Vault token ==="

kubectl -n $FLUX_NAMESPACE delete secret vault-token --ignore-not-found
kubectl -n $FLUX_NAMESPACE create secret generic vault-token --from-literal=token="$VAULT_TOKEN"

echo "Vault token secret created in namespace $FLUX_NAMESPACE."

echo "=== Applying Flux Kustomization with sops decryption config ==="

cat > infra/fluxCD/flux-system/dev-kustomization.yaml <<EOF
apiVersion: kustomize.toolkit.fluxcd.io/v1beta1
kind: Kustomization
metadata:
  name: dev
  namespace: $FLUX_NAMESPACE
spec:
  interval: 1m
  path: ./dev
  prune: true
  sourceRef:
    kind: GitRepository
    name: flux-system
  targetNamespace: default
  timeout: 2m
  decryption:
    provider: sops
    secretRef:
      name: vault-token
      namespace: $FLUX_NAMESPACE
EOF

kubectl apply -f infra/fluxCD/flux-system/dev-kustomization.yaml

echo "Flux Kustomization applied."

echo "=== Waiting for Flux reconciliation (up to 5 mins) ==="
for i in {1..30}; do
  status=$(kubectl get kustomization dev -n $FLUX_NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
  if [[ "$status" == "True" ]]; then
    echo "Flux Kustomization reconciled successfully."
    exit 0
  fi
  echo "Waiting... ($i/30)"
  sleep 10
done

echo "Timeout waiting for Flux reconciliation."
kubectl get kustomization dev -n $FLUX_NAMESPACE
exit 1
