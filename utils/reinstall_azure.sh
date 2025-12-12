#!/usr/bin/env bash
set -euo pipefail

# ===============================
# CONFIG — pin your version here
# ===============================
AZ_CLI_VERSION="${AZ_CLI_VERSION:-2.81.0-1~jammy}"

echo ">>> Installing Azure CLI version: ${AZ_CLI_VERSION}"

# --- Ensure required tools ---
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends curl gnupg ca-certificates

# --- Add Microsoft keyring (idempotent) ---
sudo mkdir -p /usr/share/keyrings
curl -sL https://packages.microsoft.com/keys/microsoft.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg

# --- Add azure-cli repo (idempotent, overwrite allowed) ---
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] \
https://packages.microsoft.com/repos/azure-cli/ jammy main" \
  | sudo tee /etc/apt/sources.list.d/azure-cli.list >/dev/null

# --- Refresh package lists ---
sudo apt-get update -y

# --- Install pinned Azure CLI ---
sudo apt-get install -y --allow-downgrades azure-cli="${AZ_CLI_VERSION}"

# --- Freeze version (prevent future upgrades) ---
sudo apt-mark hold azure-cli

# --- Disable auto-update and dynamic extension install ---
if command -v az >/dev/null 2>&1; then
  az config set auto_update.disable=true >/dev/null || true
  az config set extension.use_dynamic_install=no >/dev/null || true
  echo ">>> Azure CLI installed successfully:"
  az version | sed -n "1,3p"
else
  echo "ERROR: Azure CLI installation failed" >&2
  exit 1
fi

echo ">>> Done."
