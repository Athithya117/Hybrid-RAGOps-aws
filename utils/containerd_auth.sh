#!/usr/bin/env bash
# scripts/setup_containerd_ghcr_auth.sh
set -euo pipefail

GHCR_USER="${GITHUB_USER:?Set GITHUB_USER}"
GHCR_PAT="${GHCR_PAT:?Set GHCR_PAT}"

CONFIG_PATH="/etc/containerd/config.toml"

sudo mkdir -p /etc/containerd

if ! grep -q '\[plugins."io.containerd.grpc.v1.cri".registry.configs."ghcr.io".auth\]' "$CONFIG_PATH"; then
  echo "[INFO] Updating containerd config for GHCR auth..."

  sudo tee -a "$CONFIG_PATH" > /dev/null <<EOF

[plugins."io.containerd.grpc.v1.cri".registry.configs."ghcr.io".auth]
  username = "${GHCR_USER}"
  password = "${GHCR_PAT}"
EOF

  sudo systemctl restart containerd
else
  echo "[INFO] GHCR auth already configured"
fi
