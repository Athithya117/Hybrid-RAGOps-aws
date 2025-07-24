#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { echo "[+] $1"; }
err() { echo "[!] $1" >&2; }

# Ensure dependencies
command -v gh >/dev/null 2>&1 || { err "Install: apt install -y gh"; exit 1; }
command -v git >/dev/null 2>&1 || { err "Install: apt install -y git"; exit 1; }

# Prompt or use env vars
[[ -z "${GH_USERNAME:-}" ]] && read -rp "GitHub username: " GH_USERNAME
[[ -z "${GH_EMAIL:-}" ]] && read -rp "GitHub email: " GH_EMAIL

# The token stored in GH_TOKEN env is used by gh automatically.
if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "👉 Generate a PAT with repo scope: https://github.com/settings/tokens/new"
  read -rsp "Enter your GitHub PAT: " TOKEN
  echo ""
  export GH_TOKEN="$TOKEN"
fi

log "Using GH_TOKEN for authentication"

# Configure Git via gh credential helper :contentReference[oaicite:7]{index=7}
git config --global credential.https://github.com.helper ""
git config --global credential.https://github.com.helper "!$(which gh) auth git-credential"

# Git configs
git config --global user.name "$GH_USERNAME"
git config --global user.email "$GH_EMAIL"

log "✅ GitHub CLI and Git configured for non‑interactive PAT use"
