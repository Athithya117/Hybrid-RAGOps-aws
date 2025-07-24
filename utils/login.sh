#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { echo "[+] $1"; }
err() { echo "[!] $1" >&2; }

# Ensure dependencies
command -v gh >/dev/null 2>&1 || { err "Missing gh CLI. Install with: apt install -y gh"; exit 1; }
command -v git >/dev/null 2>&1 || { err "Missing Git. Install with: apt install -y git"; exit 1; }

# Prompt if environment variables not set
[[ -z "${GH_USERNAME:-}" ]] && read -rp "Enter your GitHub username: " GH_USERNAME
[[ -z "${GH_EMAIL:-}" ]] && read -rp "Enter your GitHub email: " GH_EMAIL
if [[ -z "${GH_PAT:-}" ]]; then
  echo "👉 Generate a PAT with repo scope at:"
  echo "   https://github.com/settings/tokens/new"
  read -rsp "Enter your GitHub Personal Access Token (PAT): " GH_PAT
  echo ""
fi

export GH_USERNAME GH_EMAIL GH_PAT
log "Environment variables set."

# Logout current session if present (no --yes flag) :contentReference[oaicite:1]{index=1}
if gh auth status --hostname github.com &>/dev/null; then
  log "Logging out previous GitHub CLI session..."
  echo | gh auth logout --hostname github.com || true
fi

# Non-interactive login via PAT :contentReference[oaicite:2]{index=2}
log "Logging in GitHub CLI with PAT..."
echo "$GH_PAT" | gh auth login --with-token --hostname github.com

# Configure Git for push over HTTPS
git config --global user.name "$GH_USERNAME"
git config --global user.email "$GH_EMAIL"
git config --global credential.helper store

# Store HTTPS credentials
echo "https://${GH_USERNAME}:${GH_PAT}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

log "✅ GitHub login and Git config complete."
