#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Logging
log() { echo -e "[+] $1"; }
err() { echo -e "[!] $1" >&2; }

# Ensure required tools
command -v gh >/dev/null 2>&1 || { err "Missing gh CLI. Install with: apt install -y gh"; exit 1; }
command -v git >/dev/null 2>&1 || { err "Missing Git. Install with: apt install -y git"; exit 1; }

# Accept inputs or fallback to env
GH_USERNAME="${1:-${GH_USERNAME:-}}"
GH_EMAIL="${2:-${GH_EMAIL:-}}"
GH_PAT="${3:-${GH_PAT:-}}"

if [[ -z "$GH_USERNAME" ]]; then
  read -rp "Enter your GitHub username: " GH_USERNAME
fi
if [[ -z "$GH_EMAIL" ]]; then
  read -rp "Enter your GitHub email: " GH_EMAIL
fi
if [[ -z "$GH_PAT" ]]; then
  echo "👉 If you don't have a PAT, generate one at:"
  echo "   https://github.com/settings/tokens/new"
  read -rsp "Enter your GitHub Personal Access Token (PAT): " GH_PAT
  echo ""
fi

# Idempotent logout (no '-s' flag)
if gh auth status &>/dev/null; then
  log "Logging out existing GitHub session..."
  gh auth logout --hostname github.com || true
fi

# Authenticate via PAT
echo "$GH_PAT" | gh auth login --with-token
log "GitHub CLI authenticated via PAT."

# Configure Git
git config --global user.name "$GH_USERNAME"
git config --global user.email "$GH_EMAIL"
git config --global credential.helper store

log "Git config updated: $GH_USERNAME <$GH_EMAIL>"

# Persist HTTPS credentials
echo "https://${GH_USERNAME}:${GH_PAT}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

log "GitHub credentials stored in ~/.git-credentials"
log "GitHub login and config complete."
