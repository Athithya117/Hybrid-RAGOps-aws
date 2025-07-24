#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

log() { echo "[+] $1"; }
err() { echo "[!] $1" >&2; }

# Check dependencies
command -v gh >/dev/null || { err "Install: apt install -y gh"; exit 1; }
command -v git >/dev/null || { err "Install: apt install -y git"; exit 1; }

# Ask user input if not set
read -rp "Enter your GitHub username: " GH_USERNAME
read -rp "Enter your GitHub email: " GH_EMAIL
echo "👉 Generate a PAT with repo scope: https://github.com/settings/tokens/new"
read -rsp "Enter your GitHub Personal Access Token (PAT): " GH_TOKEN && echo

# Export to current session
export GH_USERNAME GH_EMAIL GH_TOKEN
log "Exported GH_USERNAME, GH_EMAIL, GH_TOKEN as environment variables."

# Authenticate headlessly
log "Logging in GitHub CLI with PAT..."
echo "$GH_TOKEN" | gh auth login --with-token --hostname github.com

# Setup Git to use gh for credentials
gh auth setup-git --hostname github.com

# Configure Git identity
git config --global user.name "$GH_USERNAME"
git config --global user.email "$GH_EMAIL"

log "✅ GitHub CLI authenticated and Git config complete."
