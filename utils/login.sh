#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Logging helpers
log() { echo -e "[+] $1"; }
err() { echo -e "[!] $1" >&2; }

# Ensure tools
command -v gh >/dev/null 2>&1 || { err "Missing gh CLI. Install with: apt install -y gh"; exit 1; }
command -v git >/dev/null 2>&1 || { err "Missing Git. Install with: apt install -y git"; exit 1; }

# Prompt if env vars not set
[[ -z "${GH_USERNAME:-}" ]] && read -rp "Enter your GitHub username: " GH_USERNAME
[[ -z "${GH_EMAIL:-}" ]] && read -rp "Enter your GitHub email: " GH_EMAIL
[[ -z "${GH_PAT:-}" ]] && {
  echo "👉 If you don't have a PAT, generate one at:"
  echo "   https://github.com/settings/tokens/new"
  read -rsp "Enter your GitHub Personal Access Token (PAT): " GH_PAT
  echo ""
}

# Export to env for current session (optional)
export GH_USERNAME
export GH_EMAIL
export GH_PAT

log "Exported GH_USERNAME, GH_EMAIL, GH_PAT as environment variables."

# Logout existing GitHub CLI session if any
if gh auth status --hostname github.com &>/dev/null; then
  log "Logging out previous GitHub CLI session..."
  echo | gh auth logout --hostname github.com --yes || true
fi

# Login using PAT (non-interactive)
log "Logging in GitHub CLI with PAT..."
echo "$GH_PAT" | gh auth login --with-token

# Set git global configs
git config --global user.name "$GH_USERNAME"
git config --global user.email "$GH_EMAIL"
git config --global credential.helper store

# Store GitHub credentials for HTTPS push
echo "https://${GH_USERNAME}:${GH_PAT}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

log "GitHub CLI authenticated and Git config complete."
