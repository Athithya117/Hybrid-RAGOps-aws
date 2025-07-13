#!/usr/bin/env bash

IFS=$'\n\t'

# Logging
log() { echo -e "\e[32m[+]\e[0m $1"; }
err() { echo -e "\e[31m[!]\e[0m $1" >&2; }

# Ensure required tools
command -v gh >/dev/null 2>&1 || { err "Missing gh CLI. Install with: sudo apt install gh"; exit 1; }
command -v git >/dev/null 2>&1 || { err "Missing Git. Install with: sudo apt install git"; exit 1; }

# Accept inputs or prompt
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

# Always logout first to ensure idempotency
if gh auth status &>/dev/null; then
  log "Logging out existing GitHub session..."
  gh auth logout -h github.com -s all || true
fi

# Authenticate via GitHub CLI
echo "$GH_PAT" | gh auth login --with-token || { err "GitHub CLI login failed."; exit 1; }
log "GitHub CLI authenticated."

# Git global config
git config --global user.name "$GH_USERNAME"
git config --global user.email "$GH_EMAIL"
git config --global credential.helper store

log "Git config updated: $GH_USERNAME <$GH_EMAIL>"

# Persist HTTPS credentials
cat <<EOF > ~/.git-credentials
https://$GH_USERNAME:$GH_PAT@github.com
EOF

chmod 600 ~/.git-credentials
log "GitHub credentials stored in ~/.git-credentials."

log " GitHub login and config complete."
