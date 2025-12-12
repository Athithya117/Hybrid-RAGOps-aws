#!/usr/bin/env bash
# Minimal, deterministic Pulumi bootstrap for infra/pulumi_azure
set -euo pipefail
IFS=$'\n\t'

# --- Defaults / minimal exports (overridable by env)
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-mypassword}"
export PROJECT_DIR="${PROJECT_DIR:-infra/pulumi_azure}"
export PULUMI_STACK="${PULUMI_STACK:-staging}"
export PULUMI_AZ_CONTAINER="${PULUMI_AZ_CONTAINER:-pulumi-state}"

# export minimal required envs (adjust per env)
export AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
export AZURE_TENANT_ID="${AZURE_TENANT_ID:-}"
export AZURE_LOCATION="${AZURE_LOCATION:-centralindia}"
export AZURE_RESOURCE_GROUP_NAME="${AZURE_RESOURCE_GROUP_NAME:-rg-rag-prod}"
export AZURE_STORAGE_ACCOUNT_NAME="${AZURE_STORAGE_ACCOUNT_NAME:-storeragprod42}"
export AKS_K8S_VERSION="1.29"

export AZ_CLI_BIN="${AZ_CLI_BIN:-az}"
export PULUMI_BIN="${PULUMI_BIN:-pulumi}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export REQ_FILE="${REQ_FILE:-${PROJECT_DIR}/requirements.txt}"

# Helper
log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found"; }

# Fast checks
require_cmd "$AZ_CLI_BIN"
require_cmd "$PYTHON_BIN"

# Validate required Azure envs (fail fast)
: "${AZURE_SUBSCRIPTION_ID:?Please export AZURE_SUBSCRIPTION_ID}"
: "${AZURE_TENANT_ID:?Please export AZURE_TENANT_ID}"
: "${AZURE_STORAGE_ACCOUNT_NAME:?Please export AZURE_STORAGE_ACCOUNT_NAME}"
: "${AZURE_RESOURCE_GROUP_NAME:?Please export AZURE_RESOURCE_GROUP_NAME}"

# Locate project dir
PROJECT_DIR="$(cd "$PROJECT_DIR" 2>/dev/null && pwd || die "project dir $PROJECT_DIR not found")"
REQ_FILE="$(cd "$(dirname "$REQ_FILE")" >/dev/null 2>&1 && printf '%s' "$REQ_FILE" || die "requirements path invalid")"

# Ensure requirements file exists (per policy: do not create it)
if [ ! -f "$REQ_FILE" ]; then
  die "requirements.txt missing at $REQ_FILE — per policy, script will not create it. Add it and re-run."
fi

# Ensure Pulumi.yaml & program exist
if [ ! -f "${PROJECT_DIR}/Pulumi.yaml" ]; then
  die "Pulumi.yaml missing in ${PROJECT_DIR}. Add it and re-run."
fi

# Ensure storage container existence (do not create containers)
container_exists() {
  local sa="$1" c="$2"
  # use az CLI auth-mode login; if permission problems this will fail
  if $AZ_CLI_BIN storage container show -n "$c" --account-name "$sa" --auth-mode login >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

for c in "${PULUMI_AZ_CONTAINER}" ; do
  if ! container_exists "$AZURE_STORAGE_ACCOUNT_NAME" "$c"; then
    die "Required container '$c' not found in storage account '$AZURE_STORAGE_ACCOUNT_NAME'. Per policy, will not create it."
  fi
done

# Create venv and install deps
VENV_DIR="${PROJECT_DIR}/.venv-pulumi"
if [ ! -d "$VENV_DIR" ]; then
  log "creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || die "venv creation failed"
fi
# activate
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
log "installing python deps from $REQ_FILE"
python -m pip install -r "$REQ_FILE" >/dev/null 2>&1 || die "pip install failed; inspect ${VENV_DIR}/.pulumi-logs or run manually"

# Ensure pulumi CLI
if ! command -v "$PULUMI_BIN" >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    log "installing pulumi CLI"
    curl -fsSL https://get.pulumi.com | sh -s -- --quiet || die "pulumi install failed"
    export PATH="$HOME/.pulumi/bin:${PATH}"
  else
    die "pulumi not found and curl not available to install"
  fi
fi

# Pulumi backend login (azblob)
PULUMI_BACKEND="azblob://${PULUMI_AZ_CONTAINER}?storage_account=${AZURE_STORAGE_ACCOUNT_NAME}"
log "logging into pulumi backend $PULUMI_BACKEND"
$PULUMI_BIN login "$PULUMI_BACKEND" >/dev/null 2>&1 || die "pulumi login failed (ensure az login & permissions)"

# Work from project dir for all pulumi commands (deterministic)
pushd "$PROJECT_DIR" >/dev/null

# Ensure stack exists: select or init (non-interactive)
if $PULUMI_BIN stack select "$PULUMI_STACK" --non-interactive >/dev/null 2>&1; then
  log "pulumi: selected stack $PULUMI_STACK"
else
  log "pulumi: creating stack $PULUMI_STACK"
  # use --yes to avoid interactive prompts
  $PULUMI_BIN stack init --stack "$PULUMI_STACK" --yes >/dev/null 2>&1 || {
    # final fallback: try interactive init for clearer error to user
    log "pulumi: non-interactive init failed; attempting interactive init so you see the error"
    $PULUMI_BIN stack init --stack "$PULUMI_STACK" || die "pulumi: stack init failed"
  }
  log "pulumi: stack $PULUMI_STACK created"
fi

# Optional: preview then up
if [ "${1:-}" = "--preview" ]; then
  $PULUMI_BIN preview --non-interactive --diff || die "pulumi preview failed"
  popd >/dev/null
  exit 0
fi

log "running pulumi up (non-interactive)"
$PULUMI_BIN up --yes --non-interactive || die "pulumi up failed"

# capture outputs
log "exporting pulumi outputs to ../pulumi-outputs.json"
$PULUMI_BIN stack output --json > ../pulumi-outputs.json || true

popd >/dev/null
log "done"
exit 0
