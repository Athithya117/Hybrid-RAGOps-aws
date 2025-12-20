#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

export AZURE_RESOURCE_GROUP_NAME="rg-rag-prod"
export AZURE_LOCATION="centralindia"
export AZURE_STORAGE_ACCOUNT_NAME="storeragprod42"
export AZURE_CONTAINER="rag-data-prod"
export PULUMI_AZ_CONTAINER="pulumi-state"
export FLOW_LOG_CONTAINER="flow-logs"
export BACKUP_AZ_CONTAINER="backups"
export AZURE_DELETE_ACCOUNT="0"
export FORCE_DELETE="1"
export AZ_CLI_BIN="${AZ_CLI_BIN:-az}"
export PULUMI_BIN="${PULUMI_BIN:-pulumi}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PROJECT_DIR="${PROJECT_DIR:-infra/pulumi_azure}"
export REQ_FILE="${REQ_FILE:-${PROJECT_DIR}/requirements.txt}"
export PULUMI_STACK="${PULUMI_STACK:-staging}"
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-}"
export PULUMI_STATE_CONTAINER="${PULUMI_STATE_CONTAINER:-pulumi-state}"
export PULUMI_STORAGE_ACCOUNT_NAME="${PULUMI_STORAGE_ACCOUNT_NAME:-pulumistate${ENV:-staging}01}"
export FORCE="${FORCE:-0}"
export ENV="${ENV:-staging}"
export ARM_USE_AZURE_CLI="${ARM_USE_AZURE_CLI:-true}"
export ARM_SUBSCRIPTION_ID="${ARM_SUBSCRIPTION_ID:-${AZURE_SUBSCRIPTION_ID:-}}"
export ARM_TENANT_ID="${ARM_TENANT_ID:-${AZURE_TENANT_ID:-}}"
export AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID must already be exported}"
export AZURE_TENANT_ID="${AZURE_TENANT_ID:?AZURE_TENANT_ID must already be exported}"


export RESOURCE_NAME_PREFIX=rag
export AZURE_LOCATION=eastus
export VNET_CIDR=10.1.0.0/16
export AKS_SUBNET_PREFIX=10.1.1.0/24
export APPGW_SUBNET_PREFIX=10.1.2.0/24
export CREATE_NAT=false
export FRONTEND_HOSTNAME="ui.athithya.site"

export CLOUDFLARE_CREATE_K8S="true"
export CLOUDFLARE_IMAGE="cloudflare/cloudflared:2025.11.1"
export CLOUDFLARE_TUNNEL_REPLICAS="2"


log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found"; }
require_cmd "$AZ_CLI_BIN"
require_cmd "$PYTHON_BIN"
require_cmd "$PULUMI_BIN"
require_cmd jq
PROJECT_DIR="$(cd "$PROJECT_DIR" 2>/dev/null && pwd || die "project dir $PROJECT_DIR not found")"
REQ_FILE="$(cd "$(dirname "$REQ_FILE")" >/dev/null 2>&1 && printf '%s' "$REQ_FILE" || die "requirements path invalid")"
[ -f "$REQ_FILE" ] || die "requirements.txt missing at $REQ_FILE"
[ -f "${PROJECT_DIR}/Pulumi.yaml" ] || die "Pulumi.yaml missing in ${PROJECT_DIR}"
PULUMI_OUTPUT_FILE="${PROJECT_DIR%/}/pulumi-outputs.json"
[ ! -d "$PULUMI_OUTPUT_FILE" ] || die "PULUMI output path $PULUMI_OUTPUT_FILE is a directory; must be a file path"
container_blob_list_ok(){ local sa="$1"; local c="$2"; $AZ_CLI_BIN storage blob list --account-name "$sa" --container-name "$c" --auth-mode login -o json >/dev/null 2>&1 && return 0 || return 1; }
storage_account_exists(){ local sa="$1"; local rg="$2"; $AZ_CLI_BIN storage account show -n "$sa" -g "$rg" -o json >/dev/null 2>&1 && return 0 || return 1; }
if ! storage_account_exists "$PULUMI_STORAGE_ACCOUNT_NAME" "$AZURE_RESOURCE_GROUP_NAME"; then
  die "Storage account ${PULUMI_STORAGE_ACCOUNT_NAME} not found in RG ${AZURE_RESOURCE_GROUP_NAME}. Create it or set PULUMI_STORAGE_ACCOUNT_NAME correctly."
fi
VENV_DIR="${PROJECT_DIR}/.venv-pulumi"
if [ ! -d "$VENV_DIR" ]; then
  log "creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || die "venv creation failed"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log "installing python deps from $REQ_FILE"
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1
python -m pip install -r "$REQ_FILE" >/dev/null 2>&1
if ! command -v "$PULUMI_BIN" >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    log "installing pulumi CLI"
    curl -fsSL https://get.pulumi.com | sh -s -- --quiet || die "pulumi install failed"
    export PATH="$HOME/.pulumi/bin:${PATH}"
  else
    die "pulumi not found and curl not available to install"
  fi
fi
PULUMI_BACKEND_URL="azblob://${PULUMI_STATE_CONTAINER}?storage_account=${PULUMI_STORAGE_ACCOUNT_NAME}"
log "checking access to Pulumi backend container ${PULUMI_STATE_CONTAINER} in ${PULUMI_STORAGE_ACCOUNT_NAME}"
if container_blob_list_ok "$PULUMI_STORAGE_ACCOUNT_NAME" "$PULUMI_STATE_CONTAINER"; then
  log "backend container blob access OK with current identity"
else
  log "backend container blob access failed; attempting to ensure minimal data-plane role at container scope"
  ACCOUNT_JSON="$($AZ_CLI_BIN account show -o json 2>/dev/null || true)"
  USER_TYPE="$(printf '%s' "$ACCOUNT_JSON" | jq -r '.user.type' 2>/dev/null || echo "user")"
  if [ "$USER_TYPE" = "user" ]; then
    ASSIGNEE_ID="$($AZ_CLI_BIN ad signed-in-user show --query id -o tsv 2>/dev/null || true)"
  else
    SP_APP_ID="$(printf '%s' "$ACCOUNT_JSON" | jq -r '.user.name' 2>/dev/null || true)"
    ASSIGNEE_ID="$($AZ_CLI_BIN ad sp show --id "$SP_APP_ID" --query objectId -o tsv 2>/dev/null || true)"
  fi
  if [ -z "${ASSIGNEE_ID:-}" ]; then
    log "cannot determine principal id for role assignment; will attempt storage key fallback"
    ROLE_ASSIGNED=0
  else
    SCOPE="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AZURE_RESOURCE_GROUP_NAME}/providers/Microsoft.Storage/storageAccounts/${PULUMI_STORAGE_ACCOUNT_NAME}/blobServices/default/containers/${PULUMI_STATE_CONTAINER}"
    EXISTING_RA="$($AZ_CLI_BIN role assignment list --assignee "$ASSIGNEE_ID" --scope "$SCOPE" --role "Storage Blob Data Contributor" -o json 2>/dev/null || echo '[]')"
    if [ "$(printf '%s' "$EXISTING_RA" | jq 'length' 2>/dev/null || echo 0)" -gt 0 ]; then
      log "Storage Blob Data Contributor already assigned at container scope"
      ROLE_ASSIGNED=1
    else
      set +e
      $AZ_CLI_BIN role assignment create --role "Storage Blob Data Contributor" --assignee "$ASSIGNEE_ID" --scope "$SCOPE" >/dev/null 2>&1
      RC=$?
      set -e
      if [ "$RC" -eq 0 ]; then
        log "assigned Storage Blob Data Contributor at container scope"
        ROLE_ASSIGNED=1
      else
        log "role assignment failed or not permitted by caller; will attempt storage key fallback"
        ROLE_ASSIGNED=0
      fi
    fi
  fi
  if [ "${ROLE_ASSIGNED:-0}" -eq 1 ]; then
    tries=0
    until container_blob_list_ok "$PULUMI_STORAGE_ACCOUNT_NAME" "$PULUMI_STATE_CONTAINER" || [ "$tries" -ge 12 ]; do
      sleep 5
      tries=$((tries+1))
      log "waiting for role propagation ($tries/12)"
    done
    container_blob_list_ok "$PULUMI_STORAGE_ACCOUNT_NAME" "$PULUMI_STATE_CONTAINER" || log "container still not accessible after role assignment; will fallback to storage key"
  fi
fi
log "attempting pulumi login to $PULUMI_BACKEND_URL (azblob)"
set +e
$PULUMI_BIN login "$PULUMI_BACKEND_URL"
LOGIN_RC=$?
set -e
if [ "$LOGIN_RC" -ne 0 ]; then
  log "pulumi login via Entra failed; attempting fallback using storage account key"
  AZ_KEY="$($AZ_CLI_BIN storage account keys list -g "$AZURE_RESOURCE_GROUP_NAME" -n "$PULUMI_STORAGE_ACCOUNT_NAME" --query '[0].value' -o tsv 2>/dev/null || true)"
  if [ -n "$AZ_KEY" ]; then
    export AZURE_STORAGE_ACCOUNT="$PULUMI_STORAGE_ACCOUNT_NAME"
    export AZURE_STORAGE_KEY="$AZ_KEY"
    log "retrying pulumi login using AZURE_STORAGE_KEY"
    set +e
    $PULUMI_BIN login "$PULUMI_BACKEND_URL"
    LOGIN_RC=$?
    set -e
    if [ "$LOGIN_RC" -ne 0 ]; then
      die "pulumi login failed using AZURE_STORAGE_KEY; please verify storage account keys and network access"
    fi
  else
    die "unable to obtain storage account key for fallback; ensure caller can list keys or assign roles appropriately"
  fi
fi
pushd "$PROJECT_DIR" >/dev/null
if $PULUMI_BIN stack select "$PULUMI_STACK" --non-interactive >/dev/null 2>&1; then
  log "pulumi: selected stack $PULUMI_STACK"
else
  log "pulumi: creating stack $PULUMI_STACK"
  if ! $PULUMI_BIN stack init --stack "$PULUMI_STACK"; then
    log "interactive stack init fallback"
    $PULUMI_BIN stack init --stack "$PULUMI_STACK" || die "pulumi: stack init failed"
  fi
  log "pulumi: stack $PULUMI_STACK created"
fi
if [ -n "${PULUMI_CONFIG_PASSPHRASE:-}" ]; then
  if ! $PULUMI_BIN config get --path pulumi:configPassphrase >/dev/null 2>&1; then
    export PULUMI_CONFIG_PASSPHRASE="$PULUMI_CONFIG_PASSPHRASE"
  fi
fi
MODE="${1:---create}"
if [ "$MODE" = "--preview" ]; then
  log "running pulumi preview (human)"
  $PULUMI_BIN preview --non-interactive --diff
  popd >/dev/null
  exit 0
fi
trap 'rm -f "${PREVIEW_TMP:-}"' EXIT
PREVIEW_TMP="$(mktemp)"
set +e
$PULUMI_BIN preview --non-interactive --json > "$PREVIEW_TMP" 2>&1
PREV_EXIT=$?
set -e
PREVIEW_JSON="$(cat "$PREVIEW_TMP" || true)"
if [ "$PREV_EXIT" -ne 0 ]; then
  log "pulumi preview returned non-zero ($PREV_EXIT) — showing preview output for debugging"
  sed -n '1,200p' "$PREVIEW_TMP"
  rm -f "$PREVIEW_TMP"
  popd >/dev/null
  die "pulumi preview failed (see above)"
fi
DELETES="[]"
if printf '%s' "$PREVIEW_JSON" | jq -e . >/dev/null 2>&1; then
  DELETES="$(printf '%s' "$PREVIEW_JSON" | jq -r '.resourceChanges[]? | select(.op=="delete" or .op=="deleteReplace") | {urn:.urn,op:.op,type:.type} ' | jq -s '.' 2>/dev/null || echo "[]")"
else
  log "pulumi preview JSON parsing failed; aborting to be safe"
  rm -f "$PREVIEW_TMP"
  popd >/dev/null
  die "pulumi preview returned invalid JSON"
fi
DELETE_COUNT="$(printf '%s' "$DELETES" | jq 'length' 2>/dev/null || echo 0)"
if [ "$MODE" = "--delete" ]; then
  log "mode=delete detected"
  if [ "$DELETE_COUNT" -eq 0 ]; then
    log "preview shows no deletions; proceeding to run 'pulumi destroy --yes' to remove all resources in stack"
  else
    log "preview indicates $DELETE_COUNT delete actions:"
    printf '%s\n' "$DELETES" | jq -r '.[] | "op=\(.op) type=\(.type) urn=\(.urn)"'
  fi
  if [ "${FORCE:-0}" != "1" ]; then
    log "To actually perform destroy, re-run with FORCE=1 environment variable set"
    rm -f "$PREVIEW_TMP"
    popd >/dev/null
    exit 2
  fi
  log "performing pulumi destroy (non-interactive)"
  if ! $PULUMI_BIN destroy --yes --non-interactive; then
    log "pulumi destroy failed; attempting to capture debug preview --diff"
    $PULUMI_BIN preview --diff || true
    rm -f "$PREVIEW_TMP"
    popd >/dev/null
    die "pulumi destroy failed"
  fi
  log "pulumi destroy completed successfully"
  if [ -f "$PULUMI_OUTPUT_FILE" ]; then
    rm -f "$PULUMI_OUTPUT_FILE" || true
    log "removed $PULUMI_OUTPUT_FILE"
  fi
  popd >/dev/null
  exit 0
fi
if [ "$DELETE_COUNT" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  log "Detected $DELETE_COUNT destructive change(s) in preview. Aborting to avoid accidental deletes."
  printf '%s\n' "$DELETES" | jq -r '.[] | "op=\(.op) type=\(.type) urn=\(.urn)"'
  echo ""
  echo "If you intend to apply these deletes, re-run with FORCE=1 to skip this safety check."
  rm -f "$PREVIEW_TMP"
  popd >/dev/null
  exit 2
fi
[ "$DELETE_COUNT" -gt 0 ] && log "FORCE=1 set or zero deletes detected — continuing"
log "applying pulumi up (non-interactive) - streaming output"
if ! $PULUMI_BIN up --yes --non-interactive; then
  log "pulumi up failed; capturing debug preview --diff"
  $PULUMI_BIN preview --diff || true
  rm -f "$PREVIEW_TMP"
  popd >/dev/null
  die "pulumi up failed"
fi
log "exporting pulumi outputs to $PULUMI_OUTPUT_FILE"
$PULUMI_BIN stack output --json > "$PULUMI_OUTPUT_FILE" || log "warning: failed to write pulumi outputs to $PULUMI_OUTPUT_FILE"
chmod 0600 "$PULUMI_OUTPUT_FILE" || true
rm -f "$PREVIEW_TMP"
popd >/dev/null
log "done"
exit 0
