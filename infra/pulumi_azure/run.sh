#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
: "${AZ_CLI_BIN:=az}"
: "${PULUMI_BIN:=pulumi}"
: "${PYTHON_BIN:=python3}"
: "${PROJECT_DIR:=infra/pulumi_azure}"
: "${REQ_FILE:=${PROJECT_DIR}/requirements.txt}"
: "${PULUMI_STACK:=staging}"
: "${PULUMI_CONFIG_PASSPHRASE:=}"
: "${PULUMI_STATE_CONTAINER:=pulumi-state}"
: "${PULUMI_STORAGE_ACCOUNT_NAME:=pulumistate${ENV:-staging}01}"
: "${FORCE:=0}"
log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found"; }
require_cmd "$AZ_CLI_BIN"
require_cmd "$PYTHON_BIN"
require_cmd "$PULUMI_BIN"
require_cmd jq
: "${AZURE_SUBSCRIPTION_ID:?Please export AZURE_SUBSCRIPTION_ID}"
: "${AZURE_TENANT_ID:?Please export AZURE_TENANT_ID}"
PROJECT_DIR="$(cd "$PROJECT_DIR" 2>/dev/null && pwd || die "project dir $PROJECT_DIR not found")"
REQ_FILE="$(cd "$(dirname "$REQ_FILE")" >/dev/null 2>&1 && printf '%s' "$REQ_FILE" || die "requirements path invalid")"
[ -f "$REQ_FILE" ] || die "requirements.txt missing at $REQ_FILE"
[ -f "${PROJECT_DIR}/Pulumi.yaml" ] || die "Pulumi.yaml missing in ${PROJECT_DIR}"
PULUMI_OUTPUT_FILE="${PROJECT_DIR%/}/pulumi-outputs.json"
[ ! -d "$PULUMI_OUTPUT_FILE" ] || die "PULUMI output path $PULUMI_OUTPUT_FILE is a directory; must be a file path"
container_exists(){ local sa="$1"; local c="$2"; if $AZ_CLI_BIN storage container show -n "$c" --account-name "$sa" --auth-mode login >/dev/null 2>&1; then return 0; fi; return 1; }
container_exists "$PULUMI_STORAGE_ACCOUNT_NAME" "$PULUMI_STATE_CONTAINER" || die "Pulumi backend container '$PULUMI_STATE_CONTAINER' not found in storage account '$PULUMI_STORAGE_ACCOUNT_NAME'"
VENV_DIR="${PROJECT_DIR}/.venv-pulumi"
[ -d "$VENV_DIR" ] || { log "creating venv at $VENV_DIR"; "$PYTHON_BIN" -m venv "$VENV_DIR" || die "venv creation failed"; }
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
log "installing python deps from $REQ_FILE"
python -m pip install -r "$REQ_FILE" >/dev/null 2>&1 || die "pip install failed; inspect venv logs"
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
log "logging into pulumi backend: $PULUMI_BACKEND_URL"
$PULUMI_BIN login "$PULUMI_BACKEND_URL" >/dev/null 2>&1 || die "pulumi login failed (ensure az login & that identity has blob/container access)"
pushd "$PROJECT_DIR" >/dev/null
if $PULUMI_BIN stack select "$PULUMI_STACK" --non-interactive >/dev/null 2>&1; then
  log "pulumi: selected stack $PULUMI_STACK"
else
  log "pulumi: creating stack $PULUMI_STACK"
  $PULUMI_BIN stack init --stack "$PULUMI_STACK" >/dev/null 2>&1 || { log "interactive stack init fallback"; $PULUMI_BIN stack init --stack "$PULUMI_STACK" || die "pulumi: stack init failed"; }
  log "pulumi: stack $PULUMI_STACK created"
fi
if [ -n "${PULUMI_CONFIG_PASSPHRASE:-}" ]; then
  if ! $PULUMI_BIN config get --path pulumi:configPassphrase >/dev/null 2>&1; then
    export PULUMI_CONFIG_PASSPHRASE="$PULUMI_CONFIG_PASSPHRASE"
  fi
fi
if [ "${1:-}" = "--preview" ]; then
  log "running pulumi preview (human)"
  $PULUMI_BIN preview --non-interactive --diff || die "pulumi preview failed"
  popd >/dev/null
  exit 0
fi
log "running pulumi preview (json) to detect destructive changes (streaming to terminal)"
set +e
PREVIEW_RAW="$(bash -lc 'pulumi preview --non-interactive --json 2>&1 | tee /dev/stderr; printf "\n__PULUMI_EXIT_CODE__%s" "$?"')"
PREV_EXIT="$(printf '%s' "$PREVIEW_RAW" | sed -n 's/.*__PULUMI_EXIT_CODE__\([0-9]*\)$/\1/p' || echo 1)"
PREVIEW_JSON="$(printf '%s' "$PREVIEW_RAW" | sed 's/\n__PULUMI_EXIT_CODE__[0-9]*$//' || true)"
set -e
if [ -z "$PREVIEW_JSON" ]; then
  log "pulumi preview produced no JSON output; raw preview exit=$PREV_EXIT"
  [ "$PREV_EXIT" -eq 0 ] || die "pulumi preview failed (no JSON)"
fi
if [ "$PREV_EXIT" -ne 0 ]; then
  log "pulumi preview returned non-zero ($PREV_EXIT) — showing truncated preview JSON for debugging"
  printf '%s\n' "$PREVIEW_JSON" | sed -n '1,200p'
  popd >/dev/null
  die "pulumi preview returned non-zero (see above)"
fi
DELETES="$(printf '%s' "$PREVIEW_JSON" | jq -r '.resourceChanges[]? | select(.op=="delete" or .op=="deleteReplace") | {urn:.urn,op:.op,type:.type} ' | jq -s '.' || echo "[]")"
DELETE_COUNT="$(printf '%s' "$DELETES" | jq 'length' 2>/dev/null || echo 0)"
if [ "$DELETE_COUNT" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  log "Detected $DELETE_COUNT destructive change(s) in preview. Aborting to avoid accidental deletes."
  printf '%s\n' "$DELETES" | jq -r '.[] | "op=\(.op) type=\(.type) urn=\(.urn)"'
  echo ""
  echo "If you intend to apply these deletes, re-run with FORCE=1 to skip this safety check."
  popd >/dev/null
  exit 2
fi
[ "$DELETE_COUNT" -gt 0 ] && log "FORCE=1 set or zero deletes detected — continuing"
log "applying pulumi up (non-interactive)"
$PULUMI_BIN up --yes --non-interactive || { log "pulumi up failed; running pulumi preview --diff for context"; $PULUMI_BIN preview --diff || true; popd >/dev/null; die "pulumi up failed"; }
log "exporting pulumi outputs to $PULUMI_OUTPUT_FILE"
$PULUMI_BIN stack output --json > "$PULUMI_OUTPUT_FILE" || log "warning: failed to write pulumi outputs to $PULUMI_OUTPUT_FILE"
chmod 0600 "$PULUMI_OUTPUT_FILE" || true
popd >/dev/null
log "done"
exit 0
