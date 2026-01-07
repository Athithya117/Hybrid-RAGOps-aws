#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: backup_and_restore.sh <backup|restore>"
  exit 1
fi

PYTHON="${PYTHON:-python3}"

BACKUP_RUNNER="infra/runners/run_qdrant_backup.py"
RESTORE_RUNNER="infra/runners/run_qdrant_restore.py"

ENV="${ENV:-STAGING}"
NAMESPACE="${QDRANT_NAMESPACE:-qdrant}"
PORT_BASE="${PORT_BASE:-7000}"

AZ_CONTAINER="${QDRANT_AZURE_CONTAINER:-qdrant-backups}"
AZ_PREFIX="${QDRANT_AZURE_PREFIX:-qdrant/backups}"

PER_POD="${QDRANT_PER_POD:-true}"
BACKUP_ID="${BACKUP_ID:-}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not found"; exit 1; }
}

require_env_any() {
  for v in "$@"; do
    [[ -n "${!v:-}" ]] && return 0
  done
  echo "ERROR: one of [$*] must be set"
  exit 1
}

require_cmd kubectl
require_cmd "$PYTHON"

if [[ "$ENV" == "STAGING" ]]; then
  require_env_any AZURE_STORAGE_CONNECTION_STRING AZURE_STORAGE_ACCOUNT_NAME
fi

if [[ -z "$AZ_CONTAINER" || -z "$AZ_PREFIX" ]]; then
  echo "ERROR: QDRANT_AZURE_CONTAINER and QDRANT_AZURE_PREFIX are required"
  exit 1
fi

COMMON_BACKUP_ARGS=(
  --backup
  --azure-container "$AZ_CONTAINER"
  --azure-prefix "$AZ_PREFIX"
  --namespace "$NAMESPACE"
)

COMMON_RESTORE_ARGS=(
  --azure-container "$AZ_CONTAINER"
  --azure-prefix "$AZ_PREFIX"
  --namespace "$NAMESPACE"
)

if [[ "$PER_POD" == "true" ]]; then
  COMMON_BACKUP_ARGS+=(--per-pod --port-base "$PORT_BASE")
  COMMON_RESTORE_ARGS+=(--per-pod --port-base "$PORT_BASE")
fi

case "$MODE" in
  backup)
    echo "==> Qdrant backup started (ENV=$ENV, per-pod=$PER_POD)"
    exec "$PYTHON" "$BACKUP_RUNNER" "${COMMON_BACKUP_ARGS[@]}"
    ;;
  restore)
    echo "==> Qdrant restore started (ENV=$ENV, per-pod=$PER_POD)"
    if [[ -n "$BACKUP_ID" ]]; then
      COMMON_RESTORE_ARGS+=(--backup-id "$BACKUP_ID")
    fi
    exec "$PYTHON" "$RESTORE_RUNNER" "${COMMON_RESTORE_ARGS[@]}"
    ;;
  *)
    echo "ERROR: unknown mode '$MODE' (expected backup|restore)"
    exit 1
    ;;
esac
