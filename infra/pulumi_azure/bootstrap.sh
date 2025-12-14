#!/usr/bin/env bash
# pulumi_prerequisites.sh
# Idempotent creation of Pulumi backend resources: Resource Group, Storage Account, Blob Container.
#
# Usage:
#   export AZURE_SUBSCRIPTION_ID=...
#   export AZURE_TENANT_ID=...
#   # optionally override defaults below
#   ./pulumi_prerequisites.sh
#
set -euo pipefail
IFS=$'\n\t'

# -----------------------
# Defaults (override by env)
# -----------------------
: "${AZ_CLI_BIN:=az}"
: "${AZURE_SUBSCRIPTION_ID:=""}"
: "${AZURE_TENANT_ID:=""}"
: "${AZURE_LOCATION:=centralindia}"

# Pulumi backend resource names (deterministic)
: "${PULUMI_BACKEND_RG:=${AZURE_RESOURCE_GROUP_NAME}}"
: "${PULUMI_STORAGE_ACCOUNT_NAME:=pulumistate${ENV:-staging}01}" # MUST be globally unique; override in CI when necessary
: "${PULUMI_STATE_CONTAINER:=pulumi-state}"
: "${PULUMI_BACKEND_SKU:=Standard_LRS}"

# Logging helpers
log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found"; }

# -----------------------
# Preflight checks
# -----------------------
require_cmd "$AZ_CLI_BIN"

# Validate subscription/tenant set
: "${AZURE_SUBSCRIPTION_ID:?Please export AZURE_SUBSCRIPTION_ID}"
: "${AZURE_TENANT_ID:?Please export AZURE_TENANT_ID}"

# Ensure az login
if ! $AZ_CLI_BIN account show --subscription "$AZURE_SUBSCRIPTION_ID" >/dev/null 2>&1; then
  die "az login required and/or AZURE_SUBSCRIPTION_ID invalid. Run 'az login' or configure SP credentials."
fi

# -----------------------
# Helpers: idempotent creation
# -----------------------
ensure_rg(){
  local rg="$1"
  if $AZ_CLI_BIN group show -n "$rg" >/dev/null 2>&1; then
    log "resource group exists: $rg"
  else
    log "creating resource group: $rg (location: $AZURE_LOCATION)"
    $AZ_CLI_BIN group create -n "$rg" -l "$AZURE_LOCATION" --subscription "$AZURE_SUBSCRIPTION_ID" --output none
    log "resource group created: $rg"
  fi
}

storage_account_name_valid(){
  # Azure storage account name rules: 3-24 lowercase letters and numbers only
  local name="$1"
  if [[ "$name" =~ ^[a-z0-9]{3,24}$ ]]; then
    return 0
  fi
  return 1
}

ensure_storage_account(){
  local rg="$1" sa="$2" sku="$3" kind="${4:-StorageV2}"
  # Check if storage account already exists in this subscription
  if $AZ_CLI_BIN storage account show -n "$sa" -g "$rg" >/dev/null 2>&1; then
    log "storage account exists in RG $rg: $sa"
    return 0
  fi

  # Check global name availability
  local check
  check=$($AZ_CLI_BIN storage account check-name -n "$sa" -o json 2>/dev/null) || check=""
  if [ -n "$check" ]; then
    local available
    available=$(echo "$check" | grep -o '"nameAvailable": [^,]*' | awk -F: '{print $2}' | tr -d '[:space:],')
    if [ "$available" != "true" ]; then
      die "Storage account name '$sa' is not available globally. Choose a different PULUMI_STORAGE_ACCOUNT_NAME and re-run."
    fi
  fi

  log "creating storage account: $sa in RG $rg (sku: $sku)"
  $AZ_CLI_BIN storage account create \
    --name "$sa" \
    --resource-group "$rg" \
    --location "$AZURE_LOCATION" \
    --sku "$sku" \
    --kind "$kind" \
    --https-only true \
    --allow-blob-public-access false \
    --output none

  log "storage account created: $sa"
}

ensure_blob_container(){
  local sa="$1" container="$2"
  if $AZ_CLI_BIN storage container show --account-name "$sa" -n "$container" --auth-mode login >/dev/null 2>&1; then
    log "container exists: $container in $sa"
    return 0
  fi

  log "creating container: $container in storage account: $sa"
  $AZ_CLI_BIN storage container create --account-name "$sa" -n "$container" --auth-mode login --output none
  log "container created: $container"
}

# -----------------------
# Validate storage account name pattern
# -----------------------
if ! storage_account_name_valid "$PULUMI_STORAGE_ACCOUNT_NAME"; then
  die "Invalid PULUMI_STORAGE_ACCOUNT_NAME: '$PULUMI_STORAGE_ACCOUNT_NAME'. Must be 3-24 lowercase letters and numbers."
fi

# -----------------------
# Run creation steps
# -----------------------
log "Ensuring Pulumi backend resources (RG / Storage Account / Container)"
ensure_rg "$PULUMI_BACKEND_RG"
ensure_storage_account "$PULUMI_BACKEND_RG" "$PULUMI_STORAGE_ACCOUNT_NAME" "$PULUMI_BACKEND_SKU"
ensure_blob_container "$PULUMI_STORAGE_ACCOUNT_NAME" "$PULUMI_STATE_CONTAINER"

# Optional: soft delete and immutability - commented out (uncomment to apply)
# log "Enabling soft delete on blob service (7 days)"
# $AZ_CLI_BIN storage blob service-properties delete-policy update --account-name "$PULUMI_STORAGE_ACCOUNT_NAME" --enable true --days-retained 7 --auth-mode login

log "Pulumi backend ready: storage_account=$PULUMI_STORAGE_ACCOUNT_NAME container=$PULUMI_STATE_CONTAINER (RG=$PULUMI_BACKEND_RG)"
log "You can now run bash infra/pulumi_azure/run.sh which will login Pulumi and run the stack."

exit 0
