#!/usr/bin/env bash
IFS=$'\n\t'
log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
warn(){ printf '[%s] WARN: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
err(){ printf '[%s] ERROR: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
: "${PULUMI_VERSION:=v3.214.1}"
: "${PULUMI_PY_PKG_VERSION:=${PULUMI_VERSION}}"
: "${AZURE_NATIVE_PLUGIN_VERSION:=3.11.0}"
: "${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID must be exported}"
: "${AZURE_TENANT_ID:?AZURE_TENANT_ID must be exported}"
: "${AZURE_RESOURCE_GROUP_NAME:?AZURE_RESOURCE_GROUP_NAME must be exported}"
: "${AZURE_STORAGE_ACCOUNT_NAME:?AZURE_STORAGE_ACCOUNT_NAME must be exported}"
: "${PULUMI_STACK:?PULUMI_STACK must be exported}"
: "${AZ_CLI_BIN:=az}"
: "${PULUMI_BIN:=pulumi}"
: "${PYTHON_BIN:=python3}"
: "${PROJECT_DIR:=infra/pulumi_azure}"
: "${REQ_FILE:=${PROJECT_DIR}/requirements.txt}"
: "${FORCE:=${PULUMI_FORCE_DESTROY:-1}}"
export PULUMI_DISABLE_AUTOMATIC_PLUGIN_ACQUISITION=true
if [ -n "${PULUMI_CONFIG_PASSPHRASE:-}" ]; then
  export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE}"
  log "Using PULUMI_CONFIG_PASSPHRASE from environment"
fi
for bin in "${AZ_CLI_BIN}" "${PYTHON_BIN}" jq curl; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    err "required CLI not found: ${bin}"
    exit 2
  fi
done
log "Found az: ${AZ_CLI_BIN}"
log "Found python: ${PYTHON_BIN}"
log "Found jq in PATH"
log "Found curl in PATH"
if command -v "${PULUMI_BIN}" >/dev/null 2>&1; then
  INSTALLED_VERSION="$("${PULUMI_BIN}" version 2>/dev/null | sed -n '1p' | awk '{print $1}')"
  if [ -z "${INSTALLED_VERSION}" ] || [ "${INSTALLED_VERSION}" != "${PULUMI_VERSION}" ]; then
    err "Pulumi CLI version mismatch: installed='${INSTALLED_VERSION:-none}' expected='${PULUMI_VERSION}'. This script does not install the Pulumi CLI; please install/upgrade externally."
    exit 4
  else
    log "Pulumi CLI ${PULUMI_VERSION} already present"
  fi
else
  err "Pulumi CLI not found. This script does not install the Pulumi CLI; please install Pulumi ${PULUMI_VERSION} and ensure it's on PATH."
  exit 4
fi
if ! command -v "${PULUMI_BIN}" >/dev/null 2>&1; then
  err "Pulumi CLI not available after verification"
  exit 4
fi
log "Pulumi CLI available: ${PULUMI_BIN}"
log "Pulumi CLI version: $(${PULUMI_BIN} version 2>/dev/null | sed -n '1p' || true)"
AZ_SUB="${AZURE_SUBSCRIPTION_ID}"
RG_NAME="${AZURE_RESOURCE_GROUP_NAME}"
SA_NAME="${AZURE_STORAGE_ACCOUNT_NAME}"
PULUMI_CONTAINER="${PULUMI_AZ_CONTAINER:-pulumi-state}"
STACK="${PULUMI_STACK}"
PROJ_DIR="${PROJECT_DIR}"
REQ_FILE_PATH="${REQ_FILE}"
"${AZ_CLI_BIN}" account set --subscription "${AZ_SUB}" >/dev/null 2>&1
log "Azure subscription set to ${AZ_SUB}"
if ! "${AZ_CLI_BIN}" storage account show -n "${SA_NAME}" -g "${RG_NAME}" -o none >/dev/null 2>&1; then
  err "Storage account ${SA_NAME} not found in RG ${RG_NAME}"
  exit 3
fi
log "Validated storage account ${SA_NAME} in RG ${RG_NAME}"
if "${AZ_CLI_BIN}" storage container show --account-name "${SA_NAME}" -n "${PULUMI_CONTAINER}" --auth-mode login -o none >/dev/null 2>&1; then
  log "Pulumi backend container ${PULUMI_CONTAINER} accessible to current identity"
else
  warn "Pulumi container ${PULUMI_CONTAINER} missing or not accessible; will attempt idempotent RBAC then storage-key fallback"
fi
VENV_DIR="${PROJ_DIR%/}/.venv-pulumi"
if [ ! -d "${VENV_DIR}" ]; then
  log "Creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
. "${VENV_DIR}/bin/activate"
log "Activated venv ${VENV_DIR}"
if [ -f "${REQ_FILE_PATH}" ]; then
  log "Installing python deps from ${REQ_FILE_PATH}"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || warn "pip upgrade returned non-zero"
  if ! "${PYTHON_BIN}" -m pip install -r "${REQ_FILE_PATH}" >/dev/null 2>&1; then
    warn "pip install -r ${REQ_FILE_PATH} returned non-zero; continuing to ensure pulumi python package pinned"
  fi
else
  warn "requirements file not found at ${REQ_FILE_PATH}; continuing"
fi
log "Ensuring pulumi python package version = ${PULUMI_PY_PKG_VERSION}"
if ! "${PYTHON_BIN}" -m pip install --upgrade "pulumi==${PULUMI_PY_PKG_VERSION}" >/dev/null 2>&1; then
  warn "Failed to install pulumi==${PULUMI_PY_PKG_VERSION} into venv; please ensure compatibility"
fi
if command -v "${PULUMI_BIN}" >/dev/null 2>&1; then
  log "Ensuring Pulumi resource plugin azure-native ${AZURE_NATIVE_PLUGIN_VERSION} is installed"
  if ! "${PULUMI_BIN}" plugin install resource azure-native "${AZURE_NATIVE_PLUGIN_VERSION}" >/dev/null 2>&1; then
    warn "pulumi plugin install resource azure-native ${AZURE_NATIVE_PLUGIN_VERSION} failed (network/proxy?). If disabled, ensure plugin is preinstalled on runner"
  else
    log "pulumi resource plugin azure-native ${AZURE_NATIVE_PLUGIN_VERSION} installed"
  fi
fi
PULUMI_BACKEND_URL="azblob://${PULUMI_CONTAINER}?storage_account=${SA_NAME}"
attempt_login_aad() { "${PULUMI_BIN}" login "${PULUMI_BACKEND_URL}" >/dev/null 2>&1; }
attempt_login_key() {
  local key
  key="$("${AZ_CLI_BIN}" storage account keys list -g "${RG_NAME}" -n "${SA_NAME}" --query '[0].value' -o tsv 2>/dev/null || true)"
  if [ -n "${key}" ]; then
    export AZURE_STORAGE_ACCOUNT="${SA_NAME}"
    export AZURE_STORAGE_KEY="${key}"
    "${PULUMI_BIN}" login "${PULUMI_BACKEND_URL}" >/dev/null 2>&1
  else
    return 1
  fi
}
log "Attempting Pulumi login via AAD to ${PULUMI_BACKEND_URL}"
if attempt_login_aad; then
  log "Pulumi login via AAD succeeded"
else
  warn "Pulumi login via AAD failed; ensuring account-scope Storage RBAC and retrying"
  ACCOUNT_JSON="$("${AZ_CLI_BIN}" account show -o json 2>/dev/null || true)"
  USER_TYPE="$(printf '%s' "${ACCOUNT_JSON}" | jq -r '.user.type' 2>/dev/null || echo user)"
  if [ "${USER_TYPE}" = "user" ]; then
    ASSIGNEE_ID="$("${AZ_CLI_BIN}" ad signed-in-user show --query id -o tsv 2>/dev/null || true)"
  else
    ASSIGNEE_ID="$("${AZ_CLI_BIN}" ad sp show --id "$(printf '%s' "${ACCOUNT_JSON}" | jq -r '.user.name')" --query objectId -o tsv 2>/dev/null || true)"
  fi
  if [ -z "${ASSIGNEE_ID}" ]; then
    err "Unable to resolve assignee identity for RBAC; trying storage-key fallback"
    if attempt_login_key; then
      log "Pulumi login via storage-key fallback succeeded"
    else
      err "Storage-key fallback failed; set AZURE_STORAGE_KEY or request admin to grant Storage Blob Data Contributor on the storage account"
      exit 5
    fi
  else
    SCOPE="/subscriptions/${AZ_SUB}/resourceGroups/${RG_NAME}/providers/Microsoft.Storage/storageAccounts/${SA_NAME}"
    if az role assignment list --assignee "${ASSIGNEE_ID}" --scope "${SCOPE}" --role "Storage Blob Data Contributor" -o tsv 2>/dev/null | grep -q .; then
      log "Account-scoped Storage Blob Data Contributor already present for current principal"
    else
      if az role assignment create --assignee "${ASSIGNEE_ID}" --role "Storage Blob Data Contributor" --scope "${SCOPE}" >/dev/null 2>&1; then
        log "Role assignment created; waiting for propagation"
      else
        warn "Failed to create role assignment (insufficient privileges?) — attempting storage-key fallback"
        if attempt_login_key; then
          log "Pulumi login via storage-key fallback succeeded"
        else
          err "Storage-key fallback failed; set AZURE_STORAGE_KEY or ask admin to grant role"
          exit 5
        fi
      fi
    fi
    tries=0
    while ! "${AZ_CLI_BIN}" storage container show --account-name "${SA_NAME}" -n "${PULUMI_CONTAINER}" --auth-mode login -o none >/dev/null 2>&1; do
      tries=$((tries+1))
      if [ "${tries}" -ge 24 ]; then
        warn "Role propagation timed out (~120s); attempting storage-key fallback"
        if attempt_login_key; then
          log "Pulumi login via storage-key fallback succeeded"
          break
        else
          err "Storage-key fallback failed; set AZURE_STORAGE_KEY or ask admin to grant role"
          exit 5
        fi
      fi
      sleep 5
    done
    if attempt_login_aad; then
      log "Pulumi login via AAD succeeded after RBAC reconciliation"
    else
      if attempt_login_key; then
        log "Pulumi login via storage-key fallback succeeded"
      else
        err "Pulumi login failed after RBAC reconciliation and storage-key fallback"
        exit 5
      fi
    fi
  fi
fi
if [ ! -d "${PROJ_DIR}" ]; then
  err "Pulumi project directory not found: ${PROJ_DIR}"
  exit 6
fi
PUSHED=0
pushd "${PROJ_DIR}" >/dev/null && PUSHED=1
PREVIEW_TMP="$(mktemp -t pulumi_preview.XXXXXX.txt)"
cleanup() {
  rm -f "${PREVIEW_TMP}" >/dev/null 2>&1 || true
  if [ "${PUSHED}" -eq 1 ]; then
    popd >/dev/null 2>&1 || true
  fi
}
trap 'cleanup' EXIT
OUTPUTS_FILE="./pulumi-outputs.json"
if "${PULUMI_BIN}" stack select "${STACK}" --non-interactive >/dev/null 2>&1; then
  log "Pulumi: selected existing stack ${STACK}"
else
  log "Pulumi: stack ${STACK} not selectable non-interactively. Inspecting remote stacks."
  STACKS_JSON="$("${PULUMI_BIN}" stack ls --json 2>/dev/null || true)"
  if [ -n "${STACKS_JSON}" ] && printf '%s' "${STACKS_JSON}" | jq -e . >/dev/null 2>&1; then
    if printf '%s' "${STACKS_JSON}" | jq -e --arg s "${STACK}" '.[] | select(.name==$s)' >/dev/null 2>&1; then
      err "Pulumi stack ${STACK} exists in backend but selection failed non-interactively."
      printf '%s\n' "${STACKS_JSON}" | jq -r '.[] | "name=\(.name) lastUpdate=\(.lastUpdate)\n"' || true
      exit 7
    fi
  fi
  log "Pulumi: stack ${STACK} not found remotely; creating it now."
  set +e
  INIT_OUT="$("${PULUMI_BIN}" stack init --stack "${STACK}" 2>&1 || true)"
  INIT_RC=${?}
  set -e
  if [ "${INIT_RC}" -ne 0 ]; then
    err "pulumi stack init failed (rc=${INIT_RC}). Output (head):"
    printf '%s\n' "${INIT_OUT}" | sed -n '1,200p'
    exit 8
  fi
  if ! "${PULUMI_BIN}" stack select "${STACK}" --non-interactive >/dev/null 2>&1; then
    err "Pulumi stack ${STACK} not selected after init."
    printf '%s\n' "${INIT_OUT}" | sed -n '1,200p'
    exit 9
  fi
  log "Pulumi: created and selected stack ${STACK}"
fi
MODE="${1:---create}"
if [ "${MODE}" = "--preview" ]; then
  log "Running pulumi preview (human-facing)"
  set +e
  "${PULUMI_BIN}" preview --non-interactive --diff 2>&1 | tee "${PREVIEW_TMP}"
  PREV_RC=${?}
  set -e
  if [ "${PREV_RC}" -ne 0 ]; then
    err "pulumi preview failed (exit ${PREV_RC})"
    exit 10
  fi
  log "pulumi preview succeeded"
  exit 0
fi
log "Running pulumi preview (showing native logs)..."
set +e
"${PULUMI_BIN}" preview --non-interactive --diff 2>&1 | tee "${PREVIEW_TMP}"
PREV_RC=${?}
set -e
if [ "${PREV_RC}" -ne 0 ]; then
  err "pulumi preview failed (exit ${PREV_RC})"
  exit 11
fi

if [ "${MODE}" = "--delete" ]; then
  log "Destroy mode requested; removing ManagedCluster(s) in selected stack (no guessing)"

  STACK_JSON="$("${PULUMI_BIN}" stack export 2>/dev/null)" || { err "pulumi stack export failed"; exit 14; }

  mapfile -t CLUSTER_LINES < <(printf '%s' "$STACK_JSON" | jq -c '.deployment.resources[] | select(.type == "azure-native:containerservice:ManagedCluster") | {urn:.urn, name:.inputs.name, rg:.inputs.resourceGroupName}')

  if [ "${#CLUSTER_LINES[@]}" -eq 0 ]; then
    log "No ManagedCluster resources found in stack ${STACK}; skipping targeted cluster deletion"
  else
    log "Found ${#CLUSTER_LINES[@]} ManagedCluster(s) in stack; destroying them sequentially"
    for line in "${CLUSTER_LINES[@]}"; do
      URN="$(printf '%s' "$line" | jq -r '.urn')"
      CL_NAME="$(printf '%s' "$line" | jq -r '.name')"
      CL_RG="$(printf '%s' "$line" | jq -r '.rg')"
      if [ -z "${URN}" ] || [ "${URN}" = "null" ]; then
        warn "Skipping an entry with no URN: ${line}"
        continue
      fi
      log "Destroying ManagedCluster resource urn=${URN} name=${CL_NAME} resourceGroup=${CL_RG}"
      if ! "${PULUMI_BIN}" destroy --target "${URN}" --yes --non-interactive; then
        err "Targeted destroy failed for urn=${URN}"
        exit 14
      fi
      log "Triggered destroy for ${CL_NAME}; waiting for AKS-managed RGs (MC_*) for this cluster to disappear (if resourceGroupName available)"
      if [ -n "${CL_RG}" ] && [ "${CL_RG}" != "null" ]; then
        tries=0
        while : ; do
          FOUND_RG="$("${AZ_CLI_BIN}" group list --query "[?starts_with(name, 'MC_${CL_RG}_${CL_NAME}_')].name" -o tsv 2>/dev/null || true)"
          if [ -z "${FOUND_RG}" ]; then
            log "No MC_* RGs found for cluster ${CL_NAME} in RG ${CL_RG}; proceeding"
            break
          fi
          tries=$((tries+1))
          if [ "${tries}" -ge 60 ]; then
            warn "MC_* RGs still present for ${CL_NAME} after ~10 minutes: ${FOUND_RG}. Proceeding anyway."
            break
          fi
          log "Waiting for MC_* RGs for ${CL_NAME} to disappear (found: ${FOUND_RG}); sleeping 10s"
          sleep 10
        done
      else
        log "No resourceGroupName for cluster ${CL_NAME} in stack metadata; skipping MC_* wait"
      fi
    done
  fi

  log "All ManagedCluster targets processed; performing full pulumi destroy"
  if ! "${PULUMI_BIN}" destroy --yes --non-interactive; then
    err "Final pulumi destroy failed"
    exit 14
  fi
  log "pulumi destroy completed"
  rm -f "${OUTPUTS_FILE}" || true
  exit 0
fi

log "Applying pulumi up (non-interactive)"
if ! "${PULUMI_BIN}" up --yes --non-interactive; then
  warn "pulumi up failed; showing debug preview (--diff)"
  "${PULUMI_BIN}" preview --diff || true
  err "pulumi up failed"
fi

log "pulumi up completed; exporting outputs"
if ! "${PULUMI_BIN}" stack output --json > "${OUTPUTS_FILE}" 2>/dev/null; then
  warn "failed to write outputs file"
else
  chmod 0600 "${OUTPUTS_FILE}" || true
  log "Wrote stack outputs to ${OUTPUTS_FILE}"
fi

log "run.sh completed successfully"
exit 0
