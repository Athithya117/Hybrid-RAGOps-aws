#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# mode_a.sh — create rag-api + rag-spa app registrations + API client secret + SPA redirect URIs
# Usage: ./mode_a.sh [--append-secret]
# Prereqs: az CLI logged in (az login) and jq installed.

PREFIX="${PREFIX:-rag}"
API_NAME="${PREFIX}-api"
SPA_NAME="${PREFIX}-spa"
AZ="{{AZ:-}}" # placeholder not used; left for template readers

APP_DESCRIPTION="RAG platform apps created by mode_a.sh"

# Redirect URIs to add to the SPA registration (idempotent add)
FRONTEND_URL="${FRONTEND_URL:-http://localhost:8000}"
REDIRECTS=(
  "${FRONTEND_URL%/}/auth/callback"
  "${FRONTEND_URL%/}/auth/callback/entra"
  "${FRONTEND_URL%/}/auth/callback/external-id"
  "http://frontend.default.svc.cluster.local/auth/callback"
  "http://frontend.default.svc/auth/callback"
  "https://frontend.default.svc.cluster.local/auth/callback"
  "https://frontend.default.svc/auth/callback"
)

# CLI knobs
AZ_CLI="${AZ_CLI:-az}"
JQ="${JQ:-jq}"
APPEND_SECRET="${1:-}" # if "--append-secret", attempt to append secret instead of replacing

log(){ printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found"; }

require_cmd "$AZ_CLI"
require_cmd "$JQ"

# Ensure logged in
if ! $AZ_CLI account show -o json >/dev/null 2>&1; then
  die "az CLI not logged in. Run 'az login' with an account that can create app registrations."
fi

log "Using az CLI: $(command -v $AZ_CLI)"

# Helper: find app by display name, return json for first match
find_app_by_name(){
  local name="$1"
  # az ad app list supports --display-name in many versions; fallback to list+jq
  $AZ_CLI ad app list --display-name "$name" -o json 2>/dev/null | $JQ '.[0] // empty'
}

# Create or reuse API app
if [ -n "${API_EXISTING_CLIENT_ID:-}" ]; then
  API_APP_ID="$API_EXISTING_CLIENT_ID"
  log "Using provided API_EXISTING_CLIENT_ID=$API_APP_ID"
  API_APP_JSON="$($AZ_CLI ad app show --id "$API_APP_ID" -o json 2>/dev/null || true)"
  if [ -z "$API_APP_JSON" ]; then
    die "API_EXISTING_CLIENT_ID set but application not found in current tenant: $API_APP_ID"
  fi
else
  existing_api="$(find_app_by_name "$API_NAME" || true)"
  if [ -n "$existing_api" ]; then
    API_APP_ID="$(printf '%s' "$existing_api" | $JQ -r '.appId')"
    API_APP_OBJECT_ID="$(printf '%s' "$existing_api" | $JQ -r '.id')"
    log "Found existing API app ($API_NAME) appId=$API_APP_ID objectId=$API_APP_OBJECT_ID"
  else
    log "Creating API app registration: $API_NAME"
    API_APP_JSON="$($AZ_CLI ad app create --display-name "$API_NAME" --sign-in-audience AzureADMyOrg -o json)"
    API_APP_ID="$(printf '%s' "$API_APP_JSON" | $JQ -r '.appId')"
    API_APP_OBJECT_ID="$(printf '%s' "$API_APP_JSON" | $JQ -r '.id')"
    log "Created API app appId=$API_APP_ID objectId=$API_APP_OBJECT_ID"
  fi
fi

# Create or reuse SPA app
if [ -n "${SPA_EXISTING_CLIENT_ID:-}" ]; then
  SPA_APP_ID="$SPA_EXISTING_CLIENT_ID"
  log "Using provided SPA_EXISTING_CLIENT_ID=$SPA_APP_ID"
  SPA_APP_JSON="$($AZ_CLI ad app show --id "$SPA_APP_ID" -o json 2>/dev/null || true)"
  if [ -z "$SPA_APP_JSON" ]; then
    die "SPA_EXISTING_CLIENT_ID set but application not found in current tenant: $SPA_APP_ID"
  fi
else
  existing_spa="$(find_app_by_name "$SPA_NAME" || true)"
  if [ -n "$existing_spa" ]; then
    SPA_APP_ID="$(printf '%s' "$existing_spa" | $JQ -r '.appId')"
    SPA_APP_OBJECT_ID="$(printf '%s' "$existing_spa" | $JQ -r '.id')"
    log "Found existing SPA app ($SPA_NAME) appId=$SPA_APP_ID objectId=$SPA_APP_OBJECT_ID"
  else
    log "Creating SPA app registration: $SPA_NAME"
    SPA_APP_JSON="$($AZ_CLI ad app create --display-name "$SPA_NAME" --sign-in-audience AzureADMyOrg -o json)"
    SPA_APP_ID="$(printf '%s' "$SPA_APP_JSON" | $JQ -r '.appId')"
    SPA_APP_OBJECT_ID="$(printf '%s' "$SPA_APP_JSON" | $JQ -r '.id')"
    log "Created SPA app appId=$SPA_APP_ID objectId=$SPA_APP_OBJECT_ID"
  fi
fi

# Ensure service principals (idempotent)
ensure_sp(){
  local appid="$1"
  local name="$2"
  if $AZ_CLI ad sp show --id "$appid" -o json >/dev/null 2>&1; then
    log "Service principal already exists for appId=$appid"
  else
    log "Creating service principal for appId=$appid"
    $AZ_CLI ad sp create --id "$appid" >/dev/null
    log "Created service principal for appId=$appid"
  fi
}
ensure_sp "$API_APP_ID" "$API_NAME"
ensure_sp "$SPA_APP_ID" "$SPA_NAME"

# Patch SPA redirect URIs (idempotent add)
# Build current set, union with our list, write back using az ad app update --id <objectId> --set spa.redirectUris=...
current_redirects_json="$($AZ_CLI ad app show --id "$SPA_APP_ID" -o json)"
if [ -z "$current_redirects_json" ]; then
  die "Failed to read SPA application object for appId=$SPA_APP_ID"
fi
# try to get object id
SPA_APP_OBJECT_ID="$(printf '%s' "$current_redirects_json" | $JQ -r '.id')"
existing_spa_spas="$(printf '%s' "$current_redirects_json" | $JQ -r '.spa.redirectUris // [] | @json')"
# Merge lists (jq)
all_redirects_json="$(jq -n --argjson a "$existing_spa_spas" --argfile b /dev/stdin '[$a, $b] | add | unique' <<<"$(printf '%s\n' "${REDIRECTS[@]}" | jq -R -s -c 'split("\n")[:-1]')" )" || true

# Fallback simpler merge: create combined list via shell and dedupe
combine_and_dedupe(){
  local arr=()
  # load existing redirects (spa.redirectUris + web.redirectUris) to arr
  mapfile -t existing < <(printf '%s' "$current_redirects_json" | $JQ -r '.spa.redirectUris[]? // empty; .web.redirectUris[]? // empty' 2>/dev/null || true)
  for u in "${existing[@]}"; do arr+=("$u"); done
  for u in "${REDIRECTS[@]}"; do arr+=("$u"); done
  # dedupe preserving order
  declare -A seen
  out=()
  for v in "${arr[@]}"; do
    if [ -z "${v:-}" ]; then continue; fi
    if [ -z "${seen[$v]:-}" ]; then seen[$v]=1; out+=("$v"); fi
  done
  printf '%s\n' "${out[@]}" | jq -R -s -c 'split("\n")[:-1]'
}

redirects_json="$(combine_and_dedupe)"

log "Updating SPA redirect URIs (object id: $SPA_APP_OBJECT_ID)"
# Try using az ad app update with --set spa.redirectUris
set_cmd_success=0
if $AZ_CLI ad app update --id "$SPA_APP_OBJECT_ID" --set "spa.redirectUris=${redirects_json}" >/dev/null 2>&1; then
  set_cmd_success=1
  log "SPA redirect URIs updated via az ad app update (spa.redirectUris)."
else
  # Try web.redirectUris fallback
  if $AZ_CLI ad app update --id "$SPA_APP_OBJECT_ID" --set "web.redirectUris=${redirects_json}" >/dev/null 2>&1; then
    set_cmd_success=1
    log "SPA redirect URIs updated via az ad app update (web.redirectUris)."
  else
    log "az ad app update --set failed for spa/web redirectUris; attempting REST fallback (Microsoft Graph) via az rest."
    # Construct patch body
    body="$(jq -n --argjson r "$redirects_json" '{spa: {redirectUris: $r}, web: {redirectUris: $r}}')"
    # Use az rest to patch the application object (requires Graph permissions for caller; fallback may fail)
    PATCH_URL="https://graph.microsoft.com/v1.0/applications/${SPA_APP_OBJECT_ID}"
    if az rest --method PATCH --uri "$PATCH_URL" --headers "Content-Type=application/json" --body "$body" >/dev/null 2>&1; then
      set_cmd_success=1
      log "SPA redirect URIs updated via az rest (Graph PATCH)."
    else
      log "Failed to update redirect URIs via az ad app update and az rest. Please update SPA redirect URIs manually in the Entra portal."
    fi
  fi
fi

# Ensure API identifier URI (api://<appId>) is set as identifierUris (best-effort)
API_IDENTIFIER_URI="api://${API_APP_ID}"
log "Ensuring API identifier URI set to ${API_IDENTIFIER_URI} (best-effort)"
# fetch current identifierUris
cur_identifier_uris="$(printf '%s' "$API_APP_JSON" | $JQ -r '.identifierUris // [] | @json' 2>/dev/null || echo '[]')"
if printf '%s' "$cur_identifier_uris" | $JQ -e --arg id "$API_IDENTIFIER_URI" 'index($id) | . >= 0' >/dev/null 2>&1; then
  log "API identifierUris already contains ${API_IDENTIFIER_URI}"
else
  # attempt to set identifierUris with az ad app update --set
  if $AZ_CLI ad app update --id "$API_APP_ID" --set "identifierUris=[\"${API_IDENTIFIER_URI}\"]" >/dev/null 2>&1; then
    log "Set identifierUris on API app to ${API_IDENTIFIER_URI}."
  else
    log "Could not set identifierUris via az ad app update; skipping. This is non-fatal."
  fi
fi

# Create API client secret (idempotent attempt)
log "Creating API client secret (will print secret; store securely)."
# Use --append if requested and supported; otherwise let CLI create a new password
CREDS_JSON=""
if [ "${APPEND_SECRET:-}" = "--append-secret" ]; then
  if $AZ_CLI ad app credential reset --id "$API_APP_ID" --append --years 5 -o json >/dev/null 2>&1; then
    CREDS_JSON="$($AZ_CLI ad app credential reset --id "$API_APP_ID" --append --years 5 -o json)"
  else
    CREDS_JSON="$($AZ_CLI ad app credential reset --id "$API_APP_ID" --years 5 -o json)"
  fi
else
  CREDS_JSON="$($AZ_CLI ad app credential reset --id "$API_APP_ID" --years 5 -o json)"
fi

API_CLIENT_SECRET="$(printf '%s' "$CREDS_JSON" | $JQ -r '.password // .secretText // empty')"
if [ -z "$API_CLIENT_SECRET" ]; then
  log "Warning: CLI did not return a secret value (some CLI versions may not echo secrets). You may need to create/rotate a secret manually."
else
  log "API client secret created."
fi

# Final outputs (export lines). Print safe guidance and a block you can eval.
cat <<EOF

# --- Export these into your shell to test locally (copy-paste or eval) ---
export SPA_EXISTING_CLIENT_ID="${SPA_APP_ID}"
export API_EXISTING_CLIENT_ID="${API_APP_ID}"
EOF

if [ -n "${API_CLIENT_SECRET:-}" ]; then
  echo "export API_CLIENT_SECRET=\"${API_CLIENT_SECRET}\""
else
  echo "# API_CLIENT_SECRET not returned by CLI; create a secret in portal or run 'az ad app credential reset --id ${API_APP_ID} --years 5 -o json' and export API_CLIENT_SECRET accordingly."
fi

cat <<'EOF'

# Recommended next steps:
# - Store API_CLIENT_SECRET in a secure secret store (Key Vault, etc).
# - If you want the SPA to request access tokens for the API, request the scope "api://<API_APP_ID>/.default" or define explicit OAuth2 scopes in the API app.
# - For production, register exact HTTPS redirect URIs (use a real domain & TLS).
EOF

log "Done."
