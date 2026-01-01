#!/usr/bin/env bash
CLOUDFLARED_VERSION="${CLOUDFLARED_VERSION:-2025.11.1}"
INSTALL_BIN="${INSTALL_BIN:-/usr/local/bin/cloudflared}"
TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-rag-frontend}"
CLOUDLARED_HOME="${HOME}/.cloudflared"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
JQ_CMD="$(command -v jq 2>/dev/null || true)"
ENABLE_DNS_BINDING="${ENABLE_DNS_BINDING:-"true"}"
PUBLIC_HOSTNAME="${FRONTEND_HOSTNAME:-"ui.athithya.site"}"
DNS_OVERWRITE="${DNS_OVERWRITE:-"true"}"
BASHRC_FILE="${BASHRC_FILE:-$HOME/.bashrc}"
mkdir -p "$CLOUDLARED_HOME" 2>/dev/null || true
log(){ printf "%s INFO: %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"; }
warn(){ printf "%s WARN: %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >&2; }
append_to_bashrc(){
  key="$1"
  value="$2"
  file="${BASHRC_FILE}"
  touch "$file" 2>/dev/null || true
  sed -i "/^export ${key}=/d" "$file" 2>/dev/null || true
  printf 'export %s="%s"\n' "$key" "$value" >> "$file"
  printf "%s INFO: persisted %s to %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$key" "$file"
}
b64_of(){
  f="$1"
  if [ ! -f "$f" ]; then
    printf ""
    return 0
  fi
  if base64 --help >/dev/null 2>&1 && base64 --help 2>&1 | grep -q -- '-w'; then
    base64 -w0 "$f" 2>/dev/null || true
  else
    base64 "$f" 2>/dev/null | tr -d '\n' || true
  fi
}
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
  log "CLOUDFLARE_TUNNEL_TOKEN present in environment; ensuring persisted in ${BASHRC_FILE}"
  append_to_bashrc "CLOUDFLARE_TUNNEL_TOKEN" "$CLOUDFLARE_TUNNEL_TOKEN"
  printf "%s INFO: Done. Source %s to use the token.\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BASHRC_FILE"
  return 0 2>/dev/null || true
fi
need_install=0
if command -v cloudflared >/dev/null 2>&1; then
  cur="$(cloudflared --version 2>/dev/null | head -n1 || true)"
  if echo "$cur" | grep -q "$CLOUDFLARED_VERSION"; then
    log "cloudflared present and matches pinned version: $CLOUDFLARED_VERSION"
  else
    log "cloudflared present but version mismatch ($cur). Will install pinned $CLOUDFLARED_VERSION."
    need_install=1
  fi
else
  log "cloudflared not found; will install pinned $CLOUDFLARED_VERSION."
  need_install=1
fi
if [ "$need_install" -eq 1 ]; then
  TMPDL="$(mktemp /tmp/cloudflared.XXXX)" || true
  URL="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
  log "Downloading cloudflared $CLOUDFLARED_VERSION"
  if curl -fsSL -o "$TMPDL" "$URL"; then
    chmod +x "$TMPDL" 2>/dev/null || true
    if [ -w "$(dirname "$INSTALL_BIN")" ] || [ "$(id -u)" -eq 0 ]; then
      mv -f "$TMPDL" "$INSTALL_BIN" 2>/dev/null || true
    else
      sudo mv -f "$TMPDL" "$INSTALL_BIN" 2>/dev/null || true
    fi
    chmod 0755 "$INSTALL_BIN" 2>/dev/null || true
    log "Installed cloudflared at $INSTALL_BIN"
  else
    warn "Failed to download cloudflared from $URL; please install manually and re-run"
    rm -f "$TMPDL" 2>/dev/null || true
  fi
fi
if [ -f "${CLOUDLARED_HOME}/cert.pem" ]; then
  log "cloudflared account cert present at ${CLOUDLARED_HOME}/cert.pem"
else
  log "No account cert found; starting interactive 'cloudflared tunnel login'"
  LOGIN_LOG="$(mktemp /tmp/cloudflared-login.XXXX.log)" || true
  cloudflared tunnel login >"$LOGIN_LOG" 2>&1 &
  printed_url=0
  for i in $(seq 1 "$WAIT_SECONDS"); do
    sleep 1
    if [ -f "$LOGIN_LOG" ]; then
      url="$(grep -Eo 'https?://[^ )\"'\'']+' "$LOGIN_LOG" 2>/dev/null | grep -E 'dash.cloudflare.com|login.cloudflareaccess.org|trycloudflare' | head -n1 || true)"
      if [ -n "$url" ] && [ "$printed_url" -eq 0 ]; then
        printf "\n%s INFO: Open the following URL in a browser to complete login (within %ss):\n%s\n\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$WAIT_SECONDS" "$url"
        printed_url=1
      fi
    fi
    if [ -f "${CLOUDLARED_HOME}/cert.pem" ]; then
      break
    fi
  done
  if [ -f "$LOGIN_LOG" ]; then
    tail -n 20 "$LOGIN_LOG" 2>/dev/null || true
    rm -f "$LOGIN_LOG" 2>/dev/null || true
  fi
  if [ -f "${CLOUDLARED_HOME}/cert.pem" ]; then
    log "Account login succeeded"
  else
    warn "Login did not complete within ${WAIT_SECONDS}s; re-run after completing interactive authorization"
  fi
fi
TUNNEL_ID=""
if command -v cloudflared >/dev/null 2>&1; then
  if [ -n "$JQ_CMD" ]; then
    TUNNEL_ID="$(cloudflared tunnel list --output json 2>&1 | jq -r --arg n "$TUNNEL_NAME" '.[]? | select(.name==$n) | .id' 2>/dev/null || true)"
  else
    TUNNEL_ID="$(cloudflared tunnel list 2>&1 | awk -v n="$TUNNEL_NAME" '$0 ~ n {print $1; exit}' || true)"
  fi
  if [ -n "$TUNNEL_ID" ]; then
    log "Found existing tunnel '$TUNNEL_NAME' (id=$TUNNEL_ID)"
  else
    log "Creating tunnel '$TUNNEL_NAME'"
    pre_files="$(ls -1 "${CLOUDLARED_HOME}"/*.json 2>/dev/null || true)"
    create_out="$(cloudflared tunnel create "$TUNNEL_NAME" 2>&1 || true)"
    printf "%s\n" "$create_out"
    TUNNEL_ID="$(printf "%s\n" "$create_out" | grep -Eo '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' | head -n1 || true)"
    if [ -z "$TUNNEL_ID" ]; then
      post_files="$(ls -1 "${CLOUDLARED_HOME}"/*.json 2>/dev/null || true)"
      new_file=""
      for f in $post_files; do
        if ! printf "%s\n" "$pre_files" | grep -Fxq "$f" 2>/dev/null; then
          new_file="$f"
          break
        fi
      done
      if [ -n "$new_file" ] && [ -f "$new_file" ]; then
        TUNNEL_ID="$(basename "$new_file" .json)"
      fi
    fi
    if [ -z "$TUNNEL_ID" ] && [ -n "$JQ_CMD" ]; then
      TUNNEL_ID="$(cloudflared tunnel list --output json 2>&1 | jq -r --arg n "$TUNNEL_NAME" '.[]? | select(.name==$n) | .id' 2>/dev/null || true)"
    fi
    if [ -n "$TUNNEL_ID" ]; then
      log "Created tunnel id=$TUNNEL_ID"
    else
      warn "Tunnel creation did not return an ID; re-run after confirming login/permissions"
    fi
  fi
  token_out="$(cloudflared tunnel token "$TUNNEL_NAME" 2>&1 || true)"
  if printf "%s\n" "$token_out" | grep -Eq '^[A-Za-z0-9+/=_-]+$'; then
    CLOUDFLARE_TUNNEL_TOKEN="$(printf "%s\n" "$token_out" | tr -d '\r\n')"
    append_to_bashrc "CLOUDFLARE_TUNNEL_TOKEN" "$CLOUDFLARE_TUNNEL_TOKEN"
    log "Tunnel token persisted to ${BASHRC_FILE}"
  else
    creds_candidate=""
    if [ -n "$TUNNEL_ID" ] && [ -f "${CLOUDLARED_HOME}/${TUNNEL_ID}.json" ]; then
      creds_candidate="${CLOUDLARED_HOME}/${TUNNEL_ID}.json"
    else
      creds_candidate="$(ls -1 "${CLOUDLARED_HOME}"/*.json 2>/dev/null | head -n1 || true)"
    fi
    if [ -n "$creds_candidate" ] && [ -f "$creds_candidate" ]; then
      B64="$(b64_of "$creds_candidate")"
      if [ -n "$B64" ]; then
        append_to_bashrc "CLOUDFLARE_TUNNEL_CREDENTIALS_B64" "$B64"
        log "Tunnel credentials persisted to ${BASHRC_FILE}"
      else
        warn "Found credentials file but failed to base64 it: $creds_candidate"
      fi
    else
      warn "No token and no credentials.json found; complete interactive login and re-run"
    fi
  fi
  if [ "${ENABLE_DNS_BINDING}" = "true" ]; then
    if [ -z "${PUBLIC_HOSTNAME}" ]; then
      warn "ENABLE_DNS_BINDING=true but PUBLIC_HOSTNAME not set; skipping DNS binding"
    else
      log "Attempting DNS binding ${PUBLIC_HOSTNAME} -> ${TUNNEL_NAME}"
      if [ "${DNS_OVERWRITE}" = "true" ]; then
        route_cmd=(cloudflared tunnel route dns --overwrite-dns "$TUNNEL_NAME" "$PUBLIC_HOSTNAME")
      else
        route_cmd=(cloudflared tunnel route dns "$TUNNEL_NAME" "$PUBLIC_HOSTNAME")
      fi
      route_out="$("${route_cmd[@]}" 2>&1 || true)"
      printf "%s\n" "$route_out"
      if printf "%s\n" "$route_out" | grep -iqE 'added|created|success|already exists|record exists|already configured|configured to route|Bound|bound'; then
        log "DNS binding succeeded or already existed for ${PUBLIC_HOSTNAME}"
      else
        warn "DNS binding did not clearly succeed; inspect command output above"
      fi
    fi
  fi
else
  warn "cloudflared unavailable; install and re-run"
fi
printf "\n%s SUMMARY:\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if grep -q -E '^export CLOUDFLARE_TUNNEL_TOKEN=' "$BASHRC_FILE" 2>/dev/null; then
  printf "  CLOUDFLARE_TUNNEL_TOKEN persisted to %s\n" "$BASHRC_FILE"
elif grep -q -E '^export CLOUDFLARE_TUNNEL_CREDENTIALS_B64=' "$BASHRC_FILE" 2>/dev/null; then
  printf "  CLOUDFLARE_TUNNEL_CREDENTIALS_B64 persisted to %s\n" "$BASHRC_FILE"
else
  printf "  No token or credentials persisted. Complete login and re-run.\n"
fi
printf "To use in this shell: source %s\n" "$BASHRC_FILE"
printf "%s INFO: Done.\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
