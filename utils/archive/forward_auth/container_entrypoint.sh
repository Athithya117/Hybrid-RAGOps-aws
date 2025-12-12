#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=8000}"
AUTH_ALLOW_MISSING=${AUTH_ALLOW_MISSING:-1}
if [ -z "${JWT_SECRET:-}" ] || [ -z "${SESSION_SECRET:-}" ]; then
  if [ "${AUTH_ALLOW_MISSING}" = "1" ] || [ "${AUTH_ALLOW_MISSING:-}" = "true" ]; then
    echo "Warning: JWT_SECRET/SESSION_SECRET missing but AUTH_ALLOW_MISSING_SECRETS is set. Running in permissive mode."
  else
    echo "Error: JWT_SECRET and SESSION_SECRET must be set (or set AUTH_ALLOW_MISSING_SECRETS=1)."
    exit 2
  fi
fi

COOKIE_MODE=${COOKIE_MODE:-both}
if ! echo "cookie localstorage both" | grep -qw "$COOKIE_MODE"; then
  echo "Error: COOKIE_MODE must be one of cookie|localstorage|both"
  exit 2
fi

if [ "${ENV:-DEV}" = "PROD" ] && [ "${COOKIE_MODE}" = "cookie" ] && [ "${COOKIE_SECURE:-}" != "1" ] && [ "${COOKIE_SECURE:-}" != "true" ]; then
  echo "Warning: Running in PROD with COOKIE_MODE=cookie but COOKIE_SECURE not enabled. This may prevent cookies from being sent over HTTPS."
fi

exec uvicorn stateless_openid_auth:app --host 0.0.0.0 --port "${PORT}" --proxy-headers --log-level info
