#!/usr/bin/env bash
ROOT_DIR="$(pwd)"
PORT="${PORT:-8000}"
LOG_FILE="/tmp/orchestrator.log"
PID_FILE="/tmp/orchestrator.pid"
TIMEOUT="${TIMEOUT:-20}"
PROBE_INTERVAL="${PROBE_INTERVAL:-5}"

required_envs=(JWT_SECRET SESSION_SECRET OAUTH_REDIRECT_BASE FRONTEND_BASE)
missing=()
for v in "${required_envs[@]}"; do
  if [ -z "${!v:-}" ]; then
    missing+=("$v")
  fi
done
if [ "${#missing[@]}" -ne 0 ]; then
  echo "ERROR: Missing required env vars: ${missing[*]}" >&2
  echo "Export them, e.g.:"
  echo "  export JWT_SECRET=... SESSION_SECRET=... OAUTH_REDIRECT_BASE=http://localhost:${PORT} FRONTEND_BASE=http://localhost:${PORT}"
  exit 2
fi

uvicorn_cmd="uvicorn app:app --host 0.0.0.0 --port ${PORT}"
UVICORN_RELOAD_FLAG="${UVICORN_RELOAD:-true}"
if [ "${UVICORN_RELOAD_FLAG}" = "true" ]; then
  uvicorn_cmd="${uvicorn_cmd} --reload"
fi

echo "[monitor] Killing any previous 'uvicorn app:app' (best-effort)..."
pkill -f "uvicorn app:app" || true
sleep 0.2
rm -f "${PID_FILE}" "${LOG_FILE}"

echo "[monitor] Starting orchestrator: ${uvicorn_cmd}"
nohup ${uvicorn_cmd} > "${LOG_FILE}" 2>&1 &
UVICORN_PID=$!
echo "${UVICORN_PID}" > "${PID_FILE}"
echo "[monitor] started pid=${UVICORN_PID}, logs -> ${LOG_FILE}"

# Wait a short time then attempt to detect readiness (non-fatal if it doesn't become ready)
echo "[monitor] Waiting up to ${TIMEOUT}s for /orchestrator/health..."
start_ts=$(date +%s)
ready=false
while true; do
  if curl -sSf "http://127.0.0.1:${PORT}/orchestrator/health" >/dev/null 2>&1; then
    ready=true
    break
  fi
  now=$(date +%s)
  if [ $((now - start_ts)) -ge "${TIMEOUT}" ]; then
    break
  fi
  sleep 0.5
done

if [ "${ready}" = true ]; then
  echo "[monitor] Orchestrator reported healthy."
else
  echo "[monitor] Orchestrator not healthy within ${TIMEOUT}s — continuing monitoring loop (will report status)."
fi

# trap to gracefully stop uvicorn when user hits Ctrl-C
cleanup() {
  echo
  echo "[monitor] Cleaning up..."
  if [ -f "${PID_FILE}" ]; then
    pid=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${pid}" ]; then
      echo "[monitor] Killing pid ${pid}..."
      kill "${pid}" 2>/dev/null || true
      sleep 0.2
    fi
    rm -f "${PID_FILE}"
  fi
  echo "[monitor] Exiting."
  exit 0
}
trap cleanup INT TERM

# continuous non-exiting monitoring loop
echo
echo "=== STARTING NON-EXITING MONITOR LOOP ==="
echo "Open the UI in your browser at: http://127.0.0.1:${PORT} (use Incognito to avoid stale cookies during OAuth flows)"
echo "Press Ctrl-C to stop."

while true; do
  ts="$(date --iso-8601=seconds 2>/dev/null || date)"
  printf "\n===== %s =====\n" "${ts}"

  # check uvicorn process alive
  if kill -0 "${UVICORN_PID}" >/dev/null 2>&1; then
    echo "[proc] orchestrator pid=${UVICORN_PID} (alive)"
  else
    echo "[proc] orchestrator pid=${UVICORN_PID} NOT RUNNING"
  fi

  # Probes: root, login, providers fragment, status fragment, me, metrics
  probe() {
    local path="$1"; local label="$2"
    set +e
    http_out="/tmp/monitor_resp.$$"
    http_code=$(curl -sS -o "${http_out}" -w '%{http_code}' "http://127.0.0.1:${PORT}${path}" 2>/dev/null || echo "000")
    set -e
    printf "%-36s %s\n" "${label}" "${http_code}"
    if [ -s "${http_out}" ]; then
      echo "---- response (first 8 lines) ----"
      head -n 8 "${http_out}" | sed -n '1,8p'
    fi
    rm -f "${http_out}" 2>/dev/null || true
  }

  probe "/" "Frontend root (HTML)"
  probe "/login" "Login page (full UI)"
  probe "/auth/fragment/providers" "Auth fragment: providers"
  probe "/auth/fragment/status" "Auth fragment: status"
  probe "/auth/me" "Auth /me (no cookie expected 401/4xx)"
  probe "/metrics" "Prometheus /metrics"

  # Print last 12 lines of orchestrator log for quick debugging
  echo "---- last 12 lines of ${LOG_FILE} ----"
  if [ -f "${LOG_FILE}" ]; then
    tail -n 12 "${LOG_FILE}" || true
  else
    echo "(no log file yet)"
  fi

  sleep "${PROBE_INTERVAL}"
done