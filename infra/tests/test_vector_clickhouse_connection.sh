#!/usr/bin/env bash
# infra/tests/test_vector_clickhouse_connection.sh
# Non-exiting end-to-end test: ClickHouse reachable, insert, deploy test pod,
# verify Vector -> ClickHouse ingestion, show Vector logs.
#
# Usage: just run the script. It will use exported env vars if present,
# otherwise fallback to kind-safe defaults.

# --- Defaults (overridable via env) ---
NAMESPACE="${NAMESPACE:-observability}"
CLICKHOUSE_SERVICE_NAME="${CLICKHOUSE_SERVICE_NAME:-clickhouse}"
CLICKHOUSE_HTTP_PORT="${CLICKHOUSE_HTTP_PORT:-8123}"
CLICKHOUSE_USER="${CLICKHOUSE_USER:-vector}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-vectorpass}"
LOCAL_PORT="${LOCAL_PORT:-8123}"
TEST_POD_NAME="${TEST_POD_NAME:-test-logger}"
INGEST_WAIT_SEC="${INGEST_WAIT_SEC:-10}"
PORTFWD_WAIT_SEC="${PORTFWD_WAIT_SEC:-5}"
CLICKHOUSE_PING_TIMEOUT="${CLICKHOUSE_PING_TIMEOUT:-30}"
VECTOR_LOG_LINES="${VECTOR_LOG_LINES:-200}"

# Internal state
PF_PID=0
TMP_TEST_MANIFEST="/tmp/${TEST_POD_NAME}.yaml"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

cleanup() {
  rc=$?
  log "cleanup: deleting test pod (if present) and stopping port-forward"
  kubectl -n "$NAMESPACE" delete pod "$TEST_POD_NAME" --ignore-not-found >/dev/null 2>&1 || true
  if [ "$PF_PID" -ne 0 ]; then
    kill "$PF_PID" >/dev/null 2>&1 || true
    wait "$PF_PID" 2>/dev/null || true
  fi
  log "cleanup done (exit code $rc)"
  return 0
}
trap cleanup EXIT

# --- helper: start port-forward and validate ClickHouse ping ---
start_port_forward() {
  log "starting port-forward: svc/${CLICKHOUSE_SERVICE_NAME} ${LOCAL_PORT}:${CLICKHOUSE_HTTP_PORT}"
  kubectl -n "$NAMESPACE" port-forward "svc/${CLICKHOUSE_SERVICE_NAME}" \
    "${LOCAL_PORT}:${CLICKHOUSE_HTTP_PORT}" >/dev/null 2>&1 &
  PF_PID=$!
  sleep 1

  # wait for ping up to timeout
  log "waiting for ClickHouse HTTP ping (timeout ${CLICKHOUSE_PING_TIMEOUT}s)"
  start_ts=$(date +%s)
  while true; do
    if curl -sS "http://127.0.0.1:${LOCAL_PORT}/ping" >/dev/null 2>&1; then
      log "ClickHouse ping OK"
      return 0
    fi
    now=$(date +%s)
    if [ $((now - start_ts)) -ge "$CLICKHOUSE_PING_TIMEOUT" ]; then
      log "ClickHouse ping timed out after ${CLICKHOUSE_PING_TIMEOUT}s"
      return 1
    fi
    sleep 1
  done
}

# --- helper: run ClickHouse HTTP query (POST body), prints output ---
clickhouse_query() {
  local query="$1"
  curl -sS "http://127.0.0.1:${LOCAL_PORT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}" \
    --data-binary "$query"
}

# --- start ---
log "test script started"
log "using namespace=${NAMESPACE}"
log "clickhouse service=${CLICKHOUSE_SERVICE_NAME} port=${CLICKHOUSE_HTTP_PORT}"
log "local forward port=${LOCAL_PORT}"

# 1) Start port-forward
start_port_forward
if [ $? -ne 0 ]; then
  log "warning: port-forward failed; script will continue but tests may fail"
else
  log "sleep ${PORTFWD_WAIT_SEC}s to stabilize port-forward"
  sleep "$PORTFWD_WAIT_SEC"
fi

# 2) Direct insert test (manual insert)
log "inserting manual test row into ClickHouse (bypassing Vector)"
MANUAL_MSG="manual-insert-$(date +%s)"
MANUAL_JSON="{\"pod\":\"manual\",\"namespace\":\"default\",\"message\":\"${MANUAL_MSG}\"}"
printf '%s\n' "$MANUAL_JSON" \
  | curl -sS "http://127.0.0.1:${LOCAL_PORT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}&query=INSERT%20INTO%20logs.kube_logs%20(pod,namespace,message)%20FORMAT%20JSONEachRow" \
    --data-binary @- >/dev/null 2>&1
if [ $? -eq 0 ]; then
  log "manual insert request sent"
else
  log "manual insert request failed (continue)"
fi

# verify manual insert (may require grants; will show error if any)
log "verifying manual insert (querying ClickHouse)"
MANUAL_SELECT="SELECT ts,pod,namespace,message FROM logs.kube_logs WHERE message LIKE '%${MANUAL_MSG}%' FORMAT JSON"
res=$(clickhouse_query "$MANUAL_SELECT" 2>/dev/null) || res=""
if [ -n "$res" ]; then
  echo "$res" | sed -n '1,200p'
else
  log "no response for manual select or query failed"
fi

# 3) Deploy test-logger pod to produce logs for Vector
log "deploying test pod ${TEST_POD_NAME} that emits vector-test-* lines"
cat > "$TMP_TEST_MANIFEST" <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  name: test-logger
  labels:
    app: test-logger
spec:
  restartPolicy: Never
  containers:
    - name: logger
      image: busybox
      command:
        - sh
        - -c
        - |
          i=0
          while [ $i -lt 30 ]; do
            echo "{\"message\":\"vector-test-$i\"}"
            i=$((i+1))
            sleep 1
          done
YAML

kubectl -n "$NAMESPACE" apply -f "$TMP_TEST_MANIFEST" >/dev/null 2>&1
if [ $? -eq 0 ]; then
  log "test pod applied"
else
  log "failed to apply test pod (continue)"
fi

# wait for the pod to start
log "waiting up to 20s for test pod to enter Running or Completed"
start_ts=$(date +%s)
while true; do
  state=$(kubectl -n "$NAMESPACE" get pod "$TEST_POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
  if [ "$state" = "Running" ] || [ "$state" = "Succeeded" ] || [ "$state" = "Completed" ]; then
    log "test pod state: $state"
    break
  fi
  if [ $(( $(date +%s) - start_ts )) -ge 20 ]; then
    log "timed out waiting for test pod to start (state: $state); continuing"
    break
  fi
  sleep 1
done

# 4) Wait for Vector to ingest
log "waiting ${INGEST_WAIT_SEC}s for Vector to pick up and forward logs"
sleep "$INGEST_WAIT_SEC"

# 5) Query for vector-test rows
log "querying ClickHouse for records matching vector-test-"
VECTOR_SELECT="SELECT ts,pod,namespace,message FROM logs.kube_logs WHERE message LIKE '%vector-test-%' ORDER BY ts DESC LIMIT 20 FORMAT JSON"
vres=$(clickhouse_query "$VECTOR_SELECT" 2>/dev/null) || vres=""
if [ -n "$vres" ]; then
  log "ClickHouse query result (first 400 chars shown)"
  echo "$vres" | sed -n '1,200p'
else
  log "no result from ClickHouse for vector-test-* (possible Vector sink/auth/endpoint issue)"
fi

# 6) Show Vector logs for recent activity
log "showing last ${VECTOR_LOG_LINES} lines of Vector logs (cluster)"
kubectl -n "$NAMESPACE" logs -l app=vector --tail="$VECTOR_LOG_LINES" 2>/dev/null || {
  log "could not retrieve Vector logs (kubectl failed)"
}

# 7) Final status summary
log "final status summary:"
kubectl -n "$NAMESPACE" get pods -l app=clickhouse -o wide 2>/dev/null || true
kubectl -n "$NAMESPACE" get pods -l app=vector -o wide 2>/dev/null || true
kubectl -n "$NAMESPACE" get svc "$CLICKHOUSE_SERVICE_NAME" -o wide 2>/dev/null || true

log "test script completed (non-exiting). Cleanup will run now."

# Cleanup handled by trap on EXIT
# End of script
