#!/usr/bin/env bash
set -euo pipefail
make deploy-qdrant
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
ERR(){ printf '%s ERROR %s\n' "$(date -Iseconds)" "$*" >&2; }

QDRANT_RELEASE=${QDRANT_RELEASE:-qdrant}
QDRANT_NAMESPACE=${QDRANT_NAMESPACE:-qdrant}
VM_NAMESPACE=${VM_NAMESPACE:-monitoring}
VICTORIA_SERVICE=${VICTORIA_SERVICE:-victoria-metrics}
VICTORIA_PORT=${VICTORIA_PORT:-8428}
VMAGENT_SERVICE=${VMAGENT_SERVICE:-vmagent}
VMAGENT_PORT=${VMAGENT_PORT:-8429}
LOCAL_VICTORIA_PORT=${LOCAL_VICTORIA_PORT:-0}
LOCAL_VMAGENT_PORT=${LOCAL_VMAGENT_PORT:-0}
PORTFWD_READY_TIMEOUT=${PORTFWD_READY_TIMEOUT:-15}
PER_POD_PORTFWD_TIMEOUT=${PER_POD_PORTFWD_TIMEOUT:-8}
QUERY_RETRIES=${QUERY_RETRIES:-3}
RETRY_BACKOFF=${RETRY_BACKOFF:-2}
QUERY_SLEEP=${QUERY_SLEEP:-1}
CURL_BIN=${CURL_BIN:-curl}
PYTHON_BIN=${PYTHON_BIN:-python3}
SEED_DATA=${SEED_DATA:-false}
SEED_COLLECTION_NAME=${SEED_COLLECTION_NAME:-e2e_test_collection}
SEED_VECTOR_SIZE=${SEED_VECTOR_SIZE:-4}
SEED_POINT_ID=${SEED_POINT_ID:-1}
SEED_VECTOR=${SEED_VECTOR:-0.1,0.2,0.3,0.4}
CLEANUP_SEED=${CLEANUP_SEED:-true}
QDRANT_API_PORT=${QDRANT_API_PORT:-6333}
QUERY_ARTIFACT_DIR=${QUERY_ARTIFACT_DIR:-/tmp/test_qdrant_artifacts}

parse_bool(){
  v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${v}" in
    1|true|yes|y) echo "true";;
    0|false|no|n|"") echo "false";;
    *) echo "false";;
  esac
}

require(){ command -v "$1" >/dev/null 2>&1 || { ERR "$1 required"; exit 2; } }
require kubectl
require "${CURL_BIN}"
require "${PYTHON_BIN}"
require mktemp
require sed
require awk
require grep
require jq

SEED_DATA="$(parse_bool "${SEED_DATA}")"
CLEANUP_SEED="$(parse_bool "${CLEANUP_SEED}")"

TMPFILES=()
PFPIDS=()
mkdir -p "${QUERY_ARTIFACT_DIR}"

cleanup(){
  rc=$?
  for pid in "${PFPIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [ "${SEED_DATA}" = "true" ] && [ "${CLEANUP_SEED}" = "true" ] && [ -n "${CLEANUP_ACTION_FILE:-}" ] && [ -f "${CLEANUP_ACTION_FILE}" ]; then
    col="$(cat "${CLEANUP_ACTION_FILE}")"
    LOG "cleaning up seeded collection ${col}"
    if [ -n "${LOCAL_QDRANT_API_PORT:-}" ]; then
      LOG "deleting collection via local qdrant api at port ${LOCAL_QDRANT_API_PORT}"
      "${CURL_BIN}" -sS -X DELETE "http://127.0.0.1:${LOCAL_QDRANT_API_PORT}/collections/${col}" | jq -c . || true
    fi
  fi
  for f in "${TMPFILES[@]:-}"; do
    [ -f "$f" ] && rm -f "$f" || true
  done
  exit $rc
}
trap cleanup INT TERM EXIT

find_free_port(){
  "${PYTHON_BIN}" - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
p=s.getsockname()[1]
s.close()
print(p)
PY
}

start_portforward(){
  local ns="$1"
  local target="$2"
  local local_port="$3"
  local remote_port="$4"
  local logfile
  logfile="$(mktemp /tmp/portforward.${target//[^a-zA-Z0-9_.-]/_}.XXXXXX.log)"
  TMPFILES+=("${logfile}")
  kubectl -n "${ns}" port-forward "${target}" "${local_port}:${remote_port}" > "${logfile}" 2>&1 &
  local pid=$!
  PFPIDS+=("${pid}")
  echo "${pid}|${logfile}"
}

wait_for_http(){
  local url="$1"
  local timeout="$2"
  local end=$((SECONDS + timeout))
  while [ "${SECONDS}" -lt "${end}" ]; do
    if "${CURL_BIN}" -sS --max-time 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

LOG "starting VictoriaMetrics port-forward (svc/${VICTORIA_SERVICE} ns=${VM_NAMESPACE})"
if [ "${LOCAL_VICTORIA_PORT:-0}" -eq 0 ]; then LOCAL_VICTORIA_PORT="$(find_free_port)"; fi
pv="$(start_portforward "${VM_NAMESPACE}" "svc/${VICTORIA_SERVICE}" "${LOCAL_VICTORIA_PORT}" "${VICTORIA_PORT}")"
LOG "waiting up to ${PORTFWD_READY_TIMEOUT}s for VictoriaMetrics /metrics"
if ! wait_for_http "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
  ERR "victoria-metrics port-forward not ready; last 200 lines of port-forward log:"
  pfile="$(printf '%s' "${pv}" | awk -F'|' '{print $2}')"
  tail -n 200 "${pfile}" || true
  exit 4
fi
LOG "VictoriaMetrics port-forward ready (local:${LOCAL_VICTORIA_PORT})"

LOG "starting vmagent port-forward (svc/${VMAGENT_SERVICE} ns=${VM_NAMESPACE})"
if [ "${LOCAL_VMAGENT_PORT:-0}" -eq 0 ]; then LOCAL_VMAGENT_PORT="$(find_free_port)"; fi
pm="$(start_portforward "${VM_NAMESPACE}" "svc/${VMAGENT_SERVICE}" "${LOCAL_VMAGENT_PORT}" "${VMAGENT_PORT}")"
LOG "waiting up to ${PORTFWD_READY_TIMEOUT}s for vmagent /metrics"
if ! wait_for_http "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
  ERR "vmagent port-forward not ready; last 200 lines of port-forward log:"
  pfile="$(printf '%s' "${pm}" | awk -F'|' '{print $2}')"
  tail -n 200 "${pfile}" || true
  exit 5
fi
LOG "vmagent port-forward ready (local:${LOCAL_VMAGENT_PORT})"

LOG "discovering qdrant pods in namespace ${QDRANT_NAMESPACE}"
PODS="$(kubectl -n "${QDRANT_NAMESPACE}" get pods -l "app.kubernetes.io/name=${QDRANT_RELEASE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
if [ -z "${PODS}" ]; then
  PODS="$(kubectl -n "${QDRANT_NAMESPACE}" get pods -l "app=${QDRANT_RELEASE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
fi
if [ -z "${PODS}" ]; then
  ERR "no qdrant pods found in ${QDRANT_NAMESPACE}"
  exit 6
fi
LOG "found pods: ${PODS}"

for pod in ${PODS}; do
  LOG "validating annotations for pod ${pod}"
  ann_scrape="$(kubectl -n "${QDRANT_NAMESPACE}" get pod "${pod}" -o jsonpath='{.metadata.annotations.monitoring\.io/scrape}' 2>/dev/null || true)"
  ann_port="$(kubectl -n "${QDRANT_NAMESPACE}" get pod "${pod}" -o jsonpath='{.metadata.annotations.monitoring\.io/port}' 2>/dev/null || true)"
  ann_path="$(kubectl -n "${QDRANT_NAMESPACE}" get pod "${pod}" -o jsonpath='{.metadata.annotations.monitoring\.io/path}' 2>/dev/null || true)"
  LOG "annotations: scrape=${ann_scrape:-<unset>} port=${ann_port:-<unset>} path=${ann_path:-<unset>}"

  if [ "${ann_scrape}" != "true" ]; then
    ERR "annotation monitoring.io/scrape=true missing on ${pod}"
    exit 7
  fi
  if ! printf '%s' "${ann_port}" | grep -qE '^[0-9]+$'; then
    ERR "annotation monitoring.io/port must be numeric on ${pod} (found: ${ann_port})"
    exit 8
  fi
  if [ "${ann_port}" != "${QDRANT_API_PORT}" ]; then
    ERR "annotation monitoring.io/port mismatch expected ${QDRANT_API_PORT} found ${ann_port}"
    exit 9
  fi
  if [ "${ann_path}" != "/metrics" ]; then
    ERR "annotation monitoring.io/path must be /metrics on ${pod} (found: ${ann_path})"
    exit 10
  fi

  local_port="$(find_free_port)"
  LOG "port-forwarding pod/${pod}:${ann_port} -> localhost:${local_port}"
  pfp="$(start_portforward "${QDRANT_NAMESPACE}" "pod/${pod}" "${local_port}" "${ann_port}")"
  pfpid="$(printf '%s' "${pfp}" | awk -F'|' '{print $1}')"
  pf_log="$(printf '%s' "${pfp}" | awk -F'|' '{print $2}')"
  if ! wait_for_http "http://127.0.0.1:${local_port}${ann_path}" "${PER_POD_PORTFWD_TIMEOUT}"; then
    ERR "port-forward to pod ${pod} failed or metrics not responding; last 200 lines of port-forward log:"
    tail -n 200 "${pf_log}" || true
    if kill -0 "${pfpid}" >/dev/null 2>&1; then kill "${pfpid}" >/dev/null 2>&1 || true; fi
    continue
  fi

  LOG "fetching /metrics from pod ${pod} (showing head)"
  "${CURL_BIN}" -sS "http://127.0.0.1:${local_port}${ann_path}" | sed -n '1,120p' || true

  if kill -0 "${pfpid}" >/dev/null 2>&1; then
    kill "${pfpid}" >/dev/null 2>&1 || true
    wait "${pfpid}" 2>/dev/null || true
  fi
done

LOG "validating vmagent target state via local vmagent /metrics (showing relevant lines)"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" | grep -E 'vm_promscrape_|vmagent_' | sed -n '1,200p' || true

BASE_VM="http://127.0.0.1:${LOCAL_VICTORIA_PORT}/api/v1/query"

query_promql(){
  local promql="$1"
  "${CURL_BIN}" -sS -G --data-urlencode "query=${promql}" "${BASE_VM}" | jq -c .
}

run_promql_with_retries(){
  local name="$1"; shift
  local promql="$1"; shift
  local expect="$1"; shift
  local last_json=""
  for attempt in $(seq 1 "${QUERY_RETRIES}"); do
    LOG "PromQL ${name} attempt ${attempt}/${QUERY_RETRIES}: ${promql}"
    json="$(query_promql "${promql}")" || json=""
    echo "${json}"
    last_json="${json}"
    ok=$(echo "${json}" | jq -r 'select(.status=="success") | .data.result | length' 2>/dev/null || echo "0")
    if [ "${ok}" != "0" ]; then
      LOG "PromQL ${name} returned ${ok} result(s)"
      val=$(echo "${json}" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "")
      if [ "${expect}" = "gt0" ]; then
        if [ -n "${val}" ] && "${PYTHON_BIN}" - <<PY "${val}"
import sys
try:
    v=float(sys.argv[1])
    sys.exit(0 if v>0 else 1)
except:
    sys.exit(2)
PY
        then LOG "PASS ${name} -> ${val}"; return 0; fi
      elif [ "${expect}" = "eq1" ]; then
        if [ -n "${val}" ] && "${PYTHON_BIN}" - <<PY "${val}"
import sys,math
try:
    v=float(sys.argv[1])
    sys.exit(0 if math.isclose(v,1.0,rel_tol=0,abs_tol=1e-9) else 1)
except:
    sys.exit(2)
PY
        then LOG "PASS ${name} -> ${val}"; return 0; fi
      elif [ "${expect}" = "anynum" ]; then
        if [ -n "${val}" ] && "${PYTHON_BIN}" - <<PY "${val}"
import sys
try:
    float(sys.argv[1]); sys.exit(0)
except:
    sys.exit(2)
PY
        then LOG "PASS ${name} -> ${val}"; return 0; fi
      elif [ "${expect}" = "gt0_seriesfetched" ]; then
        sf=$(echo "${json}" | jq -r '.stats.seriesFetched // empty' 2>/dev/null || echo "")
        if [ -n "${sf}" ] && [ "${sf}" -gt 0 ] 2>/dev/null; then
          LOG "PASS ${name} -> seriesFetched=${sf}"; return 0
        fi
      else
        ERR "unknown expectation ${expect}"; return 20
      fi
    else
      LOG "PromQL ${name} produced no results; retrying"
    fi
    sleep $((RETRY_BACKOFF * attempt + QUERY_SLEEP))
  done
  ERR "FAIL ${name} after ${QUERY_RETRIES} attempts; last response:"
  echo "${last_json}" | jq -c . || echo "${last_json}"
  return 1
}

run_promql_with_retries "qdrant_metrics_exist" 'count({__name__=~"app_info|collections_total|collections_vector_total"})' gt0
run_promql_with_retries "up_namespace_qdrant" 'max(up{namespace="qdrant"})' eq1

if [ "${SEED_DATA}" = "true" ]; then
  LOG "SEED_DATA=true: seeding minimal collection '${SEED_COLLECTION_NAME}' into Qdrant"
  if [ -z "${LOCAL_QDRANT_API_PORT:-}" ] || [ "${LOCAL_QDRANT_API_PORT}" -eq 0 ]; then LOCAL_QDRANT_API_PORT="$(find_free_port)"; fi
  qpf="$(start_portforward "${QDRANT_NAMESPACE}" "svc/${QDRANT_RELEASE}" "${LOCAL_QDRANT_API_PORT}" "${QDRANT_API_PORT}")"
  LOG "waiting up to ${PORTFWD_READY_TIMEOUT}s for Qdrant API"
  if ! wait_for_http "http://127.0.0.1:${LOCAL_QDRANT_API_PORT}/collections" "${PORTFWD_READY_TIMEOUT}"; then
    ERR "qdrant API port-forward not ready; last 200 lines of port-forward log:"
    pfile="$(printf '%s' "${qpf}" | awk -F'|' '{print $2}')"
    tail -n 200 "${pfile}" || true
    exit 20
  fi

  LOG "creating collection ${SEED_COLLECTION_NAME} (idempotent)"
  create_out="$("${CURL_BIN}" -sS -X PUT "http://127.0.0.1:${LOCAL_QDRANT_API_PORT}/collections/${SEED_COLLECTION_NAME}" -H "Content-Type: application/json" -d "{\"vectors\":{\"size\":${SEED_VECTOR_SIZE},\"distance\":\"Cosine\"}}")"
  echo "${create_out}" | jq -c . || echo "${create_out}"
  LOG "upserting one test point (idempotent)"
  vector_json="$(printf '[%s]' "${SEED_VECTOR}")"
  points_payload="$(printf '{"points":[{"id":%s,"vector":%s}]}' "${SEED_POINT_ID}" "${vector_json}")"
  upsert_out="$("${CURL_BIN}" -sS -X PUT "http://127.0.0.1:${LOCAL_QDRANT_API_PORT}/collections/${SEED_COLLECTION_NAME}/points?wait=true" -H "Content-Type: application/json" -d "${points_payload}")"
  echo "${upsert_out}" | jq -c . || echo "${upsert_out}"

  if [ "${CLEANUP_SEED}" = "true" ]; then
    CLEANUP_ACTION_FILE="$(mktemp /tmp/qdrant.cleanup.XXXXXX)"; TMPFILES+=("${CLEANUP_ACTION_FILE}")
    printf '%s' "${SEED_COLLECTION_NAME}" > "${CLEANUP_ACTION_FILE}"
    LOG "registered to delete seeded collection ${SEED_COLLECTION_NAME} at script exit"
  fi

  LOG "waiting for ingestion of seeded data into TSDB"
  sleep 5
  run_promql_with_retries "collections_vector_total_sum" 'sum(collections_vector_total)' gt0
else
  LOG "SEED_DATA=false; skipping seeded-data ingestion check (this run validates the pipeline only)"
fi

LOG "ALL VALIDATIONS PASSED"
exit 0
