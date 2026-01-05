#!/usr/bin/env bash
# infra/tests/monitoring/qdrant_metrics_e2e.sh
# E2E validator: Qdrant -> vmagent -> Victoria (enhanced)
# Compatible with Qdrant v1.16.x and common vmagent/Victoria installs.
set -euo pipefail
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
ERR(){ printf '%s ERROR %s\n' "$(date -Iseconds)" "$*" >&2; }

# -------------------- config (tweakable) --------------------
QDRANT_RELEASE=${QDRANT_RELEASE:-qdrant}
QDRANT_NAMESPACE=${QDRANT_NAMESPACE:-qdrant}
VM_NAMESPACE=${VM_NAMESPACE:-monitoring}
VICTORIA_SERVICE=${VICTORIA_SERVICE:-victoria-metrics}
VICTORIA_PORT=${VICTORIA_PORT:-8428}
VMAGENT_SERVICE=${VMAGENT_SERVICE:-vmagent}
VMAGENT_PORT=${VMAGENT_PORT:-8429}
QDRANT_API_PORT=${QDRANT_API_PORT:-6333}

LOCAL_VICTORIA_PORT=${LOCAL_VICTORIA_PORT:-0}
LOCAL_VMAGENT_PORT=${LOCAL_VMAGENT_PORT:-0}
LOCAL_QDRANT_PORT=${LOCAL_QDRANT_PORT:-0}

PORTFWD_READY_TIMEOUT=${PORTFWD_READY_TIMEOUT:-30}
QUERY_RETRIES=${QUERY_RETRIES:-8}
RETRY_BACKOFF=${RETRY_BACKOFF:-2}
QUERY_SLEEP=${QUERY_SLEEP:-1}

LOAD_RPS=${LOAD_RPS:-100}
LOAD_DURATION=${LOAD_DURATION:-60}
LOAD_WORKERS=${LOAD_WORKERS:-8}
FAIL_RATE=${FAIL_RATE:-0.02}
SEED_COLLECTION_NAME=${SEED_COLLECTION_NAME:-e2e_test_collection}
SEED_VECTOR_SIZE=${SEED_VECTOR_SIZE:-4}
SEED_POINTS=${SEED_POINTS:-200}

FAIL_ON_RECOMMENDED=${FAIL_ON_RECOMMENDED:-true}
REQUIRE_VM_PERSISTENCE=${REQUIRE_VM_PERSISTENCE:-false}

QUERY_ARTIFACT_DIR=${QUERY_ARTIFACT_DIR:-/tmp/qdrant_e2e_artifacts}
mkdir -p "${QUERY_ARTIFACT_DIR}"

CURL_BIN=${CURL_BIN:-curl}
PYTHON_BIN=${PYTHON_BIN:-python3}
require(){ command -v "$1" >/dev/null 2>&1 || { ERR "$1 required"; exit 2; } }
require kubectl; require "${CURL_BIN}"; require "${PYTHON_BIN}"; require jq; require mktemp; require awk; require sed; require grep

# -------------------- metric lists --------------------
ALERTING_REQUIRED=(
  "rest_responses_total"
  "rest_responses_fail_total"
  "rest_responses_duration_seconds_bucket"
  "memory_resident_bytes"
)
DASHBOARD_RECOMMENDED=(
  "collections_total"
  "collections_vector_total"
  "collection_running_optimizations"
  "collection_points"
  "process_open_fds"
  "process_threads"
  "collection_hardware_metric_vector_io_read"
  "collection_hardware_metric_vector_io_write"
  "cluster_peers_total"
  "cluster_pending_operations_total"
  "app_status_recovery_mode"
)

# vmagent candidates (from docs + common variants)
VMAGENT_PERSISTENCE_CANDIDATES=(
  "vm_persistentqueue_bytes_written_total"
  "vm_persistentqueue_written_bytes_total"
  "vm_promscrape_conn_bytes_written_total"
)
VMAGENT_SCRAPE_CANDIDATES=(
  "vm_promscrape_scraped_samples_sum"
  "vm_promscrape_scraped_samples_total"
)
VMAGENT_DISCOVERY_CANDIDATES=(
  "vm_promscrape_discovery_kubernetes_objects"
  "vm_promscrape_discovery_kubernetes_pods"
)

# -------------------- housekeeping --------------------
TMPFILES=()
PFPIDS=()
CLEANUP_COL=""
trap 'rc=$?; LOG "trap: cleaning up"; for pid in "${PFPIDS[@]:-}"; do kill "$pid" >/dev/null 2>&1 || true; done; if [ -n "${CLEANUP_COL}" ]; then curl -sS -X DELETE "http://127.0.0.1:${LOCAL_QDRANT_PORT}/collections/${CLEANUP_COL}" >/dev/null 2>&1 || true; fi; exit $rc' INT TERM EXIT

find_free_port(){
  "${PYTHON_BIN}" - <<PY
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
}

start_portforward(){
  local ns="$1" target="$2" local_port="$3" remote_port="$4"
  local logfile
  logfile="$(mktemp /tmp/portforward.${target//[^a-zA-Z0-9_.-]/_}.XXXXXX.log)"
  TMPFILES+=("${logfile}")
  kubectl -n "${ns}" port-forward "${target}" "${local_port}:${remote_port}" > "${logfile}" 2>&1 &
  local pid=$!
  PFPIDS+=("${pid}")
  echo "${pid}|${logfile}"
}

wait_for_http(){
  local url="$1" timeout_s="$2"
  local end=$((SECONDS + timeout_s))
  while [ "${SECONDS}" -lt "${end}" ]; do
    if "${CURL_BIN}" -sS --max-time 3 "${url}" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

# -------------------- small helpers --------------------
# runs a PromQL and returns 0 on success (satisfies expect), non-zero otherwise; prints brief result
BASE_VM="http://127.0.0.1:${LOCAL_VICTORIA_PORT}/api/v1/query"
query_promql(){ local promql="$1"; "${CURL_BIN}" -sS -G --data-urlencode "query=${promql}" "${BASE_VM}" | jq -c .; }

run_promql_with_retries(){
  local name="$1"; local promql="$2"; local expect="$3"
  local last_json=""
  for attempt in $(seq 1 "${QUERY_RETRIES}"); do
    LOG "PromQL ${name} attempt ${attempt}/${QUERY_RETRIES}: ${promql}"
    json="$(query_promql "${promql}")" || json=""
    last_json="${json}"
    status=$(echo "${json}" | jq -r '.status // empty' 2>/dev/null || echo "")
    if [ "${status}" != "success" ]; then
      LOG "PromQL ${name} status=${status:-<no-status>}; retrying"
      sleep $((RETRY_BACKOFF * attempt + QUERY_SLEEP)); continue
    fi
    result_len=$(echo "${json}" | jq -r '.data.result | length' 2>/dev/null || echo "0")
    sf=$(echo "${json}" | jq -r '.stats.seriesFetched // 0' 2>/dev/null || echo "0")
    if [ "${expect}" = "any" ] && [ "${result_len}" -gt 0 ]; then
      echo "${json}" >/tmp/qdrant_e2e_lastjson_"${name}.json"
      return 0
    fi
    if [ "${result_len}" -gt 0 ]; then
      val=$(echo "${json}" | jq -r '.data.result[0].value[1] // empty' 2>/dev/null || echo "")
      if [ -n "${val}" ]; then
        if "${PYTHON_BIN}" - <<PY "${val}"
import sys
try:
    v=float(sys.argv[1]); sys.exit(0 if v>0 else 1)
except:
    sys.exit(2)
PY
        then echo "${json}" >/tmp/qdrant_e2e_lastjson_"${name}.json"; return 0; fi
      fi
    fi
    if [ -n "${sf}" ] && [ "${sf}" -gt 0 ] 2>/dev/null; then
      echo "${json}" >/tmp/qdrant_e2e_lastjson_"${name}.json"; return 0
    fi
    LOG "PromQL ${name} produced no satisfying results; retrying"
    sleep $((RETRY_BACKOFF * attempt + QUERY_SLEEP))
  done
  echo "${last_json}" >/tmp/qdrant_e2e_lastjson_"${name}.json" || true
  return 1
}

# print one-line human message for a metric check (local + tsdb)
report_metric(){
  local metric_display="$1"
  local local_ok="$2"
  local tsdb_ok="$3"
  local note="$4"
  if [ "${local_ok}" = "1" ] && [ "${tsdb_ok}" = "1" ]; then
    LOG "METRIC OK: ${metric_display} — present locally and ingested into TSDB. ${note}"
  elif [ "${local_ok}" = "1" ] && [ "${tsdb_ok}" = "0" ]; then
    LOG "METRIC PARTIAL: ${metric_display} — present locally but NOT visible in TSDB. ${note}"
  else
    LOG "METRIC MISSING: ${metric_display} — not present locally. ${note}"
  fi
}

# runs local grep on last snapshot(s)
check_local_metric(){
  local m="$1"
  if grep -qE "^${m}(\{|[[:space:]]|$)" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_snapshot.txt" 2>/dev/null || grep -qE "^${m}(\{|[[:space:]]|$)" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" 2>/dev/null; then
    return 0
  fi
  return 1
}

# -------------------- main flows --------------------
LOG "port-forward VictoriaMetrics svc/${VICTORIA_SERVICE} (ns=${VM_NAMESPACE})"
if [ "${LOCAL_VICTORIA_PORT}" -eq 0 ]; then LOCAL_VICTORIA_PORT="$(find_free_port)"; fi
pv="$(start_portforward "${VM_NAMESPACE}" "svc/${VICTORIA_SERVICE}" "${LOCAL_VICTORIA_PORT}" "${VICTORIA_PORT}")"
LOG "waiting for VictoriaMetrics /metrics"
if ! wait_for_http "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
  ERR "VictoriaMetrics not reachable; tail of port-forward log:"
  tail -n 200 "$(printf '%s' "${pv}" | awk -F'|' '{print $2}')" || true
  exit 4
fi
LOG "Victoria port-forward ready (local:${LOCAL_VICTORIA_PORT})"

LOG "port-forward vmagent svc/${VMAGENT_SERVICE} (ns=${VM_NAMESPACE})"
if [ "${LOCAL_VMAGENT_PORT}" -eq 0 ]; then LOCAL_VMAGENT_PORT="$(find_free_port)"; fi
pm="$(start_portforward "${VM_NAMESPACE}" "svc/${VMAGENT_SERVICE}" "${LOCAL_VMAGENT_PORT}" "${VMAGENT_PORT}")"
LOG "waiting for vmagent /metrics"
if ! wait_for_http "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
  ERR "vmagent not reachable; tail of port-forward log:"
  tail -n 200 "$(printf '%s' "${pm}" | awk -F'|' '{print $2}')" || true
  exit 5
fi
LOG "vmagent port-forward ready (local:${LOCAL_VMAGENT_PORT})"

LOG "discovering qdrant pods in namespace ${QDRANT_NAMESPACE}"
PODS="$(kubectl -n "${QDRANT_NAMESPACE}" get pods -l "app.kubernetes.io/name=${QDRANT_RELEASE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
if [ -z "${PODS}" ]; then
  PODS="$(kubectl -n "${QDRANT_NAMESPACE}" get pods -l "app=${QDRANT_RELEASE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
fi
if [ -z "${PODS}" ]; then ERR "no qdrant pods found"; exit 6; fi
LOG "found qdrant pods: ${PODS}"

LOG "port-forwarding qdrant svc/${QDRANT_RELEASE} port ${QDRANT_API_PORT} -> localhost:${LOCAL_QDRANT_PORT}"
if [ "${LOCAL_QDRANT_PORT}" -eq 0 ]; then LOCAL_QDRANT_PORT="$(find_free_port)"; fi
qp="$(start_portforward "${QDRANT_NAMESPACE}" "svc/${QDRANT_RELEASE}" "${LOCAL_QDRANT_PORT}" "${QDRANT_API_PORT}")"
if ! wait_for_http "http://127.0.0.1:${LOCAL_QDRANT_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
  ERR "qdrant API not reachable; tail of port-forward log:"
  tail -n 200 "$(printf '%s' "${qp}" | awk -F'|' '{print $2}')" || true
  exit 7
fi
LOG "qdrant API port-forward ready (local:${LOCAL_QDRANT_PORT})"

LOG "snapshot local Qdrant /metrics"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_QDRANT_PORT}/metrics" > "${QUERY_ARTIFACT_DIR}/qdrant_metrics_snapshot.txt"

# ---- local required checks ----
for m in "${ALERTING_REQUIRED[@]}"; do
  if check_local_metric "${m}"; then
    report_metric "${m}" 1 0 "local presence verified; will check TSDB ingestion below"
  else
    report_metric "${m}" 0 0 "local metric missing; check Qdrant /metrics endpoint and instrumentation per docs."
    ERR "Required metric ${m} missing locally; aborting"
    exit 11
  fi
done

# recommended local checks
for m in "${DASHBOARD_RECOMMENDED[@]}"; do
  if check_local_metric "${m}"; then
    report_metric "${m}" 1 0 "recommended metric present locally"
  else
    if [ "${FAIL_ON_RECOMMENDED}" = "true" ]; then
      report_metric "${m}" 0 0 "recommended metric absent locally (FAIL_ON_RECOMMENDED=true)"
      ERR "Recommended metric ${m} absent locally; failing due to FAIL_ON_RECOMMENDED"
      exit 12
    else
      report_metric "${m}" 0 0 "recommended metric absent locally (ok with current config)"
    fi
  fi
done

# vmagent local detection (explicit messages)
vmagent_persist_found=false
for c in "${VMAGENT_PERSISTENCE_CANDIDATES[@]}"; do
  if "${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" | grep -qE "^${c}(\{|[[:space:]]|$)"; then
    LOG "vmagent LOCAL: persistence metric present: ${c}"
    vmagent_persist_found=true
    break
  fi
done
if [ "${vmagent_persist_found}" = "false" ]; then
  LOG "vmagent LOCAL: no persistence metrics found among candidates (${VMAGENT_PERSISTENCE_CANDIDATES[*]}); vmagent persistence may be disabled"
  if [ "${REQUIRE_VM_PERSISTENCE}" = "true" ]; then ERR "vmagent persistence required but not found locally"; exit 13; fi
fi

vmagent_scrape_found=false
for c in "${VMAGENT_SCRAPE_CANDIDATES[@]}"; do
  if "${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" | grep -qE "^${c}(\{|[[:space:]]|$)"; then
    LOG "vmagent LOCAL: scrape-sample metric present: ${c}"
    vmagent_scrape_found=true
    break
  fi
done
if [ "${vmagent_scrape_found}" = "false" ]; then
  LOG "vmagent LOCAL: no scrape-sample metrics found among candidates (${VMAGENT_SCRAPE_CANDIDATES[*]}); vmagent may be configured without those metrics"
fi

# -------------------- seed & load --------------------
LOG "creating collection ${SEED_COLLECTION_NAME} and upserting ${SEED_POINTS} points"
create_out="$("${CURL_BIN}" -sS -X PUT "http://127.0.0.1:${LOCAL_QDRANT_PORT}/collections/${SEED_COLLECTION_NAME}" -H "Content-Type: application/json" -d "{\"vectors\":{\"size\":${SEED_VECTOR_SIZE},\"distance\":\"Cosine\"}}")"
echo "${create_out}" | jq -c . || true

bsize=100; sent=0
while [ "${sent}" -lt "${SEED_POINTS}" ]; do
  chunk=$(( SEED_POINTS - sent ))
  if [ "${chunk}" -gt "${bsize}" ]; then chunk=${bsize}; fi
  pts_json="["
  for i in $(seq 1 "${chunk}"); do
    id=$((sent + i))
    vec="$("${PYTHON_BIN}" - <<PY
import json
print(json.dumps([0.1]*${SEED_VECTOR_SIZE}))
PY
)"
    pts_json="${pts_json}{\"id\":${id},\"vector\":${vec}}"
    if [ "${i}" -lt "${chunk}" ]; then pts_json="${pts_json},"; fi
  done
  pts_json="${pts_json}]"
  payload="{\"points\":${pts_json}}"
  upsert_out="$("${CURL_BIN}" -sS -X PUT "http://127.0.0.1:${LOCAL_QDRANT_PORT}/collections/${SEED_COLLECTION_NAME}/points?wait=true" -H "Content-Type: application/json" -d "${payload}")"
  echo "${upsert_out}" | jq -c . || true
  sent=$((sent + chunk))
  LOG "upserted ${sent}/${SEED_POINTS}"
done
CLEANUP_COL="${SEED_COLLECTION_NAME}"
sleep 3

LOG "starting load: rps=${LOAD_RPS}, duration=${LOAD_DURATION}s, workers=${LOAD_WORKERS}, fail_rate=${FAIL_RATE}"
"${PYTHON_BIN}" - <<PY
import threading, time, json, urllib.request, random, statistics, sys
host="http://127.0.0.1:${LOCAL_QDRANT_PORT}"
collection="${SEED_COLLECTION_NAME}"
workers=${LOAD_WORKERS}
duration=${LOAD_DURATION}
rps=${LOAD_RPS}
fail_rate=${FAIL_RATE}
limit=10
stop_at=time.time()+duration
stats={"req":0,"fails":0,"lat":[]}
lock=threading.Lock()
per_worker=max(1, rps//max(1,workers))
sleep_per=1.0/per_worker
def do_search():
    url=f"{host}/collections/{{}}/points/search"
    while time.time() < stop_at:
        target=collection
        if random.random() < fail_rate:
            target=collection+"_NO_SUCH"
        body=json.dumps({"vector":[0.1]*${SEED_VECTOR_SIZE},"limit":limit}).encode("utf-8")
        req=urllib.request.Request(url.format(target), data=body, headers={"Content-Type":"application/json"})
        t0=time.time()
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                _=r.read(); ok=True
        except Exception:
            ok=False
        lat=time.time()-t0
        with lock:
            stats["req"]+=1
            if not ok: stats["fails"]+=1
            stats["lat"].append(lat)
        time.sleep(sleep_per)
threads=[]
for i in range(workers):
    t=threading.Thread(target=do_search, daemon=True)
    threads.append(t); t.start()
for t in threads:
    t.join(timeout=duration+5)
lat_sorted=sorted(stats["lat"])
def pct(a,p):
    if not a: return None
    k=int(len(a)*p)
    k=min(k,len(a)-1)
    return a[k]
print(json.dumps({"requests":stats["req"],"fails":stats["fails"],"p50": pct(lat_sorted,0.50),"p95": pct(lat_sorted,0.95)}))
PY

sleep 5
LOG "snapshot post-load local metrics"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_QDRANT_PORT}/metrics" > "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt"

# -------------------- local post-load checks --------------------
if ! grep -qE "^rest_responses_total(\{|[[:space:]]|$)" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt"; then ERR "rest_responses_total missing"; exit 21; fi
rest_total=$(grep -E "^rest_responses_total" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" | awk '{print $2}' | head -n1)
LOG "LOCAL CHECK: rest_responses_total -> ${rest_total}"
if [ -z "${rest_total}" ] || [ "${rest_total}" -eq 0 ]; then ERR "rest_responses_total did not increase"; exit 22; fi

if ! grep -qE "^rest_responses_duration_seconds_count(\{|[[:space:]]|$)" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt"; then ERR "histogram count missing"; exit 23; fi
hist_count=$(grep -E "^rest_responses_duration_seconds_count" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" | awk '{print $2}' | head -n1)
LOG "LOCAL CHECK: rest_responses_duration_seconds_count -> ${hist_count}"

if ! grep -qE "^collections_vector_total(\{|[[:space:]]|$)" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt"; then ERR "collections_vector_total missing"; exit 24; fi
vec_total=$(grep -E "^collections_vector_total" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" | awk '{print $2}' | head -n1)
LOG "LOCAL CHECK: collections_vector_total -> ${vec_total}"

if ! grep -qE "^memory_resident_bytes(\{|[[:space:]]|$)" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt"; then ERR "memory_resident_bytes missing"; exit 25; fi
mem=$(grep -E "^memory_resident_bytes" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" | awk '{print $2}' | head -n1)
LOG "LOCAL CHECK: memory_resident_bytes -> ${mem}"

# -------------------- TSDB checks with explicit messages --------------------
# core series
if run_promql_with_retries "qdrant_core_series" 'count({__name__=~"app_info|collections_total|collections_vector_total"})' gt0; then
  LOG "TSDB CHECK: core metrics (app_info, collections_total, collections_vector_total) -> OK"
  report_metric "core_series" 1 1 "app_info/collections metrics present in TSDB"
else
  LOG "TSDB CHECK: core metrics (app_info/collections_total/collections_vector_total) -> MISSING"
  report_metric "core_series" 1 0 "local present but not in TSDB"
fi

# rest_responses_total
if run_promql_with_retries "rest_responses_total_tsdb" 'sum(rate(rest_responses_total[1m]))' gt0; then
  report_metric "rest_responses_total" 1 1 "throughput visible in TSDB"
else
  report_metric "rest_responses_total" 1 0 "local present but not visible in TSDB"
fi

# rest_responses_fail_total (may be zero)
if run_promql_with_retries "rest_fail_total_tsdb" 'sum(rate(rest_responses_fail_total[1m]))' any; then
  report_metric "rest_responses_fail_total" 1 1 "failure counter visible in TSDB (may be zero)"
else
  report_metric "rest_responses_fail_total" 1 0 "failure counter not visible in TSDB (ok if zero/absent)"
fi

# histogram buckets
if run_promql_with_retries "histogram_buckets_tsdb" 'count(rest_responses_duration_seconds_bucket)' gt0; then
  report_metric "rest_responses_duration_seconds_bucket" 1 1 "histogram buckets visible in TSDB"
else
  report_metric "rest_responses_duration_seconds_bucket" 1 0 "histogram buckets not visible in TSDB"
fi

# collections_vector_total
if run_promql_with_retries "collections_vectors_tsdb" 'sum(collections_vector_total)' gt0; then
  report_metric "collections_vector_total" 1 1 "vector count visible in TSDB"
else
  report_metric "collections_vector_total" 1 0 "vector count not visible in TSDB"
fi

# memory_resident_bytes
if run_promql_with_retries "memory_resident_tsdb" 'max(memory_resident_bytes)' gt0; then
  report_metric "memory_resident_bytes" 1 1 "memory gauge visible in TSDB"
else
  report_metric "memory_resident_bytes" 1 0 "memory gauge not visible in TSDB"
fi

# vmagent persistence (try aggregated regex query to cover all candidates)
if run_promql_with_retries "vm_persistent_any" 'count({__name__=~"vm_persistentqueue_bytes_written_total|vm_persistentqueue_written_bytes_total|vm_promscrape_conn_bytes_written_total"})' gt0; then
  LOG "TSDB CHECK: vmagent persistence metrics -> OK (one of known candidates ingested)"
  report_metric "vmagent_persistence" 1 1 "persistence metric ingested"
else
  LOG "TSDB CHECK: vmagent persistence metrics -> MISSING"
  report_metric "vmagent_persistence" ${vmagent_persist_found:+1} 0 "local:${vmagent_persist_found} - TSDB missing; ensure Victoria scrapes vmagent or vmagent remote_write paths configured."
fi

# vmagent scrape samples (aggregate regex)
if run_promql_with_retries "vm_scrape_any" 'count({__name__=~"vm_promscrape_scraped_samples_sum|vm_promscrape_scraped_samples_total"})' gt0; then
  LOG "TSDB CHECK: vmagent scrape samples -> OK"
  report_metric "vmagent_scrape_samples" ${vmagent_scrape_found:+1} 1 "scrape-sample metrics ingested"
else
  LOG "TSDB CHECK: vmagent scrape samples -> MISSING"
  report_metric "vmagent_scrape_samples" ${vmagent_scrape_found:+1} 0 "local:${vmagent_scrape_found} - TSDB missing; adjust vmagent -> Victoria configuration"
fi

# qdrant 'up' — try several fallback selectors
UP_OK=0
# 1) job common
if run_promql_with_retries "qdrant_up_job" 'max(up{job=~"qdrant|qdrant.*"})' any; then UP_OK=1; LOG "TSDB CHECK: up by job -> OK"; fi
# 2) instance port pattern (instance ending with :6333)
if [ "${UP_OK}" -eq 0 ]; then
  if run_promql_with_retries "qdrant_up_instance" 'max(up{instance=~".*:6333"})' any; then UP_OK=1; LOG "TSDB CHECK: up by instance:port -> OK"; fi
fi
# 3) pod name pattern
if [ "${UP_OK}" -eq 0 ]; then
  if run_promql_with_retries "qdrant_up_pod" 'max(up{pod=~"qdrant.*|qdrant-.*"})' any; then UP_OK=1; LOG "TSDB CHECK: up by pod label -> OK"; fi
fi
if [ "${UP_OK}" -eq 1 ]; then
  report_metric "up" 0 1 "service 'up' visible in TSDB (one of fallback selectors matched)"
else
  report_metric "up" 0 0 "service 'up' not visible in TSDB for tried selectors; find actual job/pod label and adapt dashboard"
fi

# cluster metrics (conditional)
if grep -qE "^cluster_enabled" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" 2>/dev/null || grep -qE "^cluster_peers_total" "${QUERY_ARTIFACT_DIR}/qdrant_metrics_postload.txt" 2>/dev/null; then
  if run_promql_with_retries "cluster_peers_tsdb" 'max(cluster_peers_total)' any; then
    report_metric "cluster_peers_total" 1 1 "cluster peers visible in TSDB"
  else
    report_metric "cluster_peers_total" 1 0 "cluster peers local present but not in TSDB"
  fi
  if run_promql_with_retries "cluster_pending_ops_tsdb" 'max(cluster_pending_operations_total)' any; then
    report_metric "cluster_pending_operations_total" 1 1 "cluster pending ops visible in TSDB"
  else
    report_metric "cluster_pending_operations_total" 1 0 "cluster pending ops local present but not in TSDB"
  fi
fi

# recommended enforcement (TSDB checks + message)
for m in "${DASHBOARD_RECOMMENDED[@]}"; do
  if run_promql_with_retries "recommended_${m}" "max(${m})" any; then
    report_metric "${m}" 1 1 "recommended metric present in TSDB"
  else
    if [ "${FAIL_ON_RECOMMENDED}" = "true" ]; then
      report_metric "${m}" 1 0 "recommended metric missing in TSDB (failing)"
      ERR "Recommended metric ${m} missing in TSDB; FAIL_ON_RECOMMENDED=true"
      exit 41
    else
      report_metric "${m}" 1 0 "recommended metric not in TSDB (warning)"
    fi
  fi
done

# CPU fallback checks (container -> node) with clear messages
if run_promql_with_retries "qdrant_container_cpu" 'max(container_cpu_usage_seconds_total{container=~"qdrant|qdrant.*"})' any; then
  report_metric "container_cpu_usage_seconds_total" 0 1 "container CPU visible in TSDB"
elif run_promql_with_retries "qdrant_node_cpu" 'max(node_cpu_seconds_total{mode!="idle"})' any; then
  report_metric "node_cpu_seconds_total" 0 1 "node CPU visible in TSDB"
else
  report_metric "cpu" 0 0 "CPU metrics not found in TSDB; Qdrant v1.16 does not emit cpu_ metrics — use node/container exporters instead. See docs."
fi

LOG "ALL CHECKS COMPLETE; artifacts in ${QUERY_ARTIFACT_DIR}"
LOG "If any metric shows PARTIAL or MISSING, follow the message for the precise fix (scrape config, vmagent remote_write, or relabeling)."

# cleanup seeded collection
if [ -n "${CLEANUP_COL}" ]; then
  LOG "removing seeded collection ${CLEANUP_COL}"
  "${CURL_BIN}" -sS -X DELETE "http://127.0.0.1:${LOCAL_QDRANT_PORT}/collections/${CLEANUP_COL}" >/dev/null 2>&1 || true
fi

exit 0
