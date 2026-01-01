#!/usr/bin/env bash
set -euo pipefail
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
ERR(){ printf '%s ERROR %s\n' "$(date -Iseconds)" "$*" >&2; }

VM_NAMESPACE=${VM_NAMESPACE:-monitoring}
VMAGENT_PORT=${VMAGENT_PORT:-8429}
VICTORIA_PORT=${VICTORIA_PORT:-8428}
VMAGENT_IMAGE=${VMAGENT_IMAGE:-victoriametrics/vmagent:v1.99.0}
VM_IMAGE=${VM_IMAGE:-victoriametrics/victoria-metrics:v1.99.0}
VMAGENT_REPLICAS=${VMAGENT_REPLICAS:-1}
VM_RES_CPU=${VM_RES_CPU:-100m}
VM_RES_MEM=${VM_RES_MEM:-256Mi}
VMAGENT_RES_CPU=${VMAGENT_RES_CPU:-100m}
VMAGENT_RES_MEM=${VMAGENT_RES_MEM:-256Mi}
VM_SCRAPE_INTERVAL=${VM_SCRAPE_INTERVAL:-15s}
VM_SCRAPE_TIMEOUT=${VM_SCRAPE_TIMEOUT:-10s}
REMOTE_WRITE_URL=${REMOTE_WRITE_URL:-http://victoria-metrics.${VM_NAMESPACE}.svc.cluster.local:${VICTORIA_PORT}/api/v1/write}

QDRANT_NAMESPACE=${QDRANT_NAMESPACE:-qdrant}
RETRIEVAL_NAMESPACE=${RETRIEVAL_NAMESPACE:-inference}

MANIFEST_DIR="${PWD}/infra/manifests"
MANIFEST="${MANIFEST_DIR}/00-monitoring.yaml"
mkdir -p "${MANIFEST_DIR}"

LOCAL_VICTORIA_PORT=${LOCAL_VICTORIA_PORT:-0}
LOCAL_VMAGENT_PORT=${LOCAL_VMAGENT_PORT:-0}
PORTFWD_READY_TIMEOUT=${PORTFWD_READY_TIMEOUT:-30}
PER_POD_PORTFWD_TIMEOUT=${PER_POD_PORTFWD_TIMEOUT:-8}
QUERY_RETRIES=${QUERY_RETRIES:-6}
RETRY_BACKOFF=${RETRY_BACKOFF:-3}
QUERY_SLEEP=${QUERY_SLEEP:-1}
CURL_BIN=${CURL_BIN:-curl}
PYTHON_BIN=${PYTHON_BIN:-python3}

require(){ command -v "$1" >/dev/null 2>&1 || { ERR "$1 required"; exit 2; } }
require kubectl
require "${CURL_BIN}"
require "${PYTHON_BIN}"
require jq
require mktemp
require sed
require awk
require grep

TMPFILES=()
PFPIDS=()
cleanup(){
  rc=$?
  for pid in "${PFPIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  for f in "${TMPFILES[@]:-}"; do
    [ -f "$f" ] && rm -f "$f" || true
  done
  exit $rc
}
trap cleanup INT TERM EXIT

find_free_port(){
  "${PYTHON_BIN}" - <<PY
import socket
s=socket.socket()
s.bind(('',0))
p=s.getsockname()[1]
s.close()
print(p)
PY
}

start_portforward(){
  local ns="$1"; local target="$2"; local local_port="$3"; local remote_port="$4"
  local logfile
  logfile="$(mktemp /tmp/portforward.${target//[^a-zA-Z0-9_.-]/_}.XXXXXX.log)"
  TMPFILES+=("${logfile}")
  kubectl -n "${ns}" port-forward "${target}" "${local_port}:${remote_port}" > "${logfile}" 2>&1 &
  local pid=$!
  PFPIDS+=("${pid}")
  printf '%s|%s' "${pid}" "${logfile}"
}

wait_for_http(){
  local url="$1"; local timeout="$2"; local end=$((SECONDS + timeout))
  while [ "${SECONDS}" -lt "${end}" ]; do
    if "${CURL_BIN}" -sS --max-time 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

validate_numeric_envs(){
  for ev in VMAGENT_PORT VICTORIA_PORT; do
    val="$(eval "printf '%s' \"\$${ev}\"")"
    if ! printf '%s' "${val}" | grep -qE '^[0-9]+$'; then
      ERR "ERROR: ${ev} must be numeric (found: ${val})"
      exit 2
    fi
  done
  case "${REMOTE_WRITE_URL}" in
    http://*|https://*) ;;
    *)
      ERR "ERROR: REMOTE_WRITE_URL must start with http:// or https:// (found: ${REMOTE_WRITE_URL})"
      exit 2
      ;;
  esac
}

render_manifest(){
cat > "${MANIFEST}.tmp" <<'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: __VM_NAMESPACE__
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vmagent
  namespace: __VM_NAMESPACE__
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: vmagent-clusterrole
rules:
  - apiGroups: [""]
    resources: ["pods","endpoints","services","nodes","namespaces"]
    verbs: ["get","list","watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: vmagent-clusterrolebinding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: vmagent-clusterrole
subjects:
  - kind: ServiceAccount
    name: vmagent
    namespace: __VM_NAMESPACE__
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: vmagent-config
  namespace: __VM_NAMESPACE__
data:
  scrape.yml: |
    global:
      scrape_interval: __VM_SCRAPE_INTERVAL__
      scrape_timeout: __VM_SCRAPE_TIMEOUT__
    scrape_configs:
      - job_name: k8s-pods-qdrant
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace]
            action: keep
            regex: __QDRANT_NAMESPACE__
          - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ready]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]
            action: replace
            regex: (.+);(.+)
            replacement: '$1:$2'
            target_label: __address__
          - target_label: __metrics_path__
            replacement: /metrics
      - job_name: k8s-pods-retriever
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace]
            action: keep
            regex: __RETRIEVAL_NAMESPACE__
          - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ready]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]
            action: replace
            regex: (.+);(.+)
            replacement: '$1:$2'
            target_label: __address__
          - target_label: __metrics_path__
            replacement: /metrics
      - job_name: k8s-pods-annotated
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ready]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]
            action: replace
            regex: (.+);(.+)
            replacement: '$1:$2'
            target_label: __address__
          - target_label: __metrics_path__
            replacement: /metrics
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vmagent
  namespace: __VM_NAMESPACE__
spec:
  replicas: __VMAGENT_REPLICAS__
  selector:
    matchLabels:
      app: vmagent
  template:
    metadata:
      labels:
        app: vmagent
    spec:
      serviceAccountName: vmagent
      containers:
        - name: vmagent
          image: __VMAGENT_IMAGE__
          args:
            - "-promscrape.config=/config/scrape.yml"
            - "-remoteWrite.url=__REMOTE_WRITE_URL__"
            - "-remoteWrite.tmpDataPath=/vmagent-remotewrite-data"
            - "-promscrape.suppressDuplicateScrapeTargetErrors=true"
            - "-promscrape.suppressScrapeErrors=true"
            - "-promscrape.suppressScrapeErrorsDelay=30s"
          ports:
            - containerPort: __VMAGENT_PORT__
          readinessProbe:
            httpGet:
              path: /metrics
              port: __VMAGENT_PORT__
            initialDelaySeconds: 3
            periodSeconds: 10
          resources:
            requests:
              cpu: __VMAGENT_RES_CPU__
              memory: __VMAGENT_RES_MEM__
            limits:
              cpu: __VMAGENT_RES_CPU__
              memory: __VMAGENT_RES_MEM__
          volumeMounts:
            - name: config
              mountPath: /config
            - name: tmp
              mountPath: /vmagent-remotewrite-data
      volumes:
        - name: config
          configMap:
            name: vmagent-config
        - name: tmp
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: vmagent
  namespace: __VM_NAMESPACE__
spec:
  selector:
    app: vmagent
  ports:
    - name: metrics
      port: __VMAGENT_PORT__
      targetPort: __VMAGENT_PORT__
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victoria-metrics
  namespace: __VM_NAMESPACE__
spec:
  replicas: 1
  selector:
    matchLabels:
      app: victoria-metrics
  template:
    metadata:
      labels:
        app: victoria-metrics
    spec:
      containers:
        - name: victoria-metrics
          image: __VM_IMAGE__
          args:
            - "-retentionPeriod=1d"
            - "-storageDataPath=/data"
            - "-httpListenAddr=:__VICTORIA_PORT__"
          ports:
            - containerPort: __VICTORIA_PORT__
          readinessProbe:
            httpGet:
              path: /health
              port: __VICTORIA_PORT__
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests:
              cpu: __VM_RES_CPU__
              memory: __VM_RES_MEM__
            limits:
              cpu: __VM_RES_CPU__
              memory: __VM_RES_MEM__
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: victoria-metrics
  namespace: __VM_NAMESPACE__
spec:
  selector:
    app: victoria-metrics
  ports:
    - port: __VICTORIA_PORT__
      targetPort: __VICTORIA_PORT__
EOF

  sed -e "s|__VM_NAMESPACE__|${VM_NAMESPACE}|g" \
      -e "s|__VM_SCRAPE_INTERVAL__|${VM_SCRAPE_INTERVAL}|g" \
      -e "s|__VM_SCRAPE_TIMEOUT__|${VM_SCRAPE_TIMEOUT}|g" \
      -e "s|__QDRANT_NAMESPACE__|${QDRANT_NAMESPACE}|g" \
      -e "s|__RETRIEVAL_NAMESPACE__|${RETRIEVAL_NAMESPACE}|g" \
      -e "s|__VMAGENT_IMAGE__|${VMAGENT_IMAGE}|g" \
      -e "s|__VMAGENT_REPLICAS__|${VMAGENT_REPLICAS}|g" \
      -e "s|__VMAGENT_PORT__|${VMAGENT_PORT}|g" \
      -e "s|__VMAGENT_RES_CPU__|${VMAGENT_RES_CPU}|g" \
      -e "s|__VMAGENT_RES_MEM__|${VMAGENT_RES_MEM}|g" \
      -e "s|__REMOTE_WRITE_URL__|${REMOTE_WRITE_URL}|g" \
      -e "s|__VM_IMAGE__|${VM_IMAGE}|g" \
      -e "s|__VICTORIA_PORT__|${VICTORIA_PORT}|g" \
      -e "s|__VM_RES_CPU__|${VM_RES_CPU}|g" \
      -e "s|__VM_RES_MEM__|${VM_RES_MEM}|g" \
      "${MANIFEST}.tmp" > "${MANIFEST}.tmp2" && mv "${MANIFEST}.tmp2" "${MANIFEST}.tmp"

  # build the expected literal pattern safely (avoid accidental expansion of $1/$2)
  PAT="$(printf "replacement: '%s'" '$1:$2')"
  if grep -Fq "${PAT}" "${MANIFEST}.tmp" 2>/dev/null ; then
    LOG "detected replacement literal ${PAT}"
  else
    ERR "ERROR: expected literal replacement: '\$1:$2' missing in ${MANIFEST}.tmp"
    exit 1
  fi

  # ensure no unexpected backslashes remain that could corrupt the replacement
  if grep -q "\\\\" "${MANIFEST}.tmp"; then
    ERR "ERROR: manifest contains backslash characters that may invalidate scrape addresses; aborting"
    exit 1
  fi

  mv "${MANIFEST}.tmp" "${MANIFEST}"
  LOG "rendered ${MANIFEST}"
}

apply(){
  validate_numeric_envs
  kubectl create namespace "${VM_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
  render_manifest
  kubectl apply -f "${MANIFEST}"
  kubectl -n "${VM_NAMESPACE}" rollout restart deployment vmagent >/dev/null 2>&1 || true
  LOG "waiting for vmagent availability (120s)"
  if kubectl -n "${VM_NAMESPACE}" wait --for=condition=Available deployment/vmagent --timeout=120s >/dev/null 2>&1; then LOG "vmagent available"; else LOG "warning: vmagent not marked available after 120s"; fi
  LOG "monitoring apply complete into ${VM_NAMESPACE}"
}

# Improved probe: checks vm_promscrape_series_fetched OR remote-write bytes sent
probe_vmagent_targets(){
  local tries=0 max=20
  while [ $tries -lt $max ]; do
    out="$("${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" 2>/dev/null || echo "")"
    sf="$(printf '%s' "${out}" | awk '/vm_promscrape_series_fetched/ {print $2; exit}')"
    if [ -n "${sf}" ] && printf '%s' "${sf}" | grep -qE '^[0-9]+' >/dev/null 2>&1; then
      if [ "${sf}" -gt 0 ] 2>/dev/null; then
        LOG "vmagent reports series fetched: ${sf}"
        return 0
      fi
    fi
    rw_bytes="$(printf '%s' "${out}" | awk '/vmagent_remotewrite_sent_bytes_total/ {print $2; exit}')"
    if [ -n "${rw_bytes}" ] && printf '%s' "${rw_bytes}" | grep -qE '^[0-9]+' >/dev/null 2>&1; then
      if [ "${rw_bytes}" -gt 0 ] 2>/dev/null; then
        LOG "vmagent reports remote-write bytes sent: ${rw_bytes}"
        return 0
      fi
    fi
    tries=$((tries+1))
    sleep 2
  done
  ERR "vmagent scrape/remote-write metrics not observed locally after wait"
  return 1
}

run_promql_with_retries(){
  local name="$1"; local promql="$2"; local expect="$3"
  local last_json=""
  BASE_VM="http://127.0.0.1:${LOCAL_VICTORIA_PORT}/api/v1/query"
  for attempt in $(seq 1 "${QUERY_RETRIES}"); do
    LOG "PromQL ${name} attempt ${attempt}/${QUERY_RETRIES}: ${promql}"
    json="$("${CURL_BIN}" -sS -G --data-urlencode "query=${promql}" "${BASE_VM}" 2>/dev/null || echo "")"
    last_json="${json}"
    ok=$(echo "${json}" | jq -r 'select(.status=="success") | .data.result | length' 2>/dev/null || echo "0")
    if [ "${ok}" != "0" ]; then
      LOG "PromQL ${name} returned ${ok} result(s)"
      val=$(echo "${json}" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "")
      if [ "${expect}" = "gt0" ]; then
        if [ -n "${val}" ] && "${PYTHON_BIN}" - <<PY "${val}"
import sys
try: v=float(sys.argv[1]); sys.exit(0 if v>0 else 1)
except: sys.exit(2)
PY
        then LOG "PASS ${name} -> ${val}"; return 0; fi
      elif [ "${expect}" = "eq1" ]; then
        if [ -n "${val}" ] && "${PYTHON_BIN}" - <<PY "${val}"
import sys,math
try: v=float(sys.argv[1]); sys.exit(0 if math.isclose(v,1.0,rel_tol=0,abs_tol=1e-9) else 1)
except: sys.exit(2)
PY
        then LOG "PASS ${name} -> ${val}"; return 0; fi
      elif [ "${expect}" = "anynum" ]; then
        if [ -n "${val}" ] && "${PYTHON_BIN}" - <<PY "${val}"
import sys
try: float(sys.argv[1]); sys.exit(0)
except: sys.exit(2)
PY
        then LOG "PASS ${name} -> ${val}"; return 0; fi
      fi
    else
      LOG "PromQL ${name} produced no results; retrying"
    fi
    sleep $((RETRY_BACKOFF * attempt + QUERY_SLEEP))
  done
  ERR "FAIL ${name} after ${QUERY_RETRIES} attempts; last response:"
  if [ -n "${last_json}" ]; then
    echo "${last_json}" | jq . || echo "${last_json}"
  else
    echo "{}"
  fi
  return 1
}

validate_end_to_end(){
  LOG "starting VictoriaMetrics port-forward (svc/victoria-metrics ns=${VM_NAMESPACE})"
  if [ "${LOCAL_VICTORIA_PORT:-0}" -eq 0 ]; then LOCAL_VICTORIA_PORT="$(find_free_port)"; fi
  pv="$(start_portforward "${VM_NAMESPACE}" "svc/victoria-metrics" "${LOCAL_VICTORIA_PORT}" "${VICTORIA_PORT}")"
  LOG "waiting up to ${PORTFWD_READY_TIMEOUT}s for VictoriaMetrics /metrics"
  if ! wait_for_http "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
    pfile="$(printf '%s' "${pv}" | awk -F'|' '{print $2}')"
    ERR "victoria-metrics port-forward not ready; tail ${pfile}"
    tail -n 200 "${pfile}" || true
    return 2
  fi
  LOG "VictoriaMetrics port-forward ready (local:${LOCAL_VICTORIA_PORT})"

  LOG "starting vmagent port-forward (svc/vmagent ns=${VM_NAMESPACE})"
  if [ "${LOCAL_VMAGENT_PORT:-0}" -eq 0 ]; then LOCAL_VMAGENT_PORT="$(find_free_port)"; fi
  pm="$(start_portforward "${VM_NAMESPACE}" "svc/vmagent" "${LOCAL_VMAGENT_PORT}" "${VMAGENT_PORT}")"
  LOG "waiting up to ${PORTFWD_READY_TIMEOUT}s for vmagent /metrics"
  if ! wait_for_http "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" "${PORTFWD_READY_TIMEOUT}"; then
    pfile="$(printf '%s' "${pm}" | awk -F'|' '{print $2}')"
    ERR "vmagent port-forward not ready; tail ${pfile}"
    tail -n 200 "${pfile}" || true
    return 3
  fi
  LOG "vmagent port-forward ready (local:${LOCAL_VMAGENT_PORT})"

  LOG "probing vmagent to ensure it reports scrape metrics or remote-write activity"
  if ! probe_vmagent_targets; then
    ERR "vmagent does not appear to report scrape/remote-write metrics locally; cannot proceed"
    return 4
  fi

  LOG "validated vmagent & victoria basic connectivity; now ensure kubernetes discovery works (vmagent must have RBAC)"
  return 0
}

delete(){
  kubectl -n "${VM_NAMESPACE}" delete -f "${MANIFEST}" --ignore-not-found || true
  # best-effort cleanup of cluster-scoped RBAC created by this manifest
  kubectl delete clusterrole vmagent-clusterrole --ignore-not-found || true
  kubectl delete clusterrolebinding vmagent-clusterrolebinding --ignore-not-found || true
  LOG "monitoring deleted (best-effort)"
}

usage(){ printf '%s\n' "usage: $0 --generate|--apply|--delete|--validate" ; exit 1; }

if [ "$#" -lt 1 ]; then usage; fi
cmd=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --generate) cmd="generate"; shift;;
    --apply) cmd="apply"; shift;;
    --delete) cmd="delete"; shift;;
    --validate) cmd="validate"; shift;;
    *) usage;;
  esac
done

if [ "${cmd}" = "generate" ]; then render_manifest && LOG "rendered ${MANIFEST}" ; exit 0; fi
if [ "${cmd}" = "apply" ]; then apply && LOG "applied monitoring"; exit 0; fi
if [ "${cmd}" = "delete" ]; then delete && exit 0; fi
if [ "${cmd}" = "validate" ]; then
  validate_end_to_end || exit 1
  LOG "validation succeeded"
  exit 0
fi
usage
