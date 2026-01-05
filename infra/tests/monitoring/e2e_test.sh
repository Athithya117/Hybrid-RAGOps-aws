make lc
make deploy-qdrant
make deploy-models

make run-indexing-cronjob-kind:

export BACKUP_ID="20260104T113111Z-0d243e54" # Optional explicit backup identifier to restore; leave empty to auto-select the latest backup manifest under the Azure prefix
make qdrant-restore


make deploy-retriever

bash tmp.sh

make deploy-vm 


LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
ERR(){ printf '%s ERROR %s\n' "$(date -Iseconds)" "$*" >&2; }

VM_NAMESPACE=${VM_NAMESPACE:-monitoring}
RETRIEVAL_NAMESPACE=${RETRIEVAL_NAMESPACE:-inference}
RETRIEVAL_NAME=${RETRIEVAL_NAME:-retrieval}
QDRANT_NAME=${QDRANT_NAME:-qdrant}

VICTORIA_PORT=${VICTORIA_PORT:-8428}
VMAGENT_PORT=${VMAGENT_PORT:-8429}

RETRIEVAL_METRICS_PORT=${RETRIEVAL_METRICS_PORT:-8001}
QDRANT_METRICS_PORT=${QDRANT_METRICS_PORT:-6333}

LOAD_SECONDS=${LOAD_SECONDS:-20}
PORTFWD_READY_TIMEOUT=${PORTFWD_READY_TIMEOUT:-20}

CURL_BIN=${CURL_BIN:-curl}
PYTHON_BIN=${PYTHON_BIN:-python3}

require(){ command -v "$1" >/dev/null 2>&1 || { ERR "$1 required"; exit 2; }; }
for b in kubectl jq awk sed grep "${CURL_BIN}" "${PYTHON_BIN}"; do require "$b"; done

TMPFILES=()
PFPIDS=()

cleanup(){
  rc=$?
  for pid in "${PFPIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for f in "${TMPFILES[@]:-}"; do
    [ -f "$f" ] && rm -f "$f" || true
  done
  exit "$rc"
}
trap cleanup INT TERM EXIT

find_free_port(){
  "${PYTHON_BIN}" - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
}

start_portforward(){
  ns="$1"; target="$2"; lport="$3"; rport="$4"
  log="$(mktemp /tmp/pf.${target//[^a-zA-Z0-9_.-]/_}.XXXX.log)"
  TMPFILES+=("$log")
  kubectl -n "$ns" port-forward "$target" "$lport:$rport" >"$log" 2>&1 &
  PFPIDS+=("$!")
}

wait_http(){
  url="$1"; timeout="$2"
  end=$((SECONDS+timeout))
  while [ "$SECONDS" -lt "$end" ]; do
    "${CURL_BIN}" -sf --max-time 3 "$url" >/dev/null && return 0
    sleep 1
  done
  return 1
}

promql(){
  q="$1"
  "${CURL_BIN}" -sS -G \
    --data-urlencode "query=$q" \
    "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/api/v1/query"
}

assert_gt0(){
  name="$1"; q="$2"
  LOG "PromQL ${name}: ${q}"
  json="$(promql "$q")"
  echo "$json" | jq .
  val="$(echo "$json" | jq -r '.data.result[0].value[1] // "0"')"
  "${PYTHON_BIN}" - "$val" <<'PY'
import sys
v=float(sys.argv[1])
assert v>0, v
PY
}

assert_eq1(){
  name="$1"; q="$2"
  LOG "PromQL ${name}: ${q}"
  json="$(promql "$q")"
  echo "$json" | jq .
  val="$(echo "$json" | jq -r '.data.result[0].value[1] // "0"')"
  "${PYTHON_BIN}" - "$val" <<'PY'
import sys,math
v=float(sys.argv[1])
assert math.isclose(v,1.0), v
PY
}

LOG "starting VictoriaMetrics port-forward"
LOCAL_VICTORIA_PORT="$(find_free_port)"
start_portforward "$VM_NAMESPACE" svc/victoria-metrics "$LOCAL_VICTORIA_PORT" "$VICTORIA_PORT"
wait_http "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/metrics" "$PORTFWD_READY_TIMEOUT" || { ERR "victoria not ready"; exit 10; }

LOG "starting vmagent port-forward"
LOCAL_VMAGENT_PORT="$(find_free_port)"
start_portforward "$VM_NAMESPACE" svc/vmagent "$LOCAL_VMAGENT_PORT" "$VMAGENT_PORT"
wait_http "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" "$PORTFWD_READY_TIMEOUT" || { ERR "vmagent not ready"; exit 11; }

LOG "vmagent scrape sample"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" | sed -n '1,80p'

LOG "port-forward retriever"
LOCAL_RETR_PORT="$(find_free_port)"
start_portforward "$RETRIEVAL_NAMESPACE" "svc/${RETRIEVAL_NAME}" "$LOCAL_RETR_PORT" "$RETRIEVAL_METRICS_PORT"
wait_http "http://127.0.0.1:${LOCAL_RETR_PORT}/metrics" 10 || { ERR "retriever metrics unavailable"; exit 12; }

LOG "port-forward qdrant"
LOCAL_QDR_PORT="$(find_free_port)"
start_portforward "$RETRIEVAL_NAMESPACE" "svc/${QDRANT_NAME}" "$LOCAL_QDR_PORT" "$QDRANT_METRICS_PORT"
wait_http "http://127.0.0.1:${LOCAL_QDR_PORT}/metrics" 10 || { ERR "qdrant metrics unavailable"; exit 13; }

LOG "synthetic load ${LOAD_SECONDS}s"
i=0
while [ "$i" -lt "$LOAD_SECONDS" ]; do
  "${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_RETR_PORT}/" >/dev/null || true
  i=$((i+1))
  sleep 1
done

LOG "retriever /metrics head"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_RETR_PORT}/metrics" | sed -n '1,120p'

LOG "qdrant /metrics head"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_QDR_PORT}/metrics" | sed -n '1,120p'

LOG "VictoriaMetrics label context"
promql 'count by (__name__,service)({service=~".+"})' | jq .

assert_eq1 "up_retriever" "max(up{service=\"${RETRIEVAL_NAME}\"})"
assert_gt0 "retrieval_requests" "sum(increase(retrieval_requests_total[1m]))"
assert_gt0 "qdrant_queries" "sum(increase(qdrant_query_total[1m]))"

LOG "MONITORING E2E PASSED"
