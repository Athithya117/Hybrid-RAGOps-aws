#!/usr/bin/env bash
NS="${NS:-monitoring}"
VICTORIA_SVC="${VICTORIA_SVC:-victoria-metrics}"
VMALERT_SVC="${VMALERT_SVC:-vmalert}"
ALERTM_SVC="${ALERTM_SVC:-alertmanager}"
ENABLE_PAGERDUTY="${ENABLE_PAGERDUTY:-true}"
ENABLE_SLACK="${ENABLE_SLACK:-true}"
ALERTING_PAGING_SEVERITY_LEVELS="${ALERTING_PAGING_SEVERITY_LEVELS:-critical}"
ALERTING_SLACK_SEVERITY_LEVELS="${ALERTING_SLACK_SEVERITY_LEVELS:-warning,critical}"
TMPDIR="$(mktemp -d /tmp/test_alerts.XXXXXX)"
LOG_TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PIDS=()
UUID="$(cat /proc/sys/kernel/random/uuid)"
JSON_TOOL="jq"
command -v curl >/dev/null 2>&1 || { echo "$(LOG_TS) ERROR curl required"; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "$(LOG_TS) ERROR python3 required"; exit 2; }
command -v ${JSON_TOOL} >/dev/null 2>&1 || { echo "$(LOG_TS) WARN ${JSON_TOOL} not found; outputs will be raw JSON"; JSON_TOOL=""; }

find_free_port() {
  python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
}

kill_matching_pf() {
  pgrep -af "kubectl .*port-forward.*${1}" | awk '{print $1}' | xargs -r -n1 kill || true
}

cleanup() {
  echo "$(LOG_TS) INFO cleanup: killing port-forwards"
  for p in "${PIDS[@]}"; do
    kill "${p}" 2>/dev/null || true
  done
  sleep 0.3
  for p in "${PIDS[@]}"; do
    kill -9 "${p}" 2>/dev/null || true
  done
  echo "$(LOG_TS) INFO preserving logs at ${TMPDIR}"
}

trap 'cleanup' EXIT

echo "$(LOG_TS) INFO tmpdir=${TMPDIR} test_run=${UUID}"

echo "$(LOG_TS) INFO cleaning any previous port-forwards for services"
kill_matching_pf "${VICTORIA_SVC}"
kill_matching_pf "${VMALERT_SVC}"
kill_matching_pf "${ALERTM_SVC}"
sleep 1

VICTORIA_LOCAL="$(find_free_port)"
VMALERT_LOCAL="$(find_free_port)"
ALERTM_LOCAL="$(find_free_port)"

echo "$(LOG_TS) INFO starting port-forwards victoria=${VICTORIA_LOCAL} vmalert=${VMALERT_LOCAL} alertm=${ALERTM_LOCAL}"
kubectl -n "${NS}" port-forward "svc/${VICTORIA_SVC}" "${VICTORIA_LOCAL}:8428" &
PIDS+=($!)
kubectl -n "${NS}" port-forward "svc/${VMALERT_SVC}" "${VMALERT_LOCAL}:8080" &
PIDS+=($!)
kubectl -n "${NS}" port-forward "svc/${ALERTM_SVC}" "${ALERTM_LOCAL}:9093" &
PIDS+=($!)

wait_for_http() {
  url="$1"; timeout="${2:-20}"; start=$(date +%s)
  while true; do
    if curl -sS -m2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then
      return 1
    fi
    sleep 1
  done
}

echo "$(LOG_TS) INFO waiting for Alertmanager to answer /api/v2/status"
if ! wait_for_http "http://127.0.0.1:${ALERTM_LOCAL}/api/v2/status" 20; then
  echo "$(LOG_TS) ERROR Alertmanager not responding at http://127.0.0.1:${ALERTM_LOCAL}/api/v2/status"
  echo "$(LOG_TS) INFO tailing recent port-forward and container logs for debugging"
  ps -ef | egrep "kubectl .*port-forward" || true
  kubectl -n "${NS}" get pods -o wide || true
  kubectl -n "${NS}" logs -l app=alertmanager --tail=120 2>/dev/null || true
  exit 3
fi

echo "$(LOG_TS) INFO waiting for vmalert /metrics"
if ! wait_for_http "http://127.0.0.1:${VMALERT_LOCAL}/metrics" 20; then
  echo "$(LOG_TS) ERROR vmalert not responding at http://127.0.0.1:${VMALERT_LOCAL}/metrics"
  kubectl -n "${NS}" get pods -o wide || true
  kubectl -n "${NS}" logs -l app=vmalert --tail=120 2>/dev/null || true
  exit 4
fi

echo "$(LOG_TS) INFO pushing synthetic metrics (compact)"
MET_FILE="${TMPDIR}/metrics.txt"
NOW_MS="$(date +%s000)"
T1=$((NOW_MS - 120000))
T2=$((NOW_MS - 60000))
cat > "${MET_FILE}" <<__EOF__
retrieval_requests_total{service="retrieval",test_run="${UUID}"} 100 ${T1}
retrieval_errors_total{service="retrieval",test_run="${UUID}"} 60 ${T1}
retrieval_requests_total{service="retrieval",test_run="${UUID}"} 200 ${T2}
retrieval_errors_total{service="retrieval",test_run="${UUID}"} 140 ${T2}
retrieval_requests_total{service="retrieval",test_run="${UUID}"} 300 ${NOW_MS}
retrieval_errors_total{service="retrieval",test_run="${UUID}"} 220 ${NOW_MS}
__EOF__
curl -sS --data-binary @"${MET_FILE}" "http://127.0.0.1:${VICTORIA_LOCAL}/api/v1/import/prometheus" -o "${TMPDIR}/victoria_push.json" || { echo "$(LOG_TS) ERROR push to Victoria failed"; exit 5; }
if [ -n "${JSON_TOOL}" ]; then
  echo "$(LOG_TS) INFO victoria push status:"; cat "${TMPDIR}/victoria_push.json" | ${JSON_TOOL} -c '.status'
else
  echo "$(LOG_TS) INFO victoria push raw:"; head -c 800 "${TMPDIR}/victoria_push.json"
fi

echo "$(LOG_TS) INFO posting synthetic sanity alert to Alertmanager"
curl -sS -XPOST "http://127.0.0.1:${ALERTM_LOCAL}/api/v2/alerts" -H "Content-Type: application/json" -d "[{\"labels\":{\"alertname\":\"SanityWarm\",\"severity\":\"info\",\"plane\":\"slo\",\"service\":\"sanity\",\"test_run\":\"${UUID}\"},\"annotations\":{\"summary\":\"sanity warm\"}}]" || true

sleep 1

post_alert() {
  name="$1"; plane="$2"; sev="$3"; svc="$4"
  payload="[ {\"labels\":{\"alertname\":\"${name}\",\"plane\":\"${plane}\",\"severity\":\"${sev}\",\"service\":\"${svc}\",\"test_run\":\"${UUID}\"},\"annotations\":{\"summary\":\"synthetic ${name}\"}} ]"
  echo "$(LOG_TS) INFO posting alert ${name} -> http://127.0.0.1:${ALERTM_LOCAL}/api/v2/alerts"
  curl -sS -XPOST "http://127.0.0.1:${ALERTM_LOCAL}/api/v2/alerts" -H "Content-Type: application/json" -d "${payload}" || true
}

post_alert "qdrant-paging" "safety" "critical" "qdrant"
post_alert "retriever-paging" "safety" "critical" "retriever"
post_alert "test-channel1-nonpaging" "slo" "warning" "test-channel1"
post_alert "test-channel2-nonpaging" "slo" "warning" "test-channel2"

sleep 4

echo "$(LOG_TS) INFO vmalert metrics (filtered)"
curl -s "http://127.0.0.1:${VMALERT_LOCAL}/metrics" | egrep "vmalert_alerts_(firing|pending)|vmalert_alerts_sent_total" | sed -n '1,80p'

echo "$(LOG_TS) INFO Alertmanager alerts for test_run (compact)"
if [ -n "${JSON_TOOL}" ]; then
  curl -s "http://127.0.0.1:${ALERTM_LOCAL}/api/v2/alerts" | ${JSON_TOOL} -c "[.[] | select(.labels.test_run==\"${UUID}\")]" | sed -n '1,200p'
else
  curl -s "http://127.0.0.1:${ALERTM_LOCAL}/api/v2/alerts" | sed -n '1,400p'
fi

determine_expected_receiver() {
  plane="$1"; sev="$2"
  sev_l="$(echo "${sev}" | tr '[:upper:]' '[:lower:]')"
  IFS=',' read -r -a paging_arr <<< "${ALERTING_PAGING_SEVERITY_LEVELS}"
  IFS=',' read -r -a slack_arr <<< "${ALERTING_SLACK_SEVERITY_LEVELS}"
  for s in "${paging_arr[@]}"; do
    s_trim="$(echo "$s" | xargs | tr '[:upper:]' '[:lower:]')"
    if [ "$sev_l" = "$s_trim" ]; then
      if [ "$(echo "${ENABLE_PAGERDUTY}" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
        echo "pagerduty"; return
      fi
      if [ "$(echo "${ENABLE_SLACK}" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
        echo "slack"; return
      fi
    fi
  done
  for s in "${slack_arr[@]}"; do
    s_trim="$(echo "$s" | xargs | tr '[:upper:]' '[:lower:]')"
    if [ "$sev_l" = "$s_trim" ]; then
      if [ "$(echo "${ENABLE_SLACK}" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
        echo "slack"; return
      fi
    fi
  done
  echo "default"
}

echo "$(LOG_TS) INFO verifying receiver routing and presence"
AM_JSON="$(curl -s "http://127.0.0.1:${ALERTM_LOCAL}/api/v2/alerts")"
FAIL=0
for pair in "qdrant-paging|safety|critical|qdrant" "retriever-paging|safety|critical|retriever" "test-channel1-nonpaging|slo|warning|test-channel1" "test-channel2-nonpaging|slo|warning|test-channel2"; do
  name="$(printf '%s' "$pair" | cut -d'|' -f1)"
  plane="$(printf '%s' "$pair" | cut -d'|' -f2)"
  sev="$(printf '%s' "$pair" | cut -d'|' -f3)"
  svc="$(printf '%s' "$pair" | cut -d'|' -f4)"
  if [ -n "${JSON_TOOL}" ]; then
    item="$(echo "${AM_JSON}" | ${JSON_TOOL} -c ".[] | select(.labels.alertname==\"${name}\" and .labels.service==\"${svc}\")" 2>/dev/null | head -n1)"
  else
    item="$(echo "${AM_JSON}" | grep -A6 "\"alertname\":\"${name}\"" | head -n20 || true)"
  fi
  if [ -z "${item}" ]; then
    echo "$(LOG_TS) ERROR alert ${name} not present in Alertmanager"
    FAIL=1
    continue
  fi
  if [ -n "${JSON_TOOL}" ]; then
    got_recv="$(echo "${item}" | ${JSON_TOOL} -r -c '(.receivers[0].name // "")')"
  else
    got_recv="$(echo "${item}" | sed -n '1,120p' | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n1 | sed 's/.*: *"//;s/"$//')"
  fi
  expected="$(determine_expected_receiver "${plane}" "${sev}")"
  echo "$(LOG_TS) INFO alert=${name} severity=${sev} plane=${plane} receiver=${got_recv:-'(none)'} expected=${expected}"
  if [ "${expected}" != "default" ] && [ "${got_recv}" != "${expected}" ]; then
    echo "$(LOG_TS) ERROR unexpected receiver for ${name}: got='${got_recv}' want='${expected}'"
    FAIL=1
  fi
done

echo "$(LOG_TS) INFO verify Victoria ingestion for test_run (compact)"
QRES="$(curl -sG "http://127.0.0.1:${VICTORIA_LOCAL}/api/v1/query" --data-urlencode "query=retrieval_requests_total{test_run=\"${UUID}\"}")"
if [ -n "${JSON_TOOL}" ]; then
  present="$(echo "${QRES}" | ${JSON_TOOL} -c '.data.result | length' 2>/dev/null || echo 0)"
  if [ "${present}" -gt 0 ]; then
    echo "$(LOG_TS) INFO Victoria ingestion OK (result_count=${present})"
  else
    echo "$(LOG_TS) ERROR Victoria did not ingest test series"
    FAIL=1
  fi
else
  echo "$(LOG_TS) INFO Victoria query (raw):"; echo "${QRES}" | head -c 400
fi

if [ "${FAIL}" -ne 0 ]; then
  echo "$(LOG_TS) ERROR TESTS FAILED"
  exit 10
fi

echo "$(LOG_TS) INFO ALL CHECKS PASSED; logs preserved in ${TMPDIR}"
exit 0
