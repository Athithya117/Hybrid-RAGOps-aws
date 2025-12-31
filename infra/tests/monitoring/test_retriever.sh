#!/usr/bin/env bash
set -euo pipefail

LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
LOUD(){ printf '\n==== %s %s ====\n\n' "$(date -Iseconds)" "$*"; }

VM_NAMESPACE=${VM_NAMESPACE:-monitoring}
RETRIEVAL_NAMESPACE=${RETRIEVAL_NAMESPACE:-inference}
RETRIEVAL_NAME=${RETRIEVAL_NAME:-retrieval}

VMAGENT_SVC=${VMAGENT_SVC:-vmagent}
VMAGENT_PORT=${VMAGENT_PORT:-8429}

VICTORIA_SVC=${VICTORIA_SVC:-victoria-metrics}
VICTORIA_PORT=${VICTORIA_PORT:-8428}

DEPLOY_WAIT=${DEPLOY_WAIT:-180}
SCRAPE_WAIT=${SCRAPE_WAIT:-35}
LOAD_SECONDS=${LOAD_SECONDS:-20}

has_cmd(){ command -v "$1" >/dev/null 2>&1; }
has_cmd kubectl || { echo "kubectl required"; exit 2; }

LOUD "E2E START (VictoriaMetrics-native)"

wait_ready(){
  ns="$1"; sel="$2"; timeout="${3:-$DEPLOY_WAIT}"
  start=$(date +%s)
  while true; do
    ready=$(kubectl -n "$ns" get pod -l "$sel" \
      -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
    [ "$ready" = "true" ] && return 0
    [ $(( $(date +%s) - start )) -ge "$timeout" ] && return 1
    sleep 5
  done
}

LOUD "waiting for deployments"
wait_ready "$VM_NAMESPACE" "app=${VMAGENT_SVC}" || { LOG "vmagent not ready"; exit 3; }
wait_ready "$VM_NAMESPACE" "app=${VICTORIA_SVC}" || { LOG "victoria-metrics not ready"; exit 3; }
wait_ready "$RETRIEVAL_NAMESPACE" "app=${RETRIEVAL_NAME}" || { LOG "retrieval not ready"; exit 3; }

LOUD "validating vmagent remoteWrite config"
expected_rw="http://${VICTORIA_SVC}.${VM_NAMESPACE}.svc.cluster.local:${VICTORIA_PORT}/api/v1/write"
args=$(kubectl -n "$VM_NAMESPACE" get deploy "$VMAGENT_SVC" \
  -o jsonpath='{.spec.template.spec.containers[0].args}')

echo "$args" | grep -q -- "-remoteWrite.url=${expected_rw}" || {
  LOG "FAIL: vmagent remoteWrite URL incorrect"
  exit 4
}

envs=$(kubectl -n "$VM_NAMESPACE" get deploy "$VMAGENT_SVC" \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}')
echo "$envs" | grep -q REMOTE_WRITE_URL && {
  LOG "FAIL: REMOTE_WRITE_URL env must not exist"
  exit 5
}

LOUD "generating minimal load"
kubectl -n "$RETRIEVAL_NAMESPACE" run --rm -i --restart=Never loadgen \
  --image=curlimages/curl -- sh -c "
i=0
while [ \$i -lt ${LOAD_SECONDS} ]; do
  curl -s http://${RETRIEVAL_NAME}:8001/metrics >/dev/null || true
  i=\$((i+1))
  sleep 1
done
"

LOG "waiting ${SCRAPE_WAIT}s for scrape + remote-write"
sleep "$SCRAPE_WAIT"

LOUD "asserting vmagent scrape targets"
kubectl -n "$VM_NAMESPACE" run --rm -i --restart=Never curl-vmagent \
  --image=curlimages/curl -- sh -c \
  "curl -s http://${VMAGENT_SVC}:${VMAGENT_PORT}/metrics" \
  > /tmp/vmagent.metrics

grep -q 'vm_promscrape_targets{[^}]*status="up"[^}]*} [1-9]' /tmp/vmagent.metrics || {
  LOG "FAIL: no active kubernetes scrape targets"
  exit 6
}

LOUD "asserting VictoriaMetrics ingestion"
kubectl -n "$VM_NAMESPACE" run --rm -i --restart=Never curl-vm \
  --image=curlimages/curl -- sh -c \
  "curl -s http://${VICTORIA_SVC}:${VICTORIA_PORT}/metrics" \
  > /tmp/vm.metrics

rows=$(grep '^vm_rows_inserted_total{type="promremotewrite"}' /tmp/vm.metrics | awk '{print $2}')
[ -n "$rows" ] && [ "$rows" != "0" ] || {
  LOG "FAIL: promremotewrite rows = 0"
  exit 7
}

LOG "OK: promremotewrite rows = $rows"

LOUD "OPTIONAL: querying up{} (non-gating)"
kubectl -n "$VM_NAMESPACE" run --rm -i --restart=Never curl-vm-q \
  --image=curlimages/curl -- sh -c \
  "curl -s \"http://${VICTORIA_SVC}:${VICTORIA_PORT}/api/v1/query?query=up%7Bnamespace%3D%22${RETRIEVAL_NAMESPACE}%22%7D\"" \
  || true

LOUD "E2E SUCCESS (scrape + ingest verified)"

