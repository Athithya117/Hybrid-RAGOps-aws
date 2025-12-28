APP_NS="models"
APP_LABEL="app.kubernetes.io/name=dense"
CH_NS="observability"
CH_POD="clickhouse-0"
TOKEN="dense-e2e-$(date +%s)"

echo "=== [1/6] Wait for dense pod to be Ready ==="
kubectl -n "${APP_NS}" wait pod -l "${APP_LABEL}" --for=condition=Ready --timeout=120s

POD=$(kubectl -n "${APP_NS}" get pod -l "${APP_LABEL}" -o jsonpath='{.items[0].metadata.name}')
IP=$(kubectl -n "${APP_NS}" get pod "${POD}" -o jsonpath='{.status.podIP}')

echo "Dense pod: ${POD} (${IP})"
echo "=== [2/6] Send HTTP requests to dense to generate logs ==="

for i in 1 2; do
  NAME="curltmp-${i}"
  # create a pod that runs curl and exits (portable across kubectl versions)
  kubectl -n "${APP_NS}" run "${NAME}" --restart=Never --image=curlimages/curl --command -- \
    sh -c "curl -s 'http://${IP}:8200/health?e2e=${TOKEN}&i=${i}' || true"
  # wait up to 10s for the curl pod to finish (Succeeded or Failed)
  kubectl -n "${APP_NS}" wait --for=condition=Succeeded "pod/${NAME}" --timeout=10s 2>/dev/null || true
  # remove the pod (cleanup)
  kubectl -n "${APP_NS}" delete pod "${NAME}" --ignore-not-found >/dev/null 2>&1 || true
  sleep 1
done

echo "=== [3/6] Give Vector time to ship logs ==="
sleep 15

echo "=== [4/6] Query ClickHouse for the tokened logs ==="
kubectl -n "${CH_NS}" exec -i "${CH_POD}" -- clickhouse-client --query="
SELECT
  count()        AS rows,
  any(service)   AS service,
  any(pod)       AS pod,
  any(namespace) AS namespace,
  min(ts)        AS first_ts,
  max(ts)        AS last_ts
FROM logs.kube_logs
WHERE message LIKE '%${TOKEN}%'
FORMAT Vertical
"

echo "=== [5/6] Show sample rows (if any) ==="
kubectl -n "${CH_NS}" exec -i "${CH_POD}" -- clickhouse-client --query="
SELECT ts, pod, namespace, message
FROM logs.kube_logs
WHERE message LIKE '%${TOKEN}%'
ORDER BY ts DESC
LIMIT 5
FORMAT Vertical
"

echo "=== [6/6] Done ==="
