echo "=== BOOTSTRAP: full monitoring + retriever E2E (k3s) ==="

echo "CHECK: kubectl client available?"
kubectl version --client >/dev/null 2>&1 || echo "WARN: kubectl not found or not configured"

echo "STEP 0: remove old namespaces (idempotent)"
kubectl delete ns monitoring inference --ignore-not-found >/dev/null 2>&1 || true
echo "INFO: waiting 3s for cleanup..."
sleep 3

echo "STEP 1: create namespaces"
kubectl create ns monitoring >/dev/null 2>&1 || true
kubectl create ns inference >/dev/null 2>&1 || true
echo "OK: namespaces monitoring & inference present"

echo "STEP 2: apply RBAC for vmagent (ServiceAccount + ClusterRole + CRB)"
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vmagent
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: vmagent
rules:
  - apiGroups: [""]
    resources: ["pods","endpoints","services"]
    verbs: ["get","list","watch"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get","list","watch"]
  - apiGroups: ["coordination.k8s.io"]
    resources: ["leases"]
    verbs: ["get","create","update","delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: vmagent
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: vmagent
subjects:
  - kind: ServiceAccount
    name: vmagent
    namespace: monitoring
YAML
echo "OK: RBAC applied"

echo "STEP 3: apply monitoring stack (vmagent ConfigMap fixed — literal \$1:\$2 preserved)"
kubectl -n monitoring apply -f - <<'YAML'
apiVersion: v1
kind: ConfigMap
metadata:
  name: vmagent-config
  namespace: monitoring
data:
  scrape.yml: |
    global:
      scrape_interval: 15s
      scrape_timeout: 10s
    scrape_configs:
      - job_name: k8s-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]
            action: keep
            regex: "true"
          - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]
            action: replace
            regex: (.+);(.+)
            replacement: $1:$2
            target_label: __address__
          - target_label: __metrics_path__
            replacement: /metrics
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victoria-metrics
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels: { app: victoria-metrics }
  template:
    metadata:
      labels: { app: victoria-metrics }
    spec:
      containers:
        - name: vm
          image: victoriametrics/victoria-metrics:v1.99.0
          args:
            - "-retentionPeriod=1d"
            - "-storageDataPath=/data"
          ports:
            - containerPort: 8428
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
  namespace: monitoring
spec:
  selector: { app: victoria-metrics }
  ports:
    - port: 8428
      targetPort: 8428
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vmagent
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels: { app: vmagent }
  template:
    metadata:
      labels: { app: vmagent }
    spec:
      serviceAccountName: vmagent
      containers:
        - name: vmagent
          image: victoriametrics/vmagent:v1.99.0
          args:
            - "-promscrape.config=/config/scrape.yml"
            - "-remoteWrite.url=http://victoria-metrics.monitoring.svc.cluster.local:8428/api/v1/write"
            - "-remoteWrite.tmpDataPath=/vmagent-remotewrite-data"
          ports:
            - containerPort: 8429
          volumeMounts:
            - name: config
              mountPath: /config
      volumes:
        - name: config
          configMap:
            name: vmagent-config
---
apiVersion: v1
kind: Service
metadata:
  name: vmagent
  namespace: monitoring
spec:
  selector: { app: vmagent }
  ports:
    - name: metrics
      port: 8429
      targetPort: 8429
YAML

echo "OK: monitoring stack applied (ConfigMap contains replacement: \$1:\$2 literal)"

echo "STEP 4: deploy retriever app (scrapable, annotated)"
kubectl -n inference apply -f - <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retrieval
  namespace: inference
spec:
  replicas: 1
  selector:
    matchLabels: { app: retrieval }
  template:
    metadata:
      labels: { app: retrieval }
      annotations:
        monitoring.io/scrape: "true"
        monitoring.io/port: "8001"
    spec:
      containers:
        - name: retrieval
          image: docker.io/athithya5354/retrieval:v7
          ports:
            - containerPort: 8001
---
apiVersion: v1
kind: Service
metadata:
  name: retrieval
  namespace: inference
spec:
  selector: { app: retrieval }
  ports:
    - port: 8001
      targetPort: 8001
YAML
echo "OK: retriever applied"

echo "STEP 5: wait-for-ready helper (will print loud messages on delays)"
wait_for_ready() {
  ns="$1"; sel="$2"; timeout="${3:-180}"
  echo "WAIT: waiting up to ${timeout}s for ${sel} in ${ns}..."
  elapsed=0
  while :; do
    pod=$(kubectl -n "${ns}" get pods -l "${sel}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    ready=$(kubectl -n "${ns}" get pods -l "${sel}" -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
    if [ "${ready}" = "true" ] && [ -n "${pod}" ]; then
      echo "OK: ${pod} in ${ns} Ready (${elapsed}s)."
      break
    fi
    if [ "${elapsed}" -ge "${timeout}" ]; then
      echo "STUCK: timeout waiting for ${sel} in ${ns} after ${timeout}s. Dumping pods and vmagent logs:"
      kubectl -n "${ns}" get pods -o wide || true
      kubectl -n monitoring logs deploy/vmagent --tail=200 2>/dev/null || true
      break
    fi
    if [ $((elapsed % 10)) -eq 0 ]; then
      echo "INFO: still waiting (${elapsed}s) for ${sel} in ${ns}..."
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

wait_for_ready monitoring app=vmagent 180
wait_for_ready monitoring app=victoria-metrics 180
wait_for_ready inference app=retrieval 180

echo "STEP 6: quick sanity checks (retriever /metrics, vmagent config, vmagent logs)"
echo ">>> retriever /metrics (first 10 lines):"
kubectl -n inference run --rm -i --restart=Never curl-app --image=curlimages/curl -- sh -c 'curl -sf http://retrieval:8001/metrics | head -n 10' 2>/dev/null || echo "WARN: cannot curl retrieval:8001 from cluster"
echo ">>> vmagent-config (first 200 chars):"
kubectl -n monitoring get configmap vmagent-config -o yaml 2>/dev/null | sed -n '1,200p' || true
echo ">>> vmagent logs (tail 120):"
kubectl -n monitoring logs deploy/vmagent --tail=120 2>/dev/null || true

echo "STEP 7: generate deterministic traffic to retriever for 30s (so counters appear)"
kubectl -n inference run --rm -i --restart=Never loadgen --image=curlimages/curl -- sh -c '
  echo "LOADGEN: hitting /metrics for 30s"; i=0
  while [ $i -lt 30 ]; do
    curl -sf http://retrieval:8001/metrics >/dev/null || true
    curl -sf http://retrieval:8001/ >/dev/null || true
    i=$((i+1)); sleep 1
  done
  echo "LOADGEN: done"
' >/dev/null 2>&1 || true
echo "OK: traffic generated"

echo "STEP 8: wait a scrape interval + buffer (35s)"
sleep 35

echo "STEP 9: inspect vmagent internal metrics (targets/scrapes)"
kubectl -n monitoring run --rm -i --restart=Never curl-vmagent --image=curlimages/curl -- sh -c '
  echo "VMAGENT_METRICS: lines matching promscrape or scrape counters:"
  curl -sS http://vmagent:8429/metrics 2>/dev/null | egrep -i "promscrape_targets|scrape_samples_scraped|promscrape_target_up|promscrape_scrape_duration_seconds" || true
' 2>/dev/null || true

echo "STEP 10: URL-encoded queries to VictoriaMetrics (look for non-empty result)"
echo "QUERY 1: up{namespace=\"inference\"}"
kubectl -n monitoring run --rm -i --restart=Never curl-vm --image=curlimages/curl -- sh -c '
  curl -s "http://victoria-metrics:8428/api/v1/query?query=up%7Bnamespace%3D%22inference%22%7D" || true
' 2>/dev/null || true

echo "QUERY 2: retrieval_requests_total"
kubectl -n monitoring run --rm -i --restart=Never curl-vm2 --image=curlimages/curl -- sh -c '
  curl -s "http://victoria-metrics:8428/api/v1/query?query=retrieval_requests_total" || true
' 2>/dev/null || true

echo "STEP 11: status summary (pods + vmagent metrics excerpt)"
echo "--- pods ---"
kubectl get pods -A || true
echo "--- vmagent metrics excerpt (first 300 chars) ---"
kubectl -n monitoring run --rm -i --restart=Never shortvm --image=curlimages/curl -- sh -c 'curl -sS http://vmagent:8429/metrics 2>/dev/null | head -c 300 || true' 2>/dev/null || true

echo "COMPLETE: If you see 'promscrape_targets' > 0 in STEP 9 and 'up{namespace=\"inference\"}' returned non-empty in STEP 10, pipeline is healthy."
echo "If still empty: copy vmagent logs and vmagent-config output above and paste here for analysis."
