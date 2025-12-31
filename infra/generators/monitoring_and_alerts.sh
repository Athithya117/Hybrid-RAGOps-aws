#!/usr/bin/env bash
set -euo pipefail
LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }

: "${VM_NAMESPACE:=monitoring}"
: "${ENV:=local}"
: "${RENDER_DIR:=${PWD}/infra/manifests}"
mkdir -p "${RENDER_DIR}"
MANIFEST="${RENDER_DIR}/00-monitoring.yaml"

: "${VM_PORT:=8428}"
: "${VMAGENT_PORT:=8429}"
: "${VM_USE_PVC:=false}"
: "${VM_PVC_SIZE:=10Gi}"
: "${VM_STORAGE_CLASS:=local-path}"
: "${VM_RETENTION:=1d}"
: "${VM_SCRAPE_INTERVAL:=15s}"
: "${VM_SCRAPE_TIMEOUT:=10s}"
: "${VMAGENT_IMAGE:=victoriametrics/vmagent:v1.99.0}"
: "${VM_IMAGE:=victoriametrics/victoria-metrics:v1.99.0}"
: "${VMAGENT_REPLICAS:=1}"
: "${VM_RES_CPU:=100m}"
: "${VM_RES_MEM:=256Mi}"
: "${VMAGENT_RES_CPU:=100m}"
: "${VMAGENT_RES_MEM:=256Mi}"

# Hardcoded, single source of truth for vmagent remote-write (in-cluster VM)
REMOTE_WRITE_URL="http://victoria-metrics.${VM_NAMESPACE}.svc.cluster.local:${VM_PORT}/api/v1/write"

check_kubectl(){ command -v kubectl >/dev/null 2>&1 || { LOG "kubectl required"; exit 1; } }

render_manifest(){
cat >"${MANIFEST}" <<'EOF'
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
  name: vmagent
rules:
  - apiGroups: [""]
    resources: ["pods","endpoints","services","nodes"]
    verbs: ["get","list","watch"]
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
      - job_name: k8s-pods
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
          - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_path]
            action: replace
            regex: (.+)
            target_label: __metrics_path__
          - target_label: __metrics_path__
            replacement: /metrics
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - target_label: env
            replacement: __ENV__
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vmagent
  namespace: __VM_NAMESPACE__
  labels:
    app: vmagent
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
  labels:
    app: victoria-metrics
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
            - "-retentionPeriod=__VM_RETENTION__"
            - "-storageDataPath=/data"
            - "-httpListenAddr=__:VM_PORT__"
          ports:
            - containerPort: __VM_PORT__
          readinessProbe:
            httpGet:
              path: /health
              port: __VM_PORT__
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
    - port: __VM_PORT__
      targetPort: __VM_PORT__
EOF

  sed -e "s|__VM_NAMESPACE__|${VM_NAMESPACE}|g" \
      -e "s|__VM_SCRAPE_INTERVAL__|${VM_SCRAPE_INTERVAL}|g" \
      -e "s|__VM_SCRAPE_TIMEOUT__|${VM_SCRAPE_TIMEOUT}|g" \
      -e "s|__ENV__|${ENV}|g" \
      -e "s|__VM_IMAGE__|${VM_IMAGE}|g" \
      -e "s|__VM_RETENTION__|${VM_RETENTION}|g" \
      -e "s|__VM_PORT__|${VM_PORT}|g" \
      -e "s|__:VM_PORT__|:${VM_PORT}|g" \
      -e "s|__VM_RES_CPU__|${VM_RES_CPU}|g" \
      -e "s|__VM_RES_MEM__|${VM_RES_MEM}|g" \
      -e "s|__VMAGENT_REPLICAS__|${VMAGENT_REPLICAS}|g" \
      -e "s|__VMAGENT_IMAGE__|${VMAGENT_IMAGE}|g" \
      -e "s|__REMOTE_WRITE_URL__|${REMOTE_WRITE_URL}|g" \
      -e "s|__VMAGENT_PORT__|${VMAGENT_PORT}|g" \
      -e "s|__VMAGENT_RES_CPU__|${VMAGENT_RES_CPU}|g" \
      -e "s|__VMAGENT_RES_MEM__|${VMAGENT_RES_MEM}|g" \
      "${MANIFEST}" > "${MANIFEST}.tmp" && mv "${MANIFEST}.tmp" "${MANIFEST}"

  # Ensure the replacement is present literally and no backslashes exist
  if ! grep -Fq "replacement: '\$1:\$2'" "${MANIFEST}"; then
    LOG "ERROR: expected literal replacement: '\$1:\$2' missing in ${MANIFEST}"
    exit 1
  fi

  if grep -q "\\\\" "${MANIFEST}"; then
    LOG "ERROR: manifest contains backslash characters; aborting to avoid invalid scrape addresses"
    exit 1
  fi

  # Prevent env-based override risk by ensuring REMOTE_WRITE_URL is not set as an env var
  if grep -q "name: REMOTE_WRITE_URL" "${MANIFEST}" ; then
    LOG "ERROR: REMOTE_WRITE_URL must not appear as an env var in the manifest"
    exit 1
  fi
}

apply(){
  check_kubectl
  kubectl create namespace "${VM_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
  render_manifest
  kubectl apply -f "${MANIFEST}"
  if [ "${VM_USE_PVC}" = "true" ] || [ "${VM_USE_PVC}" = "True" ] || [ "${VM_USE_PVC}" = "1" ]; then
    kubectl -n "${VM_NAMESPACE}" apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: victoria-metrics-pvc
  namespace: ${VM_NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: ${VM_STORAGE_CLASS}
  resources:
    requests:
      storage: ${VM_PVC_SIZE}
EOF
    kubectl -n "${VM_NAMESPACE}" patch deployment victoria-metrics --type='json' -p='[{"op":"replace","path":"/spec/template/spec/volumes/0","value":{"name":"data","persistentVolumeClaim":{"claimName":"victoria-metrics-pvc"}}}]' || LOG "patch pvc failed"
  fi

  # Ensure deployment uses args (not env) and restart to pick up changes
  kubectl -n "${VM_NAMESPACE}" rollout restart deployment vmagent || true

  LOG "monitoring applied into ${VM_NAMESPACE}"
}

delete(){
  check_kubectl
  kubectl delete -f "${MANIFEST}" --ignore-not-found || true
  if [ "${VM_USE_PVC}" = "true" ]; then
    kubectl -n "${VM_NAMESPACE}" delete pvc victoria-metrics-pvc --ignore-not-found || true
  fi
  LOG "monitoring deleted (best-effort)"
}

case "${1:-}" in
  --generate) render_manifest && LOG "rendered ${MANIFEST}" ;;
  --apply) apply ;;
  --delete) delete ;;
  *) LOG "usage: $0 --generate|--apply|--delete"; exit 1 ;;
esac
