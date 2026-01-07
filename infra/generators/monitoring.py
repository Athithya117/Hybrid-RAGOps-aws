#!/usr/bin/env python3
import os
import sys
import shutil
import tempfile
import subprocess
import socket
import time
import json
import re
import signal
import atexit
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

def LOG(*args):
    print(datetime.now().isoformat(), *args, flush=True)

def ERR(*args):
    print(datetime.now().isoformat(), "ERROR", *args, file=sys.stderr, flush=True)

VM_NAMESPACE = os.environ.get("VM_NAMESPACE", "monitoring")
VMAGENT_PORT = int(os.environ.get("VMAGENT_PORT", "8429"))
VICTORIA_PORT = int(os.environ.get("VICTORIA_PORT", "8428"))
VMAGENT_IMAGE = os.environ.get("VMAGENT_IMAGE", "victoriametrics/vmagent:v1.99.0")
VM_IMAGE = os.environ.get("VM_IMAGE", "victoriametrics/victoria-metrics:v1.99.0")
VMAGENT_REPLICAS = os.environ.get("VMAGENT_REPLICAS", "2")
VM_RES_CPU = os.environ.get("VM_RES_CPU", "100m")
VM_RES_MEM = os.environ.get("VM_RES_MEM", "256Mi")
VMAGENT_RES_CPU = os.environ.get("VMAGENT_RES_CPU", "100m")
VMAGENT_RES_MEM = os.environ.get("VMAGENT_RES_MEM", "256Mi")
VM_SCRAPE_INTERVAL = os.environ.get("VM_SCRAPE_INTERVAL", "15s")
VM_SCRAPE_TIMEOUT = os.environ.get("VM_SCRAPE_TIMEOUT", "10s")
REMOTE_WRITE_URL = os.environ.get(
    "REMOTE_WRITE_URL",
    f"http://victoria-metrics.{VM_NAMESPACE}.svc.cluster.local:{VICTORIA_PORT}/api/v1/write",
)
VMAGENT_PVC_STORAGE = os.environ.get("VMAGENT_PVC_STORAGE", "1Gi")
VICTORIA_PVC_STORAGE = os.environ.get("VICTORIA_PVC_STORAGE", "1Gi")
QDRANT_NAMESPACE = os.environ.get("QDRANT_NAMESPACE", "qdrant")
RETRIEVAL_NAMESPACE = os.environ.get("RETRIEVAL_NAMESPACE", "inference")
MANIFEST_DIR = Path.cwd() / "infra" / "manifests"
MANIFEST = MANIFEST_DIR / "00-monitoring.yaml"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
ENABLE_VMAGENT_SELF_SCRAPE = os.environ.get("ENABLE_VMAGENT_SELF_SCRAPE", "true").lower() == "true"
ENABLE_KUBE_STATE_METRICS = os.environ.get("ENABLE_KUBE_STATE_METRICS", "true").lower() == "true"
LOCAL_VICTORIA_PORT = int(os.environ.get("LOCAL_VICTORIA_PORT", "0"))
LOCAL_VMAGENT_PORT = int(os.environ.get("LOCAL_VMAGENT_PORT", "0"))
PORTFWD_READY_TIMEOUT = int(os.environ.get("PORTFWD_READY_TIMEOUT", "30"))
PER_POD_PORTFWD_TIMEOUT = int(os.environ.get("PER_POD_PORTFWD_TIMEOUT", "8"))
QUERY_RETRIES = int(os.environ.get("QUERY_RETRIES", "6"))
RETRY_BACKOFF = int(os.environ.get("RETRY_BACKOFF", "3"))
QUERY_SLEEP = int(os.environ.get("QUERY_SLEEP", "1"))
CURL_BIN = os.environ.get("CURL_BIN", "curl")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")
SKIP_AUTO_RESTART = os.environ.get("SKIP_AUTO_RESTART", "false").lower() == "true"
VICTORIA_WRITE_WAIT_MAX = int(os.environ.get("VICTORIA_WRITE_WAIT_MAX", "120"))
VICTORIA_WRITE_WAIT_STEP_MAX = int(os.environ.get("VICTORIA_WRITE_WAIT_STEP_MAX", "8"))
VMAGENT_READINESS_INITIAL = int(os.environ.get("VMAGENT_READINESS_INITIAL", "30"))
VMAGENT_READINESS_TIMEOUT = int(os.environ.get("VMAGENT_READINESS_TIMEOUT", "10"))
VMAGENT_READINESS_FAILURE_THRESHOLD = int(
    os.environ.get("VMAGENT_READINESS_FAILURE_THRESHOLD", "6")
)

TMPFILES = []
PFPROCS = []

def require(cmd):
    if shutil.which(cmd) is None:
        ERR(f"{cmd} required")
        sys.exit(2)

# check required tools early (will exit if missing)
for tool in ("kubectl", CURL_BIN, PYTHON_BIN, "jq", "mktemp", "sed", "awk", "grep"):
    require(tool)

def cleanup():
    for proc, tailbuf in list(PFPROCS):
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            pass
    for f in list(TMPFILES):
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))

def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    p = s.getsockname()[1]
    s.close()
    return p

def _stream_proc_output(proc, tailbuf, prefix):
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                text = line.decode('utf-8', errors='replace')
            except Exception:
                text = str(line)
            tailbuf.append(text)
            sys.stdout.write(prefix + text)
            sys.stdout.flush()
    except Exception:
        pass

def start_portforward(ns, target, local_port, remote_port):
    cmd = ["kubectl", "-n", ns, "port-forward", target, f"{local_port}:{remote_port}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tailbuf = deque(maxlen=500)
    prefix = f"[port-forward {ns}/{target}] "
    t = threading.Thread(target=_stream_proc_output, args=(proc, tailbuf, prefix), daemon=True)
    t.start()
    PFPROCS.append((proc, tailbuf))
    return proc.pid, tailbuf, proc

def wait_for_http(url, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            rc = subprocess.run([CURL_BIN, "-sS", "--max-time", "3", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def validate_numeric_envs():
    for ev_name, ev_val in (("VMAGENT_PORT", VMAGENT_PORT), ("VICTORIA_PORT", VICTORIA_PORT)):
        if not str(ev_val).isdigit():
            ERR(f"ERROR: {ev_name} must be numeric (found: {ev_val})")
            sys.exit(2)
    if not (REMOTE_WRITE_URL.startswith("http://") or REMOTE_WRITE_URL.startswith("https://")):
        ERR(f"ERROR: REMOTE_WRITE_URL must start with http:// or https:// (found: {REMOTE_WRITE_URL})")
        sys.exit(2)

def _indent_lines(text: str, indent: int) -> str:
    pad = " " * indent
    return "\n".join((pad + line) if line.strip() != "" else "" for line in text.splitlines())

def render_manifest():
    """
    Build the manifest as a list of YAML documents and write them joined with '---\n'.
    This avoids unsafe post-facto regex removals which previously corrupted the YAML.
    """
    docs = []

    # Namespace
    ns_doc = f"""apiVersion: v1
kind: Namespace
metadata:
  name: {VM_NAMESPACE}
"""
    docs.append(ns_doc)

    # vmagent pvc document: include only when replicas == 1
    try:
        replicas_int = int(VMAGENT_REPLICAS)
    except Exception:
        replicas_int = 1

    if replicas_int == 1:
        vmagent_pvc = f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vmagent-pvc
  namespace: {VM_NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: {VMAGENT_PVC_STORAGE}
"""
        docs.append(vmagent_pvc)
    else:
        LOG("VMAGENT_REPLICAS > 1: not creating vmagent-pvc; vmagent tmp will use emptyDir to avoid shared-PVC flock.lock.")

    # victoria pvc
    victoria_pvc = f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: victoria-pvc
  namespace: {VM_NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: {VICTORIA_PVC_STORAGE}
"""
    docs.append(victoria_pvc)

    # vmagent ServiceAccount and RBAC basics
    sa_vmagent = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: vmagent
  namespace: {VM_NAMESPACE}
"""
    docs.append(sa_vmagent)

    clusterrole_vmagent = """apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: vmagent-clusterrole
rules:
- apiGroups: [""]
  resources: ["pods","endpoints","services","nodes","namespaces"]
  verbs: ["get","list","watch"]
"""
    docs.append(clusterrole_vmagent)

    clusterrolebinding_vmagent = f"""apiVersion: rbac.authorization.k8s.io/v1
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
  namespace: {VM_NAMESPACE}
"""
    docs.append(clusterrolebinding_vmagent)

    # Build scrape.yml content
    scrape_yaml_lines = []
    scrape_yaml_lines.append("global:")
    scrape_yaml_lines.append(f"  scrape_interval: {VM_SCRAPE_INTERVAL}")
    scrape_yaml_lines.append(f"  scrape_timeout: {VM_SCRAPE_TIMEOUT}")
    scrape_yaml_lines.append("scrape_configs:")

    # optionally add kube-state-metrics scrape job
    if ENABLE_KUBE_STATE_METRICS:
        scrape_yaml_lines.extend([
            "  - job_name: kube-state-metrics",
            "    kubernetes_sd_configs:",
            "    - role: endpoints",
            "    relabel_configs:",
            "    - source_labels: [__meta_kubernetes_service_name]",
            "      action: keep",
            "      regex: kube-state-metrics",
            "    - source_labels: [__meta_kubernetes_namespace]",
            "      action: keep",
            f"      regex: {VM_NAMESPACE}",
            "    - source_labels: [__meta_kubernetes_endpoint_address, __meta_kubernetes_endpoint_port]",
            "      action: replace",
            "      regex: (.+);(.+)",
            "      replacement: '$1:$2'",
            "      target_label: __address__",
            "    - action: drop",
            "      source_labels: [__address__]",
            "      regex: ^$",
            "    - target_label: __metrics_path__",
            "      replacement: /metrics",
            "    - target_label: job",
            "      replacement: kube-state-metrics",
        ])

    # qdrant and retriever pod scrape jobs (structured and deterministic)
    scrape_yaml_lines.extend([
        "  - job_name: k8s-pods-qdrant",
        "    kubernetes_sd_configs:",
        "    - role: pod",
        "    relabel_configs:",
        "    - source_labels: [__meta_kubernetes_namespace]",
        "      action: keep",
        f"      regex: {QDRANT_NAMESPACE}",
        "    - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]",
        "      action: keep",
        "      regex: \"true\"",
        "    - source_labels: [__meta_kubernetes_pod_ready]",
        "      action: keep",
        "      regex: \"true\"",
        "",
        "    - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]",
        "      action: replace",
        "      regex: (.+);(.+)",
        "      replacement: '$1:$2'",
        "      target_label: __address__",
        "",
        "    - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_container_port_number]",
        "      action: replace",
        "      regex: (.+);(.+)",
        "      replacement: '$1:$2'",
        "      target_label: __address__",
        "",
        "    - action: drop",
        "      source_labels: [__address__]",
        "      regex: ^$",
        "",
        "    - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_path]",
        "      action: replace",
        "      target_label: __metrics_path__",
        "      regex: (.+)",
        "      replacement: $1",
        "    - source_labels: [__metrics_path__]",
        "      action: replace",
        "      regex: ^$",
        "      replacement: /metrics",
        "      target_label: __metrics_path__",
        "",
        "    - target_label: job",
        "      replacement: qdrant",
        "",
        "    - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]",
        "      action: replace",
        "      target_label: service",
        "      regex: (.+)",
        "    - source_labels: [__meta_kubernetes_pod_label_app]",
        "      action: replace",
        "      target_label: service",
        "      regex: (.+)",
        "    - source_labels: [__meta_kubernetes_pod_label_team]",
        "      action: replace",
        "      target_label: service",
        "      regex: (.+)",
        "",
        "  - job_name: k8s-pods-retriever",
        "    kubernetes_sd_configs:",
        "    - role: pod",
        "    relabel_configs:",
        "    - source_labels: [__meta_kubernetes_namespace]",
        "      action: keep",
        f"      regex: {RETRIEVAL_NAMESPACE}",
        "    - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]",
        "      action: keep",
        "      regex: \"true\"",
        "    - source_labels: [__meta_kubernetes_pod_ready]",
        "      action: keep",
        "      regex: \"true\"",
        "",
        "    - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]",
        "      action: replace",
        "      regex: (.+);(.+)",
        "      replacement: '$1:$2'",
        "      target_label: __address__",
        "    - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_container_port_number]",
        "      action: replace",
        "      regex: (.+);(.+)",
        "      replacement: '$1:$2'",
        "      target_label: __address__",
        "    - action: drop",
        "      source_labels: [__address__]",
        "      regex: ^$",
        "",
        "    - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_path]",
        "      action: replace",
        "      target_label: __metrics_path__",
        "      regex: (.+)",
        "      replacement: $1",
        "    - source_labels: [__metrics_path__]",
        "      action: replace",
        "      regex: ^$",
        "      replacement: /metrics",
        "      target_label: __metrics_path__",
        "",
        "    - target_label: job",
        "      replacement: retriever",
        "",
        "    - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]",
        "      action: replace",
        "      target_label: service",
        "      regex: (.+)",
        "    - source_labels: [__meta_kubernetes_pod_label_app]",
        "      action: replace",
        "      target_label: service",
        "      regex: (.+)",
        "    - source_labels: [__meta_kubernetes_pod_label_team]",
        "      action: replace",
        "      target_label: service",
        "      regex: (.+)",
    ])

    if ENABLE_VMAGENT_SELF_SCRAPE:
        scrape_yaml_lines.extend([
            "",
            "  - job_name: k8s-pods-vmagent-self",
            "    kubernetes_sd_configs:",
            "    - role: endpoints",
            "    relabel_configs:",
            "    - source_labels: [__meta_kubernetes_service_name]",
            "      action: keep",
            "      regex: vmagent",
            "    - source_labels: [__meta_kubernetes_namespace]",
            "      action: keep",
            f"      regex: {VM_NAMESPACE}",
            "",
            "    - source_labels: [__meta_kubernetes_endpoint_address, __meta_kubernetes_endpoint_port]",
            "      action: replace",
            "      regex: (.+);(.+)",
            "      replacement: '$1:$2'",
            "      target_label: __address__",
            "",
            "    - action: drop",
            "      source_labels: [__address__]",
            "      regex: ^$",
            "",
            "    - target_label: __metrics_path__",
            "      replacement: /metrics",
            "",
            "    - target_label: job",
            "      replacement: vmagent-self",
            "",
            "    - source_labels: [__meta_kubernetes_pod_label_app]",
            "      action: replace",
            "      target_label: service",
            "      regex: (.+)",
            "      replacement: vmagent",
        ])

    # join scrape lines and embed into ConfigMap YAML document
    scrape_yaml_text = "\n".join(scrape_yaml_lines)
    indented_scrape = _indent_lines(scrape_yaml_text, 4)

    configmap_doc = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: vmagent-config
  namespace: {VM_NAMESPACE}
data:
  scrape.yml: |
{indented_scrape}
"""
    docs.append(configmap_doc)

    # kube-state-metrics manifests (optional)
    if ENABLE_KUBE_STATE_METRICS:
        ksm_sa = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: kube-state-metrics
  namespace: {VM_NAMESPACE}
"""
        docs.append(ksm_sa)

        ksm_clusterrole = """apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kube-state-metrics
rules:
- apiGroups: [""]
  resources: ["nodes","pods","services","endpoints","namespaces"]
  verbs: ["get","list","watch"]
- apiGroups: ["apps"]
  resources: ["deployments","daemonsets","replicasets","statefulsets"]
  verbs: ["get","list","watch"]
- apiGroups: ["policy"]
  resources: ["poddisruptionbudgets"]
  verbs: ["get","list","watch"]
- apiGroups: ["extensions"]
  resources: ["daemonsets","replicasets"]
  verbs: ["get","list","watch"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses"]
  verbs: ["get","list","watch"]
"""
        docs.append(ksm_clusterrole)

        ksm_clusterrolebinding = f"""apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kube-state-metrics
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kube-state-metrics
subjects:
- kind: ServiceAccount
  name: kube-state-metrics
  namespace: {VM_NAMESPACE}
"""
        docs.append(ksm_clusterrolebinding)

        ksm_service = f"""apiVersion: v1
kind: Service
metadata:
  name: kube-state-metrics
  namespace: {VM_NAMESPACE}
  labels:
    app: kube-state-metrics
spec:
  ports:
  - port: 8080
    name: http-metrics
    targetPort: 8080
  selector:
    app: kube-state-metrics
"""
        docs.append(ksm_service)

        # kube-state-metrics deployment
        ksm_deploy = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: kube-state-metrics
  namespace: {VM_NAMESPACE}
  labels:
    app: kube-state-metrics
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kube-state-metrics
  template:
    metadata:
      labels:
        app: kube-state-metrics
    spec:
      serviceAccountName: kube-state-metrics
      containers:
      - name: kube-state-metrics
        image: k8s.gcr.io/kube-state-metrics/kube-state-metrics:v2.8.0
        ports:
        - containerPort: 8080
        args:
        - '--metric-labels-allowlist=pods=[*]'
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 256Mi
"""
        docs.append(ksm_deploy)

    # vmagent Deployment tmp_volume_yaml determination
    tmp_volume_yaml = ""
    if replicas_int > 1:
        tmp_volume_yaml = "      - name: tmp\n        emptyDir: {}\n"
    else:
        tmp_volume_yaml = "      - name: tmp\n        persistentVolumeClaim:\n          claimName: vmagent-pvc\n"

    vmagent_deploy = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: vmagent
  namespace: {VM_NAMESPACE}
spec:
  replicas: {VMAGENT_REPLICAS}
  selector:
    matchLabels:
      app: vmagent
  template:
    metadata:
      labels:
        app: vmagent
    spec:
      serviceAccountName: vmagent
      initContainers:
      - name: wait-victoria-write
        image: curlimages/curl:8.3.0
        command:
        - sh
        - -c
        - |
          max={VICTORIA_WRITE_WAIT_MAX}
          wait=1
          url="http://victoria-metrics.{VM_NAMESPACE}.svc.cluster.local:{VICTORIA_PORT}/api/v1/write"
          for i in $(seq 1 $max); do
            if curl -sS --max-time 5 -X POST -H 'Content-Type: application/json' --data '{{}}' "$url" >/dev/null 2>&1; then
              echo "victoria write endpoint reachable"
              exit 0
            fi
            sleep $wait
            wait=$(( wait < {VICTORIA_WRITE_WAIT_STEP_MAX} ? wait+1 : {VICTORIA_WRITE_WAIT_STEP_MAX} ))
          done
          echo "timed out waiting for victoria write endpoint" >&2
          exit 1
      containers:
      - name: vmagent
        image: {VMAGENT_IMAGE}
        args:
        - "-promscrape.config=/config/scrape.yml"
        - "-remoteWrite.url={REMOTE_WRITE_URL}"
        - "-remoteWrite.tmpDataPath=/vmagent-remotewrite-data"
        - "-promscrape.suppressDuplicateScrapeTargetErrors=true"
        - "-promscrape.suppressScrapeErrors=true"
        - "-promscrape.suppressScrapeErrorsDelay=30s"
        ports:
        - containerPort: {VMAGENT_PORT}
        readinessProbe:
          httpGet:
            path: /metrics
            port: {VMAGENT_PORT}
          initialDelaySeconds: {VMAGENT_READINESS_INITIAL}
          periodSeconds: 10
          timeoutSeconds: {VMAGENT_READINESS_TIMEOUT}
          failureThreshold: {VMAGENT_READINESS_FAILURE_THRESHOLD}
        resources:
          requests:
            cpu: {VMAGENT_RES_CPU}
            memory: {VMAGENT_RES_MEM}
          limits:
            cpu: {VMAGENT_RES_CPU}
            memory: {VMAGENT_RES_MEM}
        volumeMounts:
        - name: config
          mountPath: /config
        - name: tmp
          mountPath: /vmagent-remotewrite-data
      volumes:
      - name: config
        configMap:
          name: vmagent-config
{tmp_volume_yaml}"""
    docs.append(vmagent_deploy)

    # vmagent Service
    vmagent_svc = f"""apiVersion: v1
kind: Service
metadata:
  name: vmagent
  namespace: {VM_NAMESPACE}
spec:
  selector:
    app: vmagent
  ports:
  - name: metrics
    port: {VMAGENT_PORT}
    targetPort: {VMAGENT_PORT}
"""
    docs.append(vmagent_svc)

    # victoria deployment & service
    victoria_deploy = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: victoria-metrics
  namespace: {VM_NAMESPACE}
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
        image: {VM_IMAGE}
        args:
        - "-retentionPeriod=1d"
        - "-storageDataPath=/data"
        - "-httpListenAddr=:{VICTORIA_PORT}"
        ports:
        - containerPort: {VICTORIA_PORT}
        readinessProbe:
          httpGet:
            path: /health
            port: {VICTORIA_PORT}
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        resources:
          requests:
            cpu: {VM_RES_CPU}
            memory: {VM_RES_MEM}
          limits:
            cpu: {VM_RES_CPU}
            memory: {VM_RES_MEM}
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: victoria-pvc
"""
    docs.append(victoria_deploy)

    victoria_svc = f"""apiVersion: v1
kind: Service
metadata:
  name: victoria-metrics
  namespace: {VM_NAMESPACE}
spec:
  selector:
    app: victoria-metrics
  ports:
  - port: {VICTORIA_PORT}
    targetPort: {VICTORIA_PORT}
"""
    docs.append(victoria_svc)

    # join documents reliably with separator
    manifest_text = "\n---\n".join(doc.strip() for doc in docs if doc and doc.strip() != "") + "\n"

    # final sanity checks
    if "replacement: '$1:$2'" not in manifest_text:
        ERR("ERROR: expected literal replacement: '$1:$2' missing in manifest")
        sys.exit(1)
    if "\\" in manifest_text:
        ERR("ERROR: manifest contains backslash characters that may invalidate scrape addresses; aborting")
        sys.exit(1)

    MANIFEST.write_text(manifest_text, encoding="utf-8")
    LOG(f"rendered {MANIFEST}")

def apply():
    validate_numeric_envs()

    rc = subprocess.run(["kubectl", "get", "ns", VM_NAMESPACE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc.returncode != 0:
        create_rc = subprocess.run(["kubectl", "create", "namespace", VM_NAMESPACE], capture_output=True, text=True)
        if create_rc.returncode == 0:
            LOG(f"created namespace {VM_NAMESPACE}")
        else:
            LOG(f"namespace create returned code {create_rc.returncode}; continuing")

    # If a vmagent-scrape configmap exists, sync it (backwards compatibility)
    rc2 = subprocess.run(["kubectl", "-n", VM_NAMESPACE, "get", "configmap", "vmagent-scrape"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc2.returncode == 0:
        LOG("found vmagent-scrape configmap, syncing into vmagent-config")
        tmpf = tempfile.mktemp(suffix=".yml", prefix="/tmp/vmagent-scrape.")
        TMPFILES.append(tmpf)
        p = subprocess.run(["kubectl", "-n", VM_NAMESPACE, "get", "configmap", "vmagent-scrape", "-o", "jsonpath={.data.scrape\\.yml}"], capture_output=True, text=True)
        content = p.stdout or ""
        with open(tmpf, "w", encoding="utf-8") as fh:
            fh.write(content)
        if os.path.getsize(tmpf) > 0:
            create_proc = subprocess.run(["kubectl", "-n", VM_NAMESPACE, "create", "configmap", "vmagent-config", f"--from-file=scrape.yml={tmpf}", "--dry-run=client", "-o", "yaml"], capture_output=True, text=True)
            apply_proc = subprocess.Popen(["kubectl", "-n", VM_NAMESPACE, "apply", "-f", "-"], stdin=subprocess.PIPE)
            if create_proc.stdout:
                apply_proc.communicate(input=create_proc.stdout.encode())
            LOG("synchronized vmagent-scrape -> vmagent-config")
        else:
            LOG("vmagent-scrape exists but empty; continuing")

    render_manifest()

    # apply manifest as a whole
    rc_apply = subprocess.run(["kubectl", "apply", "-f", str(MANIFEST)], capture_output=True, text=True)
    if rc_apply.returncode != 0:
        ERR("kubectl apply failed: stdout/stderr follows")
        if rc_apply.stdout:
            print(rc_apply.stdout)
        if rc_apply.stderr:
            print(rc_apply.stderr, file=sys.stderr)
    else:
        LOG("kubectl apply completed")

    if not SKIP_AUTO_RESTART:
        LOG("waiting for victoria-metrics deployment to be available (120s)")
        roll = subprocess.run(["kubectl", "-n", VM_NAMESPACE, "rollout", "status", "deployment/victoria-metrics", "--timeout=120s"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if roll.returncode == 0:
            LOG("victoria-metrics available; verifying remote-write accept (quick check)")
            probe_cmd = [
                "kubectl", "-n", VM_NAMESPACE, "run", "--rm", "-i", "--restart=Never", "curltest",
                "--image=curlimages/curl", "--command", "--", "sh", "-c",
                "curl -sS --max-time 5 -X POST -H 'Content-Type: application/json' --data '{}' "
                f"'http://victoria-metrics.{VM_NAMESPACE}.svc.cluster.local:{VICTORIA_PORT}/api/v1/write' >/dev/null 2>&1 && echo OK || echo FAIL"
            ]
            try:
                p = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=20)
                ok = ("OK" in (p.stdout or "" ) or "OK" in (p.stderr or "" ))
            except Exception:
                ok = False
            if ok:
                LOG("victoria accepts remote-write; restarting vmagent to pick up config")
                subprocess.run(["kubectl", "-n", VM_NAMESPACE, "rollout", "restart", "deployment/vmagent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                wait = subprocess.run(["kubectl", "-n", VM_NAMESPACE, "wait", "--for=condition=Available", "deployment/vmagent", "--timeout=120s"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if wait.returncode == 0:
                    LOG("vmagent available")
                else:
                    LOG("warning: vmagent not marked available after restart")
            else:
                LOG("victoria remote-write not yet accepting connections; skipping vmagent restart to avoid retry storm")
        else:
            LOG("warning: victoria-metrics not available after 120s; skipping vmagent restart to avoid retry storm")
    else:
        LOG("SKIP_AUTO_RESTART=true; not restarting vmagent")

    LOG(f"monitoring apply complete into {VM_NAMESPACE}")

def probe_vmagent_targets():
    tries = 0
    max_tries = 30
    local_vmag_port = LOCAL_VMAGENT_PORT or find_free_port()
    pid, tailbuf, proc = start_portforward(VM_NAMESPACE, "svc/vmagent", local_vmag_port, VMAGENT_PORT)
    time.sleep(1)

    while tries < max_tries:
        try:
            p = subprocess.run([CURL_BIN, "-sS", f"http://127.0.0.1:{local_vmag_port}/metrics"], capture_output=True, text=True, timeout=5)
            out = p.stdout or ""
        except Exception:
            out = ""

        m = re.search(r'^vm_promscrape_series_fetched\s+([0-9]+)', out, re.MULTILINE)
        if m and int(m.group(1)) > 0:
            LOG(f"vmagent reports series fetched: {m.group(1)}")
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                pass
            return True

        m2 = re.search(r'^vm_promscrape_targets_scraped\s+([0-9]+)', out, re.MULTILINE)
        if m2 and int(m2.group(1)) > 0:
            LOG(f"vmagent reports targets scraped: {m2.group(1)}")
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                pass
            return True

        m3 = re.search(r'^vmagent_remotewrite_sent_bytes_total\s+([0-9]+)', out, re.MULTILINE)
        if m3 and int(m3.group(1)) > 0:
            LOG(f"vmagent reports remote-write bytes sent: {m3.group(1)}")
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                pass
            return True

        tries += 1
        time.sleep(2)

    try:
        proc.terminate(); proc.wait(timeout=2)
    except Exception:
        pass

    ERR("vmagent scrape/remote-write metrics not observed locally after wait")
    return False

def run_promql_with_retries(name, promql, expect):
    last_json = ""
    local_vict_port = LOCAL_VICTORIA_PORT or find_free_port()
    base_vm = f"http://127.0.0.1:{local_vict_port}/api/v1/query"

    for attempt in range(1, QUERY_RETRIES + 1):
        LOG(f"PromQL {name} attempt {attempt}/{QUERY_RETRIES}: {promql}")
        try:
            p = subprocess.run([CURL_BIN, "-sS", "-G", "--data-urlencode", f"query={promql}", base_vm], capture_output=True, text=True, timeout=10)
            json_text = p.stdout or ""
        except Exception:
            json_text = ""
        last_json = json_text
        try:
            j = json.loads(json_text) if json_text else {}
            ok = 0
            if j.get("status") == "success":
                ok = len(j.get("data", {}).get("result", []))
        except Exception:
            ok = 0

        if ok != 0:
            LOG(f"PromQL {name} returned {ok} result(s)")
            val = ""
            try:
                val = str(j["data"]["result"][0]["value"][1])
            except Exception:
                val = ""
            if expect == "gt0":
                try:
                    if val != "" and float(val) > 0:
                        LOG(f"PASS {name} -> {val}")
                        return True
                except Exception:
                    pass
            elif expect == "any":
                LOG(f"PASS {name} -> non-empty result")
                return True
            elif expect == "anynum":
                try:
                    float(val)
                    LOG(f"PASS {name} -> {val}")
                    return True
                except Exception:
                    pass
        else:
            LOG(f"PromQL {name} produced no results; retrying")

        time.sleep(RETRY_BACKOFF * attempt + QUERY_SLEEP)

    ERR(f"FAIL {name} after {QUERY_RETRIES} attempts; last response:")
    if last_json:
        try:
            parsed = json.loads(last_json)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(last_json)
    else:
        print("{}")
    return False

def validate_end_to_end():
    global LOCAL_VICTORIA_PORT
    LOG(f"starting VictoriaMetrics port-forward (svc/victoria-metrics ns={VM_NAMESPACE})")
    local_vict_port = LOCAL_VICTORIA_PORT or find_free_port()
    pid, tailbuf, proc = start_portforward(VM_NAMESPACE, "svc/victoria-metrics", local_vict_port, VICTORIA_PORT)

    LOG(f"waiting up to {PORTFWD_READY_TIMEOUT}s for VictoriaMetrics /metrics")
    if not wait_for_http(f"http://127.0.0.1:{local_vict_port}/metrics", PORTFWD_READY_TIMEOUT):
        ERR(f"victoria-metrics port-forward not ready; recent port-forward output follows")
        try:
            tail_text = "".join(list(tailbuf))[-1000:]
            print(tail_text)
        except Exception:
            pass
        return 2

    LOG(f"VictoriaMetrics port-forward ready (local:{local_vict_port})")
    try:
        cmd = [
            "kubectl", "-n", VM_NAMESPACE, "run", "--rm", "-i", "--restart=Never", "curltest",
            "--image=curlimages/curl", "--command", "--", "sh", "-c",
            "echo 'victoria:'; curl -sS -f http://victoria-metrics." + VM_NAMESPACE + f".svc.cluster.local:{VICTORIA_PORT}/health && echo 'OK' || echo 'FAIL'; "
            "echo 'vmagent metrics:'; curl -sS -f http://vmagent." + VM_NAMESPACE + f".svc.cluster.local:{VMAGENT_PORT}/metrics "
            "| egrep 'vm_promscrape_targets_scraped|vm_promscrape_series_fetched|vmagent_remotewrite_sent_bytes_total|vmagent_remotewrite_errors_total' || true"
        ]
        debug_out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        LOG(f"in-cluster debug check result: {debug_out.stdout.strip()}")
    except Exception as e:
        LOG(f"in-cluster debug check failed: {e}")

    LOG("probing vmagent local metrics/remote-write evidence (via temporary port-forward)")
    if not probe_vmagent_targets():
        ERR("vmagent does not appear to report scrape/remote-write metrics locally; cannot proceed")
        return 4

    LOG("verifying qdrant series visible in Victoria (PromQL)")
    prev_local_vict = LOCAL_VICTORIA_PORT
    LOCAL_VICTORIA_PORT = local_vict_port
    ok = run_promql_with_retries("qdrant_collections_vectors", 'count({__name__=~"collections_vector_total|collections_total|rest_responses_total"})', "gt0")
    LOCAL_VICTORIA_PORT = prev_local_vict
    if ok:
        LOG("Victoria shows qdrant series -> remote-write/ingestion appears to be working")
    else:
        ERR("Victoria does not show expected qdrant series; check vmagent remote-write and Victoria logs")
        return 5

    LOG("validated vmagent & victoria basic connectivity and ingestion")
    return 0

def delete():
    text = ""
    try:
        text = MANIFEST.read_text(encoding="utf-8")
    except Exception:
        text = ""
    if text:
        docs = re.split(r'\n---\s*\n', text)
        kept_docs = []
        for doc in docs:
            if re.search(r'^\s*kind:\s*Namespace\b', doc, re.I | re.M):
                LOG("skipping Namespace document in manifest; namespace will not be deleted")
                continue
            if doc.strip() == "":
                continue
            kept_docs.append(doc)

        if kept_docs:
            tmpf = tempfile.mktemp(suffix=".yaml", prefix="/tmp/monitoring-delete.")
            TMPFILES.append(tmpf)
            with open(tmpf, "w", encoding="utf-8") as fh:
                fh.write("\n---\n".join(kept_docs))
            subprocess.run(["kubectl", "-n", VM_NAMESPACE, "delete", "-f", tmpf, "--ignore-not-found"], check=False)
        else:
            LOG("no resources to delete from manifest after removing Namespace document")
    else:
        LOG("manifest not present or empty; skipping manifest delete step")

    # We no longer forcibly delete vmagent-pvc here because when replicas>1 we switch tmp to emptyDir.
    subprocess.run(["kubectl", "-n", VM_NAMESPACE, "delete", "pvc", "victoria-pvc", "--ignore-not-found"], check=False)
    subprocess.run(["kubectl", "delete", "clusterrole", "vmagent-clusterrole", "--ignore-not-found"], check=False)
    subprocess.run(["kubectl", "delete", "clusterrolebinding", "vmagent-clusterrolebinding", "--ignore-not-found"], check=False)

    # remove kube-state-metrics cluster role/binding if present (best-effort)
    if ENABLE_KUBE_STATE_METRICS:
        subprocess.run(["kubectl", "delete", "clusterrole", "kube-state-metrics", "--ignore-not-found"], check=False)
        subprocess.run(["kubectl", "delete", "clusterrolebinding", "kube-state-metrics", "--ignore-not-found"], check=False)
        subprocess.run(["kubectl", "-n", VM_NAMESPACE, "delete", "deployment", "kube-state-metrics", "--ignore-not-found"], check=False)
        subprocess.run(["kubectl", "-n", VM_NAMESPACE, "delete", "service", "kube-state-metrics", "--ignore-not-found"], check=False)
        subprocess.run(["kubectl", "-n", VM_NAMESPACE, "delete", "serviceaccount", "kube-state-metrics", "--ignore-not-found"], check=False)

    LOG("monitoring deleted (best-effort); namespace left intact")

def usage_and_exit():
    print("usage: monitoring.py --generate|--apply|--delete|--validate (multiple flags allowed, executed in order)", file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage_and_exit()

    ops = []
    for a in sys.argv[1:]:
        if a == "--generate":
            ops.append("generate")
        elif a == "--apply":
            ops.append("apply")
        elif a == "--delete":
            ops.append("delete")
        elif a == "--validate":
            ops.append("validate")
        else:
            usage_and_exit()

    if not ops:
        usage_and_exit()

    for op in ops:
        if op == "generate":
            render_manifest()
            LOG(f"rendered {MANIFEST}")
        elif op == "apply":
            apply()
            LOG("applied monitoring")
        elif op == "delete":
            delete()
            LOG("deleted monitoring resources (namespace preserved)")
        elif op == "validate":
            rc = validate_end_to_end()
            if rc == 0:
                LOG("validation succeeded")
            else:
                ERR(f"validation failed with code {rc}")
                sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
