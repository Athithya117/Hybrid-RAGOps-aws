#!/usr/bin/env python3
"""
Hardcoded, k3s-only monitoring + vmagent bootstrap generator.

- Single YAML render to infra/manifests/monitoring/monitoring_and_alerts.yaml (overwrites).
- Applies secret directly to the cluster (does not leave secret content in the YAML).
- Creates namespace prior to applying manifests.
- CLI: --generate, --apply, --delete
- Idempotent: uses kubectl apply / kubectl create --dry-run=client | kubectl apply -f -
"""

from __future__ import print_function
import os
import subprocess
import argparse
import textwrap
import pathlib
import sys

# -------------------
# Hardcoded config (k3s-only)
# -------------------
K8S_CLUSTER = "k3s"  # script is hardcoded for local k3s usage
OBS_NAMESPACE = "observability"
MANIFESTS_DIR = "infra/manifests/monitoring"
MANIFEST_PATH = os.path.join(MANIFESTS_DIR, "monitoring_and_alerts.yaml")

# Images / resources (staging-friendly)
VMSINGLE_IMAGE = "victoriametrics"
VMAGENT_IMAGE = "victoriametrics/vmagent:v1.99.0"
VMALERT_IMAGE = "victoriametrics/vmalert:v1.99.0"

VMSINGLE_RETENTION = "30d"
VMSINGLE_PVC_SIZE = "10Gi"
VMSINGLE_STORAGE_CLASS = ""  # leave blank for local k3s
VMAGENT_REPLICAS = 1
VMSINGLE_REPLICAS = 1
VMAGENT_CPU_REQ = "200m"
VMAGENT_MEM_REQ = "512Mi"
VMAGENT_CPU_LIMIT = "1"
VMAGENT_MEM_LIMIT = "1Gi"

ENABLE_ALERTS = "false"
FAIL_ON_MISCONFIG = "false"

RETRIEVAL_SVC_SELECTOR = "app.kubernetes.io/name=retrieval"
RETRIEVAL_NAMESPACE = "inference"
RETRIEVAL_METRICS_PORT = "metrics"
QDRANT_SVC_SELECTOR = "app.kubernetes.io/name=qdrant"
QDRANT_NAMESPACE = "qdrant"
QDRANT_METRICS_PORT = "6333"

# Secret values for local staging (PLACEHOLDERS). These are applied directly to k3s.
# For production, you MUST replace with secure values and/or use your secrets manager.
SECRET_NAME = "observability-monitoring-secrets"
SECRET_LITERALS = {
    "DOCKER_PASSWORD": "changeme",
    "GIT_ASKPASS": "git",
    "GROQ_API_KEY": "changeme",
}

# -------------------
# Utility helpers
# -------------------
def run(cmd, input_data=None, check=True):
    """Run command and return CompletedProcess (text). Raise RuntimeError on failure if check=True."""
    try:
        proc = subprocess.run(cmd, input=input_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("Required binary not found: {}".format(cmd[0])) from e

    if check and proc.returncode != 0:
        raise RuntimeError("Command failed: {}\nstdout:\n{}\nstderr:\n{}".format(cmd, proc.stdout, proc.stderr))
    return proc

def ensure_dir_exists(path):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)

# -------------------
# Manifest generation
# -------------------
def build_manifest_text():
    """Return a single YAML string with multiple documents for the monitoring stack (k3s-local)."""
    manifest = textwrap.dedent(f"""\
    apiVersion: v1
    kind: Namespace
    metadata:
      name: {OBS_NAMESPACE}
    ---
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: monitoring-config
      namespace: {OBS_NAMESPACE}
    data:
      ENABLE_ALERTS: "{ENABLE_ALERTS}"
      FAIL_ON_MISCONFIG: "{FAIL_ON_MISCONFIG}"
      K8S_CLUSTER: "{K8S_CLUSTER}"
      VMSINGLE_IMAGE: "{VMSINGLE_IMAGE}"
      VMAGENT_IMAGE: "{VMAGENT_IMAGE}"
      VMALERT_IMAGE: "{VMALERT_IMAGE}"
      VMSINGLE_RETENTION: "{VMSINGLE_RETENTION}"
      VMSINGLE_PVC_SIZE: "{VMSINGLE_PVC_SIZE}"
      VMSINGLE_STORAGE_CLASS: "{VMSINGLE_STORAGE_CLASS}"
      VMAGENT_REPLICAS: "{VMAGENT_REPLICAS}"
      VMSINGLE_REPLICAS: "{VMSINGLE_REPLICAS}"
      VMAGENT_CPU_REQ: "{VMAGENT_CPU_REQ}"
      VMAGENT_MEM_REQ: "{VMAGENT_MEM_REQ}"
      VMAGENT_CPU_LIMIT: "{VMAGENT_CPU_LIMIT}"
      VMAGENT_MEM_LIMIT: "{VMAGENT_MEM_LIMIT}"
      RETRIEVAL_SVC_SELECTOR: "{RETRIEVAL_SVC_SELECTOR}"
      RETRIEVAL_NAMESPACE: "{RETRIEVAL_NAMESPACE}"
      RETRIEVAL_METRICS_PORT: "{RETRIEVAL_METRICS_PORT}"
      QDRANT_SVC_SELECTOR: "{QDRANT_SVC_SELECTOR}"
      QDRANT_NAMESPACE: "{QDRANT_NAMESPACE}"
      QDRANT_METRICS_PORT: "{QDRANT_METRICS_PORT}"
    ---
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: vmagent-config
      namespace: {OBS_NAMESPACE}
    data:
      scrape.yml: |
        global:
          scrape_interval: 15s
        scrape_configs:
          - job_name: k8s-pods
            kubernetes_sd_configs:
              - role: pod
            relabel_configs:
              - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]
                action: keep
                regex: "true"
              - source_labels: [__meta_kubernetes_pod_ip, __meta_kubernetes_pod_annotation_monitoring_io_port]
                action: replace
                target_label: __address__
                regex: (.+);(.+)
                replacement: $1:$2
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: victoria-metrics
      namespace: {OBS_NAMESPACE}
      labels:
        app: victoria-metrics
    spec:
      replicas: {VMSINGLE_REPLICAS}
      selector:
        matchLabels:
          app: victoria-metrics
      template:
        metadata:
          labels:
            app: victoria-metrics
        spec:
          containers:
            - name: vm
              image: {VMSINGLE_IMAGE}
              args:
                - "-retentionPeriod={VMSINGLE_RETENTION}"
                - "-storageDataPath=/data"
              ports:
                - containerPort: 8428
              volumeMounts:
                - name: data
                  mountPath: /data
          volumes:
            - name: data
              emptyDir: {{}}
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: victoria-metrics
      namespace: {OBS_NAMESPACE}
    spec:
      selector:
        app: victoria-metrics
      ports:
        - port: 8428
          targetPort: 8428
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vmagent
      namespace: {OBS_NAMESPACE}
      labels:
        app: vmagent
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
          containers:
            - name: vmagent
              image: {VMAGENT_IMAGE}
              args:
                - "-promscrape.config=/config/scrape.yml"
                - "-remoteWrite.url=http://victoria-metrics.{OBS_NAMESPACE}.svc.cluster.local:8428/api/v1/write"
                - "-remoteWrite.tmpDataPath=/vmagent-remotewrite-data"
                - "-remoteWrite.showURL=true"
              volumeMounts:
                - name: config
                  mountPath: /config
          volumes:
            - name: config
              configMap:
                name: vmagent-config
    ---
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: vmagent
      namespace: {OBS_NAMESPACE}
    ---
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRole
    metadata:
      name: vmagent
    rules:
      - apiGroups: [""]
        resources: ["pods"]
        verbs: ["get", "list", "watch"]
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
        namespace: {OBS_NAMESPACE}
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: fake-inference
      namespace: default
      labels:
        app: fake-inference
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: fake-inference
      template:
        metadata:
          labels:
            app: fake-inference
          annotations:
            monitoring.io/scrape: "true"
            monitoring.io/port: "9100"
        spec:
          containers:
            - name: metrics
              image: prom/node-exporter:v1.7.0
              args:
                - "--web.listen-address=:9100"
              ports:
                - containerPort: 9100
    """)
    return manifest

# -------------------
# K8s interaction helpers
# -------------------
def create_namespace():
    ns_yaml = textwrap.dedent(f"""\
    apiVersion: v1
    kind: Namespace
    metadata:
      name: {OBS_NAMESPACE}
    """)
    run(["kubectl", "apply", "-f", "-"], input_data=ns_yaml)
    print("Namespace ensured:", OBS_NAMESPACE)

def apply_secret_to_cluster():
    # Build kubectl create secret generic ... --dry-run=client -o yaml
    create_cmd = ["kubectl", "-n", OBS_NAMESPACE, "create", "secret", "generic", SECRET_NAME]
    for k, v in SECRET_LITERALS.items():
        create_cmd.append("--from-literal={0}={1}".format(k, v))
    create_cmd.extend(["--dry-run=client", "-o", "yaml"])

    proc = run(create_cmd, check=True)
    # Pipe result to kubectl apply -f -
    apply_proc = run(["kubectl", "apply", "-f", "-"], input_data=proc.stdout, check=True)
    print("Secret applied (idempotent) to namespace:", OBS_NAMESPACE)

def kubectl_apply_manifest_file(manifest_path):
    proc = run(["kubectl", "apply", "-f", manifest_path], check=True)
    print("Applied manifest:", manifest_path)
    return proc

def kubectl_delete_manifest_file(manifest_path):
    # delete resources described in manifest (idempotent)
    try:
        run(["kubectl", "delete", "-f", manifest_path, "--ignore-not-found"], check=True)
        print("Deleted resources described in", manifest_path, "(if present).")
    except RuntimeError as e:
        # If manifest doesn't exist locally, still attempt to delete by file path - ignore
        print("Warning while deleting manifest:", e)

def delete_secret_from_cluster():
    try:
        run(["kubectl", "-n", OBS_NAMESPACE, "delete", "secret", SECRET_NAME, "--ignore-not-found"], check=True)
        print("Deleted secret", SECRET_NAME, "from namespace", OBS_NAMESPACE)
    except RuntimeError as e:
        print("Warning while deleting secret:", e)

# -------------------
# CLI actions
# -------------------
def action_generate():
    ensure_dir_exists(MANIFESTS_DIR)
    manifest_text = build_manifest_text()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write(manifest_text)
    print("Generated manifest at:", MANIFEST_PATH)
    print("--- config keys rendered to ConfigMap ---")
    # list keys put under monitoring-config
    keys = [
        "ENABLE_ALERTS","FAIL_ON_MISCONFIG","K8S_CLUSTER","VMSINGLE_IMAGE","VMAGENT_IMAGE",
        "VMALERT_IMAGE","VMSINGLE_RETENTION","VMSINGLE_PVC_SIZE","VMSINGLE_STORAGE_CLASS",
        "VMAGENT_REPLICAS","VMSINGLE_REPLICAS","VMAGENT_CPU_REQ","VMAGENT_MEM_REQ",
        "VMAGENT_CPU_LIMIT","VMAGENT_MEM_LIMIT","RETRIEVAL_SVC_SELECTOR","RETRIEVAL_NAMESPACE",
        "RETRIEVAL_METRICS_PORT","QDRANT_SVC_SELECTOR","QDRANT_NAMESPACE","QDRANT_METRICS_PORT"
    ]
    print(", ".join(keys))
    print("--- secrets that will be applied to k8s (placeholder keys) ---")
    print(", ".join(list(SECRET_LITERALS.keys())))

def action_apply():
    # Generate file, ensure namespace, apply secret, then apply manifest
    action_generate()
    print("Ensuring namespace...")
    create_namespace()
    print("Applying secret to cluster (placeholder values). For production, replace secret management.")
    apply_secret_to_cluster()
    print("Applying manifest to cluster...")
    kubectl_apply_manifest_file(MANIFEST_PATH)
    print("All done. First secret key exposed to manifests as '{}' via secretKeyRef where used (placeholder).".format(list(SECRET_LITERALS.keys())[0]))

def action_delete():
    # Delete resources described in manifest and delete secret
    if os.path.exists(MANIFEST_PATH):
        kubectl_delete_manifest_file(MANIFEST_PATH)
    else:
        # still attempt a delete by applying stdin if user has previously applied same content - best-effort skip
        print("Manifest file does not exist locally; attempting to delete by path anyway (kubectl will ignore).")
        kubectl_delete_manifest_file(MANIFEST_PATH)
    delete_secret_from_cluster()
    print("Deleted resources described in {} (if present) and secret '{}' in ns '{}'.".format(MANIFEST_PATH, SECRET_NAME, OBS_NAMESPACE))

# -------------------
# Entrypoint
# -------------------
def main():
    parser = argparse.ArgumentParser(description="k3s-only monitoring+alerts generator (hardcoded for local staging).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Render YAML manifest file only (overwrite).")
    group.add_argument("--apply", action="store_true", help="Create namespace, apply secret, write manifest, apply manifest.")
    group.add_argument("--delete", action="store_true", help="Delete resources described in manifest and delete secret.")
    args = parser.parse_args()

    try:
        if args.generate:
            action_generate()
        elif args.apply:
            action_apply()
        elif args.delete:
            action_delete()
    except RuntimeError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
