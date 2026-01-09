#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import textwrap
import yaml
import subprocess
import re
import sys
from typing import Dict, Any, List

# --- Environment variables (single source, top of file, exact format requested) ---
VECTOR_IMAGE_REPO = os.getenv("VECTOR_IMAGE_REPO", "timberio/vector")
VECTOR_IMAGE_TAG = os.getenv("VECTOR_IMAGE_TAG", "0.52.0-distroless-static")
VECTOR_DATA_DIR = os.getenv("VECTOR_DATA_DIR", "/var/lib/vector")
VECTOR_BATCH_MAX_EVENTS = os.getenv("VECTOR_BATCH_MAX_EVENTS", "200")
VECTOR_BATCH_TIMEOUT_SEC = os.getenv("VECTOR_BATCH_TIMEOUT_SEC", "2.0")
VECTOR_CLICKHOUSE_DATABASE = os.getenv("VECTOR_CLICKHOUSE_DATABASE", "logs")
VECTOR_CLICKHOUSE_TABLE = os.getenv("VECTOR_CLICKHOUSE_TABLE", "kube_logs")
VECTOR_REQ_CPU = os.getenv("VECTOR_REQ_CPU", "50m")
VECTOR_REQ_MEM = os.getenv("VECTOR_REQ_MEM", "128Mi")
VECTOR_LIMIT_CPU = os.getenv("VECTOR_LIMIT_CPU", "200m")
VECTOR_LIMIT_MEM = os.getenv("VECTOR_LIMIT_MEM", "256Mi")
VECTOR_DROP_NAMESPACES = os.getenv("VECTOR_DROP_NAMESPACES", "kube-system")
CLICKHOUSE_SERVICE_NAME = os.getenv("CLICKHOUSE_SERVICE_NAME", "clickhouse")
CLICKHOUSE_HTTP_PORT = os.getenv("CLICKHOUSE_HTTP_PORT", "8123")
CLICKHOUSE_SECRET_NAME = os.getenv("CLICKHOUSE_SECRET_NAME", "clickhouse-credentials")
VECTOR_PROMETHEUS_EXPORTER = os.getenv("VECTOR_PROMETHEUS_EXPORTER", "true")
VECTOR_PROMETHEUS_EXPORTER_PORT = os.getenv("VECTOR_PROMETHEUS_EXPORTER_PORT", "8687")
NAMESPACE = os.getenv("NAMESPACE", "observability")

# typed conversions with sane fallbacks
try:
    VECTOR_BATCH_MAX_EVENTS_INT = int(VECTOR_BATCH_MAX_EVENTS)
except Exception:
    VECTOR_BATCH_MAX_EVENTS_INT = 200
try:
    VECTOR_BATCH_TIMEOUT_SEC_F = float(VECTOR_BATCH_TIMEOUT_SEC)
except Exception:
    VECTOR_BATCH_TIMEOUT_SEC_F = 2.0
try:
    CLICKHOUSE_HTTP_PORT_INT = int(CLICKHOUSE_HTTP_PORT)
except Exception:
    CLICKHOUSE_HTTP_PORT_INT = 8123
try:
    VECTOR_PROMETHEUS_EXPORTER_PORT_INT = int(VECTOR_PROMETHEUS_EXPORTER_PORT)
except Exception:
    VECTOR_PROMETHEUS_EXPORTER_PORT_INT = 8687
VECTOR_PROMETHEUS_EXPORTER_BOOL = VECTOR_PROMETHEUS_EXPORTER.lower() == "true"

# Paths and manifest constants
ROOT_MANIFEST_DIR = os.path.join("infra", "manifests", "vector")
MANIFEST_FILE = os.path.join(ROOT_MANIFEST_DIR, "vector.yaml")

VRL_PLACEHOLDER = "__VRL_REPLACEMENT_TOKEN__DO_NOT_TOUCH__"

VRL = textwrap.dedent("""\
parsed = parse_json(.message) ?? {}

if exists(parsed.timestamp) {
  if is_integer(parsed.timestamp) {
    .ts = from_unix_timestamp(parsed.timestamp) ?? now()
  } else {
    .ts = parse_timestamp(parsed.timestamp, format: "%+") ?? now()
  }
} else if exists(.timestamp) {
  .ts = parse_timestamp(.timestamp, format: "%+") ?? now()
} else {
  .ts = now()
}

formatted, fmt_err = format_timestamp(.ts, format: "%Y-%m-%d %H:%M:%S%.3f")
if fmt_err == null {
  .ts = to_string(formatted)
} else {
  now_formatted, now_err = format_timestamp(now(), format: "%Y-%m-%d %H:%M:%S%.3f")
  if now_err == null {
    .ts = to_string(now_formatted)
  } else {
    .ts = to_string(now())
  }
}

allowed_levels = __ALLOWED_LEVELS__

raw_level = ""
if exists(parsed.level) && is_string(parsed.level) {
  raw_level = parsed.level
} else if exists(.level) && is_string(.level) {
  raw_level = .level
} else {
  raw_level = ""
}

if raw_level == "" {
  .level = "INFO"
} else {
  rl_low, rl_err = downcase(raw_level)
  if rl_err == null {
    found = false
    for_each(allowed_levels) -> |_, v| {
      if v == rl_low {
        found = true
      }
    }
    if found {
      if rl_low == "debug" {
        .level = "DEBUG"
      } else if rl_low == "info" {
        .level = "INFO"
      } else if rl_low == "warn" || rl_low == "warning" {
        .level = "WARN"
      } else if rl_low == "error" || rl_low == "err" {
        .level = "ERROR"
      } else {
        .level = "INFO"
      }
    } else {
      .level = "INFO"
    }
  } else {
    .level = "INFO"
  }
}

if exists(parsed.message) {
  if is_string(parsed.message) {
    .message = parsed.message
  } else {
    .message = encode_json(parsed.message)
  }
} else if exists(.message) && is_string(.message) {
  .message = .message
} else {
  .message = ""
}

if exists(parsed.service) && is_string(parsed.service) {
  .service = parsed.service
} else if exists(.kubernetes.labels.app) && is_string(.kubernetes.labels.app) {
  .service = .kubernetes.labels.app
} else if exists(.kubernetes.container_name) && is_string(.kubernetes.container_name) {
  .service = .kubernetes.container_name
} else {
  .service = ""
}

if exists(.kubernetes.container_name) && is_string(.kubernetes.container_name) {
  .container = .kubernetes.container_name
} else {
  .container = ""
}

if exists(.kubernetes.pod_name) && is_string(.kubernetes.pod_name) {
  .pod = .kubernetes.pod_name
} else {
  .pod = ""
}

.namespace = ""
if exists(.kubernetes.pod_namespace) && is_string(.kubernetes.pod_namespace) {
  .namespace = .kubernetes.pod_namespace
} else if exists(.kubernetes.namespace) && is_string(.kubernetes.namespace) {
  .namespace = .kubernetes.namespace
} else if exists(.kubernetes.namespace_name) && is_string(.kubernetes.namespace_name) {
  .namespace = .kubernetes.namespace_name
} else if exists(.kubernetes.ns) && is_string(.kubernetes.ns) {
  .namespace = .kubernetes.ns
} else {
  .namespace = ""
}

if exists(parsed.trace_id) && is_string(parsed.trace_id) {
  .trace_id = parsed.trace_id
} else {
  .trace_id = ""
}

if exists(parsed.span_id) && is_string(parsed.span_id) {
  .span_id = parsed.span_id
} else {
  .span_id = ""
}

.fields = encode_json(parsed)

drop_namespaces = __DROP_NAMESPACES__
ns_drop = false

if .namespace != "" {
  for_each(drop_namespaces) -> |_, v| {
    if v == .namespace {
      ns_drop = true
    }
  }
}

if ns_drop {
  del(.)
}
""")

def run(cmd: str, check: bool = True):
    print("[run]", cmd)
    res = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and res.returncode != 0:
        print(res.stdout or "", end="")
        print(res.stderr or "", end="")
        raise SystemExit(res.returncode)
    return res

def ensure_manifest_dir():
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)

def inject_vrl(dumped_yaml: str, vrl: str) -> str:
    # find the placeholder line for source: "__VRL...__" and replace with a block scalar
    pattern = re.compile(r'^(\s*)source:\s*(["\'])' + re.escape(VRL_PLACEHOLDER) + r'\2\s*$', flags=re.M)
    m = pattern.search(dumped_yaml)
    if not m:
        # try without quotes
        pattern2 = re.compile(r'^(\s*)source:\s*' + re.escape(VRL_PLACEHOLDER) + r'\s*$', flags=re.M)
        m2 = pattern2.search(dumped_yaml)
        if not m2:
            raise SystemExit("VRL placeholder line not found in dumped YAML; cannot safely inject VRL.")
        m = m2
    leading = m.group(1)
    content_indent = leading + "  "
    vrl_block = "source: |\n" + textwrap.indent(vrl.rstrip("\n"), content_indent) + "\n"
    start, end = m.span(0)
    new_yaml = dumped_yaml[:start] + leading + vrl_block + dumped_yaml[end:]
    return new_yaml

def build_manifest(namespace: str) -> str:
    # derive lists from top-level env values
    drop_csv = VECTOR_DROP_NAMESPACES or ""
    if drop_csv.strip() == "":
        drop_list = ["kube-system"]
    else:
        drop_list = [p.strip() for p in drop_csv.split(",") if p.strip()]

    # keep a fixed canonical allow-list for normalization; no runtime env var
    allowed_list = ["debug", "info", "warn", "error"]

    # prepare VRL by injecting lists
    vrl = VRL.replace("__DROP_NAMESPACES__", json.dumps(drop_list)).replace("__ALLOWED_LEVELS__", json.dumps(allowed_list))

    ch_fqdn = f"{CLICKHOUSE_SERVICE_NAME}.{namespace}.svc.cluster.local"
    ch_endpoint = f"http://{ch_fqdn}:{CLICKHOUSE_HTTP_PORT_INT}"

    # Vector configuration object (will be YAML-dumped then VRL injected)
    vector_cfg: Dict[str, Any] = {
        "api": {"enabled": True, "address": "0.0.0.0:8686", "playground": False},
        "sources": {
            "kubernetes_logs": {"type": "kubernetes_logs", "insert_namespace_fields": True},
            # internal metrics source so prometheus_exporter can expose useful series
            "internal_metrics": {"type": "internal_metrics"},
        },
        "transforms": {
            "normalize_v1": {"type": "remap", "inputs": ["kubernetes_logs"], "source": VRL_PLACEHOLDER}
        },
        "sinks": {
            "clickhouse": {
                "type": "clickhouse",
                "inputs": ["normalize_v1"],
                "endpoint": ch_endpoint,
                "auth": {"strategy": "basic", "user": os.getenv("CLICKHOUSE_USER", "vector"), "password": os.getenv("CLICKHOUSE_PASSWORD", "vectorpass")},
                "database": VECTOR_CLICKHOUSE_DATABASE,
                "table": VECTOR_CLICKHOUSE_TABLE,
                "format": "json_each_row",
                "compression": "gzip",
                "skip_unknown_fields": True,
                "batch": {"max_events": VECTOR_BATCH_MAX_EVENTS_INT, "timeout_secs": VECTOR_BATCH_TIMEOUT_SEC_F},
                "healthcheck": {"enabled": True},
            }
        }
    }

    # If prometheus exporter enabled, wire it to internal_metrics (not log pipeline)
    if VECTOR_PROMETHEUS_EXPORTER_BOOL:
        exporter_port = VECTOR_PROMETHEUS_EXPORTER_PORT_INT
        vector_cfg.setdefault("sinks", {})
        vector_cfg["sinks"]["prometheus_exporter"] = {
            "type": "prometheus_exporter",
            "inputs": ["internal_metrics"],
            "address": f"0.0.0.0:{exporter_port}"
        }

    dumped = yaml.safe_dump(vector_cfg, sort_keys=False)
    vector_yaml = inject_vrl(dumped, vrl)

    # Kubernetes manifest pieces (string documents)
    sa_doc = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: vector
  namespace: {namespace}
"""
    cr_doc = """apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: vector-k8s-reader
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "namespaces", "nodes", "services", "endpoints", "events"]
    verbs: ["get", "list", "watch"]
"""
    crb_doc = f"""apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: vector-k8s-reader-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: vector-k8s-reader
subjects:
  - kind: ServiceAccount
    name: vector
    namespace: {namespace}
"""
    cfg_cm = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: vector-config
  namespace: {namespace}
data:
  vector.yaml: |
{textwrap.indent(vector_yaml.rstrip(), '    ')}
"""

    # DaemonSet: note mountPath to file via subPath so the config file is at /etc/vector/vector.yaml
    ds_doc = f"""apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: vector-agent
  namespace: {namespace}
spec:
  selector:
    matchLabels:
      app: vector
  template:
    metadata:
      labels:
        app: vector
    spec:
      serviceAccountName: vector
      tolerations:
        - operator: Exists
      volumes:
        - name: vector-config
          configMap:
            name: vector-config
            items:
              - key: vector.yaml
                path: vector.yaml
        - name: data-dir
          hostPath:
            path: {VECTOR_DATA_DIR}
            type: DirectoryOrCreate
        - name: pod-logs
          hostPath:
            path: /var/log/pods
            type: DirectoryOrCreate
      containers:
        - name: vector
          image: {VECTOR_IMAGE_REPO}:{VECTOR_IMAGE_TAG}
          args: ["-c", "/etc/vector/vector.yaml"]
          ports:
            - name: metrics
              containerPort: {VECTOR_PROMETHEUS_EXPORTER_PORT_INT}
          volumeMounts:
            - name: vector-config
              mountPath: /etc/vector/vector.yaml
              subPath: vector.yaml
            - name: data-dir
              mountPath: {VECTOR_DATA_DIR}
            - name: pod-logs
              mountPath: /var/log/pods
              readOnly: true
          resources:
            requests:
              cpu: {VECTOR_REQ_CPU}
              memory: {VECTOR_REQ_MEM}
            limits:
              cpu: {VECTOR_LIMIT_CPU}
              memory: {VECTOR_LIMIT_MEM}
          env:
            - name: CLICKHOUSE_USER
              valueFrom:
                secretKeyRef:
                  name: {CLICKHOUSE_SECRET_NAME}
                  key: username
            - name: CLICKHOUSE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {CLICKHOUSE_SECRET_NAME}
                  key: password
            - name: VECTOR_SELF_NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: VECTOR_DROP_NAMESPACES
              value: "{','.join(drop_list)}"
"""

    docs: List[str] = [sa_doc, cr_doc, crb_doc, cfg_cm, ds_doc]

    # Exporter Service: keep existing 'vector-agent' Service for backward compatibility
    if VECTOR_PROMETHEUS_EXPORTER_BOOL:
        exporter_port = VECTOR_PROMETHEUS_EXPORTER_PORT_INT
        svc_doc = f"""apiVersion: v1
kind: Service
metadata:
  name: vector-agent
  namespace: {namespace}
spec:
  selector:
    app: vector
  ports:
  - name: metrics
    port: {exporter_port}
    targetPort: {exporter_port}
"""
        # Add both the legacy-named service (vector-agent) and an alias service name (vector-prometheus-exporter)
        # so monitoring configs that expect either name will resolve.
        alias_svc_doc = f"""apiVersion: v1
kind: Service
metadata:
  name: vector-prometheus-exporter
  namespace: {namespace}
spec:
  selector:
    app: vector
  ports:
  - name: metrics
    port: {exporter_port}
    targetPort: {exporter_port}
"""
        docs.append(svc_doc)
        docs.append(alias_svc_doc)

    manifest_text = "\n---\n".join(docs)
    return manifest_text

def validate_and_write(manifest_text: str):
    try:
        list(yaml.safe_load_all(manifest_text))
    except Exception as e:
        tmp = "/tmp/vector_manifest_error.yaml"
        with open(tmp, "w") as fh:
            fh.write(manifest_text)
        raise SystemExit(f"Generated manifest YAML failed validation: {e}\nWrote manifest to {tmp} for inspection.")
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w") as fh:
        fh.write(manifest_text)
    print("[info] wrote", MANIFEST_FILE)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    ns = NAMESPACE.strip()

    if args.generate:
        manifest_text = build_manifest(ns)
        validate_and_write(manifest_text)
        print("[ok] generate complete")
        return

    if args.apply:
        if os.path.exists(MANIFEST_FILE):
            try:
                os.makedirs("/tmp/infra-backups", exist_ok=True)
                subprocess.run(f"cp {MANIFEST_FILE} /tmp/infra-backups/vector.yaml.bak", shell=True)
            except Exception:
                pass
        manifest_text = build_manifest(ns)
        validate_and_write(manifest_text)

        # ensure namespace exists and create secret for ClickHouse credentials (backwards-compatible behavior)
        subprocess.run(f"kubectl create namespace {ns} --dry-run=client -o yaml | kubectl apply -f -", shell=True)
        subprocess.run(
            f"kubectl -n {ns} create secret generic {CLICKHOUSE_SECRET_NAME} --from-literal=username={os.getenv('CLICKHOUSE_USER','vector')} --from-literal=password={os.getenv('CLICKHOUSE_PASSWORD','vectorpass')} --dry-run=client -o yaml | kubectl apply -f -",
            shell=True
        )

        # apply manifest (validate first)
        subprocess.run(f"kubectl apply --dry-run=client -f {MANIFEST_FILE}", shell=True)
        subprocess.run(f"kubectl apply -f {MANIFEST_FILE}", shell=True)

        # if exporter enabled, note to operator
        if VECTOR_PROMETHEUS_EXPORTER_BOOL:
            print("[info] Vector prometheus exporter enabled; manifest creates services 'vector-agent' and 'vector-prometheus-exporter' in namespace", ns)

        subprocess.run(f"kubectl -n {ns} rollout restart daemonset vector-agent || true", shell=True)
        print("[ok] apply complete")
        return

    if args.delete:
        if os.path.exists(MANIFEST_FILE):
            try:
                subprocess.run(f"kubectl delete -f {MANIFEST_FILE} --ignore-not-found", shell=True)
            except Exception:
                pass
            try:
                os.remove(MANIFEST_FILE)
            except Exception:
                pass
        try:
            subprocess.run(f"kubectl -n {ns} delete secret {CLICKHOUSE_SECRET_NAME} --ignore-not-found", shell=True)
            subprocess.run("kubectl delete clusterrolebinding vector-k8s-reader-binding --ignore-not-found", shell=True)
            subprocess.run("kubectl delete clusterrole vector-k8s-reader --ignore-not-found", shell=True)
            subprocess.run(f"kubectl -n {ns} delete serviceaccount vector --ignore-not-found", shell=True)
        except Exception:
            pass
        print("[ok] delete complete")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
