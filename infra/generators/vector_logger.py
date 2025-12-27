#!/usr/bin/env python3
"""
Final, corrected vector_logger.py (ready-to-run)

- Emits a single multi-document YAML at infra/manifests/vector/vector.yaml:
  ServiceAccount, ClusterRole, ClusterRoleBinding, ConfigMap, DaemonSet.
- ConfigMap.vector.yaml is written as a YAML literal block to avoid escaping.
- ClickHouse auth.strategy set to "basic" (compatible with Vector 0.52.x).
- Uses guarded VRL remap to parse JSON logs and map kubernetes metadata.
- CLI flags: --generate, --apply, --delete
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
import yaml
import textwrap
from typing import Dict, Any, List

# Hardcoded constants
CLICKHOUSE_SERVICE_NAME = "clickhouse"
CLICKHOUSE_HTTP_PORT = "8123"
CLICKHOUSE_SECRET_NAME = "clickhouse-credentials"

MANIFEST_DIR = os.path.join("infra", "manifests", "vector")
MANIFEST_FILE = os.path.join(MANIFEST_DIR, "vector.yaml")


def run(cmd: str, check: bool = True, quiet: bool = False):
    if not quiet:
        print("[run]", cmd)
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        print(r.stdout, end="")
        print(r.stderr, end="", file=sys.stderr)
        raise SystemExit(r.returncode)
    return r


def ensure_namespace(ns: str):
    run(
        f"kubectl create namespace {ns} --dry-run=client -o yaml | kubectl apply -f -"
    )


def create_secret(ns: str, name: str, username: str, password: str):
    run(
        f"kubectl -n {ns} create secret generic {name} "
        f"--from-literal=username={username} --from-literal=password={password} "
        "--dry-run=client -o yaml | kubectl apply -f -"
    )


def delete_secret(ns: str, name: str):
    run(f"kubectl -n {ns} delete secret {name} --ignore-not-found", check=False)


def ensure_manifest_dir():
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)


def write_manifest_file(
    sa: Dict[str, Any],
    cr: Dict[str, Any],
    crb: Dict[str, Any],
    configmap_cfg: str,
    daemonset: Dict[str, Any],
) -> None:
    """
    Write a YAML file containing multiple documents.
    The ConfigMap is written manually to guarantee vector.yaml is a literal block.
    """
    ensure_manifest_dir()
    parts: List[str] = []

    parts.append(yaml.safe_dump(sa, sort_keys=False))
    parts.append("---\n")
    parts.append(yaml.safe_dump(cr, sort_keys=False))
    parts.append("---\n")
    parts.append(yaml.safe_dump(crb, sort_keys=False))
    parts.append("---\n")

    cfg_cm = (
        "apiVersion: v1\nkind: ConfigMap\n"
        f"metadata:\n  name: vector-config\n  namespace: {sa['metadata']['namespace']}\n"
        "data:\n  vector.yaml: |\n"
    )
    indented = textwrap.indent(configmap_cfg.rstrip("\n"), "    ")
    cfg_cm += indented + "\n"
    parts.append(cfg_cm)
    parts.append("---\n")

    parts.append(yaml.safe_dump(daemonset, sort_keys=False))

    with open(MANIFEST_FILE, "w") as fh:
        fh.write("".join(parts))
    print("[info] wrote", MANIFEST_FILE)


def build_vector_docs(env: Dict[str, str]):
    ns = env["NAMESPACE"]
    image = f"{env['VECTOR_IMAGE_REPO']}:{env['VECTOR_IMAGE_TAG']}"
    data_dir = env["VECTOR_DATA_DIR"]
    batch_max = env["VECTOR_BATCH_MAX_EVENTS"]
    batch_to = env["VECTOR_BATCH_TIMEOUT_SEC"]

    ch_fqdn = f"{CLICKHOUSE_SERVICE_NAME}.{ns}.svc.cluster.local"
    ch_endpoint = f"http://{ch_fqdn}:{CLICKHOUSE_HTTP_PORT}"

    # Use a non-f string template so raw braces in VRL are preserved.
    cfg_template = textwrap.dedent("""\
    api:
      enabled: true
      address: "0.0.0.0:8686"
      playground: false

    sources:
      kubernetes_logs:
        type: kubernetes_logs
        self_node_name: "${VECTOR_SELF_NODE_NAME}"
        insert_namespace_fields: true

    transforms:
      parse_and_map:
        type: remap
        inputs: [kubernetes_logs]
        source: |
          # Attempt to parse message; parse_json returns null on error when used with ??.
          parsed = parse_json(.message) ?? null

          # Merge only when parsed is an object; otherwise do not attempt merge.
          if is_object(parsed) {
            if is_object(.) {
              . = merge!(., parsed)
            } else {
              . = parsed
            }
          }

          # Map kubernetes fields into top-level pod/namespace.
          if .kubernetes != null {
            if .kubernetes.pod != null && .kubernetes.pod.name != null {
              .pod = .kubernetes.pod.name
            } else if .kubernetes.pod_name != null {
              .pod = .kubernetes.pod_name
            }
            if .kubernetes.namespace != null && .kubernetes.namespace.name != null {
              .namespace = .kubernetes.namespace.name
            } else if .kubernetes.namespace != null {
              .namespace = .kubernetes.namespace
            }
          }

          # Ensure .message is a string
          if !is_string(.message) {
            .message = encode_json(.message)
          }

    sinks:
      clickhouse:
        type: clickhouse
        inputs: [parse_and_map]
        endpoint: "__CH_ENDPOINT__"
        auth:
          strategy: basic
          user: "${CLICKHOUSE_USER}"
          password: "${CLICKHOUSE_PASSWORD}"
        database: "__DB__"
        table: "__TABLE__"
        format: "json_each_row"
        compression: "gzip"
        skip_unknown_fields: true
        batch:
          max_events: __BATCH_MAX__
          timeout_secs: __BATCH_TO__
        healthcheck:
          enabled: false
    """)

    # Inject dynamic values safely (no f-string interpolation of braces).
    cfg = (
        cfg_template
        .replace("__CH_ENDPOINT__", ch_endpoint)
        .replace("__DB__", str(env["VECTOR_CLICKHOUSE_DATABASE"]))
        .replace("__TABLE__", str(env["VECTOR_CLICKHOUSE_TABLE"]))
        .replace("__BATCH_MAX__", str(batch_max))
        .replace("__BATCH_TO__", str(batch_to))
    )

    sa = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": "vector", "namespace": ns},
    }

    cr = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRole",
        "metadata": {"name": "vector-k8s-reader"},
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["pods", "namespaces", "nodes"],
                "verbs": ["get", "list", "watch"],
            }
        ],
    }

    crb = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": "vector-k8s-reader-binding"},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": "vector-k8s-reader",
        },
        "subjects": [{"kind": "ServiceAccount", "name": "vector", "namespace": ns}],
    }

    daemonset = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": "vector-agent", "namespace": ns},
        "spec": {
            "selector": {"matchLabels": {"app": "vector"}},
            "template": {
                "metadata": {"labels": {"app": "vector"}},
                "spec": {
                    "serviceAccountName": "vector",
                    "tolerations": [{"operator": "Exists"}],
                    "volumes": [
                        {
                            "name": "vector-config",
                            "configMap": {
                                "name": "vector-config",
                                "items": [{"key": "vector.yaml", "path": "vector.yaml"}],
                            },
                        },
                        {"name": "data-dir", "hostPath": {"path": data_dir, "type": "DirectoryOrCreate"}},
                        {"name": "pod-logs", "hostPath": {"path": "/var/log/pods", "type": "DirectoryOrCreate"}},
                    ],
                    "containers": [
                        {
                            "name": "vector",
                            "image": image,
                            "args": ["-c", "/etc/vector/vector.yaml"],
                            "volumeMounts": [
                                {"name": "vector-config", "mountPath": "/etc/vector/vector.yaml", "subPath": "vector.yaml"},
                                {"name": "data-dir", "mountPath": data_dir},
                                {"name": "pod-logs", "mountPath": "/var/log/pods", "readOnly": True},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": env.get("VECTOR_REQ_CPU", "50m"),
                                    "memory": env.get("VECTOR_REQ_MEM", "128Mi"),
                                },
                                "limits": {
                                    "cpu": env.get("VECTOR_LIMIT_CPU", "200m"),
                                    "memory": env.get("VECTOR_LIMIT_MEM", "256Mi"),
                                },
                            },
                            "env": [
                                {
                                    "name": "CLICKHOUSE_USER",
                                    "valueFrom": {"secretKeyRef": {"name": CLICKHOUSE_SECRET_NAME, "key": "username"}},
                                },
                                {
                                    "name": "CLICKHOUSE_PASSWORD",
                                    "valueFrom": {"secretKeyRef": {"name": CLICKHOUSE_SECRET_NAME, "key": "password"}},
                                },
                                {
                                    "name": "VECTOR_SELF_NODE_NAME",
                                    "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                                },
                            ],
                        }
                    ],
                }
            },
        },
    }

    return sa, cr, crb, cfg, daemonset



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()

    ns = os.getenv("NAMESPACE", "observability").strip()
    env = {
        "NAMESPACE": ns,
        "VECTOR_IMAGE_REPO": os.getenv("VECTOR_IMAGE_REPO", "timberio/vector"),
        "VECTOR_IMAGE_TAG": os.getenv("VECTOR_IMAGE_TAG", "0.52.0-debian"),
        "VECTOR_DATA_DIR": os.getenv("VECTOR_DATA_DIR", "/var/lib/vector"),
        "VECTOR_BATCH_MAX_EVENTS": os.getenv("VECTOR_BATCH_MAX_EVENTS", "1000"),
        "VECTOR_BATCH_TIMEOUT_SEC": os.getenv("VECTOR_BATCH_TIMEOUT_SEC", "1"),
        "VECTOR_CLICKHOUSE_DATABASE": os.getenv("VECTOR_CLICKHOUSE_DATABASE", "logs"),
        "VECTOR_CLICKHOUSE_TABLE": os.getenv("VECTOR_CLICKHOUSE_TABLE", "kube_logs"),
        "VECTOR_REQ_CPU": os.getenv("VECTOR_REQ_CPU", "50m"),
        "VECTOR_REQ_MEM": os.getenv("VECTOR_REQ_MEM", "128Mi"),
        "VECTOR_LIMIT_CPU": os.getenv("VECTOR_LIMIT_CPU", "200m"),
        "VECTOR_LIMIT_MEM": os.getenv("VECTOR_LIMIT_MEM", "256Mi"),
    }

    sa, cr, crb, cfg, daemonset = build_vector_docs(env)

    if args.generate:
        write_manifest_file(sa, cr, crb, cfg, daemonset)
        return

    if args.apply:
        ensure_namespace(env["NAMESPACE"])
        user = os.getenv("CLICKHOUSE_USER", "vector")
        password = os.getenv("CLICKHOUSE_PASSWORD", "vectorpass")
        create_secret(env["NAMESPACE"], CLICKHOUSE_SECRET_NAME, user, password)
        write_manifest_file(sa, cr, crb, cfg, daemonset)
        run(f"kubectl apply -f {MANIFEST_FILE}")
        print("[ok] vector applied")
        return

    if args.delete:
        if os.path.exists(MANIFEST_FILE):
            run(f"kubectl delete -f {MANIFEST_FILE} --ignore-not-found")
            try:
                os.remove(MANIFEST_FILE)
            except Exception:
                pass
        delete_secret(os.getenv("NAMESPACE", env["NAMESPACE"]), CLICKHOUSE_SECRET_NAME)
        run("kubectl delete clusterrolebinding vector-k8s-reader-binding --ignore-not-found", check=False)
        run("kubectl delete clusterrole vector-k8s-reader --ignore-not-found", check=False)
        run(f"kubectl -n {env['NAMESPACE']} delete serviceaccount vector --ignore-not-found", check=False)
        print("[ok] vector deleted")
        return

    p.print_help()


if __name__ == "__main__":
    main()
