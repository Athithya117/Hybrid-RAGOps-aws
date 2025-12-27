#!/usr/bin/env python3
"""
clickhouse.py

Generate / apply / delete ClickHouse Service + StatefulSet, create secret
and ensure user/database/table exist. Grants INSERT and SELECT.

Usage:
  python3 infra/manifests/clickhouse/clickhouse.py --generate
  python3 infra/manifests/clickhouse/clickhouse.py --apply
  python3 infra/manifests/clickhouse/clickhouse.py --delete
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess
import yaml
import time

# --------- static constants ----------
CLICKHOUSE_SERVICE_NAME = "clickhouse"
CLICKHOUSE_HTTP_PORT = "8123"
CLICKHOUSE_NATIVE_PORT = "9000"
CLICKHOUSE_SECRET_NAME = "clickhouse-credentials"

MANIFEST_DIR = os.path.join("infra", "manifests", "clickhouse")
MANIFEST_FILE = os.path.join(MANIFEST_DIR, "clickhouse.yaml")

# --------- helpers ----------
def run(cmd: str, capture: bool = False, check: bool = True, quiet: bool = False):
    if not quiet:
        print("[run]", cmd)
    res = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True
    )
    if check and res.returncode != 0:
        if capture:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)
    return res

def run_quiet(cmd: str, capture: bool = False, check: bool = True):
    return run(cmd, capture=capture, check=check, quiet=True)

def ensure_namespace(ns: str):
    run(
        f"kubectl create namespace {ns} --dry-run=client -o yaml"
        " | kubectl apply -f -"
    )

def write_yaml(path: str, docs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False)
    print("[info] wrote", path)

def kubectl_apply(path: str):
    run(f"kubectl apply -f {path}")

def kubectl_delete(path: str):
    run(f"kubectl delete -f {path} --ignore-not-found")

def create_secret(ns: str, name: str, username: str, password: str):
    cmd = (
        f"kubectl -n {ns} create secret generic {name} "
        f"--from-literal=username={username} "
        f"--from-literal=password={password} "
        "--dry-run=client -o yaml | kubectl apply -f -"
    )
    run(cmd)

def delete_secret(ns: str, name: str):
    run(f"kubectl -n {ns} delete secret {name} --ignore-not-found")

# --------- manifest builder ----------
def build_clickhouse_docs(env: dict) -> list:
    ns = env["NAMESPACE"]
    image = f"{env['CLICKHOUSE_IMAGE_REPO']}:{env['CLICKHOUSE_IMAGE_TAG']}"
    replicas = int(env["CLICKHOUSE_REPLICAS"])
    pvc_size = env["CLICKHOUSE_PVC_SIZE"]
    req_cpu = env["CLICKHOUSE_REQ_CPU"]
    req_mem = env["CLICKHOUSE_REQ_MEM"]
    lim_cpu = env["CLICKHOUSE_LIMIT_CPU"]
    lim_mem = env["CLICKHOUSE_LIMIT_MEM"]

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": CLICKHOUSE_SERVICE_NAME, "namespace": ns},
        "spec": {
            "selector": {"app": "clickhouse"},
            "ports": [
                {"name": "http", "port": int(CLICKHOUSE_HTTP_PORT), "targetPort": int(CLICKHOUSE_HTTP_PORT)},
                {"name": "native", "port": int(CLICKHOUSE_NATIVE_PORT), "targetPort": int(CLICKHOUSE_NATIVE_PORT)},
            ],
            "type": "ClusterIP",
        },
    }

    stateful = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": "clickhouse", "namespace": ns},
        "spec": {
            "serviceName": CLICKHOUSE_SERVICE_NAME,
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "clickhouse"}},
            "template": {
                "metadata": {"labels": {"app": "clickhouse"}},
                "spec": {
                    "containers": [
                        {
                            "name": "clickhouse",
                            "image": image,
                            "ports": [
                                {"containerPort": int(CLICKHOUSE_HTTP_PORT), "name": "http"},
                                {"containerPort": int(CLICKHOUSE_NATIVE_PORT), "name": "native"},
                            ],
                            "volumeMounts": [{"name": "data", "mountPath": "/var/lib/clickhouse"}],
                            "resources": {"requests": {"cpu": req_cpu, "memory": req_mem}, "limits": {"cpu": lim_cpu, "memory": lim_mem}},
                        }
                    ]
                },
            },
            "volumeClaimTemplates": [
                {
                    "metadata": {"name": "data"},
                    "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": pvc_size}}},
                }
            ],
        },
    }

    return [service, stateful]

# --------- wait & post actions ----------
def wait_for_pod(ns: str, label_selector: str, timeout: int):
    start = time.time()
    while True:
        try:
            r = run_quiet(
                f"kubectl -n {ns} get pods -l {label_selector} -o jsonpath='{{.items[0].metadata.name}}'",
                capture=True,
                check=False,
            )
            pod = r.stdout.strip()
            if pod:
                ready = run_quiet(
                    f"kubectl -n {ns} get pod {pod} -o jsonpath='{{.status.containerStatuses[0].ready}}'",
                    capture=True,
                    check=False,
                )
                if ready.stdout.strip() == "true":
                    return pod
        except SystemExit:
            pass
        if time.time() - start > timeout:
            raise SystemExit("timeout waiting for pod ready")
        time.sleep(3)

def create_user_and_table(ns: str, pod: str, user: str, password: str):
    q = (
        "CREATE USER IF NOT EXISTS {u} IDENTIFIED WITH plaintext_password BY '{p}'; "
        "CREATE DATABASE IF NOT EXISTS logs; "
        "CREATE TABLE IF NOT EXISTS logs.kube_logs "
        "(ts DateTime64(3) DEFAULT now(), pod String, namespace String, message String) "
        "ENGINE = MergeTree() ORDER BY ts; "
        "GRANT INSERT ON logs.* TO {u}; "
        "GRANT SELECT ON logs.* TO {u};"
    ).format(u=user, p=password)
    cmd = f"kubectl -n {ns} exec -i {pod} -- clickhouse-client --multiquery --query=\"{q}\""
    run(cmd)

# --------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()

    k8s_cluster = os.getenv("K8S_CLUSTER", "kind").strip().lower()
    ns = os.getenv("NAMESPACE", "observability").strip()

    env = {
        "NAMESPACE": ns,
        "CLICKHOUSE_IMAGE_REPO": os.getenv("CLICKHOUSE_IMAGE_REPO", "clickhouse/clickhouse-server"),
        "CLICKHOUSE_IMAGE_TAG": os.getenv("CLICKHOUSE_IMAGE_TAG", "25.8"),
        "CLICKHOUSE_REPLICAS": os.getenv("CLICKHOUSE_REPLICAS", "1"),
        "CLICKHOUSE_PVC_SIZE": os.getenv("CLICKHOUSE_PVC_SIZE", "5Gi"),
        "CLICKHOUSE_REQ_CPU": os.getenv("CLICKHOUSE_REQ_CPU", "100m"),
        "CLICKHOUSE_REQ_MEM": os.getenv("CLICKHOUSE_REQ_MEM", "512Mi"),
        "CLICKHOUSE_LIMIT_CPU": os.getenv("CLICKHOUSE_LIMIT_CPU", "500m"),
        "CLICKHOUSE_LIMIT_MEM": os.getenv("CLICKHOUSE_LIMIT_MEM", "1Gi"),
        "SETUP_TIMEOUT_SEC": int(os.getenv("SETUP_TIMEOUT_SEC", "600")),
    }

    if args.generate:
        docs = build_clickhouse_docs(env)
        write_yaml(MANIFEST_FILE, docs)
        return

    if args.apply:
        ensure_namespace(env["NAMESPACE"])
        user = os.getenv("CLICKHOUSE_USER", "vector")
        password = os.getenv("CLICKHOUSE_PASSWORD", "vectorpass")
        create_secret(env["NAMESPACE"], CLICKHOUSE_SECRET_NAME, user, password)
        docs = build_clickhouse_docs(env)
        write_yaml(MANIFEST_FILE, docs)
        kubectl_apply(MANIFEST_FILE)
        pod = wait_for_pod(env["NAMESPACE"], "app=clickhouse", env["SETUP_TIMEOUT_SEC"])
        create_user_and_table(env["NAMESPACE"], pod, user, password)
        print("[ok] clickhouse applied")
        return

    if args.delete:
        if os.path.exists(MANIFEST_FILE):
            kubectl_delete(MANIFEST_FILE)
            os.remove(MANIFEST_FILE)
            print("[info] removed", MANIFEST_FILE)
        delete_secret(os.getenv("NAMESPACE", env["NAMESPACE"]), CLICKHOUSE_SECRET_NAME)
        print("[ok] clickhouse deleted")
        return

    p.print_help()

if __name__ == "__main__":
    main()
