#!/usr/bin/env python3
"""
infra/generators/clickhouse.py

Generate / apply / delete ClickHouse Service + StatefulSet, create secret
and ensure user/database/table exist. Grants INSERT and SELECT.
This hardened generator adds readinessProbe, validates YAML and retries
the SQL creation step to tolerate transient pod initialization delays.
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
import yaml
import time
from typing import List, Dict, Any

# --------- static constants ----------
CLICKHOUSE_SERVICE_NAME = "clickhouse"
CLICKHOUSE_HTTP_PORT = "8123"
CLICKHOUSE_NATIVE_PORT = "9000"
CLICKHOUSE_SECRET_NAME = "clickhouse-credentials"

MANIFEST_DIR = os.path.join("infra", "manifests", "clickhouse")
MANIFEST_FILE = os.path.join(MANIFEST_DIR, "clickhouse.yaml")


# --------- helpers ----------
def run(cmd: str, capture: bool = False, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("[run]", cmd)
    res = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    if check and res.returncode != 0:
        if capture:
            print(res.stdout or "")
            print(res.stderr or "", file=sys.stderr)
        raise SystemExit(res.returncode)
    return res


def run_quiet(cmd: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    return run(cmd, capture=capture, check=check, quiet=True)


def ensure_namespace(ns: str) -> None:
    run(
        f"kubectl create namespace {ns} --dry-run=client -o yaml"
        " | kubectl apply -f -"
    )


def write_yaml(path: str, docs: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False)
    print("[info] wrote", path)


def kubectl_apply(path: str) -> None:
    run(f"kubectl apply -f {path}")


def kubectl_delete(path: str) -> None:
    run(f"kubectl delete -f {path} --ignore-not-found")


def create_secret(ns: str, name: str, username: str, password: str) -> None:
    cmd = (
        f"kubectl -n {ns} create secret generic {name} "
        f"--from-literal=username={username} "
        f"--from-literal=password={password} "
        "--dry-run=client -o yaml | kubectl apply -f -"
    )
    run(cmd)


def delete_secret(ns: str, name: str) -> None:
    run(f"kubectl -n {ns} delete secret {name} --ignore-not-found")


# --------- manifest builder ----------
def build_clickhouse_docs(env: Dict[str, str]) -> List[Dict[str, Any]]:
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

    # StatefulSet with a readinessProbe so Service endpoints only appear when ClickHouse accepts queries.
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
                            "resources": {
                                # slightly larger defaults to avoid OOM during startup
                                "requests": {"cpu": req_cpu, "memory": req_mem},
                                "limits": {"cpu": lim_cpu, "memory": lim_mem},
                            },
                            "readinessProbe": {
                                "exec": {"command": ["clickhouse-client", "--query", "SELECT 1"]},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 10,
                                "failureThreshold": 3,
                                "timeoutSeconds": 5,
                            },
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
def wait_for_pod(ns: str, label_selector: str, timeout: int) -> str:
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


def _shell_single_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def create_user_and_table(ns: str, pod: str, user: str, password: str, ttl_days: int) -> None:
    try:
        ttl_val = int(ttl_days)
    except Exception:
        ttl_val = 0

    ttl_clause = ""
    if ttl_val > 0:
        ttl_clause = f" TTL ts + INTERVAL {ttl_val} DAY DELETE"

    safe_password = password.replace("'", "''")

    parts: List[str] = []
    parts.append(f"CREATE USER IF NOT EXISTS {user} IDENTIFIED WITH plaintext_password BY '{safe_password}';")
    parts.append("CREATE DATABASE IF NOT EXISTS logs;")

    table_stmt = (
        "CREATE TABLE IF NOT EXISTS logs.kube_logs ("
        "ts DateTime64(3), "
        "level LowCardinality(String), "
        "message String, "
        "service LowCardinality(String), "
        "pod LowCardinality(String), "
        "namespace LowCardinality(String), "
        "container LowCardinality(String), "
        "trace_id String, "
        "span_id String, "
        "fields String"
        ") ENGINE = MergeTree() ORDER BY (ts, level)"
    )
    if ttl_clause:
        table_stmt += ttl_clause
    table_stmt += ";"

    parts.append(table_stmt)
    parts.append(f"GRANT INSERT ON logs.* TO {user};")
    parts.append(f"GRANT SELECT ON logs.* TO {user};")

    q = " ".join(parts)
    q_quoted = _shell_single_quote(q)

    # Retry loop because pod may report ready while ClickHouse still initializes
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            cmd = f"kubectl -n {ns} exec -i {pod} -- clickhouse-client --multiquery --query={q_quoted}"
            run(cmd)
            return
        except SystemExit as e:
            if attempt == attempts:
                raise
            print(f"[warn] create_user_and_table attempt {attempt} failed, retrying in 5s...")
            time.sleep(5)


# --------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()

    ns = os.getenv("NAMESPACE", "observability").strip()

    env = {
        "NAMESPACE": ns,
        "CLICKHOUSE_IMAGE_REPO": os.getenv("CLICKHOUSE_IMAGE_REPO", "clickhouse/clickhouse-server"),
        "CLICKHOUSE_IMAGE_TAG": os.getenv("CLICKHOUSE_IMAGE_TAG", "25.8"),
        "CLICKHOUSE_REPLICAS": os.getenv("CLICKHOUSE_REPLICAS", "1"),
        "CLICKHOUSE_PVC_SIZE": os.getenv("CLICKHOUSE_PVC_SIZE", "5Gi"),
        "CLICKHOUSE_REQ_CPU": os.getenv("CLICKHOUSE_REQ_CPU", "100m"),
        "CLICKHOUSE_REQ_MEM": os.getenv("CLICKHOUSE_REQ_MEM", "1Gi"),
        "CLICKHOUSE_LIMIT_CPU": os.getenv("CLICKHOUSE_LIMIT_CPU", "1"),
        "CLICKHOUSE_LIMIT_MEM": os.getenv("CLICKHOUSE_LIMIT_MEM", "2Gi"),
        "SETUP_TIMEOUT_SEC": int(os.getenv("SETUP_TIMEOUT_SEC", "600")),
        "LOGS_TTL_DAYS": os.getenv("LOGS_TTL_DAYS", "7"),
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
        # dry-run validation
        run(f"kubectl apply --dry-run=client -f {MANIFEST_FILE}")
        kubectl_apply(MANIFEST_FILE)
        pod = wait_for_pod(env["NAMESPACE"], "app=clickhouse", env["SETUP_TIMEOUT_SEC"])
        create_user_and_table(env["NAMESPACE"], pod, user, password, env["LOGS_TTL_DAYS"])
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
