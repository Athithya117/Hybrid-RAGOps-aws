#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import yaml
CLICKHOUSE_NAMESPACE = os.getenv("CH_NAMESPACE", os.getenv("NAMESPACE", "observability"))
CLICKHOUSE_SERVICE_NAME = os.getenv("CLICKHOUSE_SERVICE_NAME", "clickhouse")
CLICKHOUSE_APP_LABEL = os.getenv("CLICKHOUSE_APP_LABEL", "clickhouse")
CLICKHOUSE_STS_NAME = os.getenv("CLICKHOUSE_STS_NAME", "ch-single")
CLICKHOUSE_IMAGE = os.getenv("CLICKHOUSE_IMAGE", "clickhouse/clickhouse-server:23.12.6")
CLICKHOUSE_PVC_SIZE = os.getenv("CLICKHOUSE_PVC_SIZE", "10Gi")
CLICKHOUSE_REPLICAS = int(os.getenv("CLICKHOUSE_REPLICAS", "1"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "vector")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "vectorpass")
CLICKHOUSE_SECRET_NAME = os.getenv("CLICKHOUSE_SECRET_NAME", "clickhouse-credentials")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "logs")
CLICKHOUSE_TABLE = os.getenv("CLICKHOUSE_TABLE", "kube_logs")
CLICKHOUSE_INIT_TIMEOUT = int(os.getenv("CLICKHOUSE_INIT_TIMEOUT_SEC", os.getenv("INIT_TIMEOUT_SEC", "300")))
RENDER_DIR = Path(os.getenv("CH_MANIFESTS_DIR", "infra/manifests/clickhouse")).resolve()
STATE_DIR = Path(os.getenv("STATE_DIR", "infra/state")).resolve()
STATE_FILE = STATE_DIR / "clickhouse.json"
RENDER_FILES = {
    "namespace": RENDER_DIR / "00-namespace.yaml",
    "service": RENDER_DIR / "10-clickhouse-service.yaml",
    "single_statefulset": RENDER_DIR / "11-clickhouse-single.yaml",
    "init_sql": RENDER_DIR / "30-init.sql",
    "combined": RENDER_DIR / "clickhouse.yaml",
}
def run(cmd: List[str], timeout: int = 60, check: bool = True) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if check and proc.returncode != 0:
            print("[error] command failed:", " ".join(cmd))
            if out:
                print(out)
            if err:
                print(err, file=sys.stderr)
            raise SystemExit(proc.returncode)
        return {"rc": proc.returncode, "out": out, "err": err}
    except subprocess.TimeoutExpired as e:
        out = getattr(e, "stdout", "") or ""
        err = getattr(e, "stderr", "") or f"timeout after {timeout}s"
        return {"rc": 124, "out": out, "err": err}
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    print("[info] wrote", path)
def validate_env_cluster() -> None:
    kc = os.getenv("K8S_CLUSTER", "kind").strip().lower()
    if kc not in ("kind", "aks"):
        print(f"[error] Unsupported K8S_CLUSTER '{kc}' — allowed: kind, aks")
        raise SystemExit(2)
def render_namespace(ns: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns}}
def render_service(ns: str) -> Dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": CLICKHOUSE_SERVICE_NAME, "namespace": ns},
        "spec": {
            "selector": {"app": CLICKHOUSE_APP_LABEL},
            "ports": [
                {"name": "http", "port": 8123, "targetPort": 8123},
                {"name": "tcp", "port": 9000, "targetPort": 9000},
            ],
            "type": "ClusterIP",
        },
    }
def render_statefulset(ns: str) -> Dict[str, Any]:
    labels = {"app": CLICKHOUSE_APP_LABEL}
    pvc_size = os.getenv("CLICKHOUSE_PVC_SIZE", CLICKHOUSE_PVC_SIZE)
    req_cpu = os.getenv("CLICKHOUSE_REQ_CPU", "250m")
    req_mem = os.getenv("CLICKHOUSE_REQ_MEM", "1Gi")
    lim_cpu = os.getenv("CLICKHOUSE_LIMIT_CPU", "1")
    lim_mem = os.getenv("CLICKHOUSE_LIMIT_MEM", "2Gi")
    volume_claim = {
        "metadata": {"name": "data"},
        "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": pvc_size}}},
    }
    ss = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": CLICKHOUSE_STS_NAME, "namespace": ns},
        "spec": {
            "serviceName": CLICKHOUSE_SERVICE_NAME,
            "replicas": CLICKHOUSE_REPLICAS,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": "clickhouse",
                            "image": CLICKHOUSE_IMAGE,
                            "ports": [{"containerPort": 8123, "name": "http"}, {"containerPort": 9000, "name": "tcp"}],
                            "volumeMounts": [{"name": "data", "mountPath": "/var/lib/clickhouse"}],
                            "resources": {"requests": {"cpu": req_cpu, "memory": req_mem}, "limits": {"cpu": lim_cpu, "memory": lim_mem}},
                        }
                    ]
                },
            },
            "volumeClaimTemplates": [volume_claim],
        },
    }
    return ss
def render_init_sql(db: str = CLICKHOUSE_DB, table: str = CLICKHOUSE_TABLE, user: str = CLICKHOUSE_USER, password: str = CLICKHOUSE_PASSWORD) -> str:
    q = []
    q.append(f"CREATE DATABASE IF NOT EXISTS {db};")
    q.append(
        "CREATE TABLE IF NOT EXISTS {db}.{table} (ts DateTime64(3) DEFAULT now(), service String, pod String, namespace String, message String, fields String, level String, container String, trace_id String, span_id String) ENGINE = MergeTree() ORDER BY ts;".format(
            db=db, table=table
        )
    )
    q.append(f"CREATE USER IF NOT EXISTS {user} IDENTIFIED WITH plaintext_password BY '{password}';")
    q.append(f"GRANT INSERT ON {db}.* TO {user};")
    q.append(f"GRANT SELECT ON {db}.* TO {user};")
    return "\n".join(q) + "\n"
def generate_manifests():
    ensure_dir(RENDER_DIR)
    ns_doc = render_namespace(CLICKHOUSE_NAMESPACE)
    svc_doc = render_service(CLICKHOUSE_NAMESPACE)
    ss_doc = render_statefulset(CLICKHOUSE_NAMESPACE)
    init_sql = render_init_sql()
    atomic_write(RENDER_FILES["namespace"], yaml.safe_dump(ns_doc, sort_keys=False))
    atomic_write(RENDER_FILES["service"], yaml.safe_dump(svc_doc, sort_keys=False))
    atomic_write(RENDER_FILES["single_statefulset"], yaml.safe_dump(ss_doc, sort_keys=False))
    atomic_write(RENDER_FILES["init_sql"], init_sql)
    combined = "\n---\n".join([yaml.safe_dump(ns_doc, sort_keys=False), yaml.safe_dump(svc_doc, sort_keys=False), yaml.safe_dump(ss_doc, sort_keys=False)])
    atomic_write(RENDER_FILES["combined"], combined)
    print("[info] generated clickhouse manifests at", RENDER_DIR)
def ensure_kubectl():
    if not shutil.which("kubectl"):
        print("[error] kubectl not found in PATH")
        raise SystemExit(2)
def create_secret_if_missing(ns: str, name: str, user: str, password: str) -> None:
    rc = run(["kubectl", "-n", ns, "get", "secret", name], check=False, timeout=10)
    if rc["rc"] != 0:
        yaml_manifest = (
            "apiVersion: v1\nkind: Secret\nmetadata:\n"
            f"  name: {name}\n  namespace: {ns}\n"
            "type: Opaque\nstringData:\n"
            f"  username: \"{user}\"\n"
            f"  password: \"{password}\"\n"
        )
        run(["bash", "-lc", f"cat <<'Y' | kubectl apply -f -\n{yaml_manifest}\nY"], timeout=30, check=True)
        print(f"[info] created secret {name} in namespace {ns}")
    else:
        print("[info] secret exists; leaving as-is")
def wait_for_pod_ready(ns: str, label_selector: str, timeout: int = 180) -> str:
    start = time.time()
    while True:
        rc = run(["kubectl", "-n", ns, "get", "pods", "-l", label_selector, "-o", "json"], timeout=15, check=False)
        if rc["rc"] == 0 and rc["out"]:
            try:
                data = json.loads(rc["out"])
                items = data.get("items", [])
            except Exception:
                items = []
            if items:
                for p in items:
                    name = p.get("metadata", {}).get("name", "")
                    statuses = p.get("status", {}).get("containerStatuses", [])
                    if statuses and statuses[0].get("ready") is True:
                        return name
        if time.time() - start > timeout:
            print("[error] timeout waiting for pod ready with selector", label_selector)
            raise SystemExit(3)
        time.sleep(2)
def run_init_sql(ns: str, pod: str) -> None:
    check_cmd = ["kubectl", "-n", ns, "exec", "-i", pod, "--", "bash", "-lc", "clickhouse-client --query 'SELECT 1' || true"]
    start = time.time()
    last_out = ""
    while True:
        rc = run(check_cmd, timeout=20, check=False)
        last_out = (rc.get("out", "") or "") + "\n" + (rc.get("err", "") or "")
        if rc["rc"] == 0 and ("1" in (rc["out"] or "").split()):
            break
        if time.time() - start > CLICKHOUSE_INIT_TIMEOUT:
            print("[error] timeout waiting for clickhouse-client to accept connections; last output:")
            print(last_out)
            raise SystemExit(3)
        time.sleep(2)
    sql = Path(RENDER_FILES["init_sql"]).read_text(encoding="utf-8")
    cols = [
        ("service", "String"),
        ("pod", "String"),
        ("namespace", "String"),
        ("message", "String"),
        ("fields", "String"),
        ("level", "String"),
        ("container", "String"),
        ("trace_id", "String"),
        ("span_id", "String"),
    ]
    alter_stmts = []
    for cname, ctype in cols:
        alter_stmts.append(f"ALTER TABLE {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE} ADD COLUMN IF NOT EXISTS {cname} {ctype} DEFAULT ''")
    sql_combined = sql.strip() + "\n" + ";\n".join(alter_stmts) + ";\n"
    sql_safe = sql_combined.replace("'", "'\\''")
    cmd = ["kubectl", "-n", ns, "exec", "-i", pod, "--", "bash", "-lc", f"clickhouse-client --multiquery --query '{sql_safe}'"]
    rc = run(cmd, timeout=180, check=False)
    if rc["rc"] != 0:
        print("[warn] init SQL execution returned non-zero; stdout/err follow")
        if rc.get("out"):
            print(rc["out"])
        if rc.get("err"):
            print(rc["err"], file=sys.stderr)
    else:
        print("[info] init SQL applied successfully")
def write_state_artifact():
    ensure_dir(STATE_DIR)
    state = {
        "namespace": CLICKHOUSE_NAMESPACE,
        "service": CLICKHOUSE_SERVICE_NAME,
        "app_label": CLICKHOUSE_APP_LABEL,
        "statefulset": CLICKHOUSE_STS_NAME,
        "db": CLICKHOUSE_DB,
        "table": CLICKHOUSE_TABLE,
        "user": CLICKHOUSE_USER,
        "secret": CLICKHOUSE_SECRET_NAME,
    }
    atomic_write(STATE_FILE, json.dumps(state, indent=2))
    print("[info] wrote state artifact", STATE_FILE)
def apply_manifests():
    validate_env_cluster()
    ensure_kubectl()
    generate_manifests()
    run(["kubectl", "apply", "-f", str(RENDER_FILES["namespace"])])
    run(["kubectl", "apply", "-f", str(RENDER_FILES["service"])])
    run(["kubectl", "apply", "-f", str(RENDER_FILES["single_statefulset"])])
    create_secret_if_missing(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SECRET_NAME, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD)
    pod = wait_for_pod_ready(CLICKHOUSE_NAMESPACE, f"app={CLICKHOUSE_APP_LABEL}", timeout=300)
    print("[info] clickhouse pod ready:", pod)
    run_init_sql(CLICKHOUSE_NAMESPACE, pod)
    write_state_artifact()
    print("[ok] clickhouse apply complete")
def delete_manifests(confirm: bool = False) -> None:
    if not confirm:
        print("[error] delete requires --confirm")
        raise SystemExit(2)
    ensure_kubectl()
    for f in ("combined", "single_statefulset", "service", "namespace"):
        p = RENDER_FILES.get(f)
        if p and p.exists():
            run(["kubectl", "delete", "-f", str(p), "--ignore-not-found"], timeout=60, check=False)
            try:
                p.unlink()
            except Exception:
                pass
    run(["kubectl", "-n", CLICKHOUSE_NAMESPACE, "delete", "secret", CLICKHOUSE_SECRET_NAME, "--ignore-not-found"], check=False, timeout=30)
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except Exception:
        pass
    print("[ok] clickhouse delete complete")
def parse_args() -> Dict[str, Any]:
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true")
    return vars(p.parse_args())
def main() -> None:
    args = parse_args()
    if args.get("generate"):
        generate_manifests()
        write_state_artifact()
        return
    if args.get("apply"):
        apply_manifests()
        return
    if args.get("delete"):
        delete_manifests(confirm=args.get("confirm", False))
        return
if __name__ == "__main__":
    main()
