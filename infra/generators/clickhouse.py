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

# --- Environment variables (single source, top of file) ---
NAMESPACE = os.getenv("NAMESPACE", "observability")
CH_NAMESPACE = os.getenv("CH_NAMESPACE", NAMESPACE)
CLICKHOUSE_SERVICE_NAME = os.getenv("CLICKHOUSE_SERVICE_NAME", "clickhouse")
# NEW: optional separate service name that monitoring may expect
CLICKHOUSE_EXPORTER_SERVICE_NAME = os.getenv("CLICKHOUSE_EXPORTER_SERVICE_NAME", "clickhouse-exporter")
CLICKHOUSE_APP_LABEL = os.getenv("CLICKHOUSE_APP_LABEL", "clickhouse")
CLICKHOUSE_STS_NAME = os.getenv("CLICKHOUSE_STS_NAME", "ch-single")
CLICKHOUSE_IMAGE = os.getenv("CLICKHOUSE_IMAGE", "clickhouse/clickhouse-server:23.12.6")
CLICKHOUSE_PVC_SIZE = os.getenv("CLICKHOUSE_PVC_SIZE", "10Gi")
CLICKHOUSE_REPLICAS = os.getenv("CLICKHOUSE_REPLICAS", "1")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "vector")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "vectorpass")
CLICKHOUSE_SECRET_NAME = os.getenv("CLICKHOUSE_SECRET_NAME", "clickhouse-credentials")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "logs")
CLICKHOUSE_TABLE = os.getenv("CLICKHOUSE_TABLE", "kube_logs")
CLICKHOUSE_INIT_TIMEOUT_SEC = os.getenv("CLICKHOUSE_INIT_TIMEOUT_SEC", os.getenv("INIT_TIMEOUT_SEC", "300"))
CLICKHOUSE_ENABLE_EXPORTER = os.getenv("CLICKHOUSE_ENABLE_EXPORTER", "true")
CLICKHOUSE_EXPORTER_IMAGE = os.getenv("CLICKHOUSE_EXPORTER_IMAGE", "flant/clickhouse-exporter:nightly-10102021")
CLICKHOUSE_EXPORTER_PORT = os.getenv("CLICKHOUSE_EXPORTER_PORT", "9116")
# default exporter args: use local loopback (sidecar) and correct flag names for this exporter binary
CLICKHOUSE_EXPORTER_ARGS = os.getenv("CLICKHOUSE_EXPORTER_ARGS", f"--scrape_uri=http://127.0.0.1:8123/ --telemetry.address=:{CLICKHOUSE_EXPORTER_PORT}")
CLICKHOUSE_STRICT_EXPORTER_HEALTH = os.getenv("CLICKHOUSE_STRICT_EXPORTER_HEALTH", "true")
CH_MANIFESTS_DIR = os.getenv("CH_MANIFESTS_DIR", "infra/manifests/clickhouse")
STATE_DIR = os.getenv("STATE_DIR", "infra/state")
K8S_CLUSTER = os.getenv("K8S_CLUSTER", "kind")

# --- Typed conversions immediately after env reads ---
try:
    CLICKHOUSE_REPLICAS = int(CLICKHOUSE_REPLICAS)
except Exception:
    CLICKHOUSE_REPLICAS = 1

try:
    CLICKHOUSE_INIT_TIMEOUT = int(CLICKHOUSE_INIT_TIMEOUT_SEC)
except Exception:
    CLICKHOUSE_INIT_TIMEOUT = 300

CLICKHOUSE_ENABLE_EXPORTER = CLICKHOUSE_ENABLE_EXPORTER.lower() == "true"
CLICKHOUSE_STRICT_EXPORTER_HEALTH = CLICKHOUSE_STRICT_EXPORTER_HEALTH.lower() == "true"

try:
    CLICKHOUSE_EXPORTER_PORT = int(CLICKHOUSE_EXPORTER_PORT)
except Exception:
    CLICKHOUSE_EXPORTER_PORT = 9116

RENDER_DIR = Path(CH_MANIFESTS_DIR).resolve()
STATE_DIR = Path(STATE_DIR).resolve()
STATE_FILE = STATE_DIR / "clickhouse.json"
RENDER_FILES = {
    "namespace": RENDER_DIR / "00-namespace.yaml",
    "service": RENDER_DIR / "10-clickhouse-service.yaml",
    "metrics_service": RENDER_DIR / "20-clickhouse-exporter-service.yaml",
    "single_statefulset": RENDER_DIR / "11-clickhouse-single.yaml",
    "init_sql": RENDER_DIR / "30-init.sql",
    "combined": RENDER_DIR / "clickhouse.yaml",
}

# --- Utility functions ---
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
    kc = K8S_CLUSTER.strip().lower()
    if kc not in ("kind", "aks"):
        print(f"[error] Unsupported K8S_CLUSTER '{kc}' — allowed: kind, aks")
        raise SystemExit(2)

# --- Rendering helpers (logic preserved) ---
def render_namespace(ns: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns}}

def render_service(ns: str) -> Dict[str, Any]:
    ports = [
        {"name": "http", "port": 8123, "targetPort": 8123},
        {"name": "tcp", "port": 9000, "targetPort": 9000},
    ]
    # expose exporter port in the Service only when exporter enabled
    if CLICKHOUSE_ENABLE_EXPORTER:
        ports.append({"name": "metrics", "port": CLICKHOUSE_EXPORTER_PORT, "targetPort": CLICKHOUSE_EXPORTER_PORT})
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": CLICKHOUSE_SERVICE_NAME, "namespace": ns},
        "spec": {
            "selector": {"app": CLICKHOUSE_APP_LABEL},
            "ports": ports,
            "type": "ClusterIP",
        },
    }

def render_metrics_service(ns: str) -> Dict[str, Any]:
    """
    Render an additional Service with name CLICKHOUSE_EXPORTER_SERVICE_NAME that exposes only the metrics port.
    This makes monitoring jobs that expect service name 'clickhouse-exporter' discover the endpoints.
    """
    if not CLICKHOUSE_ENABLE_EXPORTER:
        # return a minimal no-op that won't be used
        return {}
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": CLICKHOUSE_EXPORTER_SERVICE_NAME, "namespace": ns},
        "spec": {
            "selector": {"app": CLICKHOUSE_APP_LABEL},
            "ports": [
                {"name": "metrics", "port": CLICKHOUSE_EXPORTER_PORT, "targetPort": CLICKHOUSE_EXPORTER_PORT}
            ],
            "type": "ClusterIP",
        },
    }
    return svc

def render_statefulset(ns: str) -> Dict[str, Any]:
    labels = {"app": CLICKHOUSE_APP_LABEL}
    pvc_size = CLICKHOUSE_PVC_SIZE
    req_cpu = os.getenv("CLICKHOUSE_REQ_CPU", "250m")
    req_mem = os.getenv("CLICKHOUSE_REQ_MEM", "1Gi")
    lim_cpu = os.getenv("CLICKHOUSE_LIMIT_CPU", "1")
    lim_mem = os.getenv("CLICKHOUSE_LIMIT_MEM", "2Gi")
    volume_claim = {
        "metadata": {"name": "data"},
        "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": pvc_size}}},
    }
    containers: List[Dict[str, Any]] = [
        {
            "name": "clickhouse",
            "image": CLICKHOUSE_IMAGE,
            "ports": [{"containerPort": 8123, "name": "http"}, {"containerPort": 9000, "name": "tcp"}],
            "volumeMounts": [{"name": "data", "mountPath": "/var/lib/clickhouse"}],
            "resources": {"requests": {"cpu": req_cpu, "memory": req_mem}, "limits": {"cpu": lim_cpu, "memory": lim_mem}},
        }
    ]
    # Exporter sidecar (optional, controlled by CLICKHOUSE_ENABLE_EXPORTER)
    if CLICKHOUSE_ENABLE_EXPORTER:
        # allow args override by splitting string into list (shell-like)
        exporter_args = CLICKHOUSE_EXPORTER_ARGS.strip()
        if exporter_args == "":
            exporter_args_list: List[str] = []
        else:
            # Basic split; preserve quoted segments if any (shlex)
            import shlex
            try:
                exporter_args_list = shlex.split(exporter_args)
            except Exception:
                exporter_args_list = exporter_args.split()

        containers.append({
            "name": "clickhouse-exporter",
            "image": CLICKHOUSE_EXPORTER_IMAGE,
            "args": exporter_args_list,
            "ports": [{"containerPort": CLICKHOUSE_EXPORTER_PORT, "name": "metrics"}],
            "resources": {"requests": {"cpu": "10m", "memory": "32Mi"}, "limits": {"cpu": "100m", "memory": "128Mi"}},
            # minimal startupProbe to allow ClickHouse to come up without immediate exporter exit due to transient errors
            "startupProbe": {
                "httpGet": {"path": "/", "port": CLICKHOUSE_EXPORTER_PORT},
                "failureThreshold": 12,
                "periodSeconds": 5
            }
        })
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
                    "containers": containers
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

# --- Manifest generation and state writing ---
def generate_manifests():
    ensure_dir(RENDER_DIR)
    ns_doc = render_namespace(CH_NAMESPACE)
    svc_doc = render_service(CH_NAMESPACE)
    metrics_svc_doc = render_metrics_service(CH_NAMESPACE) if CLICKHOUSE_ENABLE_EXPORTER else None
    ss_doc = render_statefulset(CH_NAMESPACE)
    init_sql = render_init_sql()
    atomic_write(RENDER_FILES["namespace"], yaml.safe_dump(ns_doc, sort_keys=False))
    atomic_write(RENDER_FILES["service"], yaml.safe_dump(svc_doc, sort_keys=False))
    if metrics_svc_doc:
        atomic_write(RENDER_FILES["metrics_service"], yaml.safe_dump(metrics_svc_doc, sort_keys=False))
    atomic_write(RENDER_FILES["single_statefulset"], yaml.safe_dump(ss_doc, sort_keys=False))
    atomic_write(RENDER_FILES["init_sql"], init_sql)
    parts = [yaml.safe_dump(ns_doc, sort_keys=False), yaml.safe_dump(svc_doc, sort_keys=False)]
    if metrics_svc_doc:
        parts.append(yaml.safe_dump(metrics_svc_doc, sort_keys=False))
    parts.append(yaml.safe_dump(ss_doc, sort_keys=False))
    combined = "\n---\n".join(parts)
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
                    # consider pod ready if all containers Ready==True
                    if statuses:
                        all_ready = True
                        for s in statuses:
                            if s.get("ready") is not True:
                                all_ready = False
                                break
                        if all_ready:
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
        "namespace": CH_NAMESPACE,
        "service": CLICKHOUSE_SERVICE_NAME,
        "exporter_service": CLICKHOUSE_EXPORTER_SERVICE_NAME,
        "app_label": CLICKHOUSE_APP_LABEL,
        "statefulset": CLICKHOUSE_STS_NAME,
        "db": CLICKHOUSE_DB,
        "table": CLICKHOUSE_TABLE,
        "user": CLICKHOUSE_USER,
        "secret": CLICKHOUSE_SECRET_NAME,
        "exporter": {
            "enabled": CLICKHOUSE_ENABLE_EXPORTER,
            "image": CLICKHOUSE_EXPORTER_IMAGE,
            "port": CLICKHOUSE_EXPORTER_PORT,
        },
    }
    atomic_write(STATE_FILE, json.dumps(state, indent=2))
    print("[info] wrote state artifact", STATE_FILE)

def apply_manifests():
    """
    Apply manifests to cluster. This function is used by both rollout (preferred)
    and the legacy --apply alias. Behavior is unchanged.
    """
    validate_env_cluster()
    ensure_kubectl()
    generate_manifests()
    # warn if monitoring scrape might be enabled without exporter present
    if not CLICKHOUSE_ENABLE_EXPORTER:
        print(f"[info] ClickHouse exporter is DISABLED. If monitoring has ENABLE_CLICKHOUSE_EXPORTER_SCRAPE=true, set CLICKHOUSE_ENABLE_EXPORTER=true to expose /metrics.")
    else:
        print("[info] ClickHouse exporter enabled; monitoring should scrape Service:", CLICKHOUSE_EXPORTER_SERVICE_NAME)
    run(["kubectl", "apply", "-f", str(RENDER_FILES["namespace"])])
    run(["kubectl", "apply", "-f", str(RENDER_FILES["service"])])
    # apply exporter service if present
    if CLICKHOUSE_ENABLE_EXPORTER and RENDER_FILES.get("metrics_service"):
        run(["kubectl", "apply", "-f", str(RENDER_FILES["metrics_service"])])
    run(["kubectl", "apply", "-f", str(RENDER_FILES["single_statefulset"])])
    create_secret_if_missing(CH_NAMESPACE, CLICKHOUSE_SECRET_NAME, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD)
    pod = wait_for_pod_ready(CH_NAMESPACE, f"app={CLICKHOUSE_APP_LABEL}", timeout=300)
    print("[info] clickhouse pod ready:", pod)
    run_init_sql(CH_NAMESPACE, pod)
    write_state_artifact()
    if CLICKHOUSE_ENABLE_EXPORTER:
        try:
            probe_cmd = [
                "kubectl", "-n", CH_NAMESPACE, "run", "--rm", "-i", "--restart=Never", "chk-exporter-test",
                "--image=curlimages/curl", "--command", "--", "sh", "-c",
                f"curl -sS --max-time 5 http://{CLICKHOUSE_EXPORTER_SERVICE_NAME}.{CH_NAMESPACE}.svc.cluster.local:{CLICKHOUSE_EXPORTER_PORT}/metrics >/dev/null && echo OK || echo FAIL"
            ]
            p = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=20)
            ok = ("OK" in (p.stdout or "" ) or "OK" in (p.stderr or "" ))
        except Exception:
            ok = False
        if ok:
            print("[info] clickhouse exporter endpoint reachable via service", CLICKHOUSE_EXPORTER_SERVICE_NAME)
        else:
            msg = "clickhouse exporter endpoint not reachable via exporter service"
            if CLICKHOUSE_STRICT_EXPORTER_HEALTH:
                print("[error]", msg)
                raise SystemExit(5)
            else:
                print("[warn]", msg, "- continuing (set CLICKHOUSE_STRICT_EXPORTER_HEALTH=true to fail)")
    print("[ok] clickhouse apply complete")

def rollout_manifests():
    """
    Preferred entrypoint for creating or converging resources to desired state.
    Uses same behavior as apply_manifests but prints an explicit rollout header.
    """
    print("[info] rollout started")
    apply_manifests()

def delete_manifests(confirm: bool = False) -> None:
    if not confirm:
        print("[error] delete requires --confirm")
        raise SystemExit(2)
    ensure_kubectl()
    for f in ("combined", "single_statefulset", "service", "metrics_service", "namespace"):
        p = RENDER_FILES.get(f)
        if p and p.exists():
            run(["kubectl", "delete", "-f", str(p), "--ignore-not-found"], timeout=60, check=False)
            try:
                p.unlink()
            except Exception:
                pass
    run(["kubectl", "-n", CH_NAMESPACE, "delete", "secret", CLICKHOUSE_SECRET_NAME, "--ignore-not-found"], check=False, timeout=30)
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except Exception:
        pass
    print("[ok] clickhouse delete complete")

def parse_args() -> Dict[str, Any]:
    import argparse
    p = argparse.ArgumentParser(description="clickhouse manifest generator")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--rollout", action="store_true", help="Create or converge resources to desired state (preferred over --apply)")
    g.add_argument("--apply", action="store_true", help="Legacy alias for --rollout (deprecated)")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true")
    return vars(p.parse_args())

def main() -> None:
    args = parse_args()
    if args.get("generate"):
        generate_manifests()
        write_state_artifact()
        return
    if args.get("rollout"):
        rollout_manifests()
        return
    if args.get("apply"):
        # backward compatibility
        print("[warn] --apply is deprecated; use --rollout")
        apply_manifests()
        return
    if args.get("delete"):
        delete_manifests(confirm=args.get("confirm", False))
        return

if __name__ == "__main__":
    main()
