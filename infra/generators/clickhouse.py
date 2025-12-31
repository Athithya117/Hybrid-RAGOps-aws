#!/usr/bin/env python3
"""
generators/clickhouse.py

Deterministic ClickHouse generator for e2e automation.

- Default mode: single-node ClickHouse (good for kind/dev)
- Optional operator mode (ALTINITY) available via CH_MODE=operator
- Auto-fallbacks for image pull failures: tries deterministic list of images
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("clickhouse_generator")

# ---------- Helpers ----------
def run_cmd(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return {"rc": proc.returncode, "out": out, "err": err}
    except subprocess.TimeoutExpired as e:
        return {"rc": 124, "out": getattr(e, "stdout", "") or "", "err": getattr(e, "stderr", "") or f"timeout after {timeout}s"}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def kubectl_available() -> bool:
    return shutil.which("kubectl") is not None

def helm_available() -> bool:
    return shutil.which("helm") is not None

def sleep_seconds(s: int) -> None:
    time.sleep(s)

def wait_for_pod_label(ns: str, label_selector: str, timeout: int = 180) -> None:
    start = time.time()
    while True:
        rc = run_cmd(["kubectl", "-n", ns, "get", "pods", "-l", label_selector, "-o", "json"], timeout=10)
        if rc["rc"] == 0 and rc["out"]:
            try:
                data = json.loads(rc["out"])
                items = data.get("items", [])
            except Exception:
                items = []
            if items:
                all_ready = True
                for p in items:
                    statuses = p.get("status", {}).get("containerStatuses", [])
                    if not statuses or statuses[0].get("ready") is not True:
                        all_ready = False
                        break
                if all_ready:
                    return
        if time.time() - start > timeout:
            raise RuntimeError(f"timeout waiting for pods with label {label_selector} in ns {ns}")
        time.sleep(2)

# ---------- Environment & defaults ----------
K8S_CLUSTER = os.getenv("K8S_CLUSTER", "kind").lower()
if K8S_CLUSTER not in ("kind", "aks"):
    LOG.error("Unsupported K8S_CLUSTER '%s' — allowed: kind, aks", K8S_CLUSTER)
    sys.exit(2)

CH_MODE = os.getenv("CH_MODE", "single").lower()  # single | operator
CH_NAMESPACE = os.getenv("CH_NAMESPACE", "clickhouse")
RENDER_DIR = Path(os.getenv("CH_MANIFESTS_DIR", "infra/manifests/clickhouse")).resolve()
RENDER_FILES = {
    "namespace": RENDER_DIR / "00-namespace.yaml",
    "single_statefulset": RENDER_DIR / "10-clickhouse-single.yaml",
    "service": RENDER_DIR / "11-clickhouse-service.yaml",
    "chi": RENDER_DIR / "20-clickhouse-installation.yaml",  # for operator mode
    "init_sql": RENDER_DIR / "30-init.sql",
}

# NOTE: changed default to a known 23.12 series tag (previously used 23.12.4.36 which was missing)
# Verified clickhouse/clickhouse-server has 23.12.* tags on Docker Hub. See Docker Hub tags for reference.
CLICKHOUSE_IMAGE = os.getenv("CLICKHOUSE_IMAGE", "clickhouse/clickhouse-server:23.12.6")
PVC_SIZE = os.getenv("CLICKHOUSE_PVC_SIZE", "10Gi")
STORAGE_CLASS = os.getenv("CLICKHOUSE_STORAGE_CLASS", "")  # if empty uses cluster default
REPLICAS = int(os.getenv("CLICKHOUSE_REPLICAS", "1"))

# operator-mode defaults (altinity)
ALTINITY_HELM_REPO = "https://helm.altinity.com"
ALTINITY_CHART = os.getenv("ALTINITY_CHART", "altinity/clickhouse-operator")
ALTINITY_CHART_VERSION = os.getenv("ALTINITY_CHART_VERSION", "0.25.6")
ALTINITY_RELEASE = os.getenv("ALTINITY_RELEASE", "clickhouse-operator")

# SQL / DB used by metrics pipeline
PLUTO_DB = os.getenv("CLICKHOUSE_DB", "pluto")

# fallback images to try deterministically when ImagePullBackOff occurs
IMAGE_FALLBACKS = [
    "clickhouse/clickhouse-server:latest",
    "altinity/clickhouse-server:latest",
]

# ---------- Renderers ----------
def render_namespace(ns: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns}}

def render_single_statefulset(ns: str, name: str = "ch-single", image: str = CLICKHOUSE_IMAGE) -> Dict[str, Any]:
    labels = {"app": name}
    volume_claim = {
        "metadata": {"name": "data"},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": PVC_SIZE}},
        },
    }
    if STORAGE_CLASS:
        volume_claim["spec"]["storageClassName"] = STORAGE_CLASS
    ss = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": name, "namespace": ns, "labels": labels},
        "spec": {
            "serviceName": name,
            "replicas": REPLICAS,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": "clickhouse",
                            "image": image,
                            "ports": [{"containerPort": 8123, "name": "http"}, {"containerPort": 9000, "name": "tcp"}],
                            "volumeMounts": [{"name": "data", "mountPath": "/var/lib/clickhouse"}],
                            "resources": {"requests": {"cpu": "250m", "memory": "1Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
                        }
                    ]
                },
            },
            "volumeClaimTemplates": [volume_claim],
        },
    }
    return ss

def render_single_service(ns: str, name: str = "ch-single") -> Dict[str, Any]:
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": ns},
        "spec": {
            "selector": {"app": name},
            "ports": [
                {"name": "http", "port": 8123, "targetPort": 8123},
                {"name": "tcp", "port": 9000, "targetPort": 9000},
            ],
        },
    }
    return svc

def render_chi(ns: str) -> Dict[str, Any]:
    chi = {
        "apiVersion": "clickhouse.altinity.com/v1",
        "kind": "ClickHouseInstallation",
        "metadata": {"name": "chi-simple", "namespace": ns},
        "spec": {
            "defaults": {},
            "configuration": {
                "clusters": [
                    {
                        "name": "cluster",
                        "layout": {"shardsCount": 1, "replicasCount": 1},
                        "hosts": [{"name": "chi-0", "podTemplate": {}, "volumeClaimTemplates": []}],
                    }
                ],
            },
        },
    }
    return chi

def render_init_sql(db: str = PLUTO_DB) -> str:
    return f"CREATE DATABASE IF NOT EXISTS {db};\n"

# ---------- Generation / Apply / Delete ----------
def ensure_dir_and_write_namespace():
    ensure_dir(RENDER_DIR)
    ns = render_namespace(CH_NAMESPACE)
    atomic_write(RENDER_FILES["namespace"], yaml.safe_dump(ns, sort_keys=False))

def generate():
    ensure_dir(RENDER_DIR)
    ensure_dir_and_write_namespace()
    ss = render_single_statefulset(CH_NAMESPACE)
    svc = render_single_service(CH_NAMESPACE)
    atomic_write(RENDER_FILES["single_statefulset"], yaml.safe_dump(ss, sort_keys=False))
    atomic_write(RENDER_FILES["service"], yaml.safe_dump(svc, sort_keys=False))
    chi = render_chi(CH_NAMESPACE)
    atomic_write(RENDER_FILES["chi"], yaml.safe_dump(chi, sort_keys=False))
    atomic_write(RENDER_FILES["init_sql"], render_init_sql(PLUTO_DB))
    LOG.info("Generated ClickHouse manifests at %s", RENDER_DIR)
    for k, v in RENDER_FILES.items():
        LOG.info(" - %s", v)

def helm_repo_add_and_update_altinity():
    if not helm_available():
        LOG.error("helm not found in PATH; required to install altinity operator")
        raise RuntimeError("helm required")
    rc = run_cmd(["helm", "repo", "add", "altinity", ALTINITY_HELM_REPO], timeout=30)
    if rc["rc"] != 0:
        LOG.warning("helm repo add altinity returned non-zero: %s", rc["err"] or rc["out"])
    run_cmd(["helm", "repo", "update"], timeout=60)

def helm_install_operator():
    if not helm_available():
        raise RuntimeError("helm required for operator mode")
    helm_repo_add_and_update_altinity()
    cmd = ["helm", "upgrade", "--install", ALTINITY_RELEASE, ALTINITY_CHART, "-n", CH_NAMESPACE, "--create-namespace", "--wait", "--timeout", "10m"]
    if ALTINITY_CHART_VERSION:
        cmd += ["--version", ALTINITY_CHART_VERSION]
    rc = run_cmd(cmd, timeout=900)
    if rc["rc"] != 0:
        raise RuntimeError(f"failed to install altinity operator: {rc['err'] or rc['out']}")

def try_statefulset_image_patch(ns: str, sts_name: str, container_name: str, new_image: str) -> bool:
    """
    Patch the image used by the StatefulSet template and trigger rolling update.
    Returns True on success (kubectl returned 0), False otherwise.
    """
    cmd = ["kubectl", "-n", ns, "set", "image", f"statefulset/{sts_name}", f"{container_name}={new_image}", "--record"]
    rc = run_cmd(cmd, timeout=60)
    if rc["rc"] == 0:
        LOG.info("Patched StatefulSet %s/%s to image %s", ns, sts_name, new_image)
        return True
    LOG.warning("Failed to patch image for StatefulSet %s/%s -> %s: %s", ns, sts_name, new_image, rc["err"] or rc["out"])
    return False

def inspect_pod_events_for_imagepull(ns: str, pod_name: str) -> List[str]:
    rc = run_cmd(["kubectl", "-n", ns, "describe", "pod", pod_name], timeout=15)
    if rc["rc"] != 0 or not rc["out"]:
        return []
    out = rc["out"]
    lines = []
    for l in out.splitlines():
        if "ImagePullBackOff" in l or "ErrImagePull" in l or "Back-off pulling image" in l or "Failed to pull image" in l:
            lines.append(l.strip())
    return lines

def apply():
    if not kubectl_available():
        raise RuntimeError("kubectl required to apply ClickHouse manifests")
    generate()
    LOG.info("Applying namespace %s", CH_NAMESPACE)
    rc = run_cmd(["kubectl", "apply", "-f", str(RENDER_FILES["namespace"])], timeout=20)
    if rc["rc"] != 0:
        raise RuntimeError(f"failed to apply namespace: {rc['err'] or rc['out']}")

    if CH_MODE == "operator":
        LOG.info("Installing ClickHouse operator via Helm (operator mode)")
        helm_install_operator()
        LOG.info("Applying ClickHouseInstallation CR")
        rc = run_cmd(["kubectl", "apply", "-f", str(RENDER_FILES["chi"])], timeout=30)
        if rc["rc"] != 0:
            raise RuntimeError(f"failed to apply ClickHouseInstallation CR: {rc['err'] or rc['out']}")
        try:
            wait_for_pod_label(CH_NAMESPACE, "app!=dummy", timeout=600)
        except Exception as e:
            LOG.warning("Timeout waiting for operator-managed ClickHouse pods: %s", e)
    else:
        # single mode
        LOG.info("Applying ClickHouse single-node StatefulSet + Service")
        for p in (RENDER_FILES["service"], RENDER_FILES["single_statefulset"]):
            rc = run_cmd(["kubectl", "apply", "-f", str(p)], timeout=60)
            if rc["rc"] != 0:
                raise RuntimeError(f"failed to apply {p}: {rc['err'] or rc['out']}")

        # Wait, but if image pull fails, attempt deterministic fallbacks
        try:
            wait_for_pod_label(CH_NAMESPACE, "app=ch-single", timeout=120)
        except Exception as first_wait_exc:
            LOG.warning("Initial wait for ch-single pods failed: %s", first_wait_exc)
            # Inspect pods
            rc = run_cmd(["kubectl", "-n", CH_NAMESPACE, "get", "pods", "-l", "app=ch-single", "-o", "jsonpath={.items[*].metadata.name}"], timeout=10)
            pod_names = (rc["out"] or "").strip().split()
            if pod_names:
                pod = pod_names[0]
                events = inspect_pod_events_for_imagepull(CH_NAMESPACE, pod)
                if events:
                    LOG.warning("Detected image-pull related events for pod %s: %s", pod, "; ".join(events))
                    # Try fallbacks deterministically
                    for img in IMAGE_FALLBACKS:
                        LOG.info("Attempting fallback image: %s", img)
                        patched = try_statefulset_image_patch(CH_NAMESPACE, "ch-single", "clickhouse", img)
                        if not patched:
                            continue
                        # wait short period for kube to act then wait for pod ready
                        sleep_seconds(5)
                        try:
                            wait_for_pod_label(CH_NAMESPACE, "app=ch-single", timeout=180)
                            LOG.info("Pod became ready after switching to image %s", img)
                            break
                        except Exception as e:
                            LOG.warning("Pod still not ready after switching to %s: %s", img, e)
                    else:
                        LOG.error("All image fallbacks failed; leaving pods as-is for manual inspection")
                else:
                    LOG.warning("Pods exist but no image-pull events detected; manual inspection advised. Pod list: %s", pod_names)
            else:
                LOG.error("No pods found after apply; something else may be wrong")

    # Database init: run SQL if we can find a pod
    pod_rc = run_cmd(["kubectl", "-n", CH_NAMESPACE, "get", "pods", "-l", "app=ch-single", "-o", "jsonpath={.items[0].metadata.name}"], timeout=10)
    pod_name = pod_rc["out"].strip() if pod_rc["rc"] == 0 and pod_rc["out"].strip() else None
    if not pod_name:
        # try any pod in namespace
        pr = run_cmd(["kubectl", "-n", CH_NAMESPACE, "get", "pods", "-o", "jsonpath={.items[0].metadata.name}"], timeout=10)
        if pr["rc"] == 0 and pr["out"].strip():
            pod_name = pr["out"].strip()
    if pod_name:
        sql = RENDER_FILES["init_sql"].read_text(encoding="utf-8")
        LOG.info("Attempting to create DB '%s' on pod %s (best-effort)", PLUTO_DB, pod_name)
        # safely escape single quotes and wrap SQL in single quotes for bash -lc
        sql_clean = sql.strip().replace("'", "'\\''")
        shell_arg = "clickhouse-client --query '{}'".format(sql_clean)
        rc = run_cmd(["kubectl", "-n", CH_NAMESPACE, "exec", pod_name, "--", "bash", "-lc", shell_arg], timeout=60)
        if rc["rc"] != 0:
            LOG.warning("clickhouse-client execution failed (non-fatal): stdout: %s stderr: %s", rc["out"], rc["err"])
        else:
            LOG.info("Database init SQL executed successfully")
    else:
        LOG.warning("No ClickHouse pod found to run init SQL; run SQL manually when ready")

    LOG.info("ClickHouse apply finished (mode=%s).", CH_MODE)

def delete(confirm: bool = False):
    if not confirm:
        raise RuntimeError("Refusing to delete without --confirm")
    if CH_MODE == "operator" and helm_available():
        LOG.info("Uninstalling Altinity ClickHouse operator release %s", ALTINITY_RELEASE)
        rc = run_cmd(["helm", "uninstall", ALTINITY_RELEASE, "-n", CH_NAMESPACE], timeout=120)
        if rc["rc"] == 0:
            LOG.info("Operator uninstalled")
        else:
            LOG.warning("helm uninstall returned non-zero: %s", rc["err"] or rc["out"])
    for key in ("single_statefulset", "service", "chi", "namespace"):
        p = RENDER_FILES.get(key)
        if not p:
            continue
        if p.exists():
            LOG.info("Deleting resources defined in %s (ignore-not-found)", p)
            rc = run_cmd(["kubectl", "delete", "-f", str(p), "--ignore-not-found"], timeout=60)
            if rc["rc"] == 0:
                LOG.info("Deleted resources from %s or they did not exist", p)
            else:
                LOG.warning("kubectl delete %s returned non-zero: stdout: %s stderr: %s", p, rc["out"], rc["err"])
    for f in RENDER_FILES.values():
        try:
            if f.exists():
                f.unlink()
                LOG.info("Removed generated file %s", f)
        except Exception as e:
            LOG.warning("Failed to remove file %s: %s", f, e)
    LOG.info("ClickHouse deletion complete (local manifests removed).")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply/delete ClickHouse manifests for dev/prod")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true", help="required to delete")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if args.generate:
            generate()
            return
        if args.apply:
            apply()
            return
        if args.delete:
            delete(confirm=args.confirm)
            return
    except Exception as e:
        LOG.error("ERROR: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
