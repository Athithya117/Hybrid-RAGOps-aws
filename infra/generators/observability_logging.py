import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("observability_logging")

def run_cmd(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: int = 300) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def kubectl_available() -> bool:
    return shutil.which("kubectl") is not None

def helm_available() -> bool:
    return shutil.which("helm") is not None

K8S_CLUSTER = os.getenv("K8S_CLUSTER", "kind").lower()
if K8S_CLUSTER not in ("kind", "aks"):
    LOG.error("Invalid K8S_CLUSTER '%s' — allowed: kind, aks", K8S_CLUSTER)
    sys.exit(2)

OBS_NAMESPACE = os.getenv("OBS_NAMESPACE", "observability")
MANIFESTS_DIR = Path(os.getenv("LOKI_MANIFESTS_DIR", "infra/manifests/loki")).resolve()
RELEASE_NAME = os.getenv("LOKI_RELEASE_NAME", "loki")
HELM_REPO_NAME = os.getenv("LOKI_HELM_REPO_NAME", "grafana")
HELM_REPO_URL = os.getenv("LOKI_HELM_REPO_URL", "https://grafana.github.io/helm-charts")
CHART_NAME = os.getenv("LOKI_CHART_NAME", "loki")
CHART_VERSION = os.getenv("LOKI_CHART_VERSION", "6.49.0")
VECTOR_IMAGE = os.getenv("VECTOR_IMAGE", "timberio/vector:0.51.1-distroless-libc")
PROMTAIL_IMAGE = os.getenv("PROMTAIL_IMAGE", "grafana/promtail:3.5.0")

LOKI_PERSISTENCE = os.getenv("LOKI_PERSISTENCE", os.getenv("LOKI_PERSISTENCE_ENABLED", "true")).lower() in ("1", "true", "yes")
LOKI_PERSISTENCE_SIZE = os.getenv("LOKI_PERSISTENCE_SIZE", "50Gi")
LOKI_STORAGE_CLASS = os.getenv("LOKI_STORAGE_CLASS", "")

LOKI_LOG_NAMESPACES = os.getenv("LOKI_LOG_NAMESPACES", "")
LOKI_LOG_OPTIN_ANNOTATION = os.getenv("LOKI_LOG_OPTIN_ANNOTATION", "logs.grafana.com/enabled=true")

HELM_TIMEOUT = int(os.getenv("HELM_TIMEOUT_SECONDS", "600"))
HELM_WAIT_ENV = os.getenv("HELM_WAIT", "")
HELM_WAIT = (HELM_WAIT_ENV.lower() in ("1","true","yes")) if HELM_WAIT_ENV else (K8S_CLUSTER == "aks")

STORAGE_SECRET_NAME = os.getenv("LOKI_STORAGE_SECRET_NAME", "loki-storage-creds")
RENDER_VALUES_NAME = os.getenv("LOKI_RENDER_VALUES", "values.yaml")
VALUES_PATH = MANIFESTS_DIR / RENDER_VALUES_NAME
NAMESPACE_FILE = MANIFESTS_DIR / "00-namespace.yaml"
SHIPPER_MANIFEST = MANIFESTS_DIR / "vector-shipper.yaml"

if not LOKI_LOG_NAMESPACES:
    LOG.error("LOKI_LOG_NAMESPACES is required and must include 'observability'. Example: export LOKI_LOG_NAMESPACES='observability,inference'")
    sys.exit(2)

ns_list = [n.strip() for n in LOKI_LOG_NAMESPACES.split(",") if n.strip()]
if "observability" not in ns_list:
    LOG.error("LOKI_LOG_NAMESPACES must include 'observability'")
    sys.exit(2)

if K8S_CLUSTER == "aks" and LOKI_PERSISTENCE and not LOKI_STORAGE_CLASS:
    LOG.error("LOKI_STORAGE_CLASS is required on AKS when persistence is enabled")
    sys.exit(2)

def render_namespace(ns: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns, "labels": {"observability-managed": "true"}}}

def render_values() -> Dict[str, Any]:
    persistence_enabled = LOKI_PERSISTENCE and K8S_CLUSTER == "aks"
    values: Dict[str, Any] = {}
    values["replicaCount"] = int(os.getenv("LOKI_REPLICA_COUNT", "3")) if K8S_CLUSTER == "aks" else 1
    values["mode"] = os.getenv("LOKI_MODE", "simple-scalable")
    values["persistence"] = {"enabled": bool(persistence_enabled), "size": LOKI_PERSISTENCE_SIZE}
    if persistence_enabled and LOKI_STORAGE_CLASS:
        values["persistence"]["storageClassName"] = LOKI_STORAGE_CLASS
    values.setdefault("loki", {})
    values["loki"].setdefault("config", {})
    values["loki"]["config"]["schema_config"] = {"configs": [{"from": "2020-10-24", "store": "boltdb-shipper", "object_store": "filesystem", "schema": "v11", "index": {"prefix": "index_", "period": "24h"}}]}
    values["metrics"] = {"enabled": True}
    values["resources"] = {"limits": {"cpu": "2000m", "memory": "8Gi"}, "requests": {"cpu": "500m", "memory": "2Gi"}}
    node_selector_key = os.getenv("OBS_NODE_SELECTOR_KEY", "observability")
    node_selector_val = os.getenv("OBS_NODE_SELECTOR_VALUE", "true")
    values["nodeSelector"] = {node_selector_key: node_selector_val}
    taint_key = os.getenv("OBS_TAINT_KEY", "CriticalAddonsOnly")
    values["tolerations"] = [{"key": taint_key, "operator": "Exists", "effect": "NoSchedule"}]
    values["vector"] = {"enabled": True, "image": {"repository": VECTOR_IMAGE.split(":", 1)[0], "tag": VECTOR_IMAGE.split(":", 1)[1] if ":" in VECTOR_IMAGE else "latest"}}
    return values

def render_vector_config(namespaces: List[str], optin_annotation: str) -> str:
    key, val = optin_annotation.split("=", 1) if "=" in optin_annotation else (optin_annotation, "true")
    cfg = {
        "sources": {
            "kubernetes_logs": {
                "type": "kubernetes_logs",
                "include_namespaces": namespaces,
                "extra_labels": {}
            }
        },
        "transforms": {
            "filter_opt_in": {
                "type": "filter",
                "condition": f'coalesce(get(.kubernetes.pod.annotations, "{key}"), "") == "{val}"'
            }
        },
        "sinks": {
            "loki": {
                "type": "http",
                "inputs": ["filter_opt_in"],
                "uri": f"http://{RELEASE_NAME}-loki.{OBS_NAMESPACE}.svc.cluster.local:3100/loki/api/v1/push",
                "encoding": {"codec": "json"}
            }
        }
    }
    return yaml.safe_dump(cfg, sort_keys=False)

def render_vector_shipper(namespaces: List[str], optin_annotation: str) -> Dict[str, Any]:
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "vector-config", "namespace": OBS_NAMESPACE, "labels": {"managed-by": "observability_logging"}}, "data": {"vector.yaml": render_vector_config(namespaces, optin_annotation)}}
    ds = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": "vector-logs-shipper", "namespace": OBS_NAMESPACE, "labels": {"managed-by": "observability_logging"}},
        "spec": {
            "selector": {"matchLabels": {"app": "vector-logs-shipper"}},
            "template": {
                "metadata": {"labels": {"app": "vector-logs-shipper"}},
                "spec": {
                    "serviceAccountName": "vector-logs-shipper",
                    "containers": [
                        {
                            "name": "vector",
                            "image": VECTOR_IMAGE,
                            "args": ["-c","/etc/vector/vector.yaml"],
                            "volumeMounts": [{"name": "config", "mountPath": "/etc/vector"}],
                            "resources": {"requests": {"cpu": "100m", "memory": "200Mi"}, "limits": {"cpu": "500m", "memory": "1Gi"}}
                        }
                    ],
                    "volumes": [{"name": "config", "configMap": {"name": "vector-config"}}]
                }
            }
        }
    }
    return {"configmap": cm, "daemonset": ds}

def build_storage_secret() -> Optional[Dict[str, Any]]:
    data: Dict[str, str] = {}
    if os.getenv("AWS_ACCESS_KEY_ID"):
        data["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        data["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    if os.getenv("AZURE_ACCOUNT_KEY"):
        data["AZURE_STORAGE_KEY"] = os.getenv("AZURE_ACCOUNT_KEY")
    if not data:
        return None
    return {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": STORAGE_SECRET_NAME, "namespace": OBS_NAMESPACE}, "type": "Opaque", "stringData": data}

def apply_secret(secret_manifest: Dict[str, Any]):
    if not kubectl_available():
        raise RuntimeError("kubectl not found")
    payload = yaml.safe_dump(secret_manifest, sort_keys=False)
    rc, out, err = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=payload.encode("utf-8"), timeout=30)
    if rc != 0:
        raise RuntimeError(f"kubectl apply secret failed: {err or out}")

def delete_secret(name: str, namespace: str):
    if not kubectl_available():
        raise RuntimeError("kubectl not found")
    rc, out, err = run_cmd(["kubectl", "delete", "secret", name, "-n", namespace, "--ignore-not-found"], timeout=30)
    if rc != 0:
        raise RuntimeError(f"kubectl delete secret failed: {err or out}")

def helm_repo_add_and_update():
    if not helm_available():
        LOG.error("helm not found in PATH")
        raise RuntimeError("helm not found")
    rc, out, err = run_cmd(["helm", "repo", "add", "--force-update", HELM_REPO_NAME, HELM_REPO_URL], timeout=30)
    if rc == 0:
        LOG.info("Helm repo %s added/updated", HELM_REPO_NAME)
    else:
        LOG.warning("helm repo add: %s %s", out, err)
    rc2, out2, err2 = run_cmd(["helm", "repo", "update"], timeout=60)
    if rc2 == 0:
        LOG.info("Helm repo update completed")
    else:
        LOG.warning("helm repo update: %s %s", out2, err2)

def helm_pending_release_cleanup(namespace: str, release: str):
    if not kubectl_available():
        LOG.warning("kubectl not available; cannot cleanup pending helm secrets")
        return
    rc, out, err = run_cmd(["kubectl", "get", "secret", "-n", namespace, "-l", "owner=helm", "-o", "json"], timeout=30)
    if rc != 0 or not out.strip():
        return
    try:
        data = json.loads(out)
    except Exception:
        return
    items = data.get("items", [])
    pending_names: List[str] = []
    for s in items:
        name = s.get("metadata", {}).get("name", "")
        if not name.startswith(f"sh.helm.release.v1.{release}.v"):
            continue
        encoded = ""
        encoded = ""
        d = s.get("data", {})
        if "release" in d:
            try:
                encoded = base64.b64decode(d["release"]).decode("utf-8", errors="ignore")
            except Exception:
                encoded = ""
        search = encoded.lower()
        if any(k in search for k in ("pending-install", "pending-upgrade", "pending-rollback")):
            pending_names.append(name)
    if not pending_names:
        return
    for n in pending_names:
        rc2, out2, err2 = run_cmd(["kubectl", "delete", "secret", n, "-n", namespace], timeout=20)
        if rc2 == 0:
            LOG.info("Deleted pending helm secret: %s", n)
        else:
            LOG.warning("Failed deleting helm pending secret %s: %s %s", n, out2, err2)

def helm_upgrade_install(values_file: Path):
    if not helm_available():
        LOG.error("helm not found")
        raise RuntimeError("helm required")
    chart_ref = f"{HELM_REPO_NAME}/{CHART_NAME}"
    cmd = ["helm", "upgrade", "--install", RELEASE_NAME, chart_ref, "--namespace", OBS_NAMESPACE, "--create-namespace", "-f", str(values_file)]
    if HELM_WAIT:
        cmd += ["--wait", "--timeout", f"{HELM_TIMEOUT}s"]
    if CHART_VERSION:
        cmd += ["--version", CHART_VERSION]
    LOG.info("Running helm command: %s", " ".join(cmd))
    rc, out, err = run_cmd(cmd, timeout=HELM_TIMEOUT + 60)
    if rc == 0:
        LOG.info("Helm upgrade/install succeeded for %s", RELEASE_NAME)
        return
    LOG.warning("Helm upgrade/install failed: rc=%d stdout=%s stderr=%s", rc, out, err)
    if "another operation" in (err or "") or "is in progress" in (err or ""):
        LOG.info("Helm reported operation in progress; attempting cleanup of pending secrets")
        helm_pending_release_cleanup(OBS_NAMESPACE, RELEASE_NAME)
        rc2, out2, err2 = run_cmd(cmd, timeout=HELM_TIMEOUT + 60)
        if rc2 == 0:
            LOG.info("Helm retry succeeded after pending-secret cleanup")
            return
    if "cannot re-use a name that is still in use" in (err or "") or "already exists" in (err or ""):
        LOG.info("Release name conflict - inspecting existing releases")
        rc_list, out_list, err_list = run_cmd(["helm", "list", "-n", OBS_NAMESPACE, "-o", "json"], timeout=30)
        if rc_list == 0 and out_list.strip():
            try:
                arr = json.loads(out_list)
                for r in arr:
                    if r.get("name") == RELEASE_NAME:
                        status = r.get("status")
                        LOG.info("Existing release %s found with status %s", RELEASE_NAME, status)
                        if status in ("deployed", "failed"):
                            cmd_upgrade = ["helm", "upgrade", RELEASE_NAME, chart_ref, "--namespace", OBS_NAMESPACE, "-f", str(values_file)]
                            if CHART_VERSION:
                                cmd_upgrade += ["--version", CHART_VERSION]
                            if HELM_WAIT:
                                cmd_upgrade += ["--wait", "--timeout", f"{HELM_TIMEOUT}s"]
                            rc3, out3, err3 = run_cmd(cmd_upgrade, timeout=HELM_TIMEOUT + 60)
                            if rc3 == 0:
                                LOG.info("Helm upgrade succeeded for existing release")
                                return
            except Exception:
                pass
    LOG.error("Helm upgrade/install ultimately failed: %s", err or out)
    raise RuntimeError(err or out or "helm failed")

def wait_for_namespace_active(ns: str, timeout_sec: int = 300, poll: int = 3):
    start = time.time()
    while True:
        rc, out, err = run_cmd(["kubectl", "get", "ns", ns, "-o", "jsonpath={.status.phase}"], timeout=10)
        if rc != 0:
            LOG.info("Namespace %s not found; will create", ns)
            return
        phase = out.strip()
        if phase == "Active":
            LOG.info("Namespace %s is Active", ns)
            return
        if phase == "Terminating":
            if time.time() - start > timeout_sec:
                LOG.error("Namespace %s stuck Terminating >%ds", ns, timeout_sec)
                raise RuntimeError("namespace terminating")
            LOG.info("Namespace %s terminating; waiting up to %ds", ns, timeout_sec - int(time.time() - start))
            time.sleep(poll)
            continue
        LOG.info("Namespace %s in phase %s; waiting", ns, phase)
        time.sleep(poll)

def ensure_namespace():
    ensure_dir(MANIFESTS_DIR)
    ns_manifest = render_namespace(OBS_NAMESPACE)
    atomic_write(NAMESPACE_FILE, yaml.safe_dump(ns_manifest, sort_keys=False))
    if not kubectl_available():
        LOG.error("kubectl not in PATH")
        raise RuntimeError("kubectl required")
    rc, out, err = run_cmd(["kubectl", "get", "ns", OBS_NAMESPACE, "-o", "jsonpath={.status.phase}"], timeout=10)
    if rc == 0 and out.strip() == "Active":
        LOG.info("Applying namespace manifest idempotently")
        rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", str(NAMESPACE_FILE)], timeout=20)
        if rc2 != 0:
            LOG.error("kubectl apply namespace failed: %s %s", out2, err2)
            raise RuntimeError("kubectl apply namespace failed")
        return
    if rc == 0 and out.strip() == "Terminating":
        LOG.warning("Namespace %s is Terminating; waiting", OBS_NAMESPACE)
        wait_for_namespace_active(OBS_NAMESPACE)
    LOG.info("Creating namespace %s", OBS_NAMESPACE)
    rc3, out3, err3 = run_cmd(["kubectl", "apply", "-f", str(NAMESPACE_FILE)], timeout=20)
    if rc3 != 0:
        LOG.error("Failed to apply namespace manifest: %s %s", out3, err3)
        raise RuntimeError("failed to apply namespace manifest")
    wait_for_namespace_active(OBS_NAMESPACE)

def generate():
    ensure_dir(MANIFESTS_DIR)
    ns_manifest = render_namespace(OBS_NAMESPACE)
    atomic_write(NAMESPACE_FILE, yaml.safe_dump(ns_manifest, sort_keys=False))
    values = render_values()
    atomic_write(VALUES_PATH, yaml.safe_dump(values, sort_keys=False))
    ship = render_vector_shipper(ns_list, LOKI_LOG_OPTIN_ANNOTATION)
    atomic_write(SHIPPER_MANIFEST, yaml.safe_dump(ship, sort_keys=False))
    LOG.info("Generated manifests at %s", str(MANIFESTS_DIR))
    LOG.info(" - %s", NAMESPACE_FILE)
    LOG.info(" - %s", VALUES_PATH)
    LOG.info(" - %s", SHIPPER_MANIFEST)

def apply():
    generate()
    ensure_namespace()
    if not kubectl_available():
        LOG.error("kubectl not found")
        raise RuntimeError("kubectl required")
    secret = build_storage_secret()
    if secret:
        LOG.info("Applying storage secret in-memory")
        apply_secret(secret)
        LOG.info("Applied storage secret %s", STORAGE_SECRET_NAME)
    helm_repo_add_and_update()
    LOG.info("Installing/upgrading Loki chart (release=%s namespace=%s)", RELEASE_NAME, OBS_NAMESPACE)
    helm_upgrade_install(VALUES_PATH)
    LOG.info("Applying shipper manifests")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", str(SHIPPER_MANIFEST)], timeout=60)
    if rc != 0:
        LOG.error("kubectl apply shipper failed: %s %s", out, err)
        raise RuntimeError("kubectl apply shipper failed")
    LOG.info("Applied shipper manifest %s", SHIPPER_MANIFEST)
    LOG.info("Apply completed")

def delete(prune_namespace: bool = False):
    if helm_available():
        LOG.info("Uninstalling helm release %s in %s", RELEASE_NAME, OBS_NAMESPACE)
        rc, out, err = run_cmd(["helm", "uninstall", RELEASE_NAME, "--namespace", OBS_NAMESPACE], timeout=120)
        if rc == 0:
            LOG.info("Helm release uninstalled")
        else:
            LOG.info("Helm uninstall returned: %s %s", out, err)
    else:
        LOG.warning("helm not available; skipping helm uninstall")
    if kubectl_available():
        rc, out, err = run_cmd(["kubectl", "delete", "-f", str(SHIPPER_MANIFEST), "--ignore-not-found"], timeout=60)
        LOG.info("Deleted shipper resources (if any)")
    try:
        delete_secret(STORAGE_SECRET_NAME, OBS_NAMESPACE)
        LOG.info("Deleted storage secret if existed")
    except Exception as e:
        LOG.warning("Warning deleting storage secret: %s", e)
    for p in [VALUES_PATH, SHIPPER_MANIFEST, NAMESPACE_FILE]:
        try:
            if p.exists():
                p.unlink()
                LOG.info("Removed %s", p)
        except Exception as e:
            LOG.warning("Failed to remove %s: %s", p, e)
    if prune_namespace:
        if not kubectl_available():
            LOG.error("kubectl not available; cannot prune namespace")
            raise RuntimeError("kubectl required")
        LOG.info("Pruning namespace %s as requested", OBS_NAMESPACE)
        rc, out, err = run_cmd(["kubectl", "delete", "ns", OBS_NAMESPACE, "--ignore-not-found"], timeout=60)
        if rc == 0:
            LOG.info("Namespace %s deletion requested", OBS_NAMESPACE)
        else:
            LOG.warning("Namespace prune returned: %s %s", out, err)
    else:
        LOG.info("Namespace retained (safe default)")

def parse_args():
    p = argparse.ArgumentParser(description="observability_logging generator (Loki + Vector).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--prune-namespace", action="store_true", help="irreversible: delete the observability namespace (use with caution)")
    return p.parse_args()

def main():
    args = parse_args()
    LOG.info("Starting observability_logging: cluster=%s release=%s namespace=%s vector=%s helm_wait=%s", K8S_CLUSTER, RELEASE_NAME, OBS_NAMESPACE, VECTOR_IMAGE, HELM_WAIT)
    try:
        if args.generate:
            generate()
            return
        if args.apply:
            apply()
            return
        if args.delete:
            delete(prune_namespace=args.prune_namespace)
            return
    except Exception as e:
        LOG.error("ERROR: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
