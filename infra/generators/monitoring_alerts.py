#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("monitoring_and_alerts")

def run_cmd(cmd: list, input_bytes: bytes | None = None, timeout: int = 60) -> Tuple[int, str, str]:
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
    LOG.error("Unsupported K8S_CLUSTER '%s' — allowed: kind, aks", K8S_CLUSTER)
    sys.exit(2)

OBS_NAMESPACE = os.getenv("OBS_NAMESPACE", "observability")
RENDER_DIR = Path(os.getenv("MONITORING_MANIFESTS_DIR", "infra/manifests/monitoring")).resolve()
KUBE_PROM_STACK_CHART = os.getenv("KUBE_PROM_STACK_CHART", "prometheus-community/kube-prometheus-stack")
KUBE_PROM_STACK_VERSION = os.getenv("KUBE_PROM_STACK_VERSION", "80.6.0")
HELM_REPO_NAME = os.getenv("PROM_HELM_REPO_NAME", "prometheus-community")
HELM_REPO_URL = os.getenv("PROM_HELM_REPO_URL", "https://prometheus-community.github.io/helm-charts")
RELEASE_NAME = os.getenv("PROM_RELEASE_NAME", "kube-prom-stack")
OBS_NODE_SELECTOR_KEY = os.getenv("OBS_NODE_SELECTOR_KEY", "observability")
OBS_NODE_SELECTOR_VALUE = os.getenv("OBS_NODE_SELECTOR_VALUE", "true")
OBS_TAINT_KEY = os.getenv("OBS_TAINT_KEY", "CriticalAddonsOnly")

PROM_ENABLED = os.getenv("PROM_ENABLED", "true").lower() in ("1", "true", "yes")
PROM_STORAGE_SIZE = os.getenv("PROM_STORAGE_SIZE", "50Gi")
PROM_RETENTION = os.getenv("PROM_RETENTION", "7d")
PROM_CPU_REQUEST = os.getenv("PROM_CPU_REQUEST", "500m")
PROM_CPU_LIMIT = os.getenv("PROM_CPU_LIMIT", "500m")
PROM_MEM_REQUEST = os.getenv("PROM_MEM_REQUEST", "1Gi")
PROM_MEM_LIMIT = os.getenv("PROM_MEM_LIMIT", "1Gi")
PROM_STORAGE_CLASS = os.getenv("PROM_STORAGE_CLASS", "")

GRAFANA_ENABLED = os.getenv("GRAFANA_ENABLED", "true").lower() in ("1", "true", "yes")
GRAFANA_ADMIN_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_ADMIN_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "change-me")
GRAFANA_PERSISTENCE = os.getenv("GRAFANA_PERSISTENCE", "true").lower() in ("1", "true", "yes")
GRAFANA_PERSISTENCE_SIZE = os.getenv("GRAFANA_PERSISTENCE_SIZE", "5Gi")
GRAFANA_STORAGE_CLASS = os.getenv("GRAFANA_STORAGE_CLASS", "")

LOKI_PERSISTENCE_SIZE = os.getenv("LOKI_PERSISTENCE_SIZE", "50Gi")
LOKI_RETENTION = os.getenv("LOKI_RETENTION", "7d")
LOKI_CPU_REQUEST = os.getenv("LOKI_CPU_REQUEST", "500m")
LOKI_MEM_REQUEST = os.getenv("LOKI_MEM_REQUEST", "1Gi")
LOKI_CPU_LIMIT = os.getenv("LOKI_CPU_LIMIT", "500m")
LOKI_MEM_LIMIT = os.getenv("LOKI_MEM_LIMIT", "2Gi")
LOKI_SHIPPER = os.getenv("LOKI_SHIPPER", "vector")

ALERTMANAGER_SLACK_WEBHOOK = os.getenv("ALERTMANAGER_SLACK_WEBHOOK", "")

PROM_REPLICAS_ENV = os.getenv("PROM_REPLICAS", "")
if PROM_REPLICAS_ENV:
    try:
        PROM_REPLICAS = int(PROM_REPLICAS_ENV)
        if PROM_REPLICAS < 1:
            raise ValueError()
    except Exception:
        LOG.error("Invalid PROM_REPLICAS '%s' — must be integer >=1", PROM_REPLICAS_ENV)
        sys.exit(2)
else:
    PROM_REPLICAS = 1 if K8S_CLUSTER == "kind" else 2

RENDER_FILES = {
    "namespace": RENDER_DIR / "00-namespace.yaml",
    "retrieval_servicemonitor": RENDER_DIR / "10-retrieval-servicemonitor.yaml",
    "qdrant_servicemonitor": RENDER_DIR / "11-qdrant-servicemonitor.yaml",
    "prometheusrule": RENDER_DIR / "20-rag-prometheusrule.yaml",
    "helm_values": RENDER_DIR / "kube-prom-values.yaml",
}

def render_namespace(ns: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns, "labels": {"name": ns, "observability-managed": "true"}}}

def render_service_monitor(name: str, monitor_ns: str, target_ns: str, selector: Dict[str, str], port: str, interval: str = "15s") -> Dict[str, Any]:
    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "ServiceMonitor",
        "metadata": {"name": name, "namespace": monitor_ns, "labels": {"app.kubernetes.io/managed-by": "monitoring_and_alerts"}},
        "spec": {"selector": {"matchLabels": selector}, "namespaceSelector": {"matchNames": [target_ns]}, "endpoints": [{"port": port, "path": "/metrics", "interval": interval}]}
    }

def render_prometheus_rule(obs_ns: str, p95: float = 0.5, err_rate: float = 0.01, pvc_thresh: float = 0.75, indexer_job: str = "indexing-backup-cronjob") -> Dict[str, Any]:
    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "PrometheusRule",
        "metadata": {"name": "rag-alerts", "namespace": obs_ns, "labels": {"app.kubernetes.io/managed-by": "monitoring_and_alerts"}},
        "spec": {
            "groups": [
                {
                    "name": "retrieval-slo",
                    "rules": [
                        {"alert": "RetrievalUnavailable", "expr": "absent(service_ready{service='retrieval'} == 1)", "for": "2m", "labels": {"severity": "critical"}, "annotations": {"summary": "Retrieval readiness missing"}},
                        {"alert": "RetrievalHigh5xx", "expr": f"sum(rate(retrieval_requests_total{{status_code=~'5..'}}[5m])) / (sum(rate(retrieval_requests_total[5m])) + 1e-12) > {err_rate}", "for": "5m", "labels": {"severity": "page"}, "annotations": {"summary": "Retrieval 5xx error rate high"}},
                        {"alert": "RetrievalP95LatencyBreach", "expr": f"histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)) > {p95}", "for": "5m", "labels": {"severity": "page"}, "annotations": {"summary": "Retrieval p95 latency exceeded"}}
                    ]
                },
                {
                    "name": "qdrant-safety",
                    "rules": [
                        {"alert": "QdrantInstanceDown", "expr": 'up{job=~"qdrant.*"} == 0', "for": "2m", "labels": {"severity": "critical"}, "annotations": {"summary": "Qdrant instance down"}},
                        {"alert": "QdrantDeadReplicas", "expr": "cluster_dead_replicas > 0", "for": "1m", "labels": {"severity": "page"}, "annotations": {"summary": "Qdrant dead replicas detected"}},
                        {"alert": "QdrantPVCHighUsage", "expr": "kubelet_volume_stats_capacity_bytes > 0 and kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes > " + str(pvc_thresh), "for": "10m", "labels": {"severity": "critical"}, "annotations": {"summary": "Qdrant PVC usage above threshold"}}
                    ]
                },
                {
                    "name": "batch",
                    "rules": [
                        {"alert": "IndexerJobFailed", "expr": f'increase(kube_job_status_failed{{job_name=~"{indexer_job}.*"}}[5m]) > 0', "for": "1m", "labels": {"severity": "page"}, "annotations": {"summary": "Indexer job failures detected"}}
                    ]
                }
            ]
        }
    }

def render_helm_values() -> Dict[str, Any]:
    node_selector = {OBS_NODE_SELECTOR_KEY: OBS_NODE_SELECTOR_VALUE}
    toleration = [{"key": OBS_TAINT_KEY, "operator": "Exists", "effect": "NoSchedule"}]
    prom_resources = {"requests": {"cpu": PROM_CPU_REQUEST, "memory": PROM_MEM_REQUEST}, "limits": {"cpu": PROM_CPU_LIMIT, "memory": PROM_MEM_LIMIT}}
    if PROM_STORAGE_CLASS:
        storage_spec = {"volumeClaimTemplate": {"spec": {"storageClassName": PROM_STORAGE_CLASS, "resources": {"requests": {"storage": PROM_STORAGE_SIZE}}}}}
    else:
        storage_spec = {"volumeClaimTemplate": {"spec": {"resources": {"requests": {"storage": PROM_STORAGE_SIZE}}}}}
    grafana_block = {}
    grafana_persistence_enabled = GRAFANA_PERSISTENCE
    if K8S_CLUSTER == "kind":
        grafana_persistence_enabled = False
    if GRAFANA_ENABLED:
        grafana_block = {
            "grafana": {
                "enabled": True,
                "adminUser": GRAFANA_ADMIN_USER,
                "adminPassword": GRAFANA_ADMIN_PASSWORD,
                "persistence": {"enabled": bool(grafana_persistence_enabled), "size": GRAFANA_PERSISTENCE_SIZE}
            }
        }
        if grafana_persistence_enabled and GRAFANA_STORAGE_CLASS:
            grafana_block["grafana"]["persistence"]["storageClassName"] = GRAFANA_STORAGE_CLASS
        grafana_block["grafana"]["nodeSelector"] = node_selector
        grafana_block["grafana"]["tolerations"] = toleration
        grafana_block["grafana"]["securityContext"] = {"fsGroup": 472}
        if K8S_CLUSTER == "kind":
            grafana_block["grafana"]["initChownData"] = {"enabled": False}
    values: Dict[str, Any] = {}
    values.update(grafana_block)
    values["prometheus"] = {
        "prometheusSpec": {
            "replicaCount": PROM_REPLICAS,
            "nodeSelector": node_selector,
            "tolerations": toleration,
            "resources": prom_resources,
            "storageSpec": storage_spec,
            "retention": PROM_RETENTION
        }
    }
    values["alertmanager"] = {"alertmanagerSpec": {"nodeSelector": node_selector, "tolerations": toleration}}
    values["nodeSelector"] = node_selector
    values["tolerations"] = toleration
    return values

def generate():
    ensure_dir(RENDER_DIR)
    ns = render_namespace(OBS_NAMESPACE)
    atomic_write(RENDER_FILES["namespace"], yaml.safe_dump(ns, sort_keys=False))
    sm_retrieval = render_service_monitor("retrieval-servicemonitor", OBS_NAMESPACE, OBS_NAMESPACE, {"app": "retrieval"}, "metrics")
    atomic_write(RENDER_FILES["retrieval_servicemonitor"], yaml.safe_dump(sm_retrieval, sort_keys=False))
    sm_qdrant = render_service_monitor("qdrant-servicemonitor", OBS_NAMESPACE, OBS_NAMESPACE, {"app": "qdrant"}, "http-metrics")
    atomic_write(RENDER_FILES["qdrant_servicemonitor"], yaml.safe_dump(sm_qdrant, sort_keys=False))
    pr = render_prometheus_rule(OBS_NAMESPACE)
    atomic_write(RENDER_FILES["prometheusrule"], yaml.safe_dump(pr, sort_keys=False))
    hv = render_helm_values()
    atomic_write(RENDER_FILES["helm_values"], yaml.safe_dump(hv, sort_keys=False))
    LOG.info("Generated manifests at %s", str(RENDER_DIR))
    for k, v in RENDER_FILES.items():
        LOG.info(" - %s", v)

def wait_for_namespace_ready(ns: str, timeout_sec: int = 300, poll: int = 3) -> None:
    start = time.time()
    while True:
        rc, out, err = run_cmd(["kubectl", "get", "ns", ns, "-o", "jsonpath={.status.phase}"], timeout=10)
        if rc != 0:
            LOG.info("Namespace %s not found (kubectl returned: %s). Will create.", ns, out or err)
            return
        phase = out.strip()
        if phase == "Active":
            LOG.info("Namespace %s is Active", ns)
            return
        if phase == "Terminating":
            if time.time() - start > timeout_sec:
                LOG.error("Namespace %s stuck Terminating for >%ds; manual intervention required", ns, timeout_sec)
                raise RuntimeError(f"namespace {ns} terminating")
            LOG.info("Namespace %s is Terminating — waiting up to %ds for removal", ns, timeout_sec - int(time.time() - start))
            time.sleep(poll)
            continue
        LOG.info("Namespace %s in phase '%s' — waiting", ns, phase)
        time.sleep(poll)

def ensure_namespace(ns: str):
    if not kubectl_available():
        LOG.error("kubectl not found in PATH; cannot ensure namespace")
        raise RuntimeError("kubectl required")
    rc, out, err = run_cmd(["kubectl", "get", "ns", ns, "-o", "jsonpath={.status.phase}"], timeout=10)
    if rc == 0 and out.strip() == "Active":
        LOG.info("Applying namespace manifest %s (idempotent)", RENDER_FILES["namespace"])
        rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", str(RENDER_FILES["namespace"])], timeout=20)
        if rc2 != 0:
            LOG.error("kubectl apply namespace failed: stdout: %s stderr: %s", out2, err2)
            raise RuntimeError("kubectl apply namespace failed")
        return
    if rc == 0 and out.strip() == "Terminating":
        LOG.warning("Namespace %s is terminating — will wait for removal before recreate", ns)
        wait_for_namespace_ready(ns)
    LOG.info("Creating namespace %s via manifest", ns)
    rc3, out3, err3 = run_cmd(["kubectl", "apply", "-f", str(RENDER_FILES["namespace"])], timeout=20)
    if rc3 != 0:
        LOG.error("Failed to apply namespace manifest: stdout: %s stderr: %s", out3, err3)
        raise RuntimeError("failed to apply namespace manifest")
    wait_for_namespace_ready(ns)

def label_nodes_for_parity():
    if not kubectl_available():
        LOG.warning("kubectl not in PATH; skipping node labeling")
        return
    rc, out, err = run_cmd(["kubectl", "get", "nodes", "-l", f"{OBS_NODE_SELECTOR_KEY}={OBS_NODE_SELECTOR_VALUE}", "-o", "name"], timeout=20)
    if rc == 0 and out.strip():
        cnt = len([l for l in out.splitlines() if l.strip()])
        LOG.info("Nodes already labeled: %s=%s (found %d)", OBS_NODE_SELECTOR_KEY, OBS_NODE_SELECTOR_VALUE, cnt)
        return
    if K8S_CLUSTER == "aks":
        rc, out, err = run_cmd(["kubectl", "get", "nodes", "-l", "kubernetes.azure.com/mode=system", "-o", "name"], timeout=20)
        if rc == 0 and out.strip():
            nodes = [n.strip().split("/", 1)[-1] for n in out.splitlines() if n.strip()]
            for n in nodes:
                rc2, o2, e2 = run_cmd(["kubectl", "label", "nodes", n, f"{OBS_NODE_SELECTOR_KEY}={OBS_NODE_SELECTOR_VALUE}", "--overwrite"], timeout=20)
                if rc2 == 0:
                    LOG.info("Labeled AKS system node %s with %s=%s", n, OBS_NODE_SELECTOR_KEY, OBS_NODE_SELECTOR_VALUE)
                else:
                    LOG.error("Failed labeling AKS node %s: stdout: %s stderr: %s", n, o2, e2)
            return
    rc, out, err = run_cmd(["kubectl", "get", "nodes", "-o", "name"], timeout=20)
    if rc != 0 or not out.strip():
        LOG.error("Cannot list cluster nodes to label: stdout: %s stderr: %s", out, err)
        raise RuntimeError("cannot list nodes")
    nodes = [n.strip().split("/", 1)[-1] for n in out.splitlines() if n.strip()]
    target = nodes[0]
    rc, o, e = run_cmd(["kubectl", "label", "nodes", target, f"{OBS_NODE_SELECTOR_KEY}={OBS_NODE_SELECTOR_VALUE}", "--overwrite"], timeout=20)
    if rc == 0:
        LOG.info("Labeled node %s with %s=%s for %s parity", target, OBS_NODE_SELECTOR_KEY, OBS_NODE_SELECTOR_VALUE, K8S_CLUSTER)
    else:
        LOG.error("Failed to label node %s: stdout: %s stderr: %s", target, o, e)
        raise RuntimeError("failed to label node")

def helm_repo_add_and_update():
    if not helm_available():
        LOG.error("helm not found in PATH; required to install charts")
        raise RuntimeError("helm required")
    rc, out, err = run_cmd(["helm", "repo", "add", "--force-update", HELM_REPO_NAME, HELM_REPO_URL], timeout=30)
    if rc == 0:
        LOG.info("Helm repo %s added/updated", HELM_REPO_NAME)
    else:
        LOG.warning("helm repo add returned non-zero: stdout: %s stderr: %s", out, err)
    rc2, out2, err2 = run_cmd(["helm", "repo", "update"], timeout=60)
    if rc2 == 0:
        LOG.info("Helm repo update completed")
    else:
        LOG.warning("helm repo update returned non-zero: stdout: %s stderr: %s", out2, err2)

def helm_release_exists(release: str, namespace: str) -> Optional[Dict[str, Any]]:
    if not helm_available():
        return None
    rc, out, err = run_cmd(["helm", "list", "-n", namespace, "-a", "-o", "json"], timeout=30)
    if rc != 0 or not out.strip():
        return None
    try:
        arr = json.loads(out)
        for item in arr:
            if item.get("name") == release:
                return item
    except Exception:
        return None
    return None

def helm_upgrade_install():
    if not helm_available():
        LOG.error("helm not in PATH; cannot run helm upgrade/install")
        raise RuntimeError("helm required")
    values_file = str(RENDER_FILES["helm_values"])
    cmd = ["helm", "upgrade", "--install", RELEASE_NAME, KUBE_PROM_STACK_CHART, "--namespace", OBS_NAMESPACE, "--create-namespace", "-f", values_file, "--wait", "--timeout", "10m"]
    if KUBE_PROM_STACK_VERSION:
        cmd += ["--version", KUBE_PROM_STACK_VERSION]
    LOG.info("Running Helm upgrade --install for release '%s' (chart=%s, version=%s)", RELEASE_NAME, KUBE_PROM_STACK_CHART, KUBE_PROM_STACK_VERSION or "latest")
    rc, out, err = run_cmd(cmd, timeout=900)
    if rc == 0:
        LOG.info("Helm upgrade/install succeeded for release '%s'", RELEASE_NAME)
        return
    LOG.warning("Helm upgrade --install failed (rc=%d). stdout: %s stderr: %s", rc, out, err)
    if "is forbidden: unable to create new content in namespace" in err or "being terminated" in err:
        LOG.error("Namespace '%s' appears to be terminating or blocking resource creation. Manual fix required.", OBS_NAMESPACE)
        raise RuntimeError("namespace terminating or blocking creation")
    rc_check, out_check, err_check = run_cmd(["helm", "list", "-A", "-o", "json"], timeout=30)
    if rc_check == 0 and out_check.strip():
        try:
            all_releases = json.loads(out_check)
            for r in all_releases:
                if r.get("name") == RELEASE_NAME and r.get("namespace") != OBS_NAMESPACE:
                    LOG.error("Release name '%s' already exists in namespace '%s'. Cannot reuse name in '%s'.", RELEASE_NAME, r.get("namespace"), OBS_NAMESPACE)
                    raise RuntimeError(f"release name conflict: exists in {r.get('namespace')}")
        except Exception:
            pass
    LOG.info("Attempting fallback: helm install with --replace to recover from failed state")
    install_cmd = ["helm", "install", RELEASE_NAME, KUBE_PROM_STACK_CHART, "--namespace", OBS_NAMESPACE, "-f", values_file, "--wait", "--timeout", "10m", "--replace"]
    if KUBE_PROM_STACK_VERSION:
        install_cmd += ["--version", KUBE_PROM_STACK_VERSION]
    rc2, out2, err2 = run_cmd(install_cmd, timeout=900)
    if rc2 == 0:
        LOG.info("Helm install --replace succeeded for release '%s'", RELEASE_NAME)
        return
    LOG.error("Helm install fallback failed (rc=%d). stdout: %s stderr: %s", rc2, out2, err2)
    raise RuntimeError(f"helm upgrade/install ultimately failed: {err2 or out2}")

def apply():
    if not kubectl_available():
        LOG.error("kubectl not found in PATH; required to apply manifests")
        raise RuntimeError("kubectl required")
    generate()
    LOG.info("Applying namespace %s", OBS_NAMESPACE)
    ensure_namespace(OBS_NAMESPACE)
    label_nodes_for_parity()
    if PROM_ENABLED:
        helm_repo_add_and_update()
        LOG.info("Installing/upgrading kube-prometheus-stack via Helm")
        helm_upgrade_install()
    for key in ("retrieval_servicemonitor", "qdrant_servicemonitor", "prometheusrule"):
        p = str(RENDER_FILES[key])
        LOG.info("Applying manifest %s", p)
        rc, out, err = run_cmd(["kubectl", "apply", "-f", p], timeout=30)
        if rc != 0:
            LOG.error("kubectl apply %s failed. stdout: %s stderr: %s", p, out, err)
            raise RuntimeError(f"kubectl apply failed for {p}")
        LOG.info("Applied %s", p)
    LOG.info("Apply completed: manifests generated and applied idempotently")

def delete():
    if helm_available():
        LOG.info("Attempting to uninstall Helm release '%s' in namespace '%s' if present", RELEASE_NAME, OBS_NAMESPACE)
        rc, out, err = run_cmd(["helm", "uninstall", RELEASE_NAME, "--namespace", OBS_NAMESPACE], timeout=120)
        if rc == 0:
            LOG.info("Helm release '%s' uninstalled", RELEASE_NAME)
        else:
            LOG.info("Helm uninstall returned non-zero: stdout: %s stderr: %s", out, err)
    else:
        LOG.warning("helm not available; skipping helm uninstall")
    for key in ("prometheusrule", "retrieval_servicemonitor", "qdrant_servicemonitor"):
        p = str(RENDER_FILES[key])
        LOG.info("Deleting resources defined in %s (ignore-not-found)", p)
        rc, out, err = run_cmd(["kubectl", "delete", "-f", p, "--ignore-not-found"], timeout=30)
        if rc == 0:
            LOG.info("Deleted resources from %s or they did not exist", p)
        else:
            LOG.warning("kubectl delete %s returned non-zero: stdout: %s stderr: %s", p, out, err)
    for f in RENDER_FILES.values():
        try:
            if f.exists():
                f.unlink()
                LOG.info("Removed generated file %s", f)
        except Exception as e:
            LOG.warning("Failed to remove file %s: %s", f, e)
    LOG.info("Delete finished. Namespace '%s' intentionally retained to avoid accidental cluster-wide deletion.", OBS_NAMESPACE)

def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply/delete monitoring and alerts for RAG platform")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    LOG.info("Starting monitoring_and_alerts with K8S_CLUSTER=%s PROM_REPLICAS=%d PROM_STORAGE_CLASS=%s", K8S_CLUSTER, PROM_REPLICAS, PROM_STORAGE_CLASS or "<cluster-default>")
    try:
        if args.generate:
            generate()
            return
        if args.apply:
            apply()
            return
        if args.delete:
            delete()
            return
    except Exception as e:
        LOG.error("ERROR: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
