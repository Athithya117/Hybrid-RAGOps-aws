#!/usr/bin/env python3
"""
monitoring_and_alerts.py

Generates Prometheus/Grafana related manifests under infra/manifests/ and optionally applies them.
- Uses os.getenv for configuration knobs.
- Applies sensitive values (alertmanager/webhook etc.) directly to the cluster as Kubernetes Secrets (kubectl).
- Three CLI modes: --generate, --apply, --delete

Requires: python3.10+, pyyaml, kubectl on PATH when --apply/--delete is used.
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------
# Helpers
# ---------------------
def run_cmd(cmd: List[str], input_bytes: Optional[bytes] = None, timeout:int = 60) -> tuple[int,str,str]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def kubectl_available() -> bool:
    return shutil.which("kubectl") is not None

# ---------------------
# Config (env-driven)
# ---------------------
CFG = {
    "RENDER_DIR": Path(os.getenv("YAML_RENDER_DIR", "infra/manifests")).resolve(),
    "MONITORING_NAMESPACE": os.getenv("MONITORING_NAMESPACE", "monitoring"),
    "RETRIEVAL_NAMESPACE": os.getenv("RETRIEVAL_NAMESPACE", "inference"),
    "RETRIEVAL_SERVICE_LABEL": os.getenv("RETRIEVAL_SERVICE_LABEL", "app=retrieval"),
    "RETRIEVAL_METRICS_PORT": os.getenv("RETRIEVAL_METRICS_PORT", "metrics"),
    "QDRANT_NAMESPACE": os.getenv("QDRANT_NAMESPACE", "qdrant"),
    "QDRANT_SERVICE_LABEL": os.getenv("QDRANT_SERVICE_LABEL", "app=qdrant"),
    "QDRANT_METRICS_PORT": os.getenv("QDRANT_METRICS_PORT", "http-metrics"),
    "SCRAPE_INTERVAL": os.getenv("SCRAPE_INTERVAL", "15s"),
    "RETRIEVAL_P95_THRESHOLD_S": float(os.getenv("RETRIEVAL_P95_THRESHOLD_S", "0.5")),
    "RETRIEVAL_ERROR_RATE": float(os.getenv("RETRIEVAL_ERROR_RATE", "0.01")),
    "QDRANT_PVC_USAGE_THRESHOLD": float(os.getenv("QDRANT_PVC_USAGE_THRESHOLD", "0.75")),
    "INDEXER_CRONJOB_NAME": os.getenv("INDEXER_CRONJOB_NAME", "indexing-backup-cronjob"),
    "INDEXER_NAMESPACE": os.getenv("INDEXER_NAMESPACE", "indexing"),
    "ALERTMANAGER_SLACK_WEBHOOK": os.getenv("ALERTMANAGER_SLACK_WEBHOOK", ""),
    "ALERTMANAGER_SMTP_PASSWORD": os.getenv("ALERTMANAGER_SMTP_PASSWORD", ""),
    "ALERTMANAGER_SECRET_NAME": os.getenv("ALERTMANAGER_SECRET_NAME", "alertmanager-credentials"),
    "ALERTMANAGER_SECRET_NAMESPACE": os.getenv("ALERTMANAGER_SECRET_NAMESPACE", ""),
    "GRAFANA_PROV_NAMESPACE": os.getenv("GRAFANA_PROV_NAMESPACE", os.getenv("MONITORING_NAMESPACE", "monitoring")),
    "GRAFANA_DATASOURCE_NAME": os.getenv("GRAFANA_DATASOURCE_NAME", "Prometheus"),
    "QDRANT_DEAD_REPLICAS_METRIC": os.getenv("QDRANT_DEAD_REPLICAS_METRIC", "cluster_dead_replicas"),
    "QDRANT_PENDING_OPS_METRIC": os.getenv("QDRANT_PENDING_OPS_METRIC", "cluster_pending_operations_total"),
    "QDRANT_SNAPSHOT_CREATED_METRIC": os.getenv("QDRANT_SNAPSHOT_CREATED_METRIC", "snapshot_created_total"),
}

# compute secret namespace default
if not CFG["ALERTMANAGER_SECRET_NAMESPACE"]:
    CFG["ALERTMANAGER_SECRET_NAMESPACE"] = CFG["MONITORING_NAMESPACE"]

# parse label selectors into dict
def parse_label_selector(sel: str) -> Dict[str,str]:
    out: Dict[str,str] = {}
    for part in filter(None, (p.strip() for p in sel.split(","))):
        if "=" in part:
            k,v = part.split("=",1)
            out[k.strip()] = v.strip()
    return out

RETRIEVAL_SELECTOR = parse_label_selector(CFG["RETRIEVAL_SERVICE_LABEL"])
QDRANT_SELECTOR = parse_label_selector(CFG["QDRANT_SERVICE_LABEL"])

# file layout
RENDER_DIR = CFG["RENDER_DIR"]
GRAFANA_DIR = RENDER_DIR / "grafana"
FILES = {
    "namespace": RENDER_DIR / "00-monitoring-namespace.yaml",
    "retrieval_servicemonitor": RENDER_DIR / "10-retrieval-servicemonitor.yaml",
    "qdrant_servicemonitor": RENDER_DIR / "11-qdrant-servicemonitor.yaml",
    "prometheusrule": RENDER_DIR / "20-rag-alerts-prometheusrule.yaml",
    "grafana_provisioning": GRAFANA_DIR / "01-grafana-provisioning.yaml",
}

# ---------------------
# Render functions
# ---------------------
def render_namespace(ns: str) -> Dict[str,Any]:
    return {"apiVersion":"v1","kind":"Namespace","metadata":{"name":ns,"labels":{"app.kubernetes.io/managed-by":"gitops-monitoring"}}}

def render_service_monitor(name: str, namespace: str, selector: Dict[str,str], port: str, path: str, interval: str) -> Dict[str,Any]:
    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "ServiceMonitor",
        "metadata": {"name": name, "namespace": namespace, "labels":{"app.kubernetes.io/managed-by":"gitops-monitoring"}},
        "spec": {
            "selector": {"matchLabels": selector},
            "namespaceSelector": {"matchNames": [namespace]},
            "endpoints": [{"port": port, "path": path, "interval": interval}],
        },
    }

def render_prometheus_rule() -> Dict[str,Any]:
    p95_t = CFG["RETRIEVAL_P95_THRESHOLD_S"]
    err_rate = CFG["RETRIEVAL_ERROR_RATE"]
    pvc_thresh = CFG["QDRANT_PVC_USAGE_THRESHOLD"]
    q_dead = CFG["QDRANT_DEAD_REPLICAS_METRIC"]
    q_pending = CFG["QDRANT_PENDING_OPS_METRIC"]
    q_snapshot = CFG["QDRANT_SNAPSHOT_CREATED_METRIC"]
    indexer_job = CFG["INDEXER_CRONJOB_NAME"]
    return {
        "apiVersion":"monitoring.coreos.com/v1",
        "kind":"PrometheusRule",
        "metadata":{"name":"rag-alerts","namespace":CFG["MONITORING_NAMESPACE"],"labels":{"app.kubernetes.io/managed-by":"gitops-monitoring"}},
        "spec":{
            "groups":[
                {"name":"retrieval-slo",
                 "rules":[
                     {"alert":"RetrievalUnavailable","expr":'absent(service_ready{service="retrieval"}==1)',"for":"2m","labels":{"severity":"critical"},"annotations":{"summary":"Retrieval readiness missing"}},
                     {"alert":"RetrievalHigh5xx","expr":f'sum(rate(retrieval_requests_total{{status_code=~"5.."}}[5m])) / sum(rate(retrieval_requests_total[5m])) > {err_rate}','for':"5m","labels":{"severity":"page"},"annotations":{"summary":"Retrieval 5xx error rate > configured threshold"}},
                     {"alert":"RetrievalP95LatencyBreach","expr":f'histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)) > {p95_t}','for':"5m","labels":{"severity":"page"},"annotations":{"summary":"Retrieval p95 latency exceeded"}},
                 ]},
                {"name":"qdrant-safety",
                 "rules":[
                    {"alert":"QdrantInstanceDown","expr":'up{job=~"qdrant.*"} == 0',"for":"2m","labels":{"severity":"critical"},"annotations":{"summary":"Qdrant instance down"}},
                    {"alert":"QdrantDeadReplicas","expr":f'{q_dead} > 0',"for":"1m","labels":{"severity":"page"},"annotations":{"summary":"Qdrant dead replicas detected"}},
                    {"alert":"QdrantPendingOpsGrowing","expr":f'increase({q_pending}[10m]) > 0',"for":"10m","labels":{"severity":"warning"},"annotations":{"summary":"Qdrant pending operations growing"}},
                    {"alert":"QdrantSnapshotStalled","expr":f'increase({q_snapshot}[1h]) == 0',"for":"1h","labels":{"severity":"critical"},"annotations":{"summary":"No qdrant snapshots created in expected window"}},
                    {"alert":"QdrantPVCHighUsage","expr":f'kubelet_volume_stats_capacity_bytes > 0 and kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes > {pvc_thresh}',"labels":{"severity":"critical"},"annotations":{"summary":"Qdrant PVC usage > configured threshold"}},
                 ]},
                {"name":"batch",
                 "rules":[
                     {"alert":"IndexerJobFailed","expr":f'increase(kube_job_status_failed{{job_name=~"{indexer_job}.*"}}[5m]) > 0',"for":"1m","labels":{"severity":"page"},"annotations":{"summary":"Indexer job failed"}}
                 ]}
            ]
        }
    }

# ---------------------
# Secret handling (apply directly, not to disk)
# ---------------------
def build_alertmanager_secret_manifest() -> Optional[Dict[str,Any]]:
    data = {}
    if CFG["ALERTMANAGER_SLACK_WEBHOOK"]:
        data["slack_webhook"] = CFG["ALERTMANAGER_SLACK_WEBHOOK"]
    if CFG["ALERTMANAGER_SMTP_PASSWORD"]:
        data["smtp_password"] = CFG["ALERTMANAGER_SMTP_PASSWORD"]
    if not data:
        return None
    return {"apiVersion":"v1","kind":"Secret","metadata":{"name":CFG["ALERTMANAGER_SECRET_NAME"],"namespace":CFG["ALERTMANAGER_SECRET_NAMESPACE"]},"type":"Opaque","stringData":data}

def apply_secret(manifest: Dict[str,Any]):
    if not kubectl_available():
        raise RuntimeError("kubectl required to apply secrets but not found in PATH")
    b = yaml.safe_dump(manifest, sort_keys=False)
    rc,out,err = run_cmd(["kubectl","apply","-f","-"], input_bytes=b.encode("utf-8"), timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl apply secret failed: {err or out}")

def delete_secret():
    if not kubectl_available():
        raise RuntimeError("kubectl required to delete secrets but not found in PATH")
    rc,out,err = run_cmd(["kubectl","delete","secret",CFG["ALERTMANAGER_SECRET_NAME"],"-n",CFG["ALERTMANAGER_SECRET_NAMESPACE"],"--ignore-not-found"], timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl delete secret returned non-zero: {err or out}")

# ---------------------
# Generate / Apply / Delete
# ---------------------
def generate_manifests():
    ensure_dir(RENDER_DIR)
    ensure_dir(GRAFANA_DIR)
    # Namespace
    ns = render_namespace(CFG["MONITORING_NAMESPACE"])
    atomic_write(FILES["namespace"], yaml.safe_dump(ns, sort_keys=False))
    # ServiceMonitors
    sm_retrieval = render_service_monitor("retrieval-servicemonitor", CFG["MONITORING_NAMESPACE"], RETRIEVAL_SELECTOR, CFG["RETRIEVAL_METRICS_PORT"], "/metrics", CFG["SCRAPE_INTERVAL"])
    atomic_write(FILES["retrieval_servicemonitor"], yaml.safe_dump(sm_retrieval, sort_keys=False))
    sm_qdrant = render_service_monitor("qdrant-servicemonitor", CFG["MONITORING_NAMESPACE"], QDRANT_SELECTOR, CFG["QDRANT_METRICS_PORT"], "/metrics", CFG["SCRAPE_INTERVAL"])
    atomic_write(FILES["qdrant_servicemonitor"], yaml.safe_dump(sm_qdrant, sort_keys=False))
    # PrometheusRule
    pr = render_prometheus_rule()
    atomic_write(FILES["prometheusrule"], yaml.safe_dump(pr, sort_keys=False))
    print("Generated manifests in:", REENDER_MSG() if False else str(RENDER_DIR))
    for k,p in FILES.items():
        if p.exists():
            print(" -", p.relative_to(Path.cwd()))

def apply_manifests():
    if not kubectl_available():
        print("ERROR: kubectl not found in PATH; cannot apply manifests", file=sys.stderr)
        raise SystemExit(2)
    # apply namespace first
    rc,out,err = run_cmd(["kubectl","apply","-f",str(FILES["namespace"])], timeout=20)
    if rc != 0:
        raise RuntimeError(f"kubectl apply namespace failed: {err or out}")
    # wait a little for namespace readiness
    # apply secret if present
    sm = build_alertmanager_secret_manifest()
    if sm:
        apply_secret(sm)
        print(f"Applied secret {CFG['ALERTMANAGER_SECRET_NAME']} to {CFG['ALERTMANAGER_SECRET_NAMESPACE']}")
    # apply servicemonitors and prometheusrule
    for key in ("retrieval_servicemonitor","qdrant_servicemonitor","prometheusrule"):
        path = FILES[key]
        rc,out,err = run_cmd(["kubectl","apply","-f",str(path)], timeout=20)
        if rc != 0:
            raise RuntimeError(f"kubectl apply {path} failed: {err or out}")
        print("Applied:", path)

def delete_manifests():
    if kubectl_available():
        # delete in reverse-ish order
        for key in ("prometheusrule","retrieval_servicemonitor","qdrant_servicemonitor"):
            path = FILES[key]
            if path.exists():
                rc,out,err = run_cmd(["kubectl","delete","-f",str(path), "--ignore-not-found"], timeout=20)
                if rc != 0:
                    print(f"kubectl delete {path} returned non-zero: {err or out}", file=sys.stderr)
                else:
                    print("Deleted resources from:", path)
        # delete secret
        try:
            delete_secret()
            print("Deleted alertmanager secret (if existed).")
        except Exception as e:
            print("Warning deleting secret:", e, file=sys.stderr)
    else:
        print("kubectl not present; skip cluster deletion step")
    # remove files
    for k,p in FILES.items():
        if p.exists():
            try:
                p.unlink()
                print("Removed file:", p)
            except Exception as e:
                print("Warning removing file:", p, e, file=sys.stderr)
    # remove grafana dir if empty
    try:
        if GRAFANA_DIR.exists() and not any(GRAFANA_DIR.iterdir()):
            GRAFANA_DIR.rmdir()
            print("Removed empty grafana dir")
    except Exception:
        pass

# ---------------------
# CLI
# ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply/delete monitoring manifests.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.generate:
        generate_manifests()
        print("Generation complete.")
        return
    if args.apply:
        generate_manifests()
        try:
            apply_manifests()
            print("Apply complete.")
        except Exception as e:
            print("ERROR during apply:", e, file=sys.stderr)
            raise
        return
    if args.delete:
        delete_manifests()
        print("Delete complete.")
        return

if __name__ == "__main__":
    main()
