"""
infra/generators/dashboard.py

Generate Grafana provisioning + dashboards for:
  - Retrieval SLO dashboard (metrics)
  - Qdrant Health dashboard (metrics + PVC usage)

Writes to infra/manifests/grafana by default.

Usage:
  python dashboard.py
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any

import yaml

# Config (env)
RENDER_DIR = Path(os.getenv("MONITORING_MANIFESTS_DIR", "infra/manifests/monitoring")).resolve()
GRAFANA_DIR = RENDER_DIR / "grafana"
DATASOURCE = os.getenv("GRAFANA_PROMETHEUS_DATASOURCE", "Prometheus")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def render_provisioning() -> Dict[str, Any]:
    return {
        "apiVersion": 1,
        "providers": [
            {
                "name": "RAG Dashboards",
                "orgId": 1,
                "folder": "RAG",
                "type": "file",
                "disableDeletion": False,
                "editable": True,
                "options": {"path": "/var/lib/grafana/dashboards/rag"},
            }
        ],
    }

def retrieval_dashboard() -> Dict[str, Any]:
    ds = DATASOURCE
    panels = []
    panels.append({
        "type": "timeseries", "title": "RPS", "gridPos": {"h": 6, "w": 12, "x": 0, "y": 0},
        "targets": [{"expr": "sum(rate(retrieval_requests_total[1m]))", "refId": "A", "datasource": ds}],
    })
    panels.append({
        "type": "timeseries", "title": "Error rate (5xx)", "gridPos": {"h": 6, "w": 12, "x": 12, "y": 0},
        "targets": [{"expr": "sum(rate(retrieval_requests_total{status_code=~\"5..\"}[5m])) / (sum(rate(retrieval_requests_total[5m])) + 1e-12)", "refId": "A", "datasource": ds}],
    })
    panels.append({
        "type": "timeseries", "title": "Latency (p50/p95/p99)", "gridPos": {"h": 8, "w": 24, "x": 0, "y": 6},
        "targets": [
            {"expr": "histogram_quantile(0.50, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))", "refId": "A", "datasource": ds},
            {"expr": "histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))", "refId": "B", "datasource": ds},
            {"expr": "histogram_quantile(0.99, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))", "refId": "C", "datasource": ds},
        ]
    })
    panels.append({
        "type": "stat", "title": "Service Ready (retrieval)", "gridPos": {"h": 4, "w": 6, "x": 0, "y": 14},
        "targets": [{"expr": "service_ready{service=\"retrieval\"}", "refId": "A", "datasource": ds}],
    })
    dashboard = {
        "title": "Retrieval SLO",
        "uid": "retrieval-slo",
        "schemaVersion": 36,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": panels,
    }
    return dashboard

def qdrant_dashboard() -> Dict[str, Any]:
    ds = DATASOURCE
    panels = []
    panels.append({
        "type": "timeseries", "title": "Qdrant instance up", "gridPos": {"h": 4, "w": 12, "x": 0, "y": 0},
        "targets": [{"expr": 'up{job=~"qdrant.*"}', "refId": "A", "datasource": ds}],
    })
    panels.append({
        "type": "timeseries", "title": "Dead replicas", "gridPos": {"h": 4, "w": 12, "x": 12, "y": 0},
        "targets": [{"expr": "cluster_dead_replicas", "refId": "A", "datasource": ds}],
    })
    panels.append({
        "type": "timeseries", "title": "Pending operations", "gridPos": {"h": 4, "w": 24, "x": 0, "y": 4},
        "targets": [{"expr": "cluster_pending_operations_total", "refId": "A", "datasource": ds}],
    })
    panels.append({
        "type": "timeseries", "title": "PVC usage (qdrant)", "gridPos": {"h": 4, "w": 24, "x": 0, "y": 8},
        "targets": [{"expr": 'kubelet_volume_stats_used_bytes{namespace=~"qdrant"} / kubelet_volume_stats_capacity_bytes{namespace=~"qdrant"}', "refId": "A", "datasource": ds}],
    })
    dashboard = {
        "title": "Qdrant Health",
        "uid": "qdrant-health",
        "schemaVersion": 36,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "panels": panels,
    }
    return dashboard

def write_json(obj: Dict[str, Any], path: Path):
    ensure_dir(path.parent)
    envelope = {"dashboard": obj, "overwrite": True}
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    print("Wrote dashboard:", path)

def write_provisioning(path: Path):
    ensure_dir(path.parent)
    prov = render_provisioning()
    path.write_text(yaml.safe_dump(prov, sort_keys=False), encoding="utf-8")
    print("Wrote provisioning:", path)

def main():
    ensure_dir(GRAFANA_DIR)
    write_json(retrieval_dashboard(), GRAFANA_DIR / "retrieval-slo.json")
    write_json(qdrant_dashboard(), GRAFANA_DIR / "qdrant-health.json")
    write_provisioning(GRAFANA_DIR / "grafana-provisioning.yaml")
    print("Grafana artifacts written to:", GRAFANA_DIR)

if __name__ == "__main__":
    main()
