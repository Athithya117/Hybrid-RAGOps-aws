#!/usr/bin/env python3
"""
dashboard.py

Generates Grafana provisioning files and two dashboards (Retrieval SLO, Qdrant Health)
under infra/manifests/grafana/. Dashboards assume a Prometheus datasource named by env GRAFANA_DATASOURCE_NAME.
They are suitable for Grafana provisioning (ConfigMap or dashboard sidecar).
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Any
import yaml

# Config
RENDER_DIR = Path(os.getenv("YAML_RENDER_DIR", "infra/manifests")).resolve()
GRAFANA_DIR = RENDER_DIR / "grafana"
DATASOURCE_NAME = os.getenv("GRAFANA_DATASOURCE_NAME", "Prometheus")
PROV_NAMESPACE = os.getenv("GRAFANA_PROV_NAMESPACE", os.getenv("MONITORING_NAMESPACE", "monitoring"))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# Minimal Grafana provisioning file (boards from ConfigMaps)
def render_grafana_provisioning() -> Dict[str,Any]:
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
                "options": {"path": "/var/lib/grafana/dashboards/rag"}
            }
        ]
    }

# Minimal retrieval SLO dashboard JSON (Grafana v9+ compatible minimal shape)
def render_retrieval_dashboard() -> Dict[str,Any]:
    title = "Retrieval SLO"
    ds = DATASOURCE_NAME
    panels = []
    # Panel: RPS
    panels.append({
        "type":"timeseries","title":"RPS","gridPos":{"h":6,"w":12,"x":0,"y":0},
        "targets":[{"expr":"sum(rate(retrieval_requests_total[1m]))","refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    # Panel: Error rate
    panels.append({
        "type":"timeseries","title":"Error rate (5xx)","gridPos":{"h":6,"w":12,"x":12,"y":0},
        "targets":[{"expr":"sum(rate(retrieval_requests_total{status_code=~\"5..\"}[5m])) / (sum(rate(retrieval_requests_total[5m])) + 1e-9)","refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    # Panel: latency p50/p95/p99
    panels.append({
        "type":"timeseries","title":"Latency p50/p95/p99 (s)","gridPos":{"h":8,"w":24,"x":0,"y":6},
        "targets":[
            {"expr":"histogram_quantile(0.50, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))","refId":"A","datasource":ds},
            {"expr":"histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))","refId":"B","datasource":ds},
            {"expr":"histogram_quantile(0.99, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))","refId":"C","datasource":ds},
        ],
        "fieldConfig":{"defaults":{}}
    })
    # Panel: service_ready
    panels.append({
        "type":"stat","title":"Service Ready","gridPos":{"h":4,"w":6,"x":0,"y":14},
        "targets":[{"expr":"service_ready{service=\"retrieval\"}","refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    dashboard = {
        "title": title,
        "uid": "retrieval-slo",
        "panels": panels,
        "schemaVersion": 36,
        "version": 1,
        "refresh": "10s",
        "time": {"from":"now-1h","to":"now"},
    }
    return dashboard

def render_qdrant_dashboard() -> Dict[str,Any]:
    title = "Qdrant Health"
    ds = DATASOURCE_NAME
    panels = []
    panels.append({
        "type":"timeseries","title":"Instance up","gridPos":{"h":4,"w":12,"x":0,"y":0},
        "targets":[{"expr":"up{job=~\"qdrant.*\"}","refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    panels.append({
        "type":"timeseries","title":"Dead replicas","gridPos":{"h":4,"w":12,"x":12,"y":0},
        "targets":[{"expr": f"{os.getenv('QDRANT_DEAD_REPLICAS_METRIC','cluster_dead_replicas')}", "refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    panels.append({
        "type":"timeseries","title":"Pending operations","gridPos":{"h":4,"w":24,"x":0,"y":4},
        "targets":[{"expr": f"{os.getenv('QDRANT_PENDING_OPS_METRIC','cluster_pending_operations_total')}", "refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    panels.append({
        "type":"timeseries","title":"Collection points (example)","gridPos":{"h":6,"w":24,"x":0,"y":8},
        "targets":[{"expr":"collection_points","refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    panels.append({
        "type":"timeseries","title":"PVC usage (qdrant namespace)","gridPos":{"h":4,"w":24,"x":0,"y":14},
        "targets":[{"expr":"kubelet_volume_stats_used_bytes{namespace=~\"qdrant\"} / kubelet_volume_stats_capacity_bytes{namespace=~\"qdrant\"}","refId":"A","datasource":ds}],
        "fieldConfig":{"defaults":{}}
    })
    dashboard = {
        "title": title,
        "uid": "qdrant-health",
        "panels": panels,
        "schemaVersion": 36,
        "version": 1,
        "refresh": "30s",
        "time": {"from":"now-6h","to":"now"},
    }
    return dashboard

def write_dashboard_json(d: Dict[str,Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Wrap into Grafana dashboard envelope
    envelope = {"dashboard": d, "overwrite": True}
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    print("Wrote dashboard:", path)

def write_provisioning_yaml(path: Path):
    prov = {
        "apiVersion": 1,
        "providers": [
            {
                "name":"RAG Dashboards",
                "orgId": 1,
                "folder":"RAG",
                "type":"file",
                "options":{"path": "/var/lib/grafana/dashboards/rag"}
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(prov, sort_keys=False), encoding="utf-8")
    print("Wrote grafana provisioning:", path)

def main():
    ensure_dir(GRAFANA_DIR)
    # dashboards
    ds1 = render_retrieval_dashboard()
    ds2 = render_qdrant_dashboard()
    write_dashboard_json(ds1, GRAFANA_DIR / "retrieval-slo.json")
    write_dashboard_json(ds2, GRAFANA_DIR / "qdrant-health.json")
    # provisioning
    write_provisioning_yaml(GRAFANA_DIR / "grafana-provisioning.yaml")
    # optional: info for operator
    print("Grafana artifacts written to:", GRAFANA_DIR)
    print("Ensure Grafana sidecar reads path:", "/var/lib/grafana/dashboards/rag (see provisioning)")

if __name__ == "__main__":
    main()
