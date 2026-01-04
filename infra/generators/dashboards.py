#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import yaml

GRAFANA_IMAGE = os.getenv("GRAFANA_IMAGE", "grafana/grafana:10.3.5")
POSTGRES_IMAGE = os.getenv("POSTGRES_IMAGE", "postgres:15.4")
GRAFANA_NAMESPACE = os.getenv("GRAFANA_NAMESPACE", "monitoring")
GRAFANA_REPLICAS = os.getenv("GRAFANA_REPLICAS", "1")
GRAFANA_USE_PVC = os.getenv("GRAFANA_USE_PVC", "false")
GRAFANA_PVC_SIZE = os.getenv("GRAFANA_PVC_SIZE", "5Gi")
GRAFANA_CPU_REQ = os.getenv("GRAFANA_CPU_REQ", "100m")
GRAFANA_MEM_REQ = os.getenv("GRAFANA_MEM_REQ", "128Mi")
GRAFANA_CPU_LIMIT = os.getenv("GRAFANA_CPU_LIMIT", "500m")
GRAFANA_MEM_LIMIT = os.getenv("GRAFANA_MEM_LIMIT", "512Mi")
GRAFANA_PROVISIONING_NAMESPACE = os.getenv("GRAFANA_PROVISIONING_NAMESPACE", GRAFANA_NAMESPACE)
GRAFANA_DASHBOARD_UID_PREFIX = os.getenv("GRAFANA_DASHBOARD_UID_PREFIX", "platform-")
DASHBOARD_SERVICES = os.getenv("DASHBOARD_SERVICES", "retriever,qdrant")
MAX_PANELS_PER_DASHBOARD = os.getenv("MAX_PANELS_PER_DASHBOARD", "48")
METRICS_DATASOURCE = os.getenv("METRICS_DATASOURCE", "VictoriaMetrics")
METRICS_DATASOURCE_URL = os.getenv("METRICS_DATASOURCE_URL", "http://victoria-metrics.monitoring.svc:8428")
CLICKHOUSE_DATASOURCE = os.getenv("CLICKHOUSE_DATASOURCE", "ClickHouse")
CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL", "http://clickhouse.clickhouse.svc:8123")
DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "monitoring")
SLO_SUCCESS_TARGET = os.getenv("SLO_SUCCESS_TARGET", "0.999")
SLO_LATENCY_QUANTILE = os.getenv("SLO_LATENCY_QUANTILE", "0.95")
DATASOURCE_URL = os.getenv("DATASOURCE_URL", METRICS_DATASOURCE_URL)
CI = os.getenv("CI", "false")
GRAFANA_MANAGE_DEPLOYMENT = os.getenv("GRAFANA_MANAGE_DEPLOYMENT", "true")
GRAFANA_ADMIN_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_ADMIN_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "")
GRAFANA_EXTERNAL_DB = os.getenv("GRAFANA_EXTERNAL_DB", "false")
GRAFANA_POSTGRES_USER = os.getenv("GRAFANA_POSTGRES_USER", "grafana")
GRAFANA_POSTGRES_DB = os.getenv("GRAFANA_POSTGRES_DB", "grafana")
GRAFANA_POSTGRES_PASSWORD = os.getenv("GRAFANA_POSTGRES_PASSWORD", "GfN9m2z!7xQpL3sV@8bR4tY1uE0kH6")
GRAFANA_POSTGRES_PVC_SIZE = os.getenv("GRAFANA_POSTGRES_PVC_SIZE", "5Gi")

LOG = logging.getLogger("dashboards_generator")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
LOG.handlers = []
LOG.addHandler(ch)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "infra" / "manifests" / "dashboards"

CLICKHOUSE_SQL_TEMPLATE = (
    "SELECT ts, level, message, fields FROM logs.kube_logs "
    "WHERE service = '$service' "
    "AND namespace = '$namespace' "
    "AND ts BETWEEN toDateTime64($__from / 1000, 3) AND toDateTime64($__to / 1000, 3) "
    "ORDER BY ts DESC LIMIT 500"
)

def run_cmd(cmd: List[str], timeout: int = 60, stdin: Optional[str] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              input=stdin.encode("utf-8") if stdin else None, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        LOG.debug("run_cmd finished: %s rc=%d out_len=%d err_len=%d", " ".join(cmd), proc.returncode, len(out), len(err))
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        out = (getattr(e, "stdout", None) or b"").decode("utf-8", errors="replace") if getattr(e, "stdout", None) else ""
        err = (getattr(e, "stderr", None) or b"").decode("utf-8", errors="replace") if getattr(e, "stderr", None) else f"timeout after {timeout}s"
        LOG.error("run_cmd timeout: %s", " ".join(cmd))
        return 124, out.strip(), err.strip()

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, str(path))
    LOG.info("wrote file %s (%d bytes)", str(path), len(content))

def coerce_bool(v: Optional[str]) -> bool:
    if not v:
        return False
    return v.strip().lower() in ("1","true","yes","on")

def safe_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def load_env() -> Dict[str, str]:
    env: Dict[str, str] = {
        "GRAFANA_IMAGE": GRAFANA_IMAGE,
        "POSTGRES_IMAGE": POSTGRES_IMAGE,
        "GRAFANA_NAMESPACE": GRAFANA_NAMESPACE,
        "GRAFANA_REPLICAS": GRAFANA_REPLICAS,
        "GRAFANA_USE_PVC": GRAFANA_USE_PVC,
        "GRAFANA_PVC_SIZE": GRAFANA_PVC_SIZE,
        "GRAFANA_CPU_REQ": GRAFANA_CPU_REQ,
        "GRAFANA_MEM_REQ": GRAFANA_MEM_REQ,
        "GRAFANA_CPU_LIMIT": GRAFANA_CPU_LIMIT,
        "GRAFANA_MEM_LIMIT": GRAFANA_MEM_LIMIT,
        "GRAFANA_PROVISIONING_NAMESPACE": GRAFANA_PROVISIONING_NAMESPACE,
        "GRAFANA_DASHBOARD_UID_PREFIX": GRAFANA_DASHBOARD_UID_PREFIX,
        "DASHBOARD_SERVICES": DASHBOARD_SERVICES,
        "MAX_PANELS_PER_DASHBOARD": MAX_PANELS_PER_DASHBOARD,
        "METRICS_DATASOURCE": METRICS_DATASOURCE,
        "METRICS_DATASOURCE_URL": METRICS_DATASOURCE_URL,
        "CLICKHOUSE_DATASOURCE": CLICKHOUSE_DATASOURCE,
        "CLICKHOUSE_URL": CLICKHOUSE_URL,
        "DEFAULT_NAMESPACE": DEFAULT_NAMESPACE,
        "SLO_SUCCESS_TARGET": SLO_SUCCESS_TARGET,
        "SLO_LATENCY_QUANTILE": SLO_LATENCY_QUANTILE,
        "DATASOURCE_URL": DATASOURCE_URL,
        "CI": CI,
        "GRAFANA_MANAGE_DEPLOYMENT": GRAFANA_MANAGE_DEPLOYMENT,
        "GRAFANA_ADMIN_USER": GRAFANA_ADMIN_USER,
        "GRAFANA_ADMIN_PASSWORD": GRAFANA_ADMIN_PASSWORD,
        "GRAFANA_EXTERNAL_DB": GRAFANA_EXTERNAL_DB,
        "GRAFANA_POSTGRES_USER": GRAFANA_POSTGRES_USER,
        "GRAFANA_POSTGRES_DB": GRAFANA_POSTGRES_DB,
        "GRAFANA_POSTGRES_PASSWORD": GRAFANA_POSTGRES_PASSWORD,
        "GRAFANA_POSTGRES_PVC_SIZE": GRAFANA_POSTGRES_PVC_SIZE,
    }
    for k in list(env.keys()):
        v = os.getenv(k)
        if v is not None:
            env[k] = v
            LOG.info("env overridden by runtime: %s", k)
        else:
            LOG.info("env default used: %s", k)
    return env

def validate_env(env: Dict[str, str]) -> None:
    try:
        sst = float(env.get("SLO_SUCCESS_TARGET", "0.999"))
        if not (0.0 < sst < 1.0):
            raise ValueError()
    except Exception:
        LOG.error("invalid SLO_SUCCESS_TARGET: %s", env.get("SLO_SUCCESS_TARGET"))
        raise RuntimeError("SLO_SUCCESS_TARGET must be a float between 0 and 1, e.g. 0.999")
    if env.get("SLO_LATENCY_QUANTILE", "0.95") not in ("0.95", "0.99"):
        LOG.error("invalid SLO_LATENCY_QUANTILE: %s", env.get("SLO_LATENCY_QUANTILE"))
        raise RuntimeError("SLO_LATENCY_QUANTILE must be '0.95' or '0.99'")

def validate_db_config(env: Dict[str, str]) -> None:
    external = coerce_bool(env.get("GRAFANA_EXTERNAL_DB", "false"))
    if external:
        if not (os.getenv("GF_DATABASE_URL") or (os.getenv("GF_DATABASE_HOST") and os.getenv("GF_DATABASE_NAME") and os.getenv("GF_DATABASE_USER") and os.getenv("GF_DATABASE_PASSWORD"))):
            LOG.error("external DB requested but GF_DATABASE_* or GF_DATABASE_URL not provided")
            raise RuntimeError("GRAFANA_EXTERNAL_DB=true requires GF_DATABASE_URL or GF_DATABASE_HOST/NAME/USER/PASSWORD environment variables to be set")
    else:
        if not env.get("GRAFANA_POSTGRES_PASSWORD"):
            LOG.error("no postgres password provided for in-cluster Postgres")
            raise RuntimeError("GRAFANA_POSTGRES_PASSWORD must be set for in-cluster Postgres")

def render_clickhouse_left(service: str, env: Dict[str, str]) -> str:
    template = os.getenv("CLICKHOUSE_SQL_TEMPLATE", CLICKHOUSE_SQL_TEMPLATE)
    left_obj = {"datasource": env["CLICKHOUSE_DATASOURCE"], "queries": [{"refId": "A", "sql": template.replace("$service", service).replace("$namespace", env.get("DEFAULT_NAMESPACE","monitoring"))}], "range": {"from": "$__from", "to": "$__to"}}
    raw = json.dumps(left_obj, separators=(",", ":"), ensure_ascii=False)
    return urllib.parse.quote_plus(raw)

def make_metric_panel(title: str, expr: str, datasource: str, gridPos: Dict[str, int], refId: str, panel_type: str = "timeseries") -> Dict[str, Any]:
    return {"type": panel_type, "title": title, "datasource": datasource, "targets": [{"expr": expr, "refId": refId}], "gridPos": gridPos}

def make_stat_panel(title: str, expr: str, datasource: str, gridPos: Dict[str, int], refId: str) -> Dict[str, Any]:
    return make_metric_panel(title, expr, datasource, gridPos, refId, panel_type="stat")

def build_service_dashboard(service: str, env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    slo_q = env.get("SLO_LATENCY_QUANTILE","0.95")
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1
    hdr = {
        "type": "text",
        "title": "Header",
        "gridPos": {"h": 2, "w": 24, "x": 0, "y": 0},
        "options": {"content": f"Service: {service}    SLO target: {env.get('SLO_SUCCESS_TARGET','0.999')}    quantile: {slo_q}"}
    }
    hdr["id"] = next_panel_id; next_panel_id += 1
    panels.append(hdr)
    if service == "retriever":
        p1 = make_metric_panel("P95 Latency", f"histogram_quantile({slo_q}, sum(rate(retrieval_request_duration_seconds_bucket{{service=~\"{service}\"}}[5m])) by (le))", metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p1["id"] = next_panel_id; next_panel_id += 1
        p2 = make_metric_panel("Error rate", f"sum(rate(retrieval_errors_total{{service=~\"{service}\"}}[5m])) / max(sum(rate(retrieval_requests_total{{service=~\"{service}\"}}[5m])),1)", metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p2["id"] = next_panel_id; next_panel_id += 1
        p3 = make_stat_panel("Service readiness", f"avg(service_ready{{service=~\"{service}\"}})", metrics_ds, {"h": 3, "w": 12, "x": 0, "y": 10}, chr(next_ref_ord)); next_ref_ord += 1
        p3["id"] = next_panel_id; next_panel_id += 1
        p4 = make_stat_panel("Requests/s", f"sum(rate(retrieval_requests_total{{service=~\"{service}\"}}[1m]))", metrics_ds, {"h": 3, "w": 12, "x": 12, "y": 10}, chr(next_ref_ord)); next_ref_ord += 1
        p4["id"] = next_panel_id; next_panel_id += 1
        panels.extend([p1, p2, p3, p4])
    elif service == "qdrant":
        p1 = make_metric_panel("Qdrant P95 Latency", f"histogram_quantile({slo_q}, sum(rate(qdrant_query_duration_seconds_bucket{{}}[5m])) by (le))", metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p1["id"] = next_panel_id; next_panel_id += 1
        p2 = make_metric_panel("Qdrant Queries/s", "sum(rate(qdrant_query_total[1m]))", metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p2["id"] = next_panel_id; next_panel_id += 1
        p3 = make_stat_panel("Collections total", "collections_total", metrics_ds, {"h": 3, "w": 12, "x": 0, "y": 10}, chr(next_ref_ord)); next_ref_ord += 1
        p3["id"] = next_panel_id; next_panel_id += 1
        p4 = make_stat_panel("Total vectors across collections", "sum(collections_vector_total)", metrics_ds, {"h": 3, "w": 12, "x": 12, "y": 10}, chr(next_ref_ord)); next_ref_ord += 1
        p4["id"] = next_panel_id; next_panel_id += 1
        panels.extend([p1, p2, p3, p4])
    else:
        p1 = make_metric_panel("P95 Latency (fallback)", f"histogram_quantile({slo_q}, sum(rate({service}_request_duration_seconds_bucket[5m])) by (le))", metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p1["id"] = next_panel_id; next_panel_id += 1
        p2 = make_metric_panel("Error rate (fallback)", f"sum(rate({service}_errors_total[5m])) / max(sum(rate({service}_requests_total[5m])),1)", metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p2["id"] = next_panel_id; next_panel_id += 1
        panels.extend([p1, p2])
    left_enc = render_clickhouse_left(service, env)
    logs_panel = {
        "type": "text",
        "title": "Logs",
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": 14},
        "options": {"content": "Open logs for this service using Explore"},
        "links": [{"title": "Open Logs", "url": f"/explore?left={left_enc}"}]
    }
    logs_panel["id"] = next_panel_id; next_panel_id += 1
    panels.append(logs_panel)
    mp = safe_int(env.get("MAX_PANELS_PER_DASHBOARD", "48"), 48)
    if len(panels) > mp:
        raise RuntimeError(f"dashboard for {service} would exceed MAX_PANELS_PER_DASHBOARD ({len(panels)} > {mp})")
    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}{service}"
    vars_list = [
        {"type": "custom", "name": "service", "options": [{"text": service, "value": service}], "current": {"text": service, "value": service}, "multi": False}
    ]
    vars_list.append({"type": "custom", "name": "namespace", "options": [{"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}], "current": {"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}, "multi": False})
    dash = {"id": None, "uid": uid, "title": f"Service Overview — {service}", "templating": {"list": vars_list}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_ingestion_dashboard(env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1
    p1 = make_metric_panel("vmagent discovery objects (pods)", 'vm_promscrape_discovery_kubernetes_objects{role="pod"}', metrics_ds, {"h": 6, "w": 24, "x": 0, "y": 0}, chr(next_ref_ord)); next_ref_ord += 1
    p1["id"] = next_panel_id; next_panel_id += 1
    p2 = make_metric_panel("vmagent remote-write bytes (increase 5m)", "increase(vmagent_remotewrite_sent_bytes_total[5m])", metrics_ds, {"h": 6, "w": 24, "x": 0, "y": 6}, chr(next_ref_ord)); next_ref_ord += 1
    p2["id"] = next_panel_id; next_panel_id += 1
    p3 = make_metric_panel("vmagent series fetched (last)", "vm_promscrape_series_fetched", metrics_ds, {"h": 6, "w": 24, "x": 0, "y": 12}, chr(next_ref_ord)); next_ref_ord += 1
    p3["id"] = next_panel_id; next_panel_id += 1
    panels.extend([p1, p2, p3])
    left_enc = render_clickhouse_left("vmagent", env)
    link_panel = {
        "type": "text",
        "title": "Logs",
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": 18},
        "options": {"content": "Open vmagent logs via Explore"},
        "links": [{"title": "Open Logs", "url": f"/explore?left={left_enc}"}]
    }
    link_panel["id"] = next_panel_id
    panels.append(link_panel)
    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}ingestion"
    dash = {"id": None, "uid": uid, "title": "Ingestion Health", "templating": {"list": [
        {"type": "custom", "name": "namespace", "options": [{"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}], "current": {"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}}]}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_platform_overview(env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1
    p1 = make_metric_panel("Total discovery (pods)", 'max(vm_promscrape_discovery_kubernetes_objects{role="pod"})', metrics_ds, {"h": 4, "w": 24, "x": 0, "y": 0}, chr(next_ref_ord)); next_ref_ord += 1
    p1["id"] = next_panel_id; next_panel_id += 1
    p2 = make_metric_panel("Services not ready (count)", 'count(service_ready==0)', metrics_ds, {"h": 4, "w": 12, "x": 0, "y": 4}, chr(next_ref_ord)); next_ref_ord += 1
    p2["id"] = next_panel_id; next_panel_id += 1
    p3 = make_metric_panel("vmagent scrape pool targets (up/down)", 'vm_promscrape_scrape_pool_targets', metrics_ds, {"h": 4, "w": 12, "x": 12, "y": 4}, chr(next_ref_ord)); next_ref_ord += 1
    p3["id"] = next_panel_id; next_panel_id += 1
    panels.extend([p1, p2, p3])
    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}platform-overview"
    dash = {"id": None, "uid": uid, "title": "Platform Overview", "templating": {"list": []}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_dashboards_configmap(dashboards: Dict[str, Dict[str, Any]], env: Dict[str, str]) -> Dict[str, Any]:
    data: Dict[str, str] = {}
    for name, db in dashboards.items():
        key = f"{name}.json"
        data[key] = json.dumps(db, separators=(",", ":"), ensure_ascii=False)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-dashboards", "namespace": ns, "labels": {"managed-by": "dashboards.py"}}, "data": data}

def build_provisioning_provider_cm(env: Dict[str, str]) -> Dict[str, Any]:
    provider_item = {
        "name": "platform-dashboards",
        "orgId": 1,
        "folder": "Platform",
        "type": "file",
        "disableDeletion": False,
        "options": {"path": "/var/lib/grafana/dashboards"}
    }
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    providers_yaml = yaml.safe_dump([provider_item], sort_keys=False)
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-provisioning", "namespace": ns, "labels": {"managed-by": "dashboards.py"}}, "data": {"providers.yaml": providers_yaml}}
    return cm

def build_datasources_cm(env: Dict[str, str]) -> Dict[str, Any]:
    ds: List[Dict[str, Any]] = []
    ds.append({
        "name": env["METRICS_DATASOURCE"],
        "type": "prometheus",
        "access": "proxy",
        "url": env["METRICS_DATASOURCE_URL"],
        "isDefault": True,
        "editable": False
    })
    if env.get("CLICKHOUSE_URL"):
        ds.append({
            "name": env["CLICKHOUSE_DATASOURCE"],
            "type": "clickhouse",
            "access": "proxy",
            "url": env["CLICKHOUSE_URL"],
            "isDefault": False,
            "editable": False
        })
    provider = {"apiVersion": 1, "datasources": ds}
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-datasources", "namespace": ns, "labels": {"managed-by": "dashboards.py"}}, "data": {"datasources.yaml": yaml.safe_dump(provider, sort_keys=False)}}

def build_postgres_secret(env: Dict[str, str]) -> Dict[str, Any]:
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    s = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "grafana-postgres-secret", "namespace": ns, "labels": {"managed-by": "dashboards.py"}},
        "stringData": {
            "postgres-user": env.get("GRAFANA_POSTGRES_USER", "grafana"),
            "postgres-password": env.get("GRAFANA_POSTGRES_PASSWORD"),
            "postgres-db": env.get("GRAFANA_POSTGRES_DB", "grafana")
        }
    }
    return s

def build_postgres_service(env: Dict[str, str]) -> Dict[str, Any]:
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "grafana-postgres", "namespace": ns, "labels": {"app": "grafana-postgres"}},
        "spec": {"ports": [{"port": 5432, "name": "postgres", "protocol": "TCP"}], "selector": {"app": "grafana-postgres"}, "clusterIP": "None"}
    }
    return svc

def build_postgres_statefulset(env: Dict[str, str]) -> Dict[str, Any]:
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    image = env.get("POSTGRES_IMAGE", "postgres:15.4")
    user = env.get("GRAFANA_POSTGRES_USER", "grafana")
    db = env.get("GRAFANA_POSTGRES_DB", "grafana")
    pvc_size = env.get("GRAFANA_POSTGRES_PVC_SIZE", "5Gi")
    sts = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": "grafana-postgres", "namespace": ns, "labels": {"app": "grafana-postgres"}},
        "spec": {
            "serviceName": "grafana-postgres",
            "replicas": 1,
            "selector": {"matchLabels": {"app": "grafana-postgres"}},
            "template": {
                "metadata": {"labels": {"app": "grafana-postgres"}},
                "spec": {
                    "containers": [
                        {
                            "name": "postgres",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "env": [
                                {"name": "POSTGRES_USER", "valueFrom": {"secretKeyRef": {"name": "grafana-postgres-secret", "key": "postgres-user"}}},
                                {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "grafana-postgres-secret", "key": "postgres-password"}}},
                                {"name": "POSTGRES_DB", "valueFrom": {"secretKeyRef": {"name": "grafana-postgres-secret", "key": "postgres-db"}}},
                                {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"}
                            ],
                            "ports": [{"containerPort": 5432, "name": "postgres"}],
                            "readinessProbe": {"exec": {"command": ["pg_isready", "-U", user]}, "initialDelaySeconds": 5, "periodSeconds": 5, "timeoutSeconds": 3},
                            "resources": {"requests": {"cpu": env.get("GRAFANA_CPU_REQ"), "memory": env.get("GRAFANA_MEM_REQ")}, "limits": {"cpu": env.get("GRAFANA_CPU_LIMIT"), "memory": env.get("GRAFANA_MEM_LIMIT")}},
                            "volumeMounts": [{"name": "pgdata", "mountPath": "/var/lib/postgresql/data"}]
                        }
                    ]
                }
            },
            "volumeClaimTemplates": [
                {"metadata": {"name": "pgdata", "labels": {"app": "grafana-postgres"}}, "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": pvc_size}}}}
            ]
        }
    }
    return sts

def build_grafana_deployment(env: Dict[str, str]) -> Dict[str, Any]:
    ns = env["GRAFANA_NAMESPACE"]
    image = env["GRAFANA_IMAGE"]
    replicas = safe_int(env.get("GRAFANA_REPLICAS","1"),1)
    use_pvc = coerce_bool(env.get("GRAFANA_USE_PVC","false"))
    volumes = [
        {"name":"dashboards","configMap":{"name":"grafana-dashboards"}},
        {"name":"provisioning","configMap":{"name":"grafana-provisioning"}},
        {"name":"datasources","configMap":{"name":"grafana-datasources"}}
    ]
    volume_mounts = [
        {"name":"dashboards","mountPath":"/var/lib/grafana/dashboards"},
        {"name":"provisioning","mountPath":"/etc/grafana/provisioning/dashboards"},
        {"name":"datasources","mountPath":"/etc/grafana/provisioning/datasources"}
    ]
    if use_pvc:
        pvc_name = "grafana-data"
        volumes.append({"name":"grafana-data","persistentVolumeClaim":{"claimName":pvc_name}})
        volume_mounts.append({"name":"grafana-data","mountPath":"/var/lib/grafana"})
    env_vars = [
        {"name":"GF_DATABASE_TYPE","value":"postgres"},
        {"name":"GF_DATABASE_HOST","value":"grafana-postgres.%s.svc.cluster.local:5432" % env.get("GRAFANA_PROVISIONING_NAMESPACE")},
        {"name":"GF_DATABASE_NAME","valueFrom":{"secretKeyRef":{"name":"grafana-postgres-secret","key":"postgres-db"}}},
        {"name":"GF_DATABASE_USER","valueFrom":{"secretKeyRef":{"name":"grafana-postgres-secret","key":"postgres-user"}}},
        {"name":"GF_DATABASE_PASSWORD","valueFrom":{"secretKeyRef":{"name":"grafana-postgres-secret","key":"postgres-password"}}},
        {"name":"GF_SECURITY_ADMIN_USER","value":env.get("GRAFANA_ADMIN_USER","admin")}
    ]
    if env.get("GRAFANA_ADMIN_PASSWORD"):
        env_vars.append({"name":"GF_SECURITY_ADMIN_PASSWORD","valueFrom":{"secretKeyRef":{"name":"grafana-admin-secret","key":"admin-password"}}})
    container = {
        "name":"grafana",
        "image":image,
        "env":env_vars,
        "ports":[{"containerPort":3000,"name":"http"}],
        "volumeMounts":volume_mounts,
        "resources":{
            "requests":{"cpu":env.get("GRAFANA_CPU_REQ"),"memory":env.get("GRAFANA_MEM_REQ")},
            "limits":{"cpu":env.get("GRAFANA_CPU_LIMIT"),"memory":env.get("GRAFANA_MEM_LIMIT")}
        },
        "readinessProbe":{"httpGet":{"path":"/api/health","port":3000},"initialDelaySeconds":5,"periodSeconds":10,"timeoutSeconds":3},
        "livenessProbe":{"httpGet":{"path":"/api/health","port":3000},"initialDelaySeconds":30,"periodSeconds":20,"timeoutSeconds":5}
    }
    pod_spec = {"containers":[container], "volumes":volumes}
    deployment = {
        "apiVersion":"apps/v1",
        "kind":"Deployment",
        "metadata":{"name":"grafana","namespace":ns,"labels":{"app":"grafana","managed-by":"dashboards.py"}},
        "spec":{
            "replicas":replicas,
            "selector":{"matchLabels":{"app":"grafana","managed-by":"dashboards.py"}},
            "template":{"metadata":{"labels":{"app":"grafana","managed-by":"dashboards.py"}},"spec":pod_spec}
        }
    }
    return deployment

def build_grafana_service(env: Dict[str, str]) -> Dict[str, Any]:
    ns = env["GRAFANA_NAMESPACE"]
    svc = {
        "apiVersion":"v1",
        "kind":"Service",
        "metadata":{"name":"grafana","namespace":ns,"labels":{"app":"grafana","managed-by":"dashboards.py"}},
        "spec":{"selector":{"app":"grafana","managed-by":"dashboards.py"},"ports":[{"port":3000,"targetPort":3000,"protocol":"TCP","name":"http"}],"type":"ClusterIP"}
    }
    return svc

def render_all(env: Dict[str, str]) -> Dict[str, Path]:
    validate_env(env)
    validate_db_config(env)
    services = [s.strip() for s in (env.get("DASHBOARD_SERVICES","") or "").split(",") if s.strip()]
    if not services:
        services = ["retriever","qdrant"]
    rendered: Dict[str, Dict[str, Any]] = {}
    for svc in services:
        db = build_service_dashboard(svc, env)
        rendered[f"service-{svc}"] = db
    rendered["ingestion-health"] = build_ingestion_dashboard(env)
    rendered["platform-overview"] = build_platform_overview(env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, Path] = {}
    for name, db in rendered.items():
        p = OUT_DIR / f"{name}.json"
        atomic_write(p, json.dumps(db, indent=2, ensure_ascii=False))
        out_paths[name] = p
    dashboards_cm = build_dashboards_configmap(rendered, env)
    prov_cm = build_provisioning_provider_cm(env)
    datasources_cm = build_datasources_cm(env)
    atomic_write(OUT_DIR / "grafana-dashboards-configmap.yaml", yaml.safe_dump(dashboards_cm, sort_keys=False))
    atomic_write(OUT_DIR / "grafana-provisioning-configmap.yaml", yaml.safe_dump(prov_cm, sort_keys=False))
    atomic_write(OUT_DIR / "grafana-datasources-configmap.yaml", yaml.safe_dump(datasources_cm, sort_keys=False))
    deployment = build_grafana_deployment(env)
    svc = build_grafana_service(env)
    atomic_write(OUT_DIR / "grafana-deployment.yaml", yaml.safe_dump(deployment, sort_keys=False))
    atomic_write(OUT_DIR / "grafana-service.yaml", yaml.safe_dump(svc, sort_keys=False))
    if not coerce_bool(env.get("GRAFANA_EXTERNAL_DB","false")):
        pg_secret = build_postgres_secret(env)
        pg_svc = build_postgres_service(env)
        pg_sts = build_postgres_statefulset(env)
        atomic_write(OUT_DIR / "grafana-postgres-secret.yaml", yaml.safe_dump(pg_secret, sort_keys=False))
        atomic_write(OUT_DIR / "grafana-postgres-service.yaml", yaml.safe_dump(pg_svc, sort_keys=False))
        atomic_write(OUT_DIR / "grafana-postgres-statefulset.yaml", yaml.safe_dump(pg_sts, sort_keys=False))
        out_paths["postgres_secret"] = OUT_DIR / "grafana-postgres-secret.yaml"
        out_paths["postgres_service"] = OUT_DIR / "grafana-postgres-service.yaml"
        out_paths["postgres_sts"] = OUT_DIR / "grafana-postgres-statefulset.yaml"
    if coerce_bool(env.get("GRAFANA_USE_PVC","false")):
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "grafana-data", "namespace": env["GRAFANA_NAMESPACE"], "labels": {"app": "grafana", "managed-by": "dashboards.py"}},
            "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": env.get("GRAFANA_PVC_SIZE","5Gi")}}}
        }
        atomic_write(OUT_DIR / "grafana-pvc.yaml", yaml.safe_dump(pvc, sort_keys=False))
        out_paths["pvc"] = OUT_DIR / "grafana-pvc.yaml"
    atomic_write(OUT_DIR / "clickhouse-explore-sql.txt", CLICKHOUSE_SQL_TEMPLATE)
    out_paths["dashboards_cm"] = OUT_DIR / "grafana-dashboards-configmap.yaml"
    out_paths["provisioning_cm"] = OUT_DIR / "grafana-provisioning-configmap.yaml"
    out_paths["datasources_cm"] = OUT_DIR / "grafana-datasources-configmap.yaml"
    out_paths["deployment"] = OUT_DIR / "grafana-deployment.yaml"
    out_paths["service"] = OUT_DIR / "grafana-service.yaml"
    LOG.info("render complete -> %s", str(OUT_DIR))
    return out_paths

def kubectl_apply_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to apply manifests")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", str(path)], timeout=60)
    if rc != 0:
        LOG.error("kubectl apply failed for %s: %s", str(path), err or out)
        raise RuntimeError(f"kubectl apply failed for {path}: {err or out}")
    LOG.info("kubectl apply succeeded: %s", str(path))

def kubectl_delete_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        LOG.warning("kubectl not present; skipping kubectl delete for %s", str(path))
        return
    rc, out, err = run_cmd(["kubectl", "delete", "-f", str(path), "--ignore-not-found"], timeout=60)
    if rc != 0:
        LOG.warning("kubectl delete returned non-zero for %s", str(path))
    else:
        LOG.info("kubectl delete succeeded: %s", str(path))

def ensure_namespace(ns: str, timeout: int = 30) -> None:
    if not shutil.which("kubectl"):
        LOG.info("kubectl not found; cannot ensure namespace %s", ns)
        return
    rc, out, err = run_cmd(["kubectl", "create", "namespace", ns, "--dry-run=client", "-o", "yaml"], timeout=10)
    if rc != 0:
        LOG.error("namespace dry-run failed for %s: %s", ns, err or out)
        raise RuntimeError(f"failed to prepare namespace {ns}: {err or out}")
    proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        LOG.error("failed to ensure namespace %s: %s", ns, stderr)
        raise RuntimeError(f"failed to create/apply namespace {ns}: {stderr}")
    start = time.time()
    while time.time() - start < timeout:
        rc2, out2, err2 = run_cmd(["kubectl", "get", "ns", ns, "-o", "json"], timeout=10)
        if rc2 == 0 and out2:
            try:
                j = json.loads(out2)
                phase = j.get("status", {}).get("phase", "")
                if phase == "Active":
                    LOG.info("namespace present %s phase=%s", ns, phase)
                    return
            except Exception:
                pass
        time.sleep(1)
    raise RuntimeError(f"namespace {ns} did not reach Active state within {timeout}s")

def create_or_update_secret_from_env(ns: str, secret_name: str, mapping: Dict[str, str]) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to create secrets")
    args = ["kubectl", "create", "secret", "generic", secret_name]
    for key, val in mapping.items():
        args.append(f"--from-literal={key}={val}")
    args.extend(["--dry-run=client", "-o", "yaml", "-n", ns])
    rc, out, err = run_cmd(args, timeout=20)
    if rc != 0:
        LOG.error("failed to build secret YAML for %s: %s", secret_name, err or out)
        raise RuntimeError(f"failed to create secret manifest: {err or out}")
    proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        LOG.error("failed to apply secret %s: %s", secret_name, stderr)
        raise RuntimeError(f"failed to apply secret {secret_name}: {stderr}")
    rc2, out2, err2 = run_cmd(["kubectl", "label", "secret", secret_name, "managed-by=dashboards.py", "-n", ns, "--overwrite"], timeout=10)
    if rc2 != 0:
        LOG.warning("failed to label secret %s", secret_name)

def apply_action() -> None:
    LOG.info("apply started")
    env = load_env()
    validate_env(env)
    validate_db_config(env)
    paths = render_all(env)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    LOG.info("ensuring namespace exists (create/apply) %s", ns)
    ensure_namespace(ns)
    if env.get("GRAFANA_ADMIN_USER") and env.get("GRAFANA_ADMIN_PASSWORD"):
        LOG.info("creating/updating grafana-admin-secret from environment in %s", ns)
        create_or_update_secret_from_env(ns, "grafana-admin-secret", {"admin-user": env["GRAFANA_ADMIN_USER"], "admin-password": env["GRAFANA_ADMIN_PASSWORD"]})
    else:
        LOG.info("no grafana admin credentials provided; skipping secret creation")
    if not coerce_bool(env.get("GRAFANA_EXTERNAL_DB","false")):
        LOG.info("creating/updating grafana-postgres-secret from environment in %s", ns)
        create_or_update_secret_from_env(ns, "grafana-postgres-secret", {"postgres-user": env.get("GRAFANA_POSTGRES_USER","grafana"), "postgres-password": env.get("GRAFANA_POSTGRES_PASSWORD"), "postgres-db": env.get("GRAFANA_POSTGRES_DB","grafana")})
        if paths.get("postgres_service"):
            LOG.info("applying postgres service %s", str(paths['postgres_service']))
            kubectl_apply_yaml(paths["postgres_service"])
        if paths.get("postgres_sts"):
            LOG.info("applying postgres statefulset %s", str(paths['postgres_sts']))
            kubectl_apply_yaml(paths["postgres_sts"])
            LOG.info("waiting for postgres statefulset rollout (best-effort, timeout=180s)")
            rc, out, err = run_cmd(["kubectl", "-n", ns, "rollout", "status", "statefulset/grafana-postgres", "--timeout=180s"], timeout=185)
            if rc != 0:
                LOG.warning("postgres rollout status check failed or timed out")
    if paths.get("datasources_cm"):
        LOG.info("applying datasources configmap %s", str(paths['datasources_cm']))
        kubectl_apply_yaml(paths["datasources_cm"])
    if paths.get("provisioning_cm"):
        LOG.info("applying provisioning provider configmap %s", str(paths['provisioning_cm']))
        kubectl_apply_yaml(paths["provisioning_cm"])
    if paths.get("dashboards_cm"):
        LOG.info("applying dashboards configmap %s", str(paths['dashboards_cm']))
        kubectl_apply_yaml(paths["dashboards_cm"])
    svc = OUT_DIR / "grafana-service.yaml"
    if svc.exists():
        LOG.info("applying service %s", str(svc))
        kubectl_apply_yaml(svc)
    if coerce_bool(env.get("GRAFANA_USE_PVC","false")):
        pvc_path = OUT_DIR / "grafana-pvc.yaml"
        if pvc_path.exists():
            LOG.info("applying pvc %s", str(pvc_path))
            kubectl_apply_yaml(pvc_path)
    if coerce_bool(env.get("GRAFANA_MANAGE_DEPLOYMENT","false")):
        dep = OUT_DIR / "grafana-deployment.yaml"
        if dep.exists():
            LOG.info("safely applying deployment %s", str(dep))
            kubectl_apply_yaml(dep)
            LOG.info("waiting for grafana deployment rollout (best-effort, timeout=180s)")
            rc, out, err = run_cmd(["kubectl", "-n", env["GRAFANA_NAMESPACE"], "rollout", "status", "deployment/grafana", "--timeout=180s"], timeout=185)
            if rc != 0:
                LOG.warning("rollout status check failed or timed out")
        else:
            LOG.info("deployment manifest not present; skipping deployment apply")
    else:
        LOG.info("management of grafana deployment skipped (GRAFANA_MANAGE_DEPLOYMENT=false)")
    LOG.info("apply complete; generator-owned resources applied. Re-run is idempotent.")

def delete_action() -> None:
    LOG.info("delete started")
    env = load_env()
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    yaml_files = [
        OUT_DIR / "grafana-dashboards-configmap.yaml",
        OUT_DIR / "grafana-provisioning-configmap.yaml",
        OUT_DIR / "grafana-datasources-configmap.yaml",
    ]
    for p in yaml_files:
        if p.exists():
            LOG.info("kubectl delete %s", str(p))
            kubectl_delete_yaml(p)
    for p in [OUT_DIR / "grafana-deployment.yaml", OUT_DIR / "grafana-service.yaml", OUT_DIR / "grafana-pvc.yaml", OUT_DIR / "grafana-postgres-statefulset.yaml", OUT_DIR / "grafana-postgres-service.yaml", OUT_DIR / "grafana-postgres-secret.yaml"]:
        if p.exists():
            kind = ("deployment" if "deployment" in p.name else ("service" if "service" in p.name else ("pvc" if "pvc" in p.name else ("statefulset" if "statefulset" in p.name else "secret"))))
            name = "grafana" if "grafana" in p.name and "postgres" not in p.name else ("grafana-postgres" if "postgres" in p.name else p.stem)
            has_label = False
            if kind in ("deployment","service","pvc","statefulset","secret"):
                rc, out, err = run_cmd(["kubectl", "-n", ns, "get", kind, name, "-o", "jsonpath={.metadata.labels.managed-by}"], timeout=10)
                if rc == 0 and out.strip() == "dashboards.py":
                    has_label = True
            if has_label:
                LOG.info("kubectl delete (managed) %s", str(p))
                kubectl_delete_yaml(p)
            else:
                LOG.warning("skipping delete of %s because resource is not labeled managed-by=dashboards.py or not present", p.name)
    try:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
            LOG.info("removed manifest directory %s", str(OUT_DIR))
        else:
            LOG.info("manifest directory not present; nothing to delete locally %s", str(OUT_DIR))
    except Exception as e:
        LOG.warning("failed to remove manifest directory: %s", e)
    LOG.info("delete complete")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/apply/delete Grafana dashboards + optional Grafana deployment + optional in-cluster Postgres.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--get-creds", action="store_true", help="Print Grafana admin username/password from grafana-admin-secret (after apply)")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        if args.delete:
            delete_action(); return
        if args.apply:
            apply_action()
            return
    except Exception as e:
        LOG.error("ERROR: %s", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
