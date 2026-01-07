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
from datetime import datetime
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
GRAFANA_POSTGRES_PVC_SIZE = os.getenv("GRAFANA_POSTGRES_PVC_SIZE", "5Gi")
GRAFANA_POSTGRES_PASSWORD = os.getenv("GRAFANA_POSTGRES_PASSWORD", "GfN9m2z!7xQpL3sV@8bR4tY1uE0kH6")

LOG = logging.getLogger("dashboards")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
ch.setFormatter(formatter)
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
                              input=stdin if stdin is not None else None, timeout=timeout, text=True)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        out = (getattr(e, "stdout", "") or "") or ""
        err = (getattr(e, "stderr", "") or "") or f"timeout after {timeout}s"
        return 124, out.strip(), err.strip()

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

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
    return env

def validate_env(env: Dict[str, str]) -> None:
    try:
        sst = float(env.get("SLO_SUCCESS_TARGET", "0.999"))
        if not (0.0 < sst < 1.0):
            raise ValueError()
    except Exception:
        raise RuntimeError("SLO_SUCCESS_TARGET must be a float between 0 and 1")
    if env.get("SLO_LATENCY_QUANTILE", "0.95") not in ("0.95", "0.99"):
        raise RuntimeError("SLO_LATENCY_QUANTILE must be '0.95' or '0.99'")

def validate_db_config(env: Dict[str, str]) -> None:
    external = coerce_bool(env.get("GRAFANA_EXTERNAL_DB", "false"))
    if external:
        if not (os.getenv("GF_DATABASE_URL") or (os.getenv("GF_DATABASE_HOST") and os.getenv("GF_DATABASE_NAME") and os.getenv("GF_DATABASE_USER") and os.getenv("GF_DATABASE_PASSWORD"))):
            raise RuntimeError("GRAFANA_EXTERNAL_DB=true requires GF_DATABASE_URL or GF_DATABASE_HOST/NAME/USER/PASSWORD")
    else:
        if not env.get("GRAFANA_POSTGRES_PASSWORD"):
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

def _log_panel_summary(panels: List[Dict[str, Any]]) -> None:
    for p in panels:
        title = p.get("title", "<no-title>")
        targets = p.get("targets", [])
        expr = ""
        if targets and isinstance(targets, list):
            expr = targets[0].get("expr", "") if isinstance(targets[0], dict) else ""
        LOG.info(" panel: %s | expr: %s", title, expr)

def build_service_dashboard(service: str, env: Dict[str, str]) -> Dict[str, Any]:
    LOG.info("Building service dashboard: %s", service)
    metrics_ds = env["METRICS_DATASOURCE"]
    slo_q = env.get("SLO_LATENCY_QUANTILE", "0.95")
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
        p_sr = make_stat_panel("Service Ready", 'max(service_ready{service="retrieval"})', metrics_ds, {"h": 3, "w": 6, "x": 0, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_sr["id"] = next_panel_id; next_panel_id += 1

        p_rps = make_stat_panel("Requests/s", "sum(rate(retrieval_requests_total[1m])) or sum(rate(http_requests_total{service=~\"retriev.*|retrieval.*\"}[1m]))", metrics_ds, {"h": 3, "w": 6, "x": 6, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_rps["id"] = next_panel_id; next_panel_id += 1

        p_p95 = make_metric_panel("P95 Latency (s)", f"histogram_quantile({slo_q}, sum by (le) (rate(retrieval_request_duration_seconds_bucket[5m])))", metrics_ds, {"h": 6, "w": 12, "x": 0, "y": 5}, chr(next_ref_ord)); next_ref_ord += 1
        p_p95["id"] = next_panel_id; next_panel_id += 1

        p_err_expr = '( sum(rate(retrieval_errors_total[5m])) / clamp_min(sum(rate(retrieval_requests_total[5m])), 1) ) * 100'
        p_err = make_metric_panel("Error Rate", p_err_expr, metrics_ds, {"h": 3, "w": 6, "x": 18, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_err["fieldConfig"] = {"defaults": {"unit": "percent"}}
        p_err["id"] = next_panel_id; next_panel_id += 1

        p_docs = make_metric_panel("Retrieved Docs (rate)", "sum(rate(retrieved_docs_count_count[1m])) or sum(rate(retrieved_docs_count[1m])) or vector(0)", metrics_ds, {"h": 3, "w": 6, "x": 12, "y": 8}, chr(next_ref_ord)); next_ref_ord += 1
        p_docs["id"] = next_panel_id; next_panel_id += 1

        repl_expr = (
            '('
            ' count(kube_pod_info{namespace="inference", pod=~"retrieval.*"})'
            ' or count(up{instance=~".*:8001|.*:8000"})'
            ' or sum(kube_deployment_status_replicas{namespace="inference", deployment=~"retrieval.*"})'
            ') or vector(0)'
        )
        p_repl = make_stat_panel("Replicas (kube-state-metrics / up / deployment)", repl_expr, metrics_ds, {"h": 3, "w": 6, "x": 12, "y": 5}, chr(next_ref_ord)); next_ref_ord += 1
        p_repl["id"] = next_panel_id; next_panel_id += 1
        p_repl["description"] = "Preferred: kube_pod_info from kube-state-metrics; fallbacks: up() by instance, controller replica count."

        p_total_increase = make_stat_panel(
            "Requests (5m increase)",
            'sum(increase(retrieval_requests_total[5m])) or sum(increase(http_requests_total{service=~"retriev.*|retrieval.*"}[5m])) or vector(0)',
            metrics_ds,
            {"h": 3, "w": 6, "x": 18, "y": 5},
            chr(next_ref_ord)
        ); next_ref_ord += 1
        p_total_increase["id"] = next_panel_id; next_panel_id += 1

        p_total_counter = make_stat_panel(
            "Total Requests (counter)",
            'sum(retrieval_requests_total) or sum(http_requests_total{service=~"retriev.*|retrieval.*"}) or vector(0)',
            metrics_ds,
            {"h": 3, "w": 6, "x": 18, "y": 8},
            chr(next_ref_ord)
        ); next_ref_ord += 1
        p_total_counter["id"] = next_panel_id; next_panel_id += 1

        p_fail_total_counter = make_stat_panel(
            "Total Failures (counter)",
            'sum(retrieval_errors_total) or vector(0)',
            metrics_ds,
            {"h": 3, "w": 6, "x": 12, "y": 8},
            chr(next_ref_ord)
        ); next_ref_ord += 1
        p_fail_total_counter["id"] = next_panel_id; next_panel_id += 1

        panels.extend([p_sr, p_rps, p_p95, p_err, p_repl, p_docs, p_total_increase, p_total_counter, p_fail_total_counter])

    elif service == "qdrant":
        p_up = make_stat_panel("Qdrant Up", 'max(up{job=~"qdrant.*"}) or max(up{instance=~".*:6333"})', metrics_ds, {"h": 3, "w": 6, "x": 0, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_up["id"] = next_panel_id; next_panel_id += 1

        p_req = make_stat_panel("Requests/s", "sum(rate(rest_responses_total[1m])) or sum(rate(rest_responses_total[5m])) or vector(0)", metrics_ds, {"h": 3, "w": 6, "x": 6, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_req["id"] = next_panel_id; next_panel_id += 1

        qdrant_p95_expr = (
            '('
            'histogram_quantile(0.95, sum by (le) (rate(rest_responses_duration_seconds_bucket[5m])))'
            ' or '
            'histogram_quantile(0.95, sum by (le) (rate(qdrant_query_duration_seconds_bucket[5m])))'
            ')'
        )
        p_p95 = make_metric_panel("P95 Latency (s)", qdrant_p95_expr, metrics_ds, {"h": 6, "w": 12, "x": 0, "y": 5}, chr(next_ref_ord)); next_ref_ord += 1
        p_p95["id"] = next_panel_id; next_panel_id += 1

        q_num = 'sum(rate(rest_responses_total{status=~"4..|5.."}[5m]))'
        q_den = '( ( sum(rate(rest_responses_total[5m])) ) or vector(1) )'
        p_err_expr = f'({q_num}) / clamp_min({q_den}, 1) * 100'
        p_err = make_metric_panel("Error Rate", p_err_expr, metrics_ds, {"h": 3, "w": 6, "x": 18, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_err["fieldConfig"] = {"defaults": {"unit": "percent"}}
        p_err["id"] = next_panel_id; next_panel_id += 1

        p_vec = make_stat_panel("Total Vectors", "sum(collections_vector_total) or vector(0)", metrics_ds, {"h": 3, "w": 6, "x": 12, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        p_vec["id"] = next_panel_id; next_panel_id += 1

        repl_expr = (
            '('
            ' count(kube_pod_info{namespace="qdrant", pod=~"qdrant.*"})'
            ' or count(up{instance=~".*:6333"})'
            ' or sum(kube_statefulset_status_replicas{namespace="qdrant", statefulset=~"qdrant.*"})'
            ') or vector(0)'
        )
        p_repl = make_stat_panel("Replicas (kube-state-metrics / up / sts)", repl_expr, metrics_ds, {"h": 3, "w": 6, "x": 12, "y": 8}, chr(next_ref_ord)); next_ref_ord += 1
        p_repl["id"] = next_panel_id; next_panel_id += 1

        p_q_fail_total_counter = make_stat_panel(
            "Total Failures (counter)",
            'sum(rest_responses_total{status=~"4..|5.."}) or vector(0)',
            metrics_ds,
            {"h": 3, "w": 6, "x": 12, "y": 5},
            chr(next_ref_ord)
        ); next_ref_ord += 1
        p_q_fail_total_counter["id"] = next_panel_id; next_panel_id += 1

        p_q_total_increase = make_stat_panel(
            "Requests (5m increase)",
            'sum(increase(rest_responses_total[5m])) or vector(0)',
            metrics_ds,
            {"h": 3, "w": 6, "x": 18, "y": 5},
            chr(next_ref_ord)
        ); next_ref_ord += 1
        p_q_total_increase["id"] = next_panel_id; next_panel_id += 1

        p_q_total_counter = make_stat_panel(
            "Total Requests (counter)",
            'sum(rest_responses_total) or vector(0)',
            metrics_ds,
            {"h": 3, "w": 6, "x": 18, "y": 8},
            chr(next_ref_ord)
        ); next_ref_ord += 1
        p_q_total_counter["id"] = next_panel_id; next_panel_id += 1

        panels.extend([p_up, p_req, p_p95, p_err, p_q_fail_total_counter, p_vec, p_repl, p_q_total_increase, p_q_total_counter])

    else:
        default_p95 = make_metric_panel("P95 Latency (fallback)", f"histogram_quantile({slo_q}, sum by (le) (rate({service}_request_duration_seconds_bucket[5m])))", metrics_ds, {"h": 6, "w": 12, "x": 0, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        default_p95["id"] = next_panel_id; next_panel_id += 1
        default_err = make_metric_panel("Error rate (fallback)", f"(sum(rate({service}_errors_total[5m])) or vector(0)) / clamp_min((sum(rate({service}_requests_total[5m])) or vector(1)),1) * 100", metrics_ds, {"h": 6, "w": 12, "x": 12, "y": 2}, chr(next_ref_ord)); next_ref_ord += 1
        default_err["id"] = next_panel_id; next_panel_id += 1
        panels.extend([default_p95, default_err])

    left_enc = render_clickhouse_left(service, env)
    logs_panel = {
        "type": "text",
        "title": "Logs",
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": 18},
        "options": {"content": "Open logs for this service using Explore"},
        "links": [{"title": "Open Logs", "url": f"/explore?left={left_enc}"}]
    }
    logs_panel["id"] = next_panel_id; next_panel_id += 1
    panels.append(logs_panel)

    LOG.info("Dashboard '%s' panels (count=%d):", service, len(panels))
    _log_panel_summary(panels)

    mp = safe_int(env.get("MAX_PANELS_PER_DASHBOARD", "48"), 48)
    if len(panels) > mp:
        raise RuntimeError(f"dashboard for {service} would exceed MAX_PANELS_PER_DASHBOARD ({len(panels)} > {mp})")

    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}{service}"
    if service == "retriever":
        ns_default = "inference"
    elif service == "qdrant":
        ns_default = "qdrant"
    else:
        ns_default = env.get("DEFAULT_NAMESPACE","monitoring")

    vars_list = [
        {"type": "custom", "name": "service", "options": [{"text": service, "value": service}], "current": {"text": service, "value": service}, "multi": False}
    ]
    vars_list.append({"type": "custom", "name": "namespace", "options": [{"text": ns_default, "value": ns_default}], "current": {"text": ns_default, "value": ns_default}, "multi": False})

    dash = {"id": None, "uid": uid, "title": f"Service Overview — {service}", "templating": {"list": vars_list}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_ingestion_dashboard(env: Dict[str, str]) -> Dict[str, Any]:
    LOG.info("Building ingestion dashboard")
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
    link_panel["id"] = next_panel_id; next_panel_id += 1
    panels.append(link_panel)

    LOG.info("Ingestion dashboard panels (count=%d):", len(panels))
    _log_panel_summary(panels)

    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}ingestion"
    dash = {"id": None, "uid": uid, "title": "Ingestion Health", "templating": {"list": [
        {"type": "custom", "name": "namespace", "options": [{"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}], "current": {"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}}]}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_platform_overview(env: Dict[str, str]) -> Dict[str, Any]:
    LOG.info("Building platform overview dashboard")
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

    LOG.info("Platform overview panels (count=%d):", len(panels))
    _log_panel_summary(panels)

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
        "metadata": {"name": "grafana-postgres", "namespace": ns, "labels": {"app": "grafana-postgres", "managed-by": "dashboards.py"}},
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
        "metadata": {"name": "grafana-postgres", "namespace": ns, "labels": {"app":"grafana-postgres","managed-by":"dashboards.py"}},
        "spec": {
            "serviceName": "grafana-postgres",
            "replicas": 1,
            "selector": {"matchLabels": {"app":"grafana-postgres"}},
            "template": {
                "metadata": {"labels": {"app":"grafana-postgres"}} ,
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
                {"metadata": {"name": "pgdata", "labels": {"app":"grafana-postgres"}}, "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": pvc_size}}}}
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

def get_k8s_resource_json(kind: str, name: str, ns: str, timeout: int = 8) -> Tuple[int, str]:
    rc, out, err = run_cmd(["kubectl", "-n", ns, "get", kind, name, "-o", "json"], timeout=timeout)
    if rc != 0:
        return rc, ""
    return rc, out

def kubectl_apply_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to apply manifests")
    rc, out, err = run_cmd(["kubectl", "-n", os.getenv("GRAFANA_PROVISIONING_NAMESPACE", GRAFANA_PROVISIONING_NAMESPACE), "apply", "-f", str(path)], timeout=60)
    if rc != 0:
        raise RuntimeError(f"kubectl apply failed for {path}: {err or out}")

def kubectl_delete_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        return
    rc, out, err = run_cmd(["kubectl", "delete", "-f", str(path), "--ignore-not-found"], timeout=30)
    return

def ensure_namespace(ns: str, timeout: int = 30) -> None:
    if not shutil.which("kubectl"):
        return
    rc, out, err = run_cmd(["kubectl", "-n", ns, "create", "namespace", ns, "--dry-run=client", "-o", "yaml"], timeout=10)
    if rc != 0:
        raise RuntimeError(f"failed to prepare namespace {ns}: {err or out}")
    proc = subprocess.Popen(["kubectl", "-n", ns, "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to create/apply namespace {ns}: {stderr}")
    start = time.time()
    while time.time() - start < timeout:
        rc2, out2, err2 = run_cmd(["kubectl", "get", "ns", ns, "-o", "json"], timeout=6)
        if rc2 == 0 and out2:
            try:
                j = json.loads(out2)
                phase = j.get("status", {}).get("phase", "")
                if phase == "Active":
                    return
            except Exception:
                pass
        time.sleep(1)

def create_or_update_secret_from_env(ns: str, secret_name: str, mapping: Dict[str, str]) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to create secrets")
    args = ["kubectl", "create", "secret", "generic", secret_name]
    for key, val in mapping.items():
        args.append(f"--from-literal={key}={val}")
    args.extend(["--dry-run=client", "-o", "yaml", "-n", ns])
    rc, out, err = run_cmd(args, timeout=20)
    if rc != 0:
        raise RuntimeError(f"failed to create secret manifest: {err or out}")
    proc = subprocess.Popen(["kubectl", "-n", ns, "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to apply secret {secret_name}: {stderr}")
    run_cmd(["kubectl", "-n", ns, "label", "secret", secret_name, "managed-by=dashboards.py", "--overwrite"], timeout=10)

def render_all(env: Dict[str, str]) -> Dict[str, Path]:
    validate_env(env)
    validate_db_config(env)
    services = [s.strip() for s in (env.get("DASHBOARD_SERVICES","") or "").split(",") if s.strip()]
    if not services:
        services = ["retriever","qdrant"]
    rendered: Dict[str, Dict[str, Any]] = {}
    LOG.info("Rendering dashboards for services: %s", ", ".join(services))
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
        LOG.info("Wrote dashboard JSON: %s (%s)", name, p)
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
    LOG.info("Wrote provisioning manifests to %s", OUT_DIR)
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
        LOG.info("Wrote in-cluster Postgres manifests")
    if coerce_bool(env.get("GRAFANA_USE_PVC","false")):
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "grafana-data", "namespace": env["GRAFANA_NAMESPACE"], "labels": {"app": "grafana", "managed-by":"dashboards.py"}},
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
    LOG.info("Render complete; output paths: %s", ", ".join(f"{k}={v}" for k,v in out_paths.items()))
    return out_paths

def run_promql_check_in_cluster(promql: str, ns: str = "monitoring", metrics_url: str = METRICS_DATASOURCE_URL, timeout: int = 12) -> Tuple[int, str]:
    if not shutil.which("kubectl"):
        return 1, "kubectl not available"
    base = metrics_url.rstrip("/")
    target = f'{base}/api/v1/query'
    cmd = [
        "kubectl", "-n", ns, "run", "--rm", "-i", "--restart=Never", "promql-check",
        "--", "sh", "-c",
        f'curl -sS -G --data-urlencode "query={promql}" "{target}" || echo "__PROMQL_FAILED__"'
    ]
    rc, out, err = run_cmd(cmd, timeout=timeout)
    return rc, out.strip()

def verify_post_apply(env: Dict[str, str]) -> None:
    checks = [
        ("replicas_qdrant", 'count(kube_pod_info{namespace="qdrant",pod=~"qdrant.*"}) or count(up{instance=~".*:6333"}) or sum(kube_statefulset_status_replicas{namespace="qdrant", statefulset=~"qdrant.*"})'),
        ("replicas_retriever", 'count(kube_pod_info{namespace="inference",pod=~"retrieval.*"}) or count(up{instance=~".*:8001|.*:8000"}) or sum(kube_deployment_status_replicas{namespace="inference",deployment=~"retrieval.*"})'),
        ("collections_vectors", "sum(collections_vector_total) or vector(0)"),
        ("qdrant_histogram_buckets_count", 'count(rest_responses_duration_seconds_bucket) or count(qdrant_query_duration_seconds_bucket)'),
        ("retriever_requests_rate", 'sum(rate(retrieval_requests_total[1m])) or sum(rate(http_requests_total{service=~"retriev.*|retrieval.*"}[1m])) or vector(0)'),
    ]
    results: Dict[str, Any] = {}
    metrics_url = env.get("METRICS_DATASOURCE_URL", METRICS_DATASOURCE_URL)
    ns = env.get("GRAFANA_PROVISIONING_NAMESPACE","monitoring")
    for name, q in checks:
        rc, out = run_promql_check_in_cluster(q, ns=ns, metrics_url=metrics_url)
        if rc != 0 or not out or "__PROMQL_FAILED__" in out:
            results[name] = {"ok": False, "raw": out}
            LOG.info("verify_post_apply: check %s -> FAILED or empty (raw: %s)", name, out)
            continue
        try:
            j = json.loads(out)
            status = j.get("status", "")
            if status == "success":
                rr = j.get("data", {}).get("result", [])
                results[name] = {"ok": True, "raw": out, "result_count": len(rr)}
                LOG.info("verify_post_apply: check %s -> OK (result_count=%d)", name, len(rr))
            else:
                results[name] = {"ok": False, "raw": out}
                LOG.info("verify_post_apply: check %s -> not-success (raw: %s)", name, out)
        except Exception as e:
            results[name] = {"ok": False, "raw": out, "parse_error": str(e)}
            LOG.info("verify_post_apply: check %s -> parse error: %s", name, str(e))
    return

def apply_action() -> None:
    env = load_env()
    validate_env(env)
    validate_db_config(env)
    paths = render_all(env)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    ensure_namespace(ns)
    if env.get("GRAFANA_ADMIN_USER") and env.get("GRAFANA_ADMIN_PASSWORD"):
        create_or_update_secret_from_env(ns, "grafana-admin-secret", {"admin-user": env["GRAFANA_ADMIN_USER"], "admin-password": env["GRAFANA_ADMIN_PASSWORD"]})
    if not coerce_bool(env.get("GRAFANA_EXTERNAL_DB","false")):
        create_or_update_secret_from_env(ns, "grafana-postgres-secret", {"postgres-user": env.get("GRAFANA_POSTGRES_USER","grafana"), "postgres-password": env.get("GRAFANA_POSTGRES_PASSWORD"), "postgres-db": env.get("GRAFANA_POSTGRES_DB","grafana")})
        if paths.get("postgres_service"):
            try:
                kubectl_apply_yaml(paths["postgres_service"])
            except Exception as e:
                LOG.info("apply_action: failed to apply postgres service: %s", e)
        if paths.get("postgres_sts"):
            try:
                kubectl_apply_yaml(paths["postgres_sts"])
            except Exception as e:
                LOG.info("apply_action: failed to apply postgres statefulset: %s", e)
            run_cmd(["kubectl", "-n", ns, "rollout", "status", "statefulset/grafana-postgres", "--timeout=30s"], timeout=35)
    if paths.get("datasources_cm"):
        try:
            kubectl_apply_yaml(paths["datasources_cm"])
        except Exception as e:
            LOG.info("apply_action: failed to apply datasources configmap: %s", e)
    if paths.get("provisioning_cm"):
        try:
            kubectl_apply_yaml(paths["provisioning_cm"])
        except Exception as e:
            LOG.info("apply_action: failed to apply provisioning configmap: %s", e)
    if paths.get("dashboards_cm"):
        try:
            kubectl_apply_yaml(paths["dashboards_cm"])
        except Exception as e:
            LOG.info("apply_action: failed to apply dashboards configmap: %s", e)
    svc = OUT_DIR / "grafana-service.yaml"
    if svc.exists():
        try:
            kubectl_apply_yaml(svc)
        except Exception as e:
            LOG.info("apply_action: failed to apply grafana service: %s", e)
    if coerce_bool(env.get("GRAFANA_USE_PVC","false")):
        pvc_path = OUT_DIR / "grafana-pvc.yaml"
        if pvc_path.exists():
            try:
                kubectl_apply_yaml(pvc_path)
            except Exception as e:
                LOG.info("apply_action: failed to apply pvc: %s", e)
    dep = OUT_DIR / "grafana-deployment.yaml"
    if coerce_bool(env.get("GRAFANA_MANAGE_DEPLOYMENT","false")) and dep.exists():
        try:
            ensure_deployment_applied_safely(dep, env["GRAFANA_NAMESPACE"], name="grafana")
            run_cmd(["kubectl", "-n", env["GRAFANA_NAMESPACE"], "rollout", "status", "deployment/grafana", "--timeout=30s"], timeout=35)
        except Exception as e:
            LOG.info("apply_action: failed to manage grafana deployment: %s", e)
    try:
        verify_post_apply(env)
    except Exception as e:
        LOG.info("apply_action: verify_post_apply failed: %s", e)

def delete_action() -> None:
    env = load_env()
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    if not shutil.which("kubectl"):
        return
    resource_groups = [
        "configmap",
        "secret",
        "service",
        "statefulset",
        "deployment",
        "daemonset",
        "replicaset",
        "pod",
        "ingress",
        "pvc",
        "persistentvolumeclaim",
        "role",
        "rolebinding",
        "serviceaccount",
        "networkpolicy",
        "horizontalpodautoscaler",
        "cronjob",
        "job"
    ]
    if ns != "monitoring":
        resource_groups.append("namespace")
    for group in resource_groups:
        run_cmd(["kubectl", "-n", ns, "delete", group, "--selector=managed-by=dashboards.py", "--ignore-not-found"], timeout=30)
    try:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
            LOG.info("delete_action: removed %s", OUT_DIR)
    except Exception as e:
        LOG.info("delete_action: failed to remove out dir: %s", e)

def resource_has_managed_label(kind: str, name: str, ns: str, label_key: str = "managed-by", label_val: str = "dashboards.py") -> bool:
    rc, out = run_cmd(["kubectl", "-n", ns, "get", kind, name, "-o", "json"], timeout=8)
    if rc != 0 or not out:
        return False
    try:
        j = json.loads(out)
        labels = j.get("metadata", {}).get("labels", {}) or {}
        return labels.get(label_key) == label_val
    except Exception:
        return False

def wait_for_resource_deleted(kind: str, name: str, ns: str, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        rc, out, err = run_cmd(["kubectl", "-n", ns, "get", kind, name, "-o", "json"], timeout=4)
        if rc != 0:
            return True
        time.sleep(1)
    return False

def ensure_deployment_applied_safely(yaml_path: Path, ns: str, name: str = "grafana") -> None:
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except Exception:
        raise
    desired_sel = {}
    try:
        desired_sel = doc.get("spec", {}).get("selector", {}).get("matchLabels", {}) or {}
    except Exception:
        desired_sel = {}
    rc, out, err = run_cmd(["kubectl", "-n", ns, "get", "deployment", name, "-o", "json"], timeout=8)
    if rc != 0:
        try:
            kubectl_apply_yaml(yaml_path)
        except Exception as e:
            LOG.info("ensure_deployment_applied_safely: apply failed: %s", e)
        return
    try:
        live = json.loads(out)
        live_sel = live.get("spec", {}).get("selector", {}).get("matchLabels", {}) or {}
    except Exception:
        try:
            kubectl_apply_yaml(yaml_path)
        except Exception as e:
            LOG.info("ensure_deployment_applied_safely: apply failed during parse: %s", e)
        return
    if live_sel != desired_sel:
        managed = resource_has_managed_label("deployment", name, ns)
        force_apply = coerce_bool(os.getenv("FORCE_APPLY", "false"))
        if not managed and not force_apply:
            LOG.info("ensure_deployment_applied_safely: existing deployment selector differs, not managed and FORCE_APPLY not set -> skipping")
            return
        if not force_apply:
            return
        run_cmd(["kubectl", "-n", ns, "delete", "deployment", name, "--ignore-not-found"], timeout=30)
        ok = wait_for_resource_deleted("deployment", name, ns, timeout=30)
        try:
            kubectl_apply_yaml(yaml_path)
        except Exception as e:
            LOG.info("ensure_deployment_applied_safely: apply after delete failed: %s", e)
    else:
        try:
            kubectl_apply_yaml(yaml_path)
        except Exception as e:
            LOG.info("ensure_deployment_applied_safely: apply failed: %s", e)

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
        LOG.info("main: fatal error: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
