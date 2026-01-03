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

# ---------- structured logging ----------
class SimpleJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in {"name","msg","args","levelno","levelname","pathname","filename","module","exc_info","exc_text","stack_info","lineno","funcName","created","msecs","relativeCreated","thread","threadName","processName","process","message"}}
        payload.update({k: (v if isinstance(v, (str,int,float,bool,list,dict)) else str(v)) for k,v in extra.items()})
        return json.dumps(payload, ensure_ascii=False)

LOG = logging.getLogger("dashboards_generator")
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(SimpleJSONFormatter())
LOG.handlers = []
LOG.addHandler(ch)

# ---------- paths & defaults ----------
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "infra" / "manifests" / "dashboards"

DEFAULTS: Dict[str, str] = {
    "GRAFANA_IMAGE": "grafana/grafana:10.3.5",
    "GRAFANA_NAMESPACE": "monitoring",
    "GRAFANA_REPLICAS": "1",
    "GRAFANA_USE_PVC": "false",
    "GRAFANA_PVC_SIZE": "5Gi",
    "GRAFANA_CPU_REQ": "100m",
    "GRAFANA_MEM_REQ": "128Mi",
    "GRAFANA_CPU_LIMIT": "500m",
    "GRAFANA_MEM_LIMIT": "512Mi",
    "GRAFANA_PROVISIONING_NAMESPACE": "monitoring",
    "GRAFANA_DASHBOARD_UID_PREFIX": "platform-",
    "RUNBOOK_BASE_URL": "https://defaultsa515.z13.web.core.windows.net",
    "DASHBOARD_SERVICES": "retriever,qdrant",
    "MAX_PANELS_PER_DASHBOARD": "48",
    "METRICS_DATASOURCE": "VictoriaMetrics",
    "METRICS_DATASOURCE_URL": "http://victoria-metrics.monitoring.svc:8428",
    "CLICKHOUSE_DATASOURCE": "ClickHouse",
    "CLICKHOUSE_URL": "http://clickhouse.clickhouse.svc:8123",
    "DEFAULT_NAMESPACE": "monitoring",
    "SLO_SUCCESS_TARGET": "0.999",
    "SLO_LATENCY_QUANTILE": "0.95",
    "DATASOURCE_URL": "http://victoria-metrics.monitoring.svc:8428",
    "CI": "false",
    # safety toggle (default: do not manage cluster Deployment)
    "GRAFANA_MANAGE_DEPLOYMENT": "true",
    # required secret keys (fail if missing)
    "GRAFANA_ADMIN_USER": "",
    "GRAFANA_ADMIN_PASSWORD": "",
}

CLICKHOUSE_SQL_TEMPLATE = (
    "SELECT ts, level, message, fields FROM logs.kube_logs "
    "WHERE service = '$service' "
    "AND namespace = '$namespace' "
    "AND ts BETWEEN toDateTime64($__from / 1000, 3) AND toDateTime64($__to / 1000, 3) "
    "ORDER BY ts DESC LIMIT 500"
)

# ---------- helpers ----------
def run_cmd(cmd: List[str], timeout: int = 60, stdin: Optional[str] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=stdin.encode("utf-8") if stdin else None, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        LOG.debug("run_cmd finished", extra={"cmd": " ".join(cmd), "rc": proc.returncode, "out_len": len(out), "err_len": len(err)})
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        out = (getattr(e, "stdout", None) or b"").decode("utf-8", errors="replace") if getattr(e, "stdout", None) else ""
        err = (getattr(e, "stderr", None) or b"").decode("utf-8", errors="replace") if getattr(e, "stderr", None) else f"timeout after {timeout}s"
        LOG.error("run_cmd timeout", extra={"cmd": " ".join(cmd)})
        return 124, out.strip(), err.strip()

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, str(path))
    LOG.info("wrote file", extra={"path": str(path), "bytes": len(content)})

def coerce_bool(v: Optional[str]) -> bool:
    if not v:
        return False
    return v.lower() in ("1","true","yes","on")

def safe_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

# ---------- env & validation ----------
def load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for k, dv in DEFAULTS.items():
        v = os.getenv(k)
        env[k] = v if v is not None else dv
        if v is None:
            LOG.debug("env default used", extra={"env": k, "default": dv})
        else:
            LOG.debug("env loaded", extra={"env": k, "value": ("<redacted>" if "PASSWORD" in k or "KEY" in k or "ADMIN" in k else v)})
    return env

def validate_env(env: Dict[str, str]) -> None:
    # SLO target
    try:
        sst = float(env["SLO_SUCCESS_TARGET"])
        if not (0.0 < sst < 1.0):
            raise ValueError()
    except Exception:
        LOG.error("invalid SLO_SUCCESS_TARGET", extra={"value": env.get("SLO_SUCCESS_TARGET")})
        raise RuntimeError("SLO_SUCCESS_TARGET must be a float between 0 and 1, e.g. 0.999")
    if env["SLO_LATENCY_QUANTILE"] not in ("0.95", "0.99"):
        LOG.error("invalid SLO_LATENCY_QUANTILE", extra={"value": env.get("SLO_LATENCY_QUANTILE")})
        raise RuntimeError("SLO_LATENCY_QUANTILE must be '0.95' or '0.99'")
    # required secrets
    if not env.get("GRAFANA_ADMIN_USER"):
        raise RuntimeError("required environment variable GRAFANA_ADMIN_USER is missing or empty")
    if not env.get("GRAFANA_ADMIN_PASSWORD"):
        raise RuntimeError("required environment variable GRAFANA_ADMIN_PASSWORD is missing or empty")

# ---------- grafana datasource check (optional) ----------
def validate_grafana_datasource(env: Dict[str, str]) -> None:
    api_url = env.get("GRAFANA_API_URL") or ""
    api_key = env.get("GRAFANA_API_KEY") or ""
    ds_name = env.get("METRICS_DATASOURCE") or "VictoriaMetrics"
    if not api_url or not api_key:
        LOG.info("Grafana API credentials not provided; skipping Grafana datasource validation")
        return
    url = api_url.rstrip("/") + f"/api/datasources/name/{urllib.parse.quote(ds_name)}"
    cmd = ["curl", "-sS", "-H", f"Authorization: Bearer {api_key}", "-H", "Accept: application/json", url]
    rc, out, err = run_cmd(cmd, timeout=10)
    if rc != 0:
        raise RuntimeError(f"Grafana API datasource check failed: {err or out}")
    try:
        obj = json.loads(out)
        if obj.get("name", "").lower() != ds_name.lower():
            raise RuntimeError(f"Grafana datasource '{ds_name}' not found (returned: {obj.get('name')})")
    except Exception as e:
        raise RuntimeError(f"Grafana datasource validation failed: {e}")
    LOG.info("Grafana datasource present", extra={"datasource": ds_name})

# ---------- recording rules check (best-effort) ----------
def check_recording_rules(env: Dict[str, str]) -> None:
    candidates = [
        ROOT / "infra" / "manifests" / "alerts" / "slo.rules.yaml",
        ROOT / "monitoring" / "recording_rules.yaml",
        ROOT / "infra" / "manifests" / "monitoring" / "recording_rules.yaml",
    ]
    found = False
    for p in candidates:
        if p.exists():
            LOG.info("found recording rules candidate", extra={"path": str(p)})
            found = True
            break
    if not found:
        msg = "recording rules file not found in known locations"
        if coerce_bool(env.get("CI", "false")):
            LOG.error(msg)
            raise RuntimeError(msg)
        LOG.warning(msg + " (continuing)")
    ds = env.get("DATASOURCE_URL")
    if ds and shutil.which("curl"):
        queries = {
            "retrieval_errors_rate_1h": 'sum(rate(retrieval_errors_total[1h]))',
            "retrieval_requests_rate_1h": 'sum(rate(retrieval_requests_total[1h]))',
            "qdrant_rest_total_rate_1h": 'sum(rate(rest_responses_total[1h]))',
        }
        failures = []
        for name, q in queries.items():
            try:
                url = ds.rstrip("/") + "/api/v1/query"
                cmd = ["curl", "-sS", "-G", "--data-urlencode", f"query={q}", url]
                rc, out, err = run_cmd(cmd, timeout=10)
                if rc != 0 or not out:
                    failures.append(f"{name}:no-response")
                    continue
                j = json.loads(out)
                if j.get("status") != "success" or not j.get("data"):
                    failures.append(f"{name}:no-series")
                    continue
                LOG.info("recording rule ok", extra={"name": name})
            except Exception:
                failures.append(f"{name}:err")
        if failures:
            msg = ",".join(failures)
            if coerce_bool(env.get("CI", "false")):
                LOG.error("recording rule checks failed in CI: %s", msg)
                raise RuntimeError("recording rule checks failed: " + msg)
            LOG.warning("recording rule checks issues: %s (continuing)", msg)
    else:
        LOG.debug("Skipping PromQL recording-rule checks (DATASOURCE_URL or curl missing)")

# ---------- clickhouse left for explore link ----------
def render_clickhouse_left(service: str, env: Dict[str, str]) -> str:
    template = env.get("CLICKHOUSE_SQL_TEMPLATE", CLICKHOUSE_SQL_TEMPLATE)
    left_obj = {"datasource": env["CLICKHOUSE_DATASOURCE"], "queries": [{"refId": "A", "sql": template}], "range": {"from": "$__from", "to": "$__to"}}
    raw = json.dumps(left_obj, separators=(",", ":"), ensure_ascii=False)
    return urllib.parse.quote_plus(raw)

# ---------- panel & dashboard builders ----------
def make_metric_panel(title: str, expr: str, datasource: str, gridPos: Dict[str, int], refId: str, panel_type: str = "timeseries") -> Dict[str, Any]:
    return {"type": panel_type, "title": title, "datasource": datasource, "targets": [{"expr": expr, "refId": refId}], "gridPos": gridPos}

def make_stat_panel(title: str, expr: str, datasource: str, gridPos: Dict[str, int], refId: str) -> Dict[str, Any]:
    return make_metric_panel(title, expr, datasource, gridPos, refId, panel_type="stat")

def build_service_dashboard(service: str, env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    slo_q = env["SLO_LATENCY_QUANTILE"]
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1
    hdr = {
        "type": "text",
        "title": "Header",
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": 0},
        "options": {"content": f"Owner: platform\nRunbook base: {env.get('RUNBOOK_BASE_URL','')}\nSLO target: {env['SLO_SUCCESS_TARGET']}\nLatency quantile: {slo_q}"}
    }
    hdr["id"] = next_panel_id; next_panel_id += 1
    panels.append(hdr)

    if service == "retriever":
        p1 = make_metric_panel("P95 Latency", f"histogram_quantile({slo_q}, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))", metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 3}, chr(next_ref_ord)); next_ref_ord += 1
        p1["id"] = next_panel_id; next_panel_id += 1
        p2 = make_metric_panel("Error rate", "sum(rate(retrieval_errors_total[5m])) / sum(rate(retrieval_requests_total[5m]))", metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 3}, chr(next_ref_ord)); next_ref_ord += 1
        p2["id"] = next_panel_id; next_panel_id += 1
        p3 = make_stat_panel("Service readiness", 'avg(service_ready{service="retrieval"})', metrics_ds, {"h": 3, "w": 12, "x": 0, "y": 11}, chr(next_ref_ord)); next_ref_ord += 1
        p3["id"] = next_panel_id; next_panel_id += 1
        p4 = make_stat_panel("Requests/s", "sum(rate(retrieval_requests_total[1m]))", metrics_ds, {"h": 3, "w": 12, "x": 12, "y": 11}, chr(next_ref_ord)); next_ref_ord += 1
        p4["id"] = next_panel_id; next_panel_id += 1
        panels.extend([p1, p2, p3, p4])
    elif service == "qdrant":
        p1 = make_metric_panel("Qdrant P95 Latency", f"histogram_quantile({slo_q}, sum(rate(qdrant_query_duration_seconds_bucket[5m])) by (le))", metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 3}, chr(next_ref_ord)); next_ref_ord += 1
        p1["id"] = next_panel_id; next_panel_id += 1
        p2 = make_metric_panel("Qdrant Queries/s", "sum(rate(qdrant_query_total[1m]))", metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 3}, chr(next_ref_ord)); next_ref_ord += 1
        p2["id"] = next_panel_id; next_panel_id += 1
        p3 = make_stat_panel("Collections", "collections_total", metrics_ds, {"h": 3, "w": 12, "x": 0, "y": 11}, chr(next_ref_ord)); next_ref_ord += 1
        p3["id"] = next_panel_id; next_panel_id += 1
        p4 = make_stat_panel("Dead replicas", "max(collection_dead_replicas)", metrics_ds, {"h": 3, "w": 12, "x": 12, "y": 11}, chr(next_ref_ord)); next_ref_ord += 1
        p4["id"] = next_panel_id; next_panel_id += 1
        panels.extend([p1, p2, p3, p4])
    else:
        p1 = make_metric_panel("P95 Latency (fallback)", f"histogram_quantile({slo_q}, sum(rate({service}_request_duration_seconds_bucket[5m])) by (le))", metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 3}, chr(next_ref_ord)); next_ref_ord += 1
        p1["id"] = next_panel_id; next_panel_id += 1
        p2 = make_metric_panel("Error rate (fallback)", f"sum(rate({service}_errors_total[5m])) / max(sum(rate({service}_requests_total[5m])),1)", metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 3}, chr(next_ref_ord)); next_ref_ord += 1
        p2["id"] = next_panel_id; next_panel_id += 1
        panels.extend([p1, p2])

    left_enc = render_clickhouse_left(service, env)
    logs_panel = {
        "type": "text",
        "title": "Logs",
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": 14},
        "options": {"content": "Click 'Open Logs' to inspect ClickHouse logs for this service."},
        "links": [{"title": "Open Logs", "url": f"/explore?left={left_enc}"}]
    }
    logs_panel["id"] = next_panel_id; next_panel_id += 1
    panels.append(logs_panel)

    mp = safe_int(env.get("MAX_PANELS_PER_DASHBOARD", "32"), 32)
    if len(panels) > mp:
        raise RuntimeError(f"dashboard for {service} would exceed MAX_PANELS_PER_DASHBOARD ({len(panels)} > {mp})")

    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}{service}"
    dash = {"id": None, "uid": uid, "title": f"Service Overview — {service}", "templating": {"list": [
        {"type": "query", "name": "service", "query": service, "current": {"text": service, "value": service}},
        {"type": "query", "name": "namespace", "query": env["DEFAULT_NAMESPACE"], "current": {"text": env["DEFAULT_NAMESPACE"], "value": env["DEFAULT_NAMESPACE"]}}
    ]}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_ingestion_dashboard(env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1
    p1 = make_metric_panel("vmagent discovery objects", 'vm_promscrape_discovery_kubernetes_objects{role="pod"}', metrics_ds, {"h": 6, "w": 24, "x": 0, "y": 0}, chr(next_ref_ord)); next_ref_ord += 1
    p1["id"] = next_panel_id; next_panel_id += 1
    p2 = make_metric_panel("remote-write bytes (5m)", "increase(vm_persistentqueue_bytes_written_total[5m])", metrics_ds, {"h": 6, "w": 24, "x": 0, "y": 6}, chr(next_ref_ord)); next_ref_ord += 1
    p2["id"] = next_panel_id; next_panel_id += 1
    p3 = make_metric_panel("vmagent pending queue", "vm_persistentqueue_bytes_pending", metrics_ds, {"h": 6, "w": 24, "x": 0, "y": 12}, chr(next_ref_ord)); next_ref_ord += 1
    p3["id"] = next_panel_id; next_panel_id += 1
    panels.extend([p1, p2, p3])

    left_enc = render_clickhouse_left("vmagent", env)
    link_panel = {
        "type": "text",
        "title": "Runbook",
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": 18},
        "options": {"content": f"Runbook base: {env.get('RUNBOOK_BASE_URL','')}"},
        "links": [{"title": "Open Logs", "url": f"/explore?left={left_enc}"}]
    }
    link_panel["id"] = next_panel_id
    panels.append(link_panel)

    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}ingestion"
    dash = {"id": None, "uid": uid, "title": "Ingestion Health", "templating": {"list": [
        {"type": "query", "name": "namespace", "query": env["DEFAULT_NAMESPACE"], "current": {"text": env["DEFAULT_NAMESPACE"], "value": env["DEFAULT_NAMESPACE"]}}
    ]}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_platform_overview(env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1
    p1 = make_metric_panel("Total ingestion health (composite)", "max(vm_promscrape_discovery_kubernetes_objects{role='pod'})", metrics_ds, {"h": 4, "w": 24, "x": 0, "y": 0}, chr(next_ref_ord)); next_ref_ord += 1
    p1["id"] = next_panel_id; next_panel_id += 1
    p2 = make_metric_panel("Services not ready", "count(service_ready==0)", metrics_ds, {"h": 4, "w": 12, "x": 0, "y": 4}, chr(next_ref_ord)); next_ref_ord += 1
    p2["id"] = next_panel_id; next_panel_id += 1
    p3 = make_metric_panel("Active SLO burners", "count(firing_alerts{plane='slo',severity='critical'})", metrics_ds, {"h": 4, "w": 12, "x": 12, "y": 4}, chr(next_ref_ord)); next_ref_ord += 1
    p3["id"] = next_panel_id; next_panel_id += 1
    panels.extend([p1, p2, p3])
    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}platform-overview"
    dash = {"id": None, "uid": uid, "title": "Platform Overview", "templating": {"list": []}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboards.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

# ---------- provisioning artifacts ----------
def build_dashboards_configmap(dashboards: Dict[str, Dict[str, Any]], env: Dict[str, str]) -> Dict[str, Any]:
    data: Dict[str, str] = {}
    for name, db in dashboards.items():
        key = f"{name}.json"
        # store compact JSON (Grafana reads the JSON files)
        data[key] = json.dumps(db, separators=(",", ":"), ensure_ascii=False)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-dashboards", "namespace": ns, "labels": {"managed-by": "dashboards.py"}}, "data": data}

def build_provisioning_provider_cm(env: Dict[str, str]) -> Dict[str, Any]:
    # Grafana expects a top-level list of provider objects (not wrapped under 'providers:')
    provider_item = {
        "name": "platform-dashboards",
        "orgId": 1,
        "folder": "Platform",
        "type": "file",
        "disableDeletion": False,
        "options": {"path": "/var/lib/grafana/dashboards"}
    }
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    # produce a YAML list (each item is a provider)
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

# ---------- optional grafana deployment/service/pvc ----------
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
    # use secretKeyRef to avoid baking secret into manifests
    container = {
        "name":"grafana",
        "image":image,
        "env":[
            {"name":"GF_SECURITY_ADMIN_USER", "valueFrom": {"secretKeyRef": {"name":"grafana-admin-secret","key":"admin-user"}}},
            {"name":"GF_SECURITY_ADMIN_PASSWORD", "valueFrom": {"secretKeyRef": {"name":"grafana-admin-secret","key":"admin-password"}}},
            {"name":"GF_PATHS_DATA","value":"/var/lib/grafana/"},
            {"name":"GF_PATHS_PROVISIONING","value":"/etc/grafana/provisioning"}
        ],
        "ports":[{"containerPort":3000,"name":"http"}],
        "volumeMounts":volume_mounts,
        "resources":{
            "requests":{"cpu":env.get("GRAFANA_CPU_REQ"),"memory":env.get("GRAFANA_MEM_REQ")},
            "limits":{"cpu":env.get("GRAFANA_CPU_LIMIT"),"memory":env.get("GRAFANA_MEM_LIMIT")}
        },
        "readinessProbe":{"httpGet":{"path":"/api/health","port":3000},"initialDelaySeconds":5,"periodSeconds":10}
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

def build_grafana_pvc(env: Dict[str, str]) -> Dict[str, Any]:
    ns = env["GRAFANA_NAMESPACE"]
    size = env.get("GRAFANA_PVC_SIZE","5Gi")
    pvc = {
        "apiVersion":"v1",
        "kind":"PersistentVolumeClaim",
        "metadata":{"name":"grafana-data","namespace":ns,"labels":{"app":"grafana","managed-by":"dashboards.py"}},
        "spec":{"accessModes":["ReadWriteOnce"],"resources":{"requests":{"storage":size}}}
    }
    return pvc

# ---------- render pipeline (always overwrite files atomically) ----------
def render_all(env: Dict[str, str]) -> Dict[str, Path]:
    validate_env(env)
    try:
        validate_grafana_datasource(env)
    except Exception as e:
        LOG.warning("grafana datasource validation issue: %s", str(e))
        if coerce_bool(env.get("CI", "false")):
            raise
    try:
        check_recording_rules(env)
    except Exception as e:
        LOG.warning("recording rule checks had issues: %s", str(e))
        if coerce_bool(env.get("CI", "false")):
            raise
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
    if coerce_bool(env.get("GRAFANA_USE_PVC","false")):
        pvc = build_grafana_pvc(env)
        atomic_write(OUT_DIR / "grafana-pvc.yaml", yaml.safe_dump(pvc, sort_keys=False))
        out_paths["pvc"] = OUT_DIR / "grafana-pvc.yaml"
    atomic_write(OUT_DIR / "clickhouse-explore-sql.txt", CLICKHOUSE_SQL_TEMPLATE)
    out_paths["dashboards_cm"] = OUT_DIR / "grafana-dashboards-configmap.yaml"
    out_paths["provisioning_cm"] = OUT_DIR / "grafana-provisioning-configmap.yaml"
    out_paths["datasources_cm"] = OUT_DIR / "grafana-datasources-configmap.yaml"
    out_paths["deployment"] = OUT_DIR / "grafana-deployment.yaml"
    out_paths["service"] = OUT_DIR / "grafana-service.yaml"
    LOG.info("rendering manifests to disk", extra={"out_dir": str(OUT_DIR), "files": [str(p) for p in out_paths.values()]})
    return out_paths

def json_validate(path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
    except Exception as e:
        LOG.error("dashboard JSON invalid", extra={"path": str(path), "error": str(e)})
        raise

# ---------- kubectl helpers ----------
def kubectl_apply_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to apply manifests")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", str(path)], timeout=60)
    if rc != 0:
        LOG.error("kubectl apply failed", extra={"file": str(path), "stdout": out, "stderr": err})
        raise RuntimeError(f"kubectl apply failed for {path}: {err or out}")
    LOG.info("kubectl apply succeeded", extra={"file": str(path)})

def kubectl_delete_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        LOG.warning("kubectl not present; skipping kubectl delete for %s", str(path))
        return
    rc, out, err = run_cmd(["kubectl", "delete", "-f", str(path), "--ignore-not-found"], timeout=60)
    if rc != 0:
        LOG.warning("kubectl delete returned non-zero", extra={"file": str(path), "stdout": out, "stderr": err})
    else:
        LOG.info("kubectl delete succeeded", extra={"file": str(path)})

def ensure_namespace(ns: str, timeout: int = 30) -> None:
    if not shutil.which("kubectl"):
        LOG.debug("kubectl not found; cannot ensure namespace %s", ns)
        return
    # produce namespace YAML and apply it (atomic)
    rc, out, err = run_cmd(["kubectl", "create", "namespace", ns, "--dry-run=client", "-o", "yaml"], timeout=10)
    if rc != 0:
        LOG.error("namespace dry-run failed", extra={"ns": ns, "err": err or out})
        raise RuntimeError(f"failed to prepare namespace {ns}: {err or out}")
    proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        LOG.error("failed to ensure namespace", extra={"ns": ns, "stderr": stderr})
        raise RuntimeError(f"failed to create/apply namespace {ns}: {stderr}")
    # wait until namespace is active
    start = time.time()
    while time.time() - start < timeout:
        rc2, out2, err2 = run_cmd(["kubectl", "get", "ns", ns, "-o", "json"], timeout=10)
        if rc2 == 0 and out2:
            try:
                j = json.loads(out2)
                phase = j.get("status", {}).get("phase", "")
                if phase == "Active":
                    LOG.info("namespace present", extra={"namespace": ns, "phase": phase})
                    return
            except Exception:
                pass
        time.sleep(1)
    raise RuntimeError(f"namespace {ns} did not reach Active state within {timeout}s")

def create_or_update_secret_from_env(ns: str, secret_name: str, mapping: Dict[str, str]) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to create secrets")
    # Build kubectl secret create command (dry-run) then apply via stdin
    args = ["kubectl", "create", "secret", "generic", secret_name]
    for key, val in mapping.items():
        args.append(f"--from-literal={key}={val}")
    args.extend(["--dry-run=client", "-o", "yaml", "-n", ns])
    rc, out, err = run_cmd(args, timeout=20)
    if rc != 0:
        LOG.error("failed to build secret YAML", extra={"secret": secret_name, "err": err or out})
        raise RuntimeError(f"failed to create secret manifest: {err or out}")
    # apply the secret YAML via kubectl apply -f -
    proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        LOG.error("failed to apply secret", extra={"secret": secret_name, "stderr": stderr})
        raise RuntimeError(f"failed to apply secret {secret_name}: {stderr}")
    # label the secret as managed-by
    rc2, out2, err2 = run_cmd(["kubectl", "label", "secret", secret_name, "managed-by=dashboards.py", "-n", ns, "--overwrite"], timeout=10)
    if rc2 != 0:
        LOG.warning("failed to label secret", extra={"secret": secret_name, "err": err2 or out2})

def resource_has_managed_label(ns: str, kind: str, name: str) -> bool:
    rc, out, err = run_cmd(["kubectl", "-n", ns, "get", kind, name, "-o", "jsonpath={.metadata.labels.managed-by}"], timeout=10)
    if rc != 0:
        return False
    return (out.strip() == "dashboards.py")

def deployment_exists(namespace: str, name: str) -> Optional[Dict[str, Any]]:
    rc, out, err = run_cmd(["kubectl", "-n", namespace, "get", "deployment", name, "-o", "json"], timeout=10)
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None

def safe_apply_deployment(deployment_path: Path, env: Dict[str, str]) -> None:
    ns = env["GRAFANA_NAMESPACE"]
    name = "grafana"
    existing = deployment_exists(ns, name)
    intended = yaml.safe_load(deployment_path.read_text(encoding="utf-8"))
    intended_selector = intended.get("spec", {}).get("selector", {}).get("matchLabels", {}) or {}
    if existing is None:
        LOG.info("No existing grafana deployment found; creating", extra={"namespace": ns})
        kubectl_apply_yaml(deployment_path)
        return
    labels = existing.get("metadata", {}).get("labels", {}) or {}
    if labels.get("managed-by") != "dashboards.py":
        LOG.warning("existing grafana deployment is not managed-by=dashboards.py; skipping apply to avoid mutating another installation", extra={"namespace": ns})
        return
    existing_selector = existing.get("spec", {}).get("selector", {}).get("matchLabels", {}) or {}
    if existing_selector != intended_selector:
        LOG.error("deployment selector mismatch; selector is immutable. Skipping apply.", extra={"namespace": ns, "existing_selector": existing_selector, "intended_selector": intended_selector})
        return
    kubectl_apply_yaml(deployment_path)

# ---------- CLI actions (only --apply and --delete) ----------
def apply_action() -> None:
    LOG.info("apply started")
    env = load_env()
    validate_env(env)
    try:
        validate_grafana_datasource(env)
    except Exception as e:
        LOG.warning("Grafana API validation skipped/failed: %s", str(e))
    try:
        check_recording_rules(env)
    except Exception as e:
        LOG.warning("recording rule checks had issues: %s", str(e))
    paths = render_all(env)

    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    # create/ensure namespace
    LOG.info("ensuring namespace exists (create/apply)", extra={"namespace": ns})
    ensure_namespace(ns)

    # secrets (imperative, never written to disk)
    LOG.info("creating/updating grafana-admin-secret from environment", extra={"namespace": ns})
    create_or_update_secret_from_env(ns, "grafana-admin-secret", {"admin-user": env["GRAFANA_ADMIN_USER"], "admin-password": env["GRAFANA_ADMIN_PASSWORD"]})

    # apply configmaps in deterministic order
    if paths.get("datasources_cm"):
        LOG.info("applying datasources configmap", extra={"file": str(paths['datasources_cm'])})
        kubectl_apply_yaml(paths["datasources_cm"])
    if paths.get("provisioning_cm"):
        LOG.info("applying provisioning provider configmap", extra={"file": str(paths['provisioning_cm'])})
        kubectl_apply_yaml(paths["provisioning_cm"])
    if paths.get("dashboards_cm"):
        LOG.info("applying dashboards configmap", extra={"file": str(paths['dashboards_cm'])})
        kubectl_apply_yaml(paths["dashboards_cm"])

    # apply service
    svc = OUT_DIR / "grafana-service.yaml"
    if svc.exists():
        LOG.info("applying service", extra={"file": str(svc)})
        kubectl_apply_yaml(svc)

    # optional PVC
    if coerce_bool(env.get("GRAFANA_USE_PVC","false")):
        pvc_path = OUT_DIR / "grafana-pvc.yaml"
        if pvc_path.exists():
            LOG.info("applying pvc", extra={"file": str(pvc_path)})
            kubectl_apply_yaml(pvc_path)

    # optional manage deployment (safe)
    if coerce_bool(env.get("GRAFANA_MANAGE_DEPLOYMENT","false")):
        dep = OUT_DIR / "grafana-deployment.yaml"
        if dep.exists():
            LOG.info("safely applying deployment", extra={"file": str(dep)})
            safe_apply_deployment(dep, env)
            # wait for rollout
            LOG.info("waiting for grafana deployment rollout (best-effort, timeout=120s)")
            rc, out, err = run_cmd(["kubectl", "-n", env["GRAFANA_NAMESPACE"], "rollout", "status", "deployment/grafana", "--timeout=120s"], timeout=125)
            if rc != 0:
                LOG.warning("rollout status check failed or timed out", extra={"stdout": out, "stderr": err})
        else:
            LOG.info("deployment manifest not present; skipping deployment apply")
    else:
        LOG.info("management of grafana deployment skipped (GRAFANA_MANAGE_DEPLOYMENT=false)", extra={"manage_deployment": "False"})

    LOG.info("apply complete; generator-owned resources applied. Re-run is idempotent.")

def delete_action() -> None:
    LOG.info("delete started")
    env = load_env()
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    # delete non-cluster-critical resources first
    yaml_files = [
        OUT_DIR / "grafana-dashboards-configmap.yaml",
        OUT_DIR / "grafana-provisioning-configmap.yaml",
        OUT_DIR / "grafana-datasources-configmap.yaml",
    ]
    for p in yaml_files:
        if p.exists():
            LOG.info("kubectl delete", extra={"file": str(p)})
            kubectl_delete_yaml(p)
    # deployment/service/pvc: only delete if resource exists and managed-by label present
    for p in [OUT_DIR / "grafana-deployment.yaml", OUT_DIR / "grafana-service.yaml", OUT_DIR / "grafana-pvc.yaml"]:
        if p.exists():
            kind = ("deployment" if "deployment" in p.name else ("service" if "service" in p.name else "pvc"))
            # check resource existence
            name = "grafana"
            has_label = resource_has_managed_label(ns, kind, name)
            if has_label:
                LOG.info("kubectl delete (managed) file", extra={"file": str(p)})
                kubectl_delete_yaml(p)
            else:
                LOG.warning("skipping delete of %s because resource is not labeled managed-by=dashboards.py or not present", p.name)
    # delete secret managed-by label
    rc, out, err = run_cmd(["kubectl", "-n", ns, "get", "secret", "grafana-admin-secret", "-o", "jsonpath={.metadata.labels.managed-by}"], timeout=10)
    if rc == 0 and out.strip() == "dashboards.py":
        LOG.info("deleting grafana-admin-secret (managed-by dashboards.py)")
        run_cmd(["kubectl", "delete", "secret", "grafana-admin-secret", "-n", ns, "--ignore-not-found"], timeout=10)
    else:
        LOG.info("not deleting grafana-admin-secret: not labeled managed-by=dashboards.py or not found")
    # remove local manifests dir
    try:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
            LOG.info("removed manifest directory", extra={"path": str(OUT_DIR)})
        else:
            LOG.warning("manifest directory not present; nothing to delete locally", extra={"path": str(OUT_DIR)})
    except Exception as e:
        LOG.warning("failed to remove manifest directory: %s", e)
    LOG.info("delete complete")

# ---------- arg parsing (only --apply and --delete) ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/apply/delete Grafana dashboards + optional Grafana deployment (no Helm). Only --apply or --delete allowed.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        if args.apply:
            apply_action(); return
        if args.delete:
            delete_action(); return
    except Exception as e:
        LOG.error("ERROR: %s", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
