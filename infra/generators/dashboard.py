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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import yaml

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

LOG = logging.getLogger("dashboard_generator")
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(SimpleJSONFormatter())
LOG.handlers = []
LOG.addHandler(ch)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "infra" / "manifests" / "dashboards"

DEFAULTS: Dict[str, str] = {
    "GRAFANA_PROVISIONING_NAMESPACE": "monitoring",
    "GRAFANA_DASHBOARD_UID_PREFIX": "platform-",
    "DASHBOARD_SERVICES": "retriever,qdrant",
    "METRICS_DATASOURCE": "VictoriaMetrics",
    "METRICS_DATASOURCE_URL": "http://victoria-metrics.monitoring.svc:8428",
    "CLICKHOUSE_DATASOURCE": "ClickHouse",
    "CLICKHOUSE_URL": "http://clickhouse.clickhouse.svc:8123",
    "DEFAULT_NAMESPACE": "monitoring",
    "SLO_SUCCESS_TARGET": "0.999",
    "SLO_LATENCY_QUANTILE": "0.95",
    "CI": "false",
    "GRAFANA_MANAGE_DEPLOYMENT": "false",
    "GRAFANA_ADMIN_USER": "",
    "GRAFANA_ADMIN_PASSWORD": "",
    "MAX_PANELS_PER_DASHBOARD": "48",
}

CLICKHOUSE_SQL_TEMPLATE = (
    "SELECT ts, level, message, fields FROM logs.kube_logs "
    "WHERE service = '$service' "
    "AND namespace = '$namespace' "
    "AND ts BETWEEN toDateTime64($__from / 1000, 3) AND toDateTime64($__to / 1000, 3) "
    "ORDER BY ts DESC LIMIT 500"
)

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
    try:
        sst = float(env["SLO_SUCCESS_TARGET"])
        if not (0.0 < sst < 1.0):
            raise ValueError()
    except Exception:
        LOG.error("invalid SLO_SUCCESS_TARGET", extra={"value": env.get("SLO_SUCCESS_TARGET")})
        raise RuntimeError("SLO_SUCCESS_TARGET must be a float between 0 and 1, e.g. 0.999")
    if env["SLO_LATENCY_QUANTILE"] not in ("0.95", "0.99"):
        LOG.warning("SLO_LATENCY_QUANTILE is unexpected, accepting value but recommend 0.95 or 0.99", extra={"value": env.get("SLO_LATENCY_QUANTILE")})
    if coerce_bool(env.get("GRAFANA_MANAGE_DEPLOYMENT")):
        if not env.get("GRAFANA_ADMIN_USER"):
            raise RuntimeError("GRAFANA_ADMIN_USER required when GRAFANA_MANAGE_DEPLOYMENT=true")
        if not env.get("GRAFANA_ADMIN_PASSWORD"):
            raise RuntimeError("GRAFANA_ADMIN_PASSWORD required when GRAFANA_MANAGE_DEPLOYMENT=true")

def render_clickhouse_left(service: str, env: Dict[str, str]) -> str:
    template = os.getenv("CLICKHOUSE_SQL_TEMPLATE", CLICKHOUSE_SQL_TEMPLATE)
    left_obj = {"datasource": env["CLICKHOUSE_DATASOURCE"], "queries": [{"refId": "A", "sql": template.replace("$service", service).replace("$namespace", env.get("DEFAULT_NAMESPACE","monitoring"))}], "range": {"from": "$__from", "to": "$__to"}}
    raw = json.dumps(left_obj, separators=(",", ":"), ensure_ascii=False)
    return urllib.parse.quote_plus(raw)

def make_metric_panel(title: str, expr: str, datasource: str, gridPos: Dict[str, int], refId: str, panel_type: str = "timeseries") -> Dict[str, Any]:
    p = {"type": panel_type, "title": title, "datasource": datasource, "targets": [{"expr": expr, "refId": refId}], "gridPos": gridPos}
    return p

def make_stat_panel(title: str, expr: str, datasource: str, gridPos: Dict[str, int], refId: str) -> Dict[str, Any]:
    return make_metric_panel(title, expr, datasource, gridPos, refId, panel_type="stat")

def build_platform_overview(env: Dict[str, str]) -> Dict[str, Any]:
    metrics_ds = env["METRICS_DATASOURCE"]
    slo_q = env["SLO_LATENCY_QUANTILE"]
    services = [s.strip() for s in (env.get("DASHBOARD_SERVICES","") or "").split(",") if s.strip()]
    if not services:
        services = ["retriever","qdrant"]
    panels: List[Dict[str, Any]] = []
    next_ref_ord = ord("A")
    next_panel_id = 1

    header = {"type": "text", "title": "Platform Overview", "gridPos": {"h": 2, "w": 24, "x": 0, "y": 0}, "options": {"content": "Platform overview. Use the service selector to focus."}}
    header["id"] = next_panel_id; next_panel_id += 1
    panels.append(header)

    p95_expr = ("histogram_quantile(" + slo_q + ", sum(rate({__name__=~\".*_request_duration_seconds_bucket|.*_query_duration_seconds_bucket\",service=~\"$service\"}[5m])) by (le))")
    p95 = make_metric_panel("P95 Latency", p95_expr, metrics_ds, {"h": 8, "w": 12, "x": 0, "y": 2}, chr(next_ref_ord))
    next_ref_ord += 1
    p95["id"] = next_panel_id; next_panel_id += 1
    p95["repeat"] = "service"
    panels.append(p95)

    err_expr = ("sum(rate({__name__=~\".*_errors_total|.*_error_total\",service=~\"$service\"}[5m])) / max(sum(rate({__name__=~\".*_requests_total|.*_request_total\",service=~\"$service\"}[5m])),1)")
    err = make_metric_panel("Error rate", err_expr, metrics_ds, {"h": 8, "w": 12, "x": 12, "y": 2}, chr(next_ref_ord))
    next_ref_ord += 1
    err["id"] = next_panel_id; next_panel_id += 1
    err["repeat"] = "service"
    panels.append(err)

    ready_expr = 'avg(service_ready{service=~"$service"})'
    ready = make_stat_panel("Service readiness", ready_expr, metrics_ds, {"h": 3, "w": 12, "x": 0, "y": 10}, chr(next_ref_ord))
    next_ref_ord += 1
    ready["id"] = next_panel_id; next_panel_id += 1
    ready["repeat"] = "service"
    panels.append(ready)

    rps_expr = 'sum(rate({__name__=~".*_requests_total|.*_request_total",service=~"$service"}[1m]))'
    rps = make_stat_panel("Requests/s", rps_expr, metrics_ds, {"h": 3, "w": 12, "x": 12, "y": 10}, chr(next_ref_ord))
    next_ref_ord += 1
    rps["id"] = next_panel_id; next_panel_id += 1
    rps["repeat"] = "service"
    panels.append(rps)

    left_enc = render_clickhouse_left("${service}", env)
    logs_panel = {"type": "text", "title": "Logs", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 13}, "options": {"content": "Open logs for selected service"}, "links": [{"title": "Open Logs", "url": f"/explore?left={left_enc}"}]}
    logs_panel["id"] = next_panel_id; next_panel_id += 1
    panels.append(logs_panel)

    templating_list = []
    svc_options = []
    for s in services:
        svc_options.append({"text": s, "value": s})
    svc_current = services[0] if services else ""
    svc_var = {"type": "custom", "name": "service", "label": "Service", "options": svc_options, "current": {"text": svc_current, "value": svc_current}, "multi": True, "includeAll": False}
    templating_list.append(svc_var)
    ns_var = {"type": "custom", "name": "namespace", "label": "Namespace", "options": [{"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}], "current": {"text": env.get("DEFAULT_NAMESPACE","monitoring"), "value": env.get("DEFAULT_NAMESPACE","monitoring")}, "multi": False, "includeAll": False}
    templating_list.append(ns_var)

    uid = f"{env.get('GRAFANA_DASHBOARD_UID_PREFIX','platform-')}platform-overview"
    dash = {"id": None, "uid": uid, "title": "Platform Overview", "templating": {"list": templating_list}, "panels": panels, "schemaVersion": 36, "version": 1}
    dash["_meta"] = {"generator": "dashboard.py", "rendered_at": datetime.utcnow().isoformat() + "Z"}
    return dash

def build_dashboards_configmap(dashboards: Dict[str, Dict[str, Any]], env: Dict[str, str]) -> Dict[str, Any]:
    data: Dict[str, str] = {}
    for name, db in dashboards.items():
        key = f"{name}.json"
        data[key] = json.dumps(db, separators=(",", ":"), ensure_ascii=False)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-dashboards", "namespace": ns, "labels": {"managed-by": "dashboard.py"}}, "data": data}

def build_provisioning_provider_cm(env: Dict[str, str]) -> Dict[str, Any]:
    provider_item = {"name": "platform-dashboards", "orgId": 1, "folder": "Platform", "type": "file", "disableDeletion": False, "options": {"path": "/var/lib/grafana/dashboards"}}
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    providers_yaml = yaml.safe_dump([provider_item], sort_keys=False)
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-provisioning", "namespace": ns, "labels": {"managed-by": "dashboard.py"}}, "data": {"providers.yaml": providers_yaml}}
    return cm

def build_datasources_cm(env: Dict[str, str]) -> Dict[str, Any]:
    ds: List[Dict[str, Any]] = []
    ds.append({"name": env["METRICS_DATASOURCE"], "type": "prometheus", "access": "proxy", "url": env["METRICS_DATASOURCE_URL"], "isDefault": True, "editable": False})
    if env.get("CLICKHOUSE_URL"):
        ds.append({"name": env["CLICKHOUSE_DATASOURCE"], "type": "clickhouse", "access": "proxy", "url": env["CLICKHOUSE_URL"], "isDefault": False, "editable": False})
    provider = {"apiVersion": 1, "datasources": ds}
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-datasources", "namespace": ns, "labels": {"managed-by": "dashboard.py"}}, "data": {"datasources.yaml": yaml.safe_dump(provider, sort_keys=False)}}

def render_all(env: Dict[str, str]) -> Dict[str, Path]:
    validate_env(env)
    rendered: Dict[str, Dict[str, Any]] = {}
    platform = build_platform_overview(env)
    rendered["platform-overview"] = platform
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
    LOG.info("rendering manifests to disk", extra={"out_dir": str(OUT_DIR), "files": [str(p) for p in out_paths.values()]})
    out_paths["dashboards_cm"] = OUT_DIR / "grafana-dashboards-configmap.yaml"
    out_paths["provisioning_cm"] = OUT_DIR / "grafana-provisioning-configmap.yaml"
    out_paths["datasources_cm"] = OUT_DIR / "grafana-datasources-configmap.yaml"
    return out_paths

def kubectl_apply_yaml(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to apply manifests")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", str(path)], timeout=60)
    if rc != 0:
        LOG.error("kubectl apply failed", extra={"file": str(path), "stdout": out, "stderr": err})
        raise RuntimeError(f"kubectl apply failed for {path}: {err or out}")
    LOG.info("kubectl apply succeeded", extra={"file": str(path)})

def create_or_update_secret_from_env(ns: str, secret_name: str, mapping: Dict[str, str]) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to create secrets")
    args = ["kubectl", "create", "secret", "generic", secret_name]
    for key, val in mapping.items():
        args.append(f"--from-literal={key}={val}")
    args.extend(["--dry-run=client", "-o", "yaml", "-n", ns])
    rc, out, err = run_cmd(args, timeout=20)
    if rc != 0:
        LOG.error("failed to build secret YAML", extra={"secret": secret_name, "err": err or out})
        raise RuntimeError(f"failed to create secret manifest: {err or out}")
    proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(out)
    if proc.returncode != 0:
        LOG.error("failed to apply secret", extra={"secret": secret_name, "stderr": stderr})
        raise RuntimeError(f"failed to apply secret {secret_name}: {stderr}")
    rc2, out2, err2 = run_cmd(["kubectl", "label", "secret", secret_name, "managed-by=dashboard.py", "-n", ns, "--overwrite"], timeout=10)
    if rc2 != 0:
        LOG.warning("failed to label secret", extra={"secret": secret_name, "err": err2 or out2})

def apply_action() -> None:
    LOG.info("apply started")
    env = load_env()
    validate_env(env)
    paths = render_all(env)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    LOG.info("ensuring namespace exists (create/apply)", extra={"namespace": ns})
    if shutil.which("kubectl"):
        rc, out, err = run_cmd(["kubectl", "create", "namespace", ns, "--dry-run=client", "-o", "yaml"], timeout=10)
        if rc == 0:
            proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate(out)
            if proc.returncode != 0:
                raise RuntimeError(f"failed to create/apply namespace {ns}: {stderr}")
            LOG.info("namespace ensured", extra={"namespace": ns})
    if coerce_bool(env.get("GRAFANA_MANAGE_DEPLOYMENT")):
        LOG.info("creating/updating grafana-admin-secret from environment", extra={"namespace": ns})
        create_or_update_secret_from_env(ns, "grafana-admin-secret", {"admin-user": env["GRAFANA_ADMIN_USER"], "admin-password": env["GRAFANA_ADMIN_PASSWORD"]})
    if paths.get("datasources_cm"):
        LOG.info("applying datasources configmap", extra={"file": str(paths['datasources_cm'])})
        kubectl_apply_yaml(paths["datasources_cm"])
    if paths.get("provisioning_cm"):
        LOG.info("applying provisioning provider configmap", extra={"file": str(paths['provisioning_cm'])})
        kubectl_apply_yaml(paths["provisioning_cm"])
    if paths.get("dashboards_cm"):
        LOG.info("applying dashboards configmap", extra={"file": str(paths['dashboards_cm'])})
        kubectl_apply_yaml(paths["dashboards_cm"])
    LOG.info("apply complete; generator-owned resources applied. Re-run is idempotent.")

def delete_action() -> None:
    env = load_env()
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    if shutil.which("kubectl"):
        run_cmd(["kubectl", "-n", ns, "delete", "configmap", "grafana-datasources", "--ignore-not-found"], timeout=20)
        run_cmd(["kubectl", "-n", ns, "delete", "configmap", "grafana-provisioning", "--ignore-not-found"], timeout=20)
        run_cmd(["kubectl", "-n", ns, "delete", "configmap", "grafana-dashboards", "--ignore-not-found"], timeout=20)
        if coerce_bool(env.get("GRAFANA_MANAGE_DEPLOYMENT")):
            run_cmd(["kubectl", "-n", ns, "delete", "secret", "grafana-admin-secret", "--ignore-not-found"], timeout=20)
        LOG.info("deleted (or ignored missing) generator-owned resources", extra={"namespace": ns})
    else:
        LOG.warning("kubectl not available; nothing deleted")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/apply Grafana platform overview dashboard (single templated dashboard).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--render", action="store_true")
    g.add_argument("--delete", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        if args.render:
            env = load_env()
            render_all(env)
            return
        if args.apply:
            apply_action()
            return
        if args.delete:
            delete_action()
            return
    except Exception as e:
        LOG.error("ERROR: %s", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
