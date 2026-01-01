#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

class StructuredFormatter(logging.Formatter):
    def __init__(self, color_map: Dict[str, str]):
        super().__init__()
        self.color_map = color_map
        self.blacklist = {
            "name", "msg", "args", "levelno", "levelname", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process"
        }
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        level = record.levelname
        msg = record.getMessage()
        payload: Dict[str, Any] = {
            "ts": ts,
            "level": level,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "message": msg,
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in self.blacklist and k != "message"}
        safe_extras: Dict[str, Any] = {}
        for k, v in extras.items():
            try:
                json.dumps(v)
                safe_extras[k] = v
            except Exception:
                safe_extras[k] = str(v)
        payload.update(safe_extras)
        json_part = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        color = self.color_map.get(level, "")
        human = f"{color}{level}\x1b[0m {msg}"
        return f"{json_part} {human}"

ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}
LEVEL_TO_INT = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR}

def init_logger() -> logging.Logger:
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    if raw not in ALLOWED_LOG_LEVELS:
        sys.stderr.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "level": "ERROR", "message": f"unsupported LOG_LEVEL '{raw}'"}) + "\n")
        sys.exit(2)
    logger = logging.getLogger("dashboards_generator")
    logger.setLevel(LEVEL_TO_INT[raw])
    ch = logging.StreamHandler(stream=sys.stdout)
    color_map = {"DEBUG": "\x1b[37m", "INFO": "\x1b[32m", "WARN": "\x1b[33m", "ERROR": "\x1b[31m"}
    ch.setFormatter(StructuredFormatter(color_map))
    logger.handlers = []
    logger.addHandler(ch)
    return logger

LOG = init_logger()

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "infra" / "manifests" / "dashboards"

DEFAULTS: Dict[str, str] = {
    "GRAFANA_DATASOURCE": "ClickHouse",
    "DEFAULT_NAMESPACE": "monitoring",
    "SLO_SUCCESS_TARGET": "0.999",
    "SLO_LATENCY_QUANTILE": "0.95",
    "RETRIEVER_LATENCY_THRESHOLD_SECONDS": "0.5",
    "QDRANT_LATENCY_THRESHOLD_SECONDS": "0.8",
    "GRAFANA_PROVISIONING_NAMESPACE": "monitoring",
    "GRAFANA_DASHBOARD_UID_PREFIX": "platform-",
    "CLICKHOUSE_DATASOURCE_NAME": "ClickHouse",
    "RUNBOOK_RETRIEVER": "https://your.runbooks/retriever#slo",
    "RUNBOOK_QDRANT": "https://your.runbooks/qdrant#slo",
    "RUNBOOK_INGESTION": "https://your.runbooks/ingestion#guide",
}

DASHBOARD_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "service-retriever": {
        "dashboard": {
            "id": None,
            "uid": "__UID__retriever",
            "title": "Service Overview — retriever",
            "templating": {
                "list": [
                    {"type": "query", "name": "service", "query": "retriever", "current": {"text": "retriever", "value": "retriever"}},
                    {"type": "query", "name": "namespace", "query": "__DEFAULT_NAMESPACE__", "current": {"text": "__DEFAULT_NAMESPACE__", "value": "__DEFAULT_NAMESPACE__"}}
                ]
            },
            "panels": [
                {"type": "text", "title": "Header", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 0}, "options": {"content": "Owner: platform\nRunbook: __RUNBOOK_RETRIEVER__\nSLO target: __SLO_SUCCESS_TARGET__\nLatency threshold (p95): __RETRIEVER_LATENCY_THRESHOLD_SECONDS__s"}},
                {"type": "graph", "title": "P95 Latency", "targets": [{"expr": "histogram_quantile(__SLO_LATENCY_QUANTILE__, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))"}], "gridPos": {"h": 8, "w": 12, "x": 0, "y": 3}},
                {"type": "graph", "title": "Error rate", "targets": [{"expr": "sum(rate(retrieval_errors_total[5m])) / sum(rate(retrieval_requests_total[5m]))"}], "gridPos": {"h": 8, "w": 12, "x": 12, "y": 3}},
                {"type": "stat", "title": "Service readiness", "targets": [{"expr": "avg(service_ready{service=\"retrieval\"})"}], "gridPos": {"h": 3, "w": 12, "x": 0, "y": 11}},
                {"type": "stat", "title": "Requests/s", "targets": [{"expr": "sum(rate(retrieval_requests_total[1m]))"}], "gridPos": {"h": 3, "w": 12, "x": 12, "y": 11}},
                {"type": "text", "title": "Logs", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 14}, "options": {"content": "Click 'Open Logs' to inspect ClickHouse logs for this service."}, "links": [{"title": "Open Logs", "url": "/explore?left={\"datasource\":\"__GRAFANA_DATASOURCE__\",\"queries\":[{\"refId\":\"A\",\"sql\":\"__CLICKHOUSE_SQL__\"}],\"range\":{\"from\":\"$__from\",\"to\":\"$__to\"}}"}]}
            ],
            "schemaVersion": 30,
            "version": 1
        }
    },
    "service-qdrant": {
        "dashboard": {
            "id": None,
            "uid": "__UID__qdrant",
            "title": "Service Overview — qdrant",
            "templating": {
                "list": [
                    {"type": "query", "name": "service", "query": "qdrant", "current": {"text": "qdrant", "value": "qdrant"}},
                    {"type": "query", "name": "namespace", "query": "__DEFAULT_NAMESPACE__", "current": {"text": "__DEFAULT_NAMESPACE__", "value": "__DEFAULT_NAMESPACE__"}}
                ]
            },
            "panels": [
                {"type": "text", "title": "Header", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 0}, "options": {"content": "Owner: platform\nRunbook: __RUNBOOK_QDRANT__\nSLO target: __SLO_SUCCESS_TARGET__\nLatency threshold (p95): __QDRANT_LATENCY_THRESHOLD_SECONDS__s"}},
                {"type": "graph", "title": "Qdrant P95 Latency", "targets": [{"expr": "histogram_quantile(__SLO_LATENCY_QUANTILE__, sum(rate(qdrant_query_duration_seconds_bucket[5m])) by (le))"}], "gridPos": {"h": 8, "w": 12, "x": 0, "y": 3}},
                {"type": "graph", "title": "Qdrant Queries/s", "targets": [{"expr": "sum(rate(qdrant_query_total[1m]))"}], "gridPos": {"h": 8, "w": 12, "x": 12, "y": 3}},
                {"type": "stat", "title": "Collections", "targets": [{"expr": "collections_total"}], "gridPos": {"h": 3, "w": 12, "x": 0, "y": 11}},
                {"type": "stat", "title": "Dead replicas", "targets": [{"expr": "max(collection_dead_replicas)"}], "gridPos": {"h": 3, "w": 12, "x": 12, "y": 11}},
                {"type": "text", "title": "Logs", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 14}, "options": {"content": "Click 'Open Logs' to inspect ClickHouse logs for this service."}, "links": [{"title": "Open Logs", "url": "/explore?left={\"datasource\":\"__GRAFANA_DATASOURCE__\",\"queries\":[{\"refId\":\"A\",\"sql\":\"__CLICKHOUSE_SQL__\"}],\"range\":{\"from\":\"$__from\",\"to\":\"$__to\"}}"}]}
            ],
            "schemaVersion": 30,
            "version": 1
        }
    },
    "ingestion-health": {
        "dashboard": {
            "id": None,
            "uid": "__UID__ingestion",
            "title": "Ingestion Health",
            "templating": {"list": [{"type": "query", "name": "namespace", "query": "__DEFAULT_NAMESPACE__", "current": {"text": "__DEFAULT_NAMESPACE__", "value": "__DEFAULT_NAMESPACE__"}}]},
            "panels": [
                {"type": "graph", "title": "vmagent discovery objects", "targets": [{"expr": "vm_promscrape_discovery_kubernetes_objects{role=\"pod\"}"}], "gridPos": {"h": 6, "w": 24, "x": 0, "y": 0}},
                {"type": "graph", "title": "remote-write bytes (5m)", "targets": [{"expr": "increase(vm_persistentqueue_bytes_written_total[5m])"}], "gridPos": {"h": 6, "w": 24, "x": 0, "y": 6}},
                {"type": "graph", "title": "vmagent pending queue", "targets": [{"expr": "vm_persistentqueue_bytes_pending"}], "gridPos": {"h": 6, "w": 24, "x": 0, "y": 12}},
                {"type": "text", "title": "Header", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 18}, "options": {"content": "Runbook: __RUNBOOK_INGESTION__"}}
            ],
            "schemaVersion": 30,
            "version": 1
        }
    }
}

CLICKHOUSE_SQL_TEMPLATE = "SELECT ts, level, message, fields FROM logs.kube_logs WHERE service = '$service' AND namespace = '$namespace' AND ts BETWEEN toDateTime64($__from / 1000, 3) AND toDateTime64($__to / 1000, 3) ORDER BY ts DESC LIMIT 500"

PLACEHOLDER_RE = re.compile(r"__([A-Z0-9_]+)__")

def run_cmd(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
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

def load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for k, dv in DEFAULTS.items():
        v = os.getenv(k)
        env[k] = v if v is not None else dv
        if v is None:
            LOG.debug("env default used", extra={"env": k, "default": dv})
        else:
            LOG.debug("env loaded", extra={"env": k})
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
        LOG.error("invalid SLO_LATENCY_QUANTILE", extra={"value": env.get("SLO_LATENCY_QUANTILE")})
        raise RuntimeError("SLO_LATENCY_QUANTILE must be '0.95' or '0.99'")

def render_dashboard_template(template: Dict[str, Any], values: Dict[str, str]) -> Dict[str, Any]:
    text = json.dumps(template, separators=(",", ":"), ensure_ascii=False)
    keys = sorted(set(PLACEHOLDER_RE.findall(text)), key=lambda s: (-len(s), s))
    for k in keys:
        token = f"__{k}__"
        val = values.get(k)
        if val is None:
            LOG.debug("no substitution value for token", extra={"token": token})
            continue
        text = text.replace(token, val)
    leftover = PLACEHOLDER_RE.findall(text)
    if leftover:
        raise RuntimeError(f"leftover placeholders after render: {leftover}")
    obj = json.loads(text)
    return obj

def build_provisioning_cm(dashboards: Dict[str, Dict[str, Any]], env: Dict[str, str]) -> Dict[str, Any]:
    cm_data = {}
    for name, db in dashboards.items():
        key = f"{name}.json"
        cm_data[key] = json.dumps(db, separators=(",", ":"), ensure_ascii=False)
    ns = env["GRAFANA_PROVISIONING_NAMESPACE"]
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "grafana-provisioning-dashboards", "namespace": ns}, "data": cm_data}
    return cm

def render_all(env: Dict[str, str]) -> Dict[str, Path]:
    validate_env(env)
    values = {k: v for k, v in env.items()}
    values["CLICKHOUSE_SQL"] = CLICKHOUSE_SQL_TEMPLATE.replace("'", "\\'")
    values["GRAFANA_DATASOURCE"] = env["GRAFANA_DATASOURCE"]
    values["DEFAULT_NAMESPACE"] = env["DEFAULT_NAMESPACE"]
    values["RUNBOOK_RETRIEVER"] = env["RUNBOOK_RETRIEVER"]
    values["RUNBOOK_QDRANT"] = env["RUNBOOK_QDRANT"]
    values["RUNBOOK_INGESTION"] = env["RUNBOOK_INGESTION"]
    rendered: Dict[str, Dict[str, Any]] = {}
    for key, tpl in DASHBOARD_TEMPLATES.items():
        db = render_dashboard_template(tpl["dashboard"], values)
        uid_prefix = env.get("GRAFANA_DASHBOARD_UID_PREFIX", "")
        if "uid" in db and isinstance(db["uid"], str):
            db["uid"] = db["uid"].replace("__UID__", uid_prefix)
        rendered[key] = db
    cm = build_provisioning_cm(rendered, env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, Path] = {}
    for name, db in rendered.items():
        p = OUT_DIR / f"{name}.json"
        atomic_write(p, json.dumps(db, indent=2, ensure_ascii=False))
        out_paths[name] = p
    cm_path = OUT_DIR / "grafana-provisioning-configmap.yaml"
    atomic_write(cm_path, yaml.safe_dump(cm, sort_keys=False))
    sql_path = OUT_DIR / "clickhouse-explore-sql.txt"
    atomic_write(sql_path, CLICKHOUSE_SQL_TEMPLATE)
    out_paths["provisioning_cm"] = cm_path
    out_paths["clickhouse_sql"] = sql_path
    LOG.info("render complete", extra={"out_dir": str(OUT_DIR), "files": [str(p) for p in out_paths.values()]})
    return out_paths

def json_validate(path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
    except Exception as e:
        LOG.error("dashboard JSON invalid", extra={"path": str(path), "error": str(e)})
        raise

def kubectl_apply(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to apply manifests")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", str(path)], timeout=60)
    if rc != 0:
        LOG.error("kubectl apply failed", extra={"file": str(path), "stdout": out, "stderr": err})
        raise RuntimeError(f"kubectl apply failed for {path}: {err or out}")
    LOG.info("kubectl apply succeeded", extra={"file": str(path)})

def kubectl_delete(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to delete manifests")
    rc, out, err = run_cmd(["kubectl", "delete", "-f", str(path), "--ignore-not-found"], timeout=60)
    if rc != 0:
        LOG.warning("kubectl delete returned non-zero", extra={"file": str(path), "stdout": out, "stderr": err})
    else:
        LOG.info("kubectl delete succeeded", extra={"file": str(path)})

def generate(args: argparse.Namespace) -> None:
    LOG.info("generate started")
    env = load_env()
    render_all(env)

def validate(args: argparse.Namespace) -> None:
    LOG.info("validate started")
    env = load_env()
    paths = render_all(env)
    for name, p in paths.items():
        if p.suffix == ".json":
            json_validate(p)
    LOG.info("validate complete")

def apply(args: argparse.Namespace) -> None:
    LOG.info("apply started")
    env = load_env()
    paths = render_all(env)
    cm = paths.get("provisioning_cm")
    if cm:
        kubectl_apply(cm)
    LOG.info("apply complete; Grafana will pick up dashboards via provisioning in the monitored namespace")

def delete(args: argparse.Namespace) -> None:
    LOG.info("delete started")
    if not args.confirm:
        raise RuntimeError("--confirm required to delete")
    files = list(OUT_DIR.glob("*"))
    for p in files:
        if p.exists() and shutil.which("kubectl"):
            try:
                kubectl_delete(p)
            except Exception as e:
                LOG.warning("kubectl delete failed for %s: %s", p, e)
    try:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
            LOG.info("removed manifest directory", extra={"path": str(OUT_DIR)})
    except Exception as e:
        LOG.warning("failed to remove manifest directory: %s", e)
    LOG.info("delete complete")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/validate/apply/delete Grafana dashboards (provisioning)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--validate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true", help="required for --delete")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        if args.generate:
            generate(args); return
        if args.validate:
            validate(args); return
        if args.apply:
            apply(args); return
        if args.delete:
            delete(args); return
    except Exception as e:
        LOG.error("ERROR: %s", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
