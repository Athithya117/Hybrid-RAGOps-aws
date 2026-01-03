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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable
import yaml

ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}
LEVEL_TO_INT = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR}

def init_logger() -> logging.Logger:
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    if raw not in ALLOWED_LOG_LEVELS:
        sys.stderr.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "level": "ERROR", "message": f"unsupported LOG_LEVEL '{raw}'"}) + "\n")
        sys.exit(2)
    logger = logging.getLogger("alerting_generator")
    logger.setLevel(LEVEL_TO_INT[raw])
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    return logger

LOG = init_logger()
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "infra" / "manifests" / "alerts"

VM_NAMESPACE = os.getenv("VM_NAMESPACE", "monitoring")
VICTORIA_PORT = os.getenv("VICTORIA_PORT", "8428")
DATASOURCE_URL = os.getenv("DATASOURCE_URL", f"http://victoria-metrics.{VM_NAMESPACE}.svc:{VICTORIA_PORT}")
VMALERT_REMOTE_WRITE_URL = os.getenv("VMALERT_REMOTE_WRITE_URL", f"http://victoria-metrics.{VM_NAMESPACE}.svc.cluster.local:{VICTORIA_PORT}/api/v1/write")
NOTIFIER_URL = os.getenv("NOTIFIER_URL", f"http://alertmanager.{VM_NAMESPACE}.svc:9093")
VMALERT_IMAGE = os.getenv("VMALERT_IMAGE", "victoriametrics/vmalert:v1.132.0")
VMALERT_REPLICAS = os.getenv("VMALERT_REPLICAS", "1")
VMALERT_EVAL_INTERVAL = os.getenv("VMALERT_EVAL_INTERVAL", "30s")
ALERTMANAGER_IMAGE = os.getenv("ALERTMANAGER_IMAGE", "prom/alertmanager:v0.27.0")
ALERTMANAGER_REPLICAS = os.getenv("ALERTMANAGER_REPLICAS", "1")
ALERTMANAGER_RES_CPU = os.getenv("ALERTMANAGER_RES_CPU", "")
ALERTMANAGER_RES_MEM = os.getenv("ALERTMANAGER_RES_MEM", "")
VMALERT_RES_CPU = os.getenv("VMALERT_RES_CPU", "")
VMALERT_RES_MEM = os.getenv("VMALERT_RES_MEM", "")
SLO_SUCCESS_TARGET = os.getenv("SLO_SUCCESS_TARGET", "0.999")
SLO_LATENCY_QUANTILE = os.getenv("SLO_LATENCY_QUANTILE", "0.95")
SLO_FAST_BURN_MULTIPLIER = os.getenv("SLO_FAST_BURN_MULTIPLIER", "2")
SLO_SLOW_BURN_MULTIPLIER = os.getenv("SLO_SLOW_BURN_MULTIPLIER", "1.2")
RETRIEVER_LATENCY_THRESHOLD_SECONDS = os.getenv("RETRIEVER_LATENCY_THRESHOLD_SECONDS", "0.5")
QDRANT_LATENCY_THRESHOLD_SECONDS = os.getenv("QDRANT_LATENCY_THRESHOLD_SECONDS", "0.8")
DEFAULT_WEBHOOK = os.getenv("DEFAULT_WEBHOOK", "")
NOTIFIER_SECRET_NAME = os.getenv("NOTIFIER_SECRET_NAME", "alertmanager-notifiers")
PAGERDUTY_ROUTING_KEY = os.getenv("PAGERDUTY_ROUTING_KEY", "") or os.getenv("PAGERDUTY_INTEGRATION_KEY", "")
ALERTMANAGER_SLACK_WEBHOOK = os.getenv("ALERTMANAGER_SLACK_WEBHOOK", "")
ALERT_DEFAULT_CHANNEL = os.getenv("ALERT_DEFAULT_CHANNEL", "")
CREATE_NOTIFIER_SECRET = os.getenv("CREATE_NOTIFIER_SECRET", "false").lower() in ("1", "true", "yes")
RUNBOOK_BASE_URL = os.getenv("RUNBOOK_BASE_URL", "")

ALERTING_SLACK_SEVERITY_LEVELS = os.getenv("ALERTING_SLACK_SEVERITY_LEVELS", "warning,critical")
ALERTING_PAGING_SEVERITY_LEVELS = os.getenv("ALERTING_PAGING_SEVERITY_LEVELS", "critical")
ALERTING_GROUP_WAIT = os.getenv("ALERTING_GROUP_WAIT", "30s")
ALERTING_GROUP_INTERVAL = os.getenv("ALERTING_GROUP_INTERVAL", "5m")
ALERTING_REPEAT_INTERVAL = os.getenv("ALERTING_REPEAT_INTERVAL", "3h")

def run_cmd(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        LOG.debug("run_cmd finished rc=%s cmd=%s out_len=%d err_len=%d", proc.returncode, " ".join(cmd), len(out), len(err))
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        out = (getattr(e, "stdout", None) or b"").decode("utf-8", errors="replace") if getattr(e, "stdout", None) else ""
        err = (getattr(e, "stderr", None) or b"").decode("utf-8", errors="replace") if getattr(e, "stderr", None) else f"timeout after {timeout}s"
        LOG.error("run_cmd timeout cmd=%s", " ".join(cmd))
        return 124, out.strip(), err.strip()

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, str(path))
    LOG.info("wrote file %s bytes=%d", str(path), len(content))

def is_k8s_manifest(path: Path) -> bool:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return False
    if re.search(r"^\s*apiVersion\s*:", txt, re.M) and re.search(r"^\s*kind\s*:", txt, re.M):
        return True
    return False

def parse_csv_to_list(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    uniq = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return uniq

def validate_inputs() -> None:
    try:
        sst = float(SLO_SUCCESS_TARGET)
        if not (0.0 < sst < 1.0):
            raise ValueError()
    except Exception:
        LOG.error("invalid SLO_SUCCESS_TARGET %s", SLO_SUCCESS_TARGET)
        raise RuntimeError("SLO_SUCCESS_TARGET must be float between 0 and 1, e.g. 0.999")
    if SLO_LATENCY_QUANTILE not in ("0.95", "0.99"):
        LOG.error("invalid SLO_LATENCY_QUANTILE %s", SLO_LATENCY_QUANTILE)
        raise RuntimeError("SLO_LATENCY_QUANTILE must be '0.95' or '0.99'")
    required = {"VMALERT_IMAGE": VMALERT_IMAGE, "DATASOURCE_URL": DATASOURCE_URL, "NOTIFIER_URL": NOTIFIER_URL}
    for k, v in required.items():
        if not v:
            LOG.error("required env missing %s", k)
            raise RuntimeError(f"{k} must be set")
    if RUNBOOK_BASE_URL:
        if not re.match(r"^https?://", RUNBOOK_BASE_URL):
            LOG.error("RUNBOOK_BASE_URL invalid %s", RUNBOOK_BASE_URL)
            raise RuntimeError("RUNBOOK_BASE_URL must be an absolute URL starting with http:// or https://")
    LOG.info("inputs validated")

def alertname_to_kebab(name: str) -> str:
    s1 = re.sub("([a-z0-9])([A-Z])", r"\1-\2", name)
    s2 = re.sub("([A-Z]+)([A-Z][a-z0-9])", r"\1-\2", s1)
    kebab = re.sub(r"[^a-zA-Z0-9\-]+", "-", s2).strip("-").lower()
    return kebab

def runbook_url_for(alert_name: str) -> str:
    base = RUNBOOK_BASE_URL.rstrip("/") if RUNBOOK_BASE_URL else ""
    filename = f"{alertname_to_kebab(alert_name)}.html"
    return f"{base}/{filename}" if base else ""

def build_slo_rules() -> Dict[str, Any]:
    sst = SLO_SUCCESS_TARGET
    sq = SLO_LATENCY_QUANTILE
    fast_mul = SLO_FAST_BURN_MULTIPLIER
    slow_mul = SLO_SLOW_BURN_MULTIPLIER
    groups: List[Dict[str, Any]] = []
    groups.append({
        "name": "recording-rules",
        "rules": [
            {"record": "retrieval_errors_rate_1h", "expr": "sum(rate(retrieval_errors_total[1h]))"},
            {"record": "retrieval_requests_rate_1h", "expr": "sum(rate(retrieval_requests_total[1h]))"},
            {"record": "retrieval_errors_rate_6h", "expr": "sum(rate(retrieval_errors_total[6h]))"},
            {"record": "retrieval_requests_rate_6h", "expr": "sum(rate(retrieval_requests_total[6h]))"},
            {"record": "qdrant_rest_fail_rate_1h", "expr": "sum(rate(rest_responses_fail_total[1h]))"},
            {"record": "qdrant_rest_total_rate_1h", "expr": "sum(rate(rest_responses_total[1h]))"},
        ],
    })
    groups.append({
        "name": "ingestion-truth",
        "rules": [
            {
                "alert": "VmagentDiscoveryEmpty",
                "expr": 'vm_promscrape_discovery_kubernetes_objects{role="pod"} == 0',
                "for": "2m",
                "labels": {"severity": "critical", "plane": "ingestion", "service": "vmagent"},
                "annotations": {
                    "summary": "vmagent pod discovery returned zero objects",
                    "description": "Verify vmagent is running and able to list Kubernetes endpoints. Check vmagent logs for discovery errors and RBAC permissions.",
                    "runbook": runbook_url_for("VmagentDiscoveryEmpty"),
                },
            },
            {
                "alert": "VmagentNoRemoteWrite",
                "expr": "increase(vm_persistentqueue_bytes_written_total[5m]) == 0",
                "for": "5m",
                "labels": {"severity": "critical", "plane": "ingestion", "service": "vmagent"},
                "annotations": {
                    "summary": "vmagent reports no remote-write bytes to Victoria in the last 5m",
                    "description": "Confirm vmagent can establish remote-write connections and that VictoriaMetrics is reachable at DATASOURCE_URL. Check persistent queue metrics and network connectivity.",
                    "runbook": runbook_url_for("VmagentNoRemoteWrite"),
                },
            },
        ],
    })
    groups.append({
        "name": "service-safety",
        "rules": [
            {
                "alert": "RetrieverNotReady",
                "expr": 'service_ready{service="retrieval"} == 0',
                "for": "2m",
                "labels": {"severity": "critical", "plane": "safety", "service": "retriever"},
                "annotations": {
                    "summary": "Retriever service reports not ready",
                    "description": "Check retriever deployment, readiness probes, recent events, and downstream dependencies.",
                    "runbook": runbook_url_for("RetrieverNotReady"),
                },
            },
            {
                "alert": "QdrantDeadReplicas",
                "expr": "collection_dead_replicas > 0",
                "for": "2m",
                "labels": {"severity": "critical", "plane": "safety", "service": "qdrant"},
                "annotations": {
                    "summary": "Qdrant reports dead replicas for at least one collection",
                    "description": "Inspect qdrant cluster health, pod logs, and storage errors. Follow cluster recovery steps in runbook.",
                    "runbook": runbook_url_for("QdrantDeadReplicas"),
                },
            },
            {
                "alert": "QdrantSnapshotStuck",
                "expr": "snapshot_creation_running > 0",
                "for": "30m",
                "labels": {"severity": "warning", "plane": "safety", "service": "qdrant"},
                "annotations": {
                    "summary": "Qdrant snapshot running for > 30m",
                    "description": "Investigate snapshot process and storage backend latency. Consider cancelling or throttling snapshot according to runbook.",
                    "runbook": runbook_url_for("QdrantSnapshotStuck"),
                },
            },
        ],
    })
    groups.append({
        "name": "retriever-slo",
        "rules": [
            {
                "alert": "RetrieverErrorBudgetFastBurn",
                "expr": f"(retrieval_errors_rate_1h / max(retrieval_requests_rate_1h, 1)) / (1 - {sst}) > {fast_mul}",
                "for": "10m",
                "labels": {"severity": "critical", "plane": "slo", "service": "retriever"},
                "annotations": {
                    "summary": "Retriever error budget fast burn (1h)",
                    "description": "Fast burn detected; investigate retriever errors, recent deploys, and backend failures.",
                    "runbook": runbook_url_for("RetrieverErrorBudgetFastBurn"),
                },
            },
            {
                "alert": "RetrieverErrorBudgetSlowBurn",
                "expr": f"(retrieval_errors_rate_6h / max(retrieval_requests_rate_6h, 1)) / (1 - {sst}) > {slow_mul}",
                "for": "30m",
                "labels": {"severity": "warning", "plane": "slo", "service": "retriever"},
                "annotations": {
                    "summary": "Retriever error budget slow burn (6h)",
                    "description": "Slow burn in error budget; review trends and mitigations in runbook.",
                    "runbook": runbook_url_for("RetrieverErrorBudgetSlowBurn"),
                },
            },
            {
                "alert": "RetrieverHighP95Latency",
                "expr": f"histogram_quantile({sq}, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)) > {RETRIEVER_LATENCY_THRESHOLD_SECONDS}",
                "for": "5m",
                "labels": {"severity": "warning", "plane": "slo", "service": "retriever"},
                "annotations": {
                    "summary": "Retriever p95 latency above threshold",
                    "description": "High p95 latency observed; check retriever CPU/memory, GC, and downstream latency.",
                    "runbook": runbook_url_for("RetrieverHighP95Latency"),
                },
            },
        ],
    })
    groups.append({
        "name": "qdrant-slo",
        "rules": [
            {
                "alert": "QdrantErrorBudgetFastBurn",
                "expr": f"(qdrant_rest_fail_rate_1h / max(qdrant_rest_total_rate_1h, 1)) / (1 - {sst}) > {fast_mul}",
                "for": "10m",
                "labels": {"severity": "critical", "plane": "slo", "service": "qdrant"},
                "annotations": {
                    "summary": "Qdrant error budget fast burn (1h)",
                    "description": "High error-rate for Qdrant; inspect cluster health and storage errors.",
                    "runbook": runbook_url_for("QdrantErrorBudgetFastBurn"),
                },
            },
            {
                "alert": "QdrantHighP95Latency",
                "expr": f"histogram_quantile({sq}, sum(rate(rest_responses_duration_seconds_bucket[5m])) by (le)) > {QDRANT_LATENCY_THRESHOLD_SECONDS}",
                "for": "5m",
                "labels": {"severity": "warning", "plane": "slo", "service": "qdrant"},
                "annotations": {
                    "summary": "Qdrant p95 latency above threshold",
                    "description": "Qdrant latency high; check indexing operations, storage I/O, and cluster load.",
                    "runbook": runbook_url_for("QdrantHighP95Latency"),
                },
            },
        ],
    })
    return {"groups": groups}

def build_vmalert_objects(rules_text: str) -> List[Dict[str, Any]]:
    ns = VM_NAMESPACE
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "vmalert-rules", "namespace": ns}, "data": {"slo.rules.yaml": rules_text}}
    replicas = 1
    try:
        replicas = max(1, int(VMALERT_REPLICAS))
    except Exception:
        replicas = 1
    container_port = 8880
    deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "vmalert", "namespace": ns, "labels": {"app": "vmalert"}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "vmalert"}},
            "template": {
                "metadata": {"labels": {"app": "vmalert"}},
                "spec": {
                    "containers": [
                        {
                            "name": "vmalert",
                            "image": VMALERT_IMAGE,
                            "args": [
                                "-rule=/etc/vmalert/slo.rules.yaml",
                                f"-datasource.url={DATASOURCE_URL}",
                                f"-notifier.url={NOTIFIER_URL}",
                                f"-evaluationInterval={VMALERT_EVAL_INTERVAL}",
                                f"-remoteWrite.url={VMALERT_REMOTE_WRITE_URL}",
                            ],
                            "volumeMounts": [{"name": "rules", "mountPath": "/etc/vmalert"}],
                            "ports": [{"containerPort": container_port, "name": "http"}],
                            "readinessProbe": {"httpGet": {"path": "/metrics", "port": container_port}, "initialDelaySeconds": 5, "periodSeconds": 10},
                        }
                    ],
                    "volumes": [{"name": "rules", "configMap": {"name": "vmalert-rules"}}],
                },
            },
        },
    }
    svc = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "vmalert", "namespace": ns}, "spec": {"selector": {"app": "vmalert"}, "ports": [{"name": "http", "port": 8080, "targetPort": container_port}]}}
    return [cm, deploy, svc]

def choose_preferred_receiver(receivers: Iterable[Dict[str, Any]]) -> str:
    names = [r.get("name") for r in receivers]
    if "default" in names:
        return "default"
    if "slack" in names:
        return "slack"
    if "pagerduty" in names:
        return "pagerduty"
    if names:
        return names[0]
    return "default-noop"

def build_alertmanager_cm() -> Dict[str, Any]:
    ns = VM_NAMESPACE
    paging = parse_csv_to_list(ALERTING_PAGING_SEVERITY_LEVELS)
    slack = parse_csv_to_list(ALERTING_SLACK_SEVERITY_LEVELS)
    receivers: List[Dict[str, Any]] = []
    if DEFAULT_WEBHOOK:
        receivers.append({"name": "default", "webhook_configs": [{"url": DEFAULT_WEBHOOK}]})
    if PAGERDUTY_ROUTING_KEY:
        receivers.append({"name": "pagerduty", "pagerduty_configs": [{"routing_key": PAGERDUTY_ROUTING_KEY, "send_resolved": True, "details": {"runbook": "{{ .CommonAnnotations.runbook }}"}}]})
    if ALERTMANAGER_SLACK_WEBHOOK:
        slack_config = {"api_url": ALERTMANAGER_SLACK_WEBHOOK, "send_resolved": True}
        if ALERT_DEFAULT_CHANNEL:
            slack_config["channel"] = ALERT_DEFAULT_CHANNEL
        receivers.append({"name": "slack", "slack_configs": [slack_config]})
    if not receivers:
        receivers.append({"name": "default-noop", "webhook_configs": [{"url": "http://127.0.0.1:9"}]})
    preferred = choose_preferred_receiver(receivers)
    base_route = {
        "group_by": ["alertname", "service", "plane"],
        "group_wait": ALERTING_GROUP_WAIT,
        "group_interval": ALERTING_GROUP_INTERVAL,
        "repeat_interval": ALERTING_REPEAT_INTERVAL,
        "receiver": preferred,
    }
    planes = ["ingestion", "safety", "slo"]
    ordered_sevs: List[str] = []
    for s in paging:
        if s not in ordered_sevs:
            ordered_sevs.append(s)
    for s in slack:
        if s not in ordered_sevs:
            ordered_sevs.append(s)
    route_children: List[Dict[str, Any]] = []
    for sev in ordered_sevs:
        sev_l = sev.lower()
        receiver = None
        if sev_l in paging and PAGERDUTY_ROUTING_KEY:
            receiver = "pagerduty"
        elif sev_l in slack and ALERTMANAGER_SLACK_WEBHOOK:
            receiver = "slack"
        if not receiver:
            continue
        for plane in planes:
            route_children.append({"match": {"plane": plane, "severity": sev_l}, "receiver": receiver, "continue": False})
    config = {
        "global": {"resolve_timeout": "5m"},
        "route": {**base_route, "routes": route_children},
        "receivers": receivers,
        "inhibit_rules": [
            {"source_match": {"plane": "ingestion", "severity": "critical"}, "target_match": {"plane": "slo"}, "equal": ["service"]},
            {"source_match": {"plane": "safety", "severity": "critical"}, "target_match": {"plane": "slo"}, "equal": ["service"]},
            {"source_match": {"plane": "slo", "severity": "critical"}, "target_match": {"plane": "slo", "severity": "warning"}, "equal": ["service"]},
        ],
    }
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "alertmanager-config", "namespace": ns, "annotations": {"notifier_secret": NOTIFIER_SECRET_NAME}}, "data": {"alertmanager.yml": yaml.safe_dump(config, sort_keys=False)}}
    return cm

def build_alertmanager_objects() -> List[Dict[str, Any]]:
    ns = VM_NAMESPACE
    replicas = 1
    try:
        replicas = max(1, int(ALERTMANAGER_REPLICAS))
    except Exception:
        replicas = 1
    deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "alertmanager", "namespace": ns, "labels": {"app": "alertmanager"}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "alertmanager"}},
            "template": {
                "metadata": {"labels": {"app": "alertmanager"}},
                "spec": {
                    "containers": [
                        {
                            "name": "alertmanager",
                            "image": ALERTMANAGER_IMAGE,
                            "args": ["--config.file=/etc/alertmanager/alertmanager.yml", "--storage.path=/alertmanager"],
                            "volumeMounts": [{"name": "config", "mountPath": "/etc/alertmanager"}],
                            "ports": [{"containerPort": 9093, "name": "web"}],
                            "readinessProbe": {"httpGet": {"path": "/api/v2/status", "port": 9093}, "initialDelaySeconds": 5, "periodSeconds": 10},
                        }
                    ],
                    "volumes": [{"name": "config", "configMap": {"name": "alertmanager-config"}}],
                },
            },
        },
    }
    svc = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "alertmanager", "namespace": ns}, "spec": {"selector": {"app": "alertmanager"}, "ports": [{"name": "web", "port": 9093, "targetPort": 9093}]}}
    return [deploy, svc]

def build_notifier_secret_manifest() -> Dict[str, Any]:
    ns = VM_NAMESPACE
    data: Dict[str, str] = {}
    if PAGERDUTY_ROUTING_KEY:
        data["pagerduty_routing_key"] = PAGERDUTY_ROUTING_KEY
    if ALERTMANAGER_SLACK_WEBHOOK:
        data["slack_api_url"] = ALERTMANAGER_SLACK_WEBHOOK
    if DEFAULT_WEBHOOK:
        data["default_webhook"] = DEFAULT_WEBHOOK
    if not data:
        return {}
    return {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": NOTIFIER_SECRET_NAME, "namespace": ns}, "type": "Opaque", "stringData": data}

def render_all() -> None:
    validate_inputs()
    rules_obj = build_slo_rules()
    rules_text = yaml.safe_dump(rules_obj, sort_keys=False)
    vmalert_objs = build_vmalert_objects(rules_text)
    alertmgr_cm = build_alertmanager_cm()
    alertmgr_objs = build_alertmanager_objects()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slo_path = OUT_DIR / "slo.rules.yaml"
    vmalert_path = OUT_DIR / "vmalert-deployment.yaml"
    alertmgr_deploy_path = OUT_DIR / "alertmanager-deployment.yaml"
    alertmgr_cm_path = OUT_DIR / "alertmanager-config.yaml"
    atomic_write(slo_path, rules_text)
    multi_vmalert = []
    for o in vmalert_objs:
        multi_vmalert.append(yaml.safe_dump(o, sort_keys=False))
    atomic_write(vmalert_path, "\n---\n".join(multi_vmalert) + "\n")
    multi_alertmgr = []
    for o in alertmgr_objs:
        multi_alertmgr.append(yaml.safe_dump(o, sort_keys=False))
    atomic_write(alertmgr_deploy_path, "\n---\n".join(multi_alertmgr) + "\n")
    atomic_write(alertmgr_cm_path, yaml.safe_dump(alertmgr_cm, sort_keys=False))
    if CREATE_NOTIFIER_SECRET:
        secret = build_notifier_secret_manifest()
        if secret:
            secret_path = OUT_DIR / f"{NOTIFIER_SECRET_NAME}-secret.yaml"
            atomic_write(secret_path, yaml.safe_dump(secret, sort_keys=False))
            LOG.info("notifier secret rendered to disk %s", str(secret_path))
        else:
            LOG.info("CREATE_NOTIFIER_SECRET set but no notifier envs present; skipping secret emission")
    LOG.info("render complete out_dir=%s files=%s", str(OUT_DIR), [str(slo_path), str(vmalert_path), str(alertmgr_deploy_path), str(alertmgr_cm_path)])

def promtool_check(rules_path: Path) -> None:
    if not shutil.which("promtool"):
        LOG.warning("promtool not found; skipping PromQL syntax check")
        return
    rc, out, err = run_cmd(["promtool", "check", "rules", str(rules_path)], timeout=30)
    if rc != 0:
        LOG.error("promtool check failed stdout=%s stderr=%s", out, err)
        raise RuntimeError(f"promtool check failed: {err or out}")
    LOG.info("promtool check passed")

def kubectl_apply(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to apply manifests")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", str(path)], timeout=60)
    if rc != 0:
        LOG.error("kubectl apply failed file=%s stdout=%s stderr=%s", str(path), out, err)
        raise RuntimeError(f"kubectl apply failed for {path}: {err or out}")
    LOG.info("kubectl apply succeeded file=%s", str(path))

def kubectl_delete(path: Path) -> None:
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl required to delete manifests")
    rc, out, err = run_cmd(["kubectl", "delete", "-f", str(path), "--ignore-not-found"], timeout=60)
    if rc != 0:
        LOG.warning("kubectl delete returned non-zero file=%s stdout=%s stderr=%s", str(path), out, err)
    else:
        LOG.info("kubectl delete succeeded file=%s", str(path))

def generate(args: argparse.Namespace) -> None:
    LOG.info("generate started")
    render_all()

def validate(args: argparse.Namespace) -> None:
    LOG.info("validate started")
    render_all()
    slo = OUT_DIR / "slo.rules.yaml"
    if not slo.exists():
        raise RuntimeError("rendered slo.rules.yaml missing")
    promtool_check(slo)
    LOG.info("validate complete")

def apply(args: argparse.Namespace) -> None:
    LOG.info("apply started")
    render_all()
    alertmgr_deploy = OUT_DIR / "alertmanager-deployment.yaml"
    alertmgr_cm = OUT_DIR / "alertmanager-config.yaml"
    vmalert_manifest = OUT_DIR / "vmalert-deployment.yaml"
    slo = OUT_DIR / "slo.rules.yaml"
    if CREATE_NOTIFIER_SECRET:
        secret_path = OUT_DIR / f"{NOTIFIER_SECRET_NAME}-secret.yaml"
        if secret_path.exists():
            kubectl_apply(secret_path)
    kubectl_apply(alertmgr_deploy)
    kubectl_apply(alertmgr_cm)
    kubectl_apply(vmalert_manifest)
    try:
        txt = slo.read_text(encoding="utf-8")
        if txt.lstrip().startswith("apiVersion:"):
            kubectl_apply(slo)
        else:
            LOG.info("slo.rules.yaml is raw rules; not applying as k8s object")
    except Exception as e:
        LOG.warning("skipping slo.rules apply: %s", e)
    LOG.info("apply complete")

def delete(args: argparse.Namespace) -> None:
    LOG.info("delete started")
    if not args.confirm:
        raise RuntimeError("--confirm required to delete")
    files = ["alertmanager-deployment.yaml", "alertmanager-config.yaml", "vmalert-deployment.yaml", "slo.rules.yaml", f"{NOTIFIER_SECRET_NAME}-secret.yaml"]
    for f in files:
        p = OUT_DIR / f
        if p.exists() and shutil.which("kubectl"):
            try:
                if is_k8s_manifest(p):
                    kubectl_delete(p)
                else:
                    LOG.info("skipping kubectl delete for non-K8s file %s", str(p))
            except Exception as e:
                LOG.warning("kubectl delete failed for %s: %s", p, e)
    if OUT_DIR.exists():
        try:
            for entry in OUT_DIR.iterdir():
                try:
                    entry.unlink()
                except Exception:
                    if entry.is_dir():
                        shutil.rmtree(entry)
            try:
                OUT_DIR.rmdir()
            except Exception:
                pass
            LOG.info("removed manifest directory path=%s", str(OUT_DIR))
        except Exception as e:
            LOG.warning("failed to remove manifest directory: %s", e)
    LOG.info("delete complete")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/validate/apply/delete alerting manifests")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--validate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--apple", action="store_true")
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
        if args.apply or args.apple:
            apply(args); return
        if args.delete:
            delete(args); return
    except Exception as e:
        LOG.error("ERROR: %s", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
