#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import yaml
import shutil
import subprocess
import hashlib
import uuid
import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("qdrant_cluster")

CANONICAL_APP_LABEL = "app.kubernetes.io/name"
METRICS_PORT_NAME = "metrics"
DEFAULT_QDRANT_METRICS_PORT = 6333
DEFAULT_QDRANT_RELEASE = "qdrant"

def run_cmd(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return {"rc": proc.returncode, "out": out, "err": err}
    except subprocess.TimeoutExpired as e:
        return {"rc": 124, "out": getattr(e, "stdout", "") or "", "err": getattr(e, "stderr", "") or f"timeout after {timeout}s"}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def load_config() -> Dict[str, Any]:
    env = os.environ.get("ENV", "STAGING").upper()
    cfg: Dict[str, Any] = {}
    cfg["ENV"] = env
    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", "infra/manifests/qdrant")).resolve()
    cfg["ARCHIVE_DIR"] = Path(os.getenv("ARCHIVE_DIR", "infra/archive/qdrant-helm-chart")).resolve()
    cfg["VENDOR_CHART_DIR"] = cfg["ARCHIVE_DIR"] / "qdrant"
    cfg["QDRANT_RELEASE"] = os.getenv("QDRANT_RELEASE", DEFAULT_QDRANT_RELEASE)
    cfg["QDRANT_NAMESPACE"] = os.getenv("QDRANT_NAMESPACE", "qdrant")
    cfg["QDRANT_IMAGE"] = os.getenv("QDRANT_IMAGE", "qdrant/qdrant:v1.16.0")
    cfg["CHART_VERSION"] = os.getenv("CHART_VERSION", "1.16.0")
    cfg["QDRANT_REPLICAS"] = int(os.getenv("QDRANT_REPLICAS", "1"))
    cfg["QDRANT_CPU"] = os.getenv("QDRANT_CPU", "1")
    cfg["QDRANT_MEMORY"] = os.getenv("QDRANT_MEMORY", "2Gi")
    cfg["QDRANT_STORAGE"] = os.getenv("QDRANT_STORAGE", "emptyDir")
    cfg["QDRANT_NODE_SELECTOR"] = os.getenv("QDRANT_NODE_SELECTOR", "")
    cfg["QDRANT_TAINT_KEY"] = os.getenv("QDRANT_TAINT_KEY", "qdrant-dedicated")
    cfg["QDRANT_TAINT_EFFECT"] = os.getenv("QDRANT_TAINT_EFFECT", "NoSchedule")
    cfg["BACKUP_AZURE_CONTAINER"] = os.getenv("BACKUP_AZ_CONTAINER", "")
    cfg["BACKUP_AZURE_PREFIX"] = os.getenv("BACKUP_AZURE_PREFIX", "qdrant/backups")
    cfg["AZURE_STORAGE_CONNECTION_STRING"] = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    cfg["AZURE_STORAGE_ACCOUNT_NAME"] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    cfg["AZURE_STORAGE_ACCOUNT_KEY"] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
    cfg["QDRANT__SERVICE__API_KEY"] = os.getenv("QDRANT__SERVICE__API_KEY", "")
    cfg["APPLY_STAGING_SECRETS"] = os.getenv("APPLY_STAGING_SECRETS", "true").lower() in ("1", "true", "yes")
    cfg["TIMEOUT_SECONDS"] = int(os.getenv("TIMEOUT_SECONDS", "600"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    cfg["HELM_PRIMARY_REPO"] = os.getenv("HELM_PRIMARY_REPO", "https://qdrant.github.io/qdrant-helm")
    cfg["HELM_FALLBACK_REPO"] = os.getenv("HELM_FALLBACK_REPO", "https://qdrant.to/helm")
    cfg["HELM_REPO_NAME"] = os.getenv("HELM_REPO_NAME", "qdrant")
    cfg["QDRANT_SHARD_NUMBER"] = int(os.getenv("QDRANT_SHARD_NUMBER", "1"))
    cfg["QDRANT_REPLICATION_FACTOR"] = int(os.getenv("QDRANT_REPLICATION_FACTOR", str(cfg["QDRANT_REPLICAS"])))
    cfg["QDRANT_WRITE_CONSISTENCY_FACTOR"] = int(os.getenv("QDRANT_WRITE_CONSISTENCY_FACTOR", "1"))
    cfg["QDRANT_LOG_LEVEL"] = os.getenv("QDRANT_LOG_LEVEL", "INFO")
    cfg["QDRANT__STORAGE__STORAGE_PATH"] = os.getenv("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")
    cfg["QDRANT__STORAGE__SNAPSHOTS_PATH"] = os.getenv("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")
    cfg["USE_LOCAL_NVME"] = os.getenv("USE_LOCAL_NVME", "false").lower() in ("1", "true", "yes")
    cfg["QDRANT_LOCAL_PATH"] = os.getenv("QDRANT_LOCAL_PATH", "/mnt/nvme/qdrant")
    cfg["TAINT_QDRANT_NODES"] = os.getenv("TAINT_QDRANT_NODES", "false").lower() in ("1", "true", "yes")
    cfg["SECRET_BACKUP_NAME"] = os.getenv("SECRET_BACKUP_NAME", "qdrant-backup-azure")
    cfg["SECRET_SERVICE_NAME"] = os.getenv("SECRET_SERVICE_NAME", "qdrant-service-creds")
    qmp = os.getenv("QDRANT_METRICS_PORT", str(DEFAULT_QDRANT_METRICS_PORT)).strip()
    if qmp.isdigit():
        cfg["QDRANT_METRICS_PORT"] = int(qmp)
        cfg["QDRANT_METRICS_PORT_NAME"] = METRICS_PORT_NAME
    else:
        cfg["QDRANT_METRICS_PORT"] = qmp
        cfg["QDRANT_METRICS_PORT_NAME"] = qmp
    cfg["MANIFESTS_SAMPLES_DIR"] = cfg["MANIFESTS_DIR"] / "_samples"
    cfg["FILES"] = {
        "values": cfg["MANIFESTS_DIR"] / "values.yaml",
        "service_patch": cfg["MANIFESTS_DIR"] / "service-patch.yaml",
        "last_deploy_summary": cfg["MANIFESTS_DIR"] / "last_deploy_summary.json",
    }
    cfg["FAIL_ON_MISCONFIG"] = os.getenv("FAIL_ON_MISCONFIG", "false").lower() in ("1", "true", "yes")
    cfg["SERVICE_VALIDATION_WAIT"] = int(os.getenv("SERVICE_VALIDATION_WAIT", "15"))
    return cfg

SENSITIVE_KEYS = {"AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_KEY", "QDRANT__SERVICE__API_KEY"}

def canonical_inputs_hash(cfg: Dict[str, Any]) -> str:
    serial: Dict[str, Any] = {}
    for k in sorted(cfg.keys()):
        if k == "INPUTS_HASH_PATH":
            continue
        if k in SENSITIVE_KEYS:
            serial[k] = bool(cfg.get(k))
        else:
            v = cfg.get(k)
            try:
                json.dumps(v)
                serial[k] = v
            except Exception:
                serial[k] = str(v)
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def detect_storageclass(kubectl_bin: str = "kubectl") -> Optional[str]:
    if shutil.which(kubectl_bin) is None:
        return None
    cmd = [kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[?(@.metadata.annotations.storageclass\\.kubernetes\\.io/is-default-class==\"true\")].metadata.name}"]
    rc = run_cmd(cmd)
    out = rc["out"].strip() if rc["rc"] == 0 else ""
    if out:
        return out
    rc2 = run_cmd([kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[0].metadata.name}"])
    out2 = rc2["out"].strip() if rc2["rc"] == 0 else ""
    return out2 if out2 else None

def helm_repo_add_if_missing(cfg: Dict[str, Any], verbose: bool = False) -> Tuple[bool, str]:
    helm = which("helm")
    if not helm:
        return False, "helm-not-found"
    cmd = [helm, "repo", "add", "--force-update", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]]
    rc = run_cmd(cmd, timeout=30)
    if rc["rc"] != 0:
        rc2 = run_cmd([helm, "repo", "add", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]], timeout=30)
        if rc2["rc"] != 0:
            return False, rc2["err"] or rc2["out"]
    rc_upd = run_cmd([helm, "repo", "update"], timeout=30)
    if rc_upd["rc"] != 0:
        return False, rc_upd["err"] or rc_upd["out"]
    return True, ""

def vendor_chart_if_missing(cfg: Dict[str, Any], verbose: bool = False) -> Tuple[bool, str]:
    vendor_dir = cfg["VENDOR_CHART_DIR"]
    helm = which("helm")
    if vendor_dir.exists() and (vendor_dir / "Chart.yaml").exists():
        return True, str(vendor_dir)
    if helm is None:
        return False, "helm-not-found"
    ok, msg = helm_repo_add_if_missing(cfg, verbose=verbose)
    if not ok:
        return False, msg
    cmd = [helm, "pull", f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--untar", "--untardir", str(cfg["ARCHIVE_DIR"])]
    rc = run_cmd(cmd, timeout=120)
    if rc["rc"] == 0:
        if vendor_dir.exists():
            return True, str(vendor_dir)
        return False, "vendor-dir-missing-after-pull"
    return False, rc["err"] or rc["out"]

def helm_upgrade_install(cfg: Dict[str, Any], values_file: Path, vendor_dir: Path, verbose: bool = False) -> Tuple[bool, str, str]:
    helm = which("helm")
    if not helm:
        return False, "helm-not-found", ""
    release = cfg["QDRANT_RELEASE"]
    ns = cfg["QDRANT_NAMESPACE"]
    timeout = "10m"
    if vendor_dir.exists():
        cmd = [helm, "upgrade", "--install", release, str(vendor_dir), "--namespace", ns, "--create-namespace", "-f", str(values_file), "--wait", "--timeout", timeout]
        rc = run_cmd(cmd, timeout=600)
        if rc["rc"] == 0:
            return True, "", rc["out"]
        vendor_err = rc["err"] or rc["out"]
    else:
        vendor_err = "vendor-not-present"
    cmd2 = [helm, "upgrade", "--install", release, f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc2 = run_cmd(cmd2, timeout=600)
    if rc2["rc"] == 0:
        return True, "", rc2["out"]
    cmd3 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_FALLBACK_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc3 = run_cmd(cmd3, timeout=600)
    if rc3["rc"] == 0:
        return True, "", rc3["out"]
    cmd4 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_PRIMARY_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc4 = run_cmd(cmd4, timeout=600)
    if rc4["rc"] == 0:
        return True, "", rc4["out"]
    combined_err = "\n--- vendor attempt ---\n" + str(vendor_err) + "\n--- primary ---\n" + (rc2["err"] or rc2["out"]) + "\n--- fallback ---\n" + (rc3["err"] or rc3["out"]) + "\n--- retry primary ---\n" + (rc4["err"] or rc4["out"])
    return False, combined_err, rc4["err"] or rc4["out"]

def render_values_yaml(cfg: Dict[str, Any], storage_class: Optional[str]) -> str:
    repo_tag = cfg["QDRANT_IMAGE"]
    if ":" in repo_tag:
        repo, tag = repo_tag.split(":", 1)
    else:
        repo, tag = repo_tag, "latest"
    peers = [f"http://{cfg['QDRANT_RELEASE']}-{i}.{cfg['QDRANT_RELEASE']}-headless:6335" for i in range(0, cfg["QDRANT_REPLICAS"])]
    values: Dict[str, Any] = {}
    values["replicaCount"] = cfg["QDRANT_REPLICAS"]
    values["image"] = {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"}
    values["service"] = {
        "type": "ClusterIP",
        "labels": {
            CANONICAL_APP_LABEL: cfg["QDRANT_RELEASE"],
            "app.kubernetes.io/component": "qdrant",
        },
        "annotations": {
            "prometheus.io/scrape": "true",
            "prometheus.io/port": str(cfg["QDRANT_METRICS_PORT"]),
            "prometheus.io/path": "/metrics",
        }
    }
    values["p2p"] = {"port": 6335}
    values["cluster"] = {"enabled": True, "peers": peers}
    values["snapshots"] = {"enabled": False, "s3": {"bucket": "", "endpoint": "", "region": "", "prefix": ""}}
    values["extraEnv"] = []
    values["resources"] = {
        "requests": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]},
        "limits": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]},
    }
    backup_secret = cfg.get("SECRET_BACKUP_NAME")
    if backup_secret:
        for k in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"):
            values["extraEnv"].append({"name": k, "valueFrom": {"secretKeyRef": {"name": backup_secret, "key": k}}})
    service_secret = cfg.get("SECRET_SERVICE_NAME")
    if service_secret and cfg.get("QDRANT__SERVICE__API_KEY"):
        values["extraEnv"].append({"name": "QDRANT__SERVICE__API_KEY", "valueFrom": {"secretKeyRef": {"name": service_secret, "key": "QDRANT__SERVICE__API_KEY"}}})
    if cfg.get("TAINT_QDRANT_NODES"):
        values["tolerations"] = [{"key": cfg["QDRANT_TAINT_KEY"], "operator": "Exists", "effect": cfg["QDRANT_TAINT_EFFECT"]}]
    else:
        values["tolerations"] = []
    if cfg.get("USE_LOCAL_NVME"):
        values["persistence"] = {"enabled": False}
        host_path = cfg.get("QDRANT_LOCAL_PATH", "/mnt/nvme/qdrant")
        values["extraVolumes"] = [
            {"name": "qdrant-storage", "hostPath": {"path": host_path, "type": "DirectoryOrCreate"}},
            {"name": "qdrant-snapshots", "hostPath": {"path": f"{host_path}/snapshots", "type": "DirectoryOrCreate"}},
        ]
        values["extraVolumeMounts"] = [
            {"name": "qdrant-storage", "mountPath": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")},
            {"name": "qdrant-snapshots", "mountPath": cfg.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")},
        ]
        if cfg.get("QDRANT_NODE_SELECTOR"):
            try:
                k, v = cfg.get("QDRANT_NODE_SELECTOR").split("=", 1)
                values["nodeSelector"] = {k: v}
            except Exception:
                values["nodeSelector"] = {}
    else:
        if storage_class:
            values["persistence"] = {"enabled": True, "storageClass": storage_class, "size": "50Gi"}
        else:
            values["persistence"] = {"enabled": False}
            values["extraVolumes"] = [{"name": "qdrant-storage", "emptyDir": {}}]
            values["extraVolumeMounts"] = [{"name": "qdrant-storage", "mountPath": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")}]
    values["config"] = {
        "params": {
            "shard_number": int(cfg.get("QDRANT_SHARD_NUMBER", 1)),
            "replication_factor": int(cfg.get("QDRANT_REPLICATION_FACTOR", cfg.get("QDRANT_REPLICAS", 1))),
            "write_consistency_factor": int(cfg.get("QDRANT_WRITE_CONSISTENCY_FACTOR", 1)),
        },
        "log_level": cfg.get("QDRANT_LOG_LEVEL", "INFO"),
        "storage": {
            "storage_path": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage"),
            "snapshots_path": cfg.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots"),
        },
    }
    return yaml.safe_dump(values, sort_keys=False)

def render_service_patch(cfg: Dict[str, Any]) -> str:
    svc_name = cfg["QDRANT_RELEASE"]
    ns = cfg["QDRANT_NAMESPACE"]
    port_num = int(cfg["QDRANT_METRICS_PORT"]) if isinstance(cfg["QDRANT_METRICS_PORT"], int) or str(cfg["QDRANT_METRICS_PORT"]).isdigit() else DEFAULT_QDRANT_METRICS_PORT
    patch = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": svc_name, "namespace": ns, "labels": {CANONICAL_APP_LABEL: cfg["QDRANT_RELEASE"], "app.kubernetes.io/component": "qdrant"}},
        "spec": {
            "selector": {CANONICAL_APP_LABEL: cfg["QDRANT_RELEASE"]},
            "ports": [
                {"name": cfg.get("QDRANT_METRICS_PORT_NAME", METRICS_PORT_NAME), "port": port_num, "targetPort": port_num, "protocol": "TCP"}
            ],
            "type": "ClusterIP"
        }
    }
    return yaml.safe_dump(patch, sort_keys=False)

def generate_manifests(cfg: Dict[str, Any], dry_run: bool = False, verbose: bool = False) -> None:
    ensure_dir(cfg["MANIFESTS_DIR"])
    inputs_hash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing = cfg["INPUTS_HASH_PATH"].read_text().strip()
    if existing == inputs_hash and not dry_run:
        LOG.info("No non-secret changes detected; generation skipped.")
        return
    storage_class = detect_storageclass() or None
    values_yaml = render_values_yaml(cfg, storage_class)
    atomic_write(cfg["FILES"]["values"], values_yaml)
    ensure_dir(cfg["MANIFESTS_SAMPLES_DIR"])
    sample_secret_placeholder = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": cfg["SECRET_BACKUP_NAME"], "namespace": cfg["QDRANT_NAMESPACE"]},
        "type": "Opaque",
        "stringData": {"AZURE_STORAGE_CONNECTION_STRING": "REPLACE_ME", "AZURE_STORAGE_ACCOUNT_NAME": "REPLACE_ME", "AZURE_STORAGE_ACCOUNT_KEY": "REPLACE_ME"},
    }
    atomic_write(cfg["MANIFESTS_SAMPLES_DIR"] / "secret-sample.placeholder.yaml", yaml.safe_dump(sample_secret_placeholder, sort_keys=False))
    service_patch_yaml = render_service_patch(cfg)
    atomic_write(cfg["FILES"]["service_patch"], service_patch_yaml)
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash)
    LOG.info("Wrote manifests to %s", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        LOG.info("--- values.yaml preview ---")
        for line in values_yaml.splitlines()[:200]:
            LOG.info(line)
        LOG.info("--- service patch preview ---")
        for line in service_patch_yaml.splitlines()[:200]:
            LOG.info(line)

def delete_manifests(cfg: Dict[str, Any]) -> None:
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception:
                pass
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except Exception:
            pass
        LOG.info("Deleted manifests at %s", str(cfg["MANIFESTS_DIR"]))
    else:
        LOG.info("Manifests dir not present: %s", str(cfg["MANIFESTS_DIR"]))

def ensure_namespace(cfg: Dict[str, Any]) -> bool:
    kubectl = which("kubectl")
    if not kubectl:
        return False
    ns = cfg["QDRANT_NAMESPACE"]
    cmd1 = [kubectl, "create", "namespace", ns, "--dry-run=client", "-o", "yaml"]
    rc = run_cmd(cmd1, timeout=20)
    if rc["rc"] != 0:
        rcg = run_cmd([kubectl, "get", "namespace", ns], timeout=20)
        return rcg["rc"] == 0
    rc2 = run_cmd([kubectl, "apply", "-f", "-"], input_bytes=rc["out"].encode("utf-8"), timeout=20)
    return rc2["rc"] == 0

def kubectl_create_secret_in_cluster(cfg: Dict[str, Any], secret_name: str, env_keys: List[str]) -> Tuple[bool, str]:
    kubectl = which("kubectl")
    if not kubectl:
        return False, "kubectl-not-found"
    ok_ns = ensure_namespace(cfg)
    if not ok_ns:
        return False, "ensure-namespace-failed"
    literals: List[str] = []
    for k in env_keys:
        val = cfg.get(k, os.environ.get(k, ""))
        if val:
            literals += ["--from-literal", f"{k}={val}"]
    if not literals:
        return False, "no-secrets-present-for-this-secret"
    cmd = [kubectl, "create", "secret", "generic", secret_name, "-n", cfg["QDRANT_NAMESPACE"], "--dry-run=client", "-o", "yaml"] + literals
    rc = run_cmd(cmd, timeout=20)
    if rc["rc"] != 0:
        return False, rc["err"] or rc["out"]
    apply_rc = run_cmd([kubectl, "apply", "-f", "-"], input_bytes=rc["out"].encode("utf-8"), timeout=20)
    if apply_rc["rc"] != 0:
        return False, apply_rc["err"] or apply_rc["out"]
    return True, ""

def validate_service_post_install(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    kubectl = which("kubectl")
    if not kubectl:
        LOG.info("kubectl not found; skipping post-install validation")
        return errors
    selector_key = CANONICAL_APP_LABEL
    selector_value = cfg["QDRANT_RELEASE"]
    rc = run_cmd([kubectl, "-n", cfg["QDRANT_NAMESPACE"], "get", "svc", "-l", f"{selector_key}={selector_value}", "-o", "json"], timeout=20)
    if rc["rc"] != 0:
        errors.append(f"kubectl get svc failed: {rc['err'] or rc['out']}")
        return errors
    try:
        data = json.loads(rc["out"] or "{}")
    except Exception:
        errors.append("failed to parse kubectl svc json")
        return errors
    items = data.get("items", []) if isinstance(data, dict) else []
    if not items:
        errors.append(f"no Service found in namespace '{cfg['QDRANT_NAMESPACE']}' with label {selector_key}={selector_value}")
        return errors
    expected_port = str(cfg["QDRANT_METRICS_PORT"]) if isinstance(cfg["QDRANT_METRICS_PORT"], int) or str(cfg["QDRANT_METRICS_PORT"]).isdigit() else str(DEFAULT_QDRANT_METRICS_PORT)
    expected_name = cfg.get("QDRANT_METRICS_PORT_NAME", METRICS_PORT_NAME)
    for svc in items:
        ports = svc.get("spec", {}).get("ports", [])
        matched = False
        for p in ports:
            name = str(p.get("name", "")).strip()
            port_num = str(p.get("port", "")).strip()
            target_port = str(p.get("targetPort", "")).strip()
            if name == expected_name or port_num == expected_port or target_port == expected_port:
                matched = True
                break
        if not matched:
            svc_name = svc.get("metadata", {}).get("name", "<unknown>")
            errors.append(f"service '{svc_name}' does not expose metrics port {expected_port} with name '{expected_name}'")
    return errors

def apply_to_cluster(cfg: Dict[str, Any], dry_run: bool = False, verbose: bool = False) -> None:
    kubectl = which("kubectl")
    helm = which("helm")
    if kubectl is None or helm is None:
        LOG.error("kubectl and helm are required in PATH to apply to cluster.")
        sys.exit(2)
    if cfg["ENV"] == "STAGING" and cfg["APPLY_STAGING_SECRETS"]:
        if not cfg.get("AZURE_STORAGE_CONNECTION_STRING") and not (cfg.get("AZURE_STORAGE_ACCOUNT_NAME") and cfg.get("AZURE_STORAGE_ACCOUNT_KEY")):
            LOG.error("ENV=STAGING requires AZURE storage secrets to be set when using --apply.")
            sys.exit(2)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    ok_ns = ensure_namespace(cfg)
    if not ok_ns:
        LOG.error("Failed to ensure namespace: %s", cfg["QDRANT_NAMESPACE"])
        sys.exit(2)
    created_any_secret = False
    if cfg["ENV"] == "STAGING" and cfg["APPLY_STAGING_SECRETS"]:
        ok_s, err_s = kubectl_create_secret_in_cluster(cfg, cfg["SECRET_BACKUP_NAME"], ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"])
        if not ok_s:
            LOG.error("Failed to create/update backup secret in-cluster: %s", err_s)
            sys.exit(2)
        LOG.info("Created/updated in-cluster secret: %s", cfg["SECRET_BACKUP_NAME"])
        created_any_secret = True
    if cfg.get("QDRANT__SERVICE__API_KEY"):
        ok_srv, err_srv = kubectl_create_secret_in_cluster(cfg, cfg["SECRET_SERVICE_NAME"], ["QDRANT__SERVICE__API_KEY"])
        if not ok_srv:
            LOG.error("Failed to create/update service secret in-cluster: %s", err_srv)
            sys.exit(2)
        LOG.info("Created/updated in-cluster secret: %s", cfg["SECRET_SERVICE_NAME"])
        created_any_secret = True
    v_ok, v_err = vendor_chart_if_missing(cfg, verbose=verbose)
    if not v_ok:
        LOG.warning("Vendor chart not available locally; will attempt remote install. vendor error: %s", v_err)
    else:
        LOG.info("Vendor chart available at %s", str(cfg["VENDOR_CHART_DIR"]))
    values_file = cfg["FILES"]["values"]
    ok, errtext, stdout_text = helm_upgrade_install(cfg, values_file, cfg["VENDOR_CHART_DIR"], verbose=verbose)
    if not ok:
        LOG.error("helm upgrade/install failed. %s", errtext)
        sys.exit(2)
    LOG.info("Helm install/upgrade succeeded for release: %s", cfg["QDRANT_RELEASE"])
    service_patch = cfg["FILES"]["service_patch"]
    rc_patch = run_cmd([kubectl, "apply", "-f", str(service_patch)], timeout=30)
    if rc_patch["rc"] != 0:
        LOG.warning("Applying service patch returned non-zero: stdout: %s stderr: %s", rc_patch["out"], rc_patch["err"])
    time.sleep(cfg.get("SERVICE_VALIDATION_WAIT", 15))
    errors = validate_service_post_install(cfg)
    if errors:
        for e in errors:
            LOG.error("Post-install validation: %s", e)
        LOG.error("Post-install validation failed; review service labels/ports.")
        if cfg.get("FAIL_ON_MISCONFIG"):
            sys.exit(2)
        else:
            LOG.warning("Continuing despite post-install validation errors because FAIL_ON_MISCONFIG=false")
    summary = {
        "release": cfg["QDRANT_RELEASE"],
        "namespace": cfg["QDRANT_NAMESPACE"],
        "replicas": cfg["QDRANT_REPLICAS"],
        "values_file": str(values_file),
        "chart_version": cfg["CHART_VERSION"],
        "image": cfg["QDRANT_IMAGE"],
        "vendor_chart_dir": str(cfg["VENDOR_CHART_DIR"]) if cfg["VENDOR_CHART_DIR"].exists() else None,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "secrets_created": created_any_secret,
    }
    atomic_write(Path(cfg["FILES"]["last_deploy_summary"]), json.dumps(summary, indent=2))
    LOG.info("Wrote deploy summary -> %s", str(cfg["FILES"]["last_deploy_summary"]))

def delete_from_cluster(cfg: Dict[str, Any]) -> None:
    if cfg["MANIFESTS_DIR"].exists():
        rc = which("kubectl")
        if rc:
            release = cfg["QDRANT_RELEASE"]
            ns = cfg["QDRANT_NAMESPACE"]
            run_cmd([rc, "delete", "deployment", f"{release}", "-n", ns, "--ignore-not-found"], timeout=30)
            run_cmd([rc, "delete", "service", f"{release}", "-n", ns, "--ignore-not-found"], timeout=30)
            run_cmd([rc, "delete", "statefulset", f"{release}", "-n", ns, "--ignore-not-found"], timeout=30)
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception:
                pass
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except Exception:
            pass
        LOG.info("Deleted local manifests in %s", str(cfg["MANIFESTS_DIR"]))

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generate/apply/delete Qdrant Helm manifests (cluster-aware, Prometheus-friendly).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true")
    grp.add_argument("--apply", action="store_true")
    grp.add_argument("--delete", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    try:
        if args.delete:
            delete_manifests(cfg)
            delete_from_cluster(cfg)
            return
        if args.generate:
            generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
            return
        if args.apply:
            apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose)
            return
    except Exception as e:
        LOG.error("ERROR: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
