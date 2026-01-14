#!/usr/bin/env python3
"""
gen_reranker.py

Deterministic generator for Reranker Kubernetes manifests.
Writes manifests to infra/manifests/reranker/

Usage:
  python gen_reranker.py --generate
  python gen_reranker.py --rollout
  python gen_reranker.py --delete
"""
from pathlib import Path
import os
import sys
import json
import yaml
import shutil
import subprocess
import argparse
import hashlib
import uuid
import datetime
import logging
import time
import tempfile

# -------------------- logging --------------------
_level_name = os.environ.get("GEN_RERANKER_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _level_name, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gen_reranker")

# -------------------- helpers --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    # write to a temp file in same dir then atomically replace
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, str(path))

def run_cmd(cmd, capture=True, check=False, timeout=None, input_text: str = None):
    """
    Run subprocess in text mode. Return (rc, stdout, stderr).
    cmd: list or string acceptable to subprocess.run (we prefer list).
    """
    try:
        proc = subprocess.run(cmd, input=input_text, capture_output=capture, text=True, check=check, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def canonical_inputs_hash(cfg: dict):
    serial = {}
    for k in sorted(cfg.keys()):
        if k == "INPUTS_HASH_PATH":
            continue
        v = cfg.get(k)
        try:
            json.dumps(v)
            serial[k] = v
        except Exception:
            serial[k] = str(v)
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def kubectl_apply_yaml(yaml_str: str, dry_run=False):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply"]
    if dry_run:
        cmd += ["--dry-run=client", "-f", "-"]
    else:
        cmd += ["-f", "-"]
    try:
        proc = subprocess.run(cmd, input=yaml_str, capture_output=True, text=True, check=True, timeout=120)
        return {"applied": True, "stdout": proc.stdout or ""}
    except subprocess.CalledProcessError as e:
        return {"applied": False, "stderr": e.stderr or str(e)}
    except subprocess.TimeoutExpired as e:
        return {"applied": False, "stderr": f"timeout: {e}"}

# -------------------- config loader (RERANKER_*) --------------------
def load_config():
    cfg = {}
    cfg["ENV"] = os.environ.get("RERANKER_ENV", os.environ.get("ENV", "STAGING")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/reranker"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["IMAGE"] = os.environ.get("RERANKER_IMAGE", "athithya5354/reranker:v1")
    cfg["NAMESPACE"] = os.environ.get("RERANKER_NAMESPACE", "models")
    cfg["SERVICE_NAME"] = os.environ.get("RERANKER_SERVICE_NAME", "reranker")
    cfg["CONTAINER_PORT"] = int(os.environ.get("RERANKER_PORT", "8202"))
    cfg["HOST"] = os.environ.get("RERANKER_HOST", "0.0.0.0")
    cfg["LOGLEVEL"] = os.environ.get("RERANKER_LOGLEVEL", "INFO")
    # replicas & resources
    if cfg["ENV"] == "PROD":
        cfg.update({
            "REPLICAS": int(os.environ.get("RERANKER_REPLICAS", "3")),
            "CPU_REQUEST": os.environ.get("RERANKER_CPU_REQUEST", "1000m"),
            "CPU_LIMIT": os.environ.get("RERANKER_CPU_LIMIT", "4000m"),
            "MEMORY_REQUEST": os.environ.get("RERANKER_MEMORY_REQUEST", "1Gi"),
            "MEMORY_LIMIT": os.environ.get("RERANKER_MEMORY_LIMIT", "4Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("RERANKER_STARTUP_FAILURE_THRESHOLD", "24")),
        })
    else:
        cfg.update({
            "REPLICAS": int(os.environ.get("RERANKER_REPLICAS", "1")),
            "CPU_REQUEST": os.environ.get("RERANKER_CPU_REQUEST", "250m"),
            "CPU_LIMIT": os.environ.get("RERANKER_CPU_LIMIT", "1000m"),
            "MEMORY_REQUEST": os.environ.get("RERANKER_MEMORY_REQUEST", "512Mi"),
            "MEMORY_LIMIT": os.environ.get("RERANKER_MEMORY_LIMIT", "1Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("RERANKER_STARTUP_FAILURE_THRESHOLD", "60")),
        })
    # probes
    cfg["PROBE_PERIOD_SECONDS"] = int(os.environ.get("RERANKER_PROBE_PERIOD_SECONDS", "5"))
    cfg["READINESS_INITIAL_DELAY"] = int(os.environ.get("RERANKER_READINESS_INITIAL_DELAY", "10"))
    cfg["LIVENESS_INITIAL_DELAY"] = int(os.environ.get("RERANKER_LIVENESS_INITIAL_DELAY", "30"))
    cfg["PROBE_TIMEOUT_SECONDS"] = int(os.environ.get("RERANKER_PROBE_TIMEOUT_SECONDS", "3"))
    # GPU
    cfg["ENABLE_GPU"] = os.environ.get("RERANKER_ENABLE_GPU", "false").lower() in ("1", "true", "yes")
    cfg["GPU_RESOURCE_NAME"] = os.environ.get("RERANKER_GPU_RESOURCE", "nvidia.com/gpu")
    cfg["GPU_COUNT"] = os.environ.get("RERANKER_GPU_COUNT", "1")
    cfg["GPU_NODE_SELECTOR"] = os.environ.get("RERANKER_GPU_NODE_SELECTOR", "")
    # HPA
    cfg["HPA_ENABLED"] = os.environ.get("RERANKER_HPA_ENABLED", "false").lower() in ("1", "true", "yes")
    cfg["HPA_MIN"] = int(os.environ.get("RERANKER_HPA_MIN_REPLICAS", "1"))
    cfg["HPA_MAX"] = int(os.environ.get("RERANKER_HPA_MAX_REPLICAS", "10"))
    cfg["HPA_TARGET_CPU"] = int(os.environ.get("RERANKER_HPA_TARGET_CPU", "60"))
    # RBAC
    cfg["SA_NAME"] = os.environ.get("RERANKER_SA_NAME", f"{cfg['SERVICE_NAME']}-sa")
    cfg["ROLE_NAME"] = os.environ.get("RERANKER_ROLE_NAME", f"{cfg['SERVICE_NAME']}-role")
    cfg["ROLEBIND_NAME"] = os.environ.get("RERANKER_ROLEBIND_NAME", f"{cfg['SERVICE_NAME']}-rb")
    # labels
    cfg["LABELS"] = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "reranker",
        "app.kubernetes.io/managed-by": "gen_reranker",
        "app.kubernetes.io/instance": cfg["SERVICE_NAME"],
        "env": cfg["ENV"].lower(),
    }
    # rollout timeout
    cfg["ROLLOUT_TIMEOUT"] = int(os.environ.get("RERANKER_ROLLOUT_TIMEOUT", "300"))
    cfg["FILES"] = {
        "namespace": cfg["MANIFESTS_DIR"] / "00-namespace.yaml",
        "sa_role": cfg["MANIFESTS_DIR"] / "01-sa-role.yaml",
        "deployment": cfg["MANIFESTS_DIR"] / "02-deployment.yaml",
        "service": cfg["MANIFESTS_DIR"] / "03-service.yaml",
        "hpa": cfg["MANIFESTS_DIR"] / "04-hpa.yaml",
    }
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    return cfg

# -------------------- YAML renderers --------------------
def render_namespace(cfg):
    ns = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "gen_reranker"}},
    }
    return yaml.safe_dump(ns, sort_keys=False)

def render_sa_role(cfg):
    sa = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": cfg["SA_NAME"], "namespace": cfg["NAMESPACE"]},
    }
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": cfg["ROLE_NAME"], "namespace": cfg["NAMESPACE"]},
        "rules": [
            {"apiGroups": [""], "resources": ["pods", "services", "endpoints", "configmaps"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get"]},
        ],
    }
    rb = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": cfg["ROLEBIND_NAME"], "namespace": cfg["NAMESPACE"]},
        "subjects": [{"kind": "ServiceAccount", "name": cfg["SA_NAME"], "namespace": cfg["NAMESPACE"]}],
        "roleRef": {"kind": "Role", "name": cfg["ROLE_NAME"], "apiGroup": "rbac.authorization.k8s.io"},
    }
    return "\n---\n".join([yaml.safe_dump(x, sort_keys=False) for x in (sa, role, rb)])

def render_deployment(cfg, config_checksum: str = ""):
    labels = cfg["LABELS"].copy()
    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["CONTAINER_PORT"]}],
        "env": [
            {"name": "RERANKER_PORT", "value": str(cfg["CONTAINER_PORT"])},
            {"name": "ENV", "value": cfg["ENV"]},
            {"name": "RERANKER_LOGLEVEL", "value": cfg["LOGLEVEL"]},
        ],
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": cfg["CONTAINER_PORT"]},
            "initialDelaySeconds": cfg["LIVENESS_INITIAL_DELAY"],
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": 6,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": cfg["CONTAINER_PORT"]},
            "initialDelaySeconds": cfg["READINESS_INITIAL_DELAY"],
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": 3,
        },
        "startupProbe": {
            "httpGet": {"path": "/health", "port": cfg["CONTAINER_PORT"]},
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": cfg["STARTUP_FAILURE_THRESHOLD"],
        },
        "resources": {
            "requests": {"cpu": cfg["CPU_REQUEST"], "memory": cfg["MEMORY_REQUEST"]},
            "limits": {"cpu": cfg["CPU_LIMIT"], "memory": cfg["MEMORY_LIMIT"]},
        },
    }

    if cfg["ENABLE_GPU"]:
        try:
            gcount = int(cfg["GPU_COUNT"])
        except Exception:
            gcount = 1
        container["resources"]["limits"][cfg["GPU_RESOURCE_NAME"]] = gcount

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-deployment", "namespace": cfg["NAMESPACE"], "labels": labels},
        "spec": {
            "replicas": cfg["REPLICAS"],
            "selector": {"matchLabels": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "serviceAccountName": cfg["SA_NAME"],
                    "containers": [container],
                },
            },
        },
    }

    # GPU nodeSelector handling
    if cfg["ENABLE_GPU"] and cfg["GPU_NODE_SELECTOR"]:
        if "=" in cfg["GPU_NODE_SELECTOR"]:
            k, v = cfg["GPU_NODE_SELECTOR"].split("=", 1)
            deployment["spec"]["template"]["spec"]["nodeSelector"] = {k: v}
        else:
            deployment["spec"]["template"]["spec"]["nodeSelector"] = {cfg["GPU_NODE_SELECTOR"]: "true"}

    # ensure annotations exist and include monitoring and config checksum to force rollout on config changes
    deployment["spec"]["template"]["metadata"].setdefault("annotations", {})
    deployment["spec"]["template"]["metadata"]["annotations"].update({
        "prometheus.io/scrape": "true",
        "prometheus.io/port": str(cfg["CONTAINER_PORT"]),
        "prometheus.io/path": "/metrics",
    })
    if config_checksum:
        # use a valid DNS-compatible prefix (no underscores)
        deployment["spec"]["template"]["metadata"]["annotations"]["gen-reranker/config-checksum"] = config_checksum

    return yaml.safe_dump(deployment, sort_keys=False)

def render_service(cfg):
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-svc", "namespace": cfg["NAMESPACE"], "labels": cfg["LABELS"]},
        "spec": {
            "type": "ClusterIP",
            "ports": [{"port": cfg["CONTAINER_PORT"], "targetPort": cfg["CONTAINER_PORT"], "protocol": "TCP", "name": "http"}],
            "selector": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]},
        },
    }
    return yaml.safe_dump(svc, sort_keys=False)

def render_hpa(cfg):
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-hpa", "namespace": cfg["NAMESPACE"]},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": f"{cfg['SERVICE_NAME']}-deployment"},
            "minReplicas": cfg["HPA_MIN"],
            "maxReplicas": cfg["HPA_MAX"],
            "metrics": [
                {"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": cfg["HPA_TARGET_CPU"]}}}
            ],
        },
    }
    return yaml.safe_dump(hpa, sort_keys=False)

# -------------------- generate / rollout / delete --------------------
def generate_manifests(cfg, dry_run=False, verbose=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    inputs_hash = canonical_inputs_hash(cfg)
    existing = None
    try:
        if cfg["INPUTS_HASH_PATH"].exists():
            existing = cfg["INPUTS_HASH_PATH"].read_text().strip()
    except Exception:
        existing = None
    if existing == inputs_hash and not dry_run:
        log.info("No non-secret changes detected; generation skipped.")
        return

    config_checksum = inputs_hash
    ns_yaml = render_namespace(cfg)
    sa_role_yaml = render_sa_role(cfg)
    deploy_yaml = render_deployment(cfg, config_checksum=config_checksum)
    svc_yaml = render_service(cfg)
    atomic_write(cfg["FILES"]["namespace"], ns_yaml)
    atomic_write(cfg["FILES"]["sa_role"], sa_role_yaml)
    atomic_write(cfg["FILES"]["deployment"], deploy_yaml)
    atomic_write(cfg["FILES"]["service"], svc_yaml)
    if cfg["HPA_ENABLED"]:
        hpa_yaml = render_hpa(cfg)
        atomic_write(cfg["FILES"]["hpa"], hpa_yaml)
    # save inputs hash
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash, encoding="utf-8")
    log.info("Wrote manifests to %s", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        log.info("Namespace (head):\n%s", "\n".join(ns_yaml.splitlines()[:20]))
        log.info("Deployment (head):\n%s", "\n".join(deploy_yaml.splitlines()[:60]))

def wait_for_rollout(cfg, timeout_seconds: int = None):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot wait for rollout")
        return 2
    if timeout_seconds is None:
        timeout_seconds = cfg.get("ROLLOUT_TIMEOUT", 300)
    deployment_name = f"{cfg['SERVICE_NAME']}-deployment"
    cmd = [kubectl, "rollout", "status", f"deployment/{deployment_name}", "-n", cfg["NAMESPACE"], f"--timeout={timeout_seconds}s"]
    rc, out, err = run_cmd(cmd, capture=True, timeout=timeout_seconds + 10)
    if rc == 0:
        log.info("Deployment %s rolled out successfully", deployment_name)
        return 0

    # gather diagnostics
    log.error("Rollout status failed (rc=%d). Gathering diagnostics...", rc)
    d_cmds = [
        [kubectl, "get", "pods", "-n", cfg["NAMESPACE"]],
        [kubectl, "describe", "pod", "-l", f"app.kubernetes.io/name={cfg['SERVICE_NAME']}", "-n", cfg["NAMESPACE"]],
        [kubectl, "logs", "-l", f"app.kubernetes.io/name={cfg['SERVICE_NAME']}", "-n", cfg["NAMESPACE"], "--tail=200"],
    ]
    for c in d_cmds:
        rco, outo, erro = run_cmd(c, capture=True, timeout=30)
        if outo:
            log.error("CMD %s output:\n%s", " ".join(c), outo.strip())
        if erro:
            log.error("CMD %s error:\n%s", " ".join(c), erro.strip())
    return rc

def apply_to_cluster(cfg, dry_run=False, verbose=False, mode_label: str = "rollout"):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot apply")
        sys.exit(2)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    if dry_run:
        log.info("Dry-run: skipping kubectl apply")
        return
    files = [cfg["FILES"]["namespace"], cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text(encoding="utf-8") + "\n"
    res = kubectl_apply_yaml(combined, dry_run=False)
    if not res.get("applied", False):
        log.error("%s failed: %s", mode_label, res.get("stderr") or res.get("error"))
        sys.exit(2)

    rc = wait_for_rollout(cfg, timeout_seconds=cfg.get("ROLLOUT_TIMEOUT", 300))
    if rc != 0:
        log.error("%s: rollout failed (rc=%d)", mode_label, rc)
        sys.exit(2)

    summary = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image": cfg["IMAGE"],
        "namespace": cfg["NAMESPACE"],
        "replicas": cfg["REPLICAS"],
        "files": {k: str(v) for k, v in cfg["FILES"].items()},
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    log.info("%s complete; applied manifests to cluster and wrote deploy summary", mode_label)

def delete_manifests(cfg):
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
            if cfg["INPUTS_HASH_PATH"].exists():
                cfg["INPUTS_HASH_PATH"].unlink()
        except Exception:
            pass
        log.info("Deleted manifests at %s", str(cfg["MANIFESTS_DIR"]))
    else:
        log.info("Manifests dir not present: %s", str(cfg["MANIFESTS_DIR"]))

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/rollout/delete Reranker Kubernetes manifests.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--rollout", action="store_true", help="Create or converge resources to desired state (preferred over --apply).")
    grp.add_argument("--apply", action="store_true", help="Legacy alias for --rollout (deprecated).")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests.")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    if args.delete:
        delete_manifests(cfg)
        return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
    if args.rollout:
        apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose, mode_label="rollout")
        return
    if args.apply:
        log.warning("--apply is deprecated; use --rollout")
        apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose, mode_label="apply")
        return

if __name__ == "__main__":
    main()
