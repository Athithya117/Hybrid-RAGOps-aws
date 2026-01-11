#!/usr/bin/env python3
"""
gen_sparse.py

Deterministic generator for Sparse embedder Kubernetes manifests.
Writes manifests to infra/manifests/sparse/

Usage:
  # generate files
  python gen_sparse.py --generate

  # rollout to cluster (create or converge resources)
  python gen_sparse.py --rollout

  # delete generated manifests
  python gen_sparse.py --delete
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

# -------------------- logging --------------------
logging.basicConfig(level=os.environ.get("GEN_SPARSE_LOGLEVEL", "INFO"))
log = logging.getLogger("gen_sparse")

# -------------------- helpers --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def run_cmd(cmd, capture=True, check=False, timeout=None, input_bytes=None):
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=capture, text=True, check=check, timeout=timeout)
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
        proc = subprocess.run(cmd, input=yaml_str.encode("utf-8"), capture_output=True, check=True, timeout=120)
        return {"applied": True, "stdout": proc.stdout.decode() if proc.stdout else ""}
    except subprocess.CalledProcessError as e:
        return {"applied": False, "stderr": e.stderr.decode() if e.stderr else str(e)}
    except subprocess.TimeoutExpired as e:
        return {"applied": False, "stderr": f"timeout: {e}"}

# -------------------- config loader (SPARSE_*) --------------------
def load_config():
    cfg = {}
    # environment context
    cfg["ENV"] = os.environ.get("SPARSE_ENV", os.environ.get("ENV", "STAGING")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/sparse"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    # image and runtime
    cfg["IMAGE"] = os.environ.get("SPARSE_IMAGE", "athithya5354/sparse:v1")
    cfg["NAMESPACE"] = os.environ.get("SPARSE_NAMESPACE", "models")
    cfg["SERVICE_NAME"] = os.environ.get("SPARSE_SERVICE_NAME", "sparse")
    cfg["CONTAINER_PORT"] = int(os.environ.get("SPARSE_PORT", "8201"))
    cfg["HOST"] = os.environ.get("SPARSE_HOST", "0.0.0.0")
    cfg["LOGLEVEL"] = os.environ.get("SPARSE_LOGLEVEL", "INFO")
    # replicas & resources (defaults differ by ENV)
    if cfg["ENV"] == "PROD":
        cfg.update({
            "REPLICAS": int(os.environ.get("SPARSE_REPLICAS", "3")),
            "CPU_REQUEST": os.environ.get("SPARSE_CPU_REQUEST", "1000m"),
            "CPU_LIMIT": os.environ.get("SPARSE_CPU_LIMIT", "4000m"),
            "MEMORY_REQUEST": os.environ.get("SPARSE_MEMORY_REQUEST", "1Gi"),
            "MEMORY_LIMIT": os.environ.get("SPARSE_MEMORY_LIMIT", "4Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("SPARSE_STARTUP_FAILURE_THRESHOLD", "24")),  # ~120s with 5s period
        })
    else:
        # staging defaults tuned for typical sparse model
        cfg.update({
            "REPLICAS": int(os.environ.get("SPARSE_REPLICAS", "1")),
            "CPU_REQUEST": os.environ.get("SPARSE_CPU_REQUEST", "250m"),
            "CPU_LIMIT": os.environ.get("SPARSE_CPU_LIMIT", "1000m"),
            "MEMORY_REQUEST": os.environ.get("SPARSE_MEMORY_REQUEST", "512Mi"),
            "MEMORY_LIMIT": os.environ.get("SPARSE_MEMORY_LIMIT", "1Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("SPARSE_STARTUP_FAILURE_THRESHOLD", "60")),  # ~300s with 5s period
        })
    # probe timings (overridable)
    cfg["PROBE_PERIOD_SECONDS"] = int(os.environ.get("SPARSE_PROBE_PERIOD_SECONDS", "5"))
    cfg["READINESS_INITIAL_DELAY"] = int(os.environ.get("SPARSE_READINESS_INITIAL_DELAY", "10"))
    cfg["LIVENESS_INITIAL_DELAY"] = int(os.environ.get("SPARSE_LIVENESS_INITIAL_DELAY", "30"))
    cfg["PROBE_TIMEOUT_SECONDS"] = int(os.environ.get("SPARSE_PROBE_TIMEOUT_SECONDS", "3"))
    # GPU support
    cfg["ENABLE_GPU"] = os.environ.get("SPARSE_ENABLE_GPU", "false").lower() in ("1", "true", "yes")
    cfg["GPU_RESOURCE_NAME"] = os.environ.get("SPARSE_GPU_RESOURCE", "nvidia.com/gpu")
    cfg["GPU_COUNT"] = os.environ.get("SPARSE_GPU_COUNT", "1")
    cfg["GPU_NODE_SELECTOR"] = os.environ.get("SPARSE_GPU_NODE_SELECTOR", "")
    # HPA
    cfg["HPA_ENABLED"] = os.environ.get("SPARSE_HPA_ENABLED", "false").lower() in ("1", "true", "yes")
    cfg["HPA_MIN"] = int(os.environ.get("SPARSE_HPA_MIN_REPLICAS", "1"))
    cfg["HPA_MAX"] = int(os.environ.get("SPARSE_HPA_MAX_REPLICAS", "10"))
    cfg["HPA_TARGET_CPU"] = int(os.environ.get("SPARSE_HPA_TARGET_CPU", "60"))
    # RBAC/serviceaccount names
    cfg["SA_NAME"] = os.environ.get("SPARSE_SA_NAME", f"{cfg['SERVICE_NAME']}-sa")
    cfg["ROLE_NAME"] = os.environ.get("SPARSE_ROLE_NAME", f"{cfg['SERVICE_NAME']}-role")
    cfg["ROLEBIND_NAME"] = os.environ.get("SPARSE_ROLEBIND_NAME", f"{cfg['SERVICE_NAME']}-rb")
    # metadata and labels
    cfg["LABELS"] = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "embedder",
        "app.kubernetes.io/managed-by": "gen_sparse",
        "app.kubernetes.io/instance": cfg["SERVICE_NAME"],
        "env": cfg["ENV"].lower(),
    }
    # output filenames
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
        "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "gen_sparse"}},
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

def render_deployment(cfg):
    labels = cfg["LABELS"].copy()
    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["CONTAINER_PORT"]}],
        "env": [
            {"name": "SPARSE_PORT", "value": str(cfg["CONTAINER_PORT"])},
            {"name": "ENV", "value": cfg["ENV"]},
            {"name": "SPARSE_LOGLEVEL", "value": cfg["LOGLEVEL"]},
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

    # GPU support: add resource limits if enabled
    if cfg["ENABLE_GPU"]:
        try:
            gcount = int(cfg["GPU_COUNT"])
        except Exception:
            gcount = 1
        container["resources"]["limits"][cfg["GPU_RESOURCE_NAME"]] = gcount

    pod_spec = {
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

    # If GPU node selector provided, add to pod spec (best-effort)
    if cfg["ENABLE_GPU"] and cfg["GPU_NODE_SELECTOR"]:
        pod_spec["spec"]["template"]["spec"]["nodeSelector"] = {k: v for k, v in [cfg["GPU_NODE_SELECTOR"].split("=", 1)]} if "=" in cfg["GPU_NODE_SELECTOR"] else {cfg["GPU_NODE_SELECTOR"]: "true"}

    # Add prometheus scraping annotations (optional standard)
    pod_spec["spec"]["template"]["metadata"].setdefault("annotations", {})
    pod_spec["spec"]["template"]["metadata"]["annotations"].update({
        "prometheus.io/scrape": "true",
        "prometheus.io/port": str(cfg["CONTAINER_PORT"]),
        "prometheus.io/path": "/metrics",
    })

    return yaml.safe_dump(pod_spec, sort_keys=False)

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
    # Generate HorizontalPodAutoscaler v2 (CPU percent)
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

# -------------------- generate / apply / delete --------------------
def generate_manifests(cfg, dry_run=False, verbose=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    inputs_hash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing = cfg["INPUTS_HASH_PATH"].read_text().strip()
    if existing == inputs_hash and not dry_run:
        log.info("No non-secret changes detected; generation skipped.")
        return

    ns_yaml = render_namespace(cfg)
    sa_role_yaml = render_sa_role(cfg)
    deploy_yaml = render_deployment(cfg)
    svc_yaml = render_service(cfg)
    atomic_write(cfg["FILES"]["namespace"], ns_yaml)
    atomic_write(cfg["FILES"]["sa_role"], sa_role_yaml)
    atomic_write(cfg["FILES"]["deployment"], deploy_yaml)
    atomic_write(cfg["FILES"]["service"], svc_yaml)
    if cfg["HPA_ENABLED"]:
        hpa_yaml = render_hpa(cfg)
        atomic_write(cfg["FILES"]["hpa"], hpa_yaml)
    # save inputs hash
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash)
    log.info("Wrote manifests to %s", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        log.info("Namespace:\n%s", ns_yaml.splitlines()[:20])
        log.info("Deployment (head):\n%s", deploy_yaml.splitlines()[:60])
    return

def apply_to_cluster(cfg, dry_run=False, verbose=False):
    # ensure kubectl exists
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot apply")
        sys.exit(2)
    # generate first (writes files)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    if dry_run:
        log.info("Dry-run: skipping kubectl apply")
        return
    # apply files in the manifest dir in deterministic order
    files = [cfg["FILES"]["namespace"], cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    res = kubectl_apply_yaml(combined, dry_run=False)
    if not res.get("applied", False):
        log.error("kubectl apply failed: %s", res.get("stderr") or res.get("error"))
        sys.exit(2)
    # write last_deploy_summary
    summary = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image": cfg["IMAGE"],
        "namespace": cfg["NAMESPACE"],
        "replicas": cfg["REPLICAS"],
        "files": {k: str(v) for k, v in cfg["FILES"].items()},
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    log.info("Applied manifests to cluster and wrote deploy summary")

# rollout is a semantic alias for apply in platform terms
def rollout_manifests(cfg, dry_run=False, verbose=False):
    log.info("rollout started")
    apply_to_cluster(cfg, dry_run=dry_run, verbose=verbose)
    log.info("rollout complete")

def delete_manifests(cfg):
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p)
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except FileNotFoundError:
            pass
        log.info("Deleted manifests at %s", str(cfg["MANIFESTS_DIR"]))
    else:
        log.info("Manifests dir not present: %s", str(cfg["MANIFESTS_DIR"]))

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/rollout/delete Sparse embedder Kubernetes manifests.")
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
        rollout_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
    if args.apply:
        log.warning("--apply is deprecated; use --rollout")
        rollout_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return

if __name__ == "__main__":
    main()
