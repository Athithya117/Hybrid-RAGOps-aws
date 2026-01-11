#!/usr/bin/env python3
"""
gen_dense.py

Deterministic generator for Dense embedder Kubernetes manifests.
Writes manifests to infra/manifests/dense/

Usage:
  # generate files
  python gen_dense.py --generate

  # rollout (create or converge resources to desired state)
  python gen_dense.py --rollout

  # delete generated manifests
  python gen_dense.py --delete
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
logging.basicConfig(level=os.environ.get("GEN_DENSE_LOGLEVEL", "INFO"))
log = logging.getLogger("gen_dense")

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

# -------------------- config loader (DENSE_*) --------------------
def load_config():
    cfg = {}
    # environment context
    cfg["ENV"] = os.environ.get("DENSE_ENV", os.environ.get("ENV", "PROD")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/dense"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    # image and runtime
    cfg["IMAGE"] = os.environ.get("DENSE_IMAGE", "athithya5354/dense:v1")
    cfg["NAMESPACE"] = os.environ.get("DENSE_NAMESPACE", "models")
    cfg["SERVICE_NAME"] = os.environ.get("DENSE_SERVICE_NAME", "dense")
    cfg["CONTAINER_PORT"] = int(os.environ.get("DENSE_PORT", "8200"))
    cfg["HOST"] = os.environ.get("DENSE_HOST", "0.0.0.0")
    cfg["LOGLEVEL"] = os.environ.get("DENSE_LOGLEVEL", "INFO")
    # replicas & resources (defaults differ by ENV)
    if cfg["ENV"] == "STAGING":
        cfg.update({
            "REPLICAS": int(os.environ.get("DENSE_REPLICAS", "3")),
            "CPU_REQUEST": os.environ.get("DENSE_CPU_REQUEST", "1000m"),
            "CPU_LIMIT": os.environ.get("DENSE_CPU_LIMIT", "4000m"),
            "MEMORY_REQUEST": os.environ.get("DENSE_MEMORY_REQUEST", "1Gi"),
            "MEMORY_LIMIT": os.environ.get("DENSE_MEMORY_LIMIT", "4Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("DENSE_STARTUP_FAILURE_THRESHOLD", "24")),  # ~120s with 5s period
        })
    else:
        # staging defaults tuned for ~100MB ONNX 368-dim model
        cfg.update({
            "REPLICAS": int(os.environ.get("DENSE_REPLICAS", "1")),
            "CPU_REQUEST": os.environ.get("DENSE_CPU_REQUEST", "250m"),
            "CPU_LIMIT": os.environ.get("DENSE_CPU_LIMIT", "1000m"),
            "MEMORY_REQUEST": os.environ.get("DENSE_MEMORY_REQUEST", "512Mi"),
            "MEMORY_LIMIT": os.environ.get("DENSE_MEMORY_LIMIT", "1Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("DENSE_STARTUP_FAILURE_THRESHOLD", "60")),  # ~300s with 5s period
        })
    # probe timings (overridable)
    cfg["PROBE_PERIOD_SECONDS"] = int(os.environ.get("DENSE_PROBE_PERIOD_SECONDS", "5"))
    cfg["READINESS_INITIAL_DELAY"] = int(os.environ.get("DENSE_READINESS_INITIAL_DELAY", "10"))
    cfg["LIVENESS_INITIAL_DELAY"] = int(os.environ.get("DENSE_LIVENESS_INITIAL_DELAY", "30"))
    cfg["PROBE_TIMEOUT_SECONDS"] = int(os.environ.get("DENSE_PROBE_TIMEOUT_SECONDS", "3"))
    # GPU support
    cfg["ENABLE_GPU"] = os.environ.get("DENSE_ENABLE_GPU", "false").lower() in ("1", "true", "yes")
    cfg["GPU_RESOURCE_NAME"] = os.environ.get("DENSE_GPU_RESOURCE", "nvidia.com/gpu")
    cfg["GPU_COUNT"] = os.environ.get("DENSE_GPU_COUNT", "1")
    cfg["GPU_NODE_SELECTOR"] = os.environ.get("DENSE_GPU_NODE_SELECTOR", "")
    # HPA
    cfg["HPA_ENABLED"] = os.environ.get("DENSE_HPA_ENABLED", "false").lower() in ("1", "true", "yes")
    cfg["HPA_MIN"] = int(os.environ.get("DENSE_HPA_MIN_REPLICAS", "1"))
    cfg["HPA_MAX"] = int(os.environ.get("DENSE_HPA_MAX_REPLICAS", "10"))
    cfg["HPA_TARGET_CPU"] = int(os.environ.get("DENSE_HPA_TARGET_CPU", "60"))
    # RBAC/serviceaccount names
    cfg["SA_NAME"] = os.environ.get("DENSE_SA_NAME", f"{cfg['SERVICE_NAME']}-sa")
    cfg["ROLE_NAME"] = os.environ.get("DENSE_ROLE_NAME", f"{cfg['SERVICE_NAME']}-role")
    cfg["ROLEBIND_NAME"] = os.environ.get("DENSE_ROLEBIND_NAME", f"{cfg['SERVICE_NAME']}-rb")
    # metadata and labels
    cfg["LABELS"] = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "embedder",
        "app.kubernetes.io/managed-by": "gen_dense",
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
        "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "gen_dense"}},
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
            {"name": "DENSE_PORT", "value": str(cfg["CONTAINER_PORT"])},
            {"name": "ENV", "value": cfg["ENV"]},
            {"name": "DENSE_LOGLEVEL", "value": cfg["LOGLEVEL"]},
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
        # startupProbe should tolerate long model warmups
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

# -------------------- generate / rollout / delete --------------------
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

def apply_to_cluster(cfg, dry_run=False, verbose=False, mode_label: str = "rollout"):
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
    log.info("%s: applying manifests to cluster", mode_label)
    # apply files in the manifest dir in deterministic order
    files = [cfg["FILES"]["namespace"], cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    res = kubectl_apply_yaml(combined, dry_run=False)
    if not res.get("applied", False):
        log.error("%s failed: %s", mode_label, res.get("stderr") or res.get("error"))
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
    log.info("%s complete; wrote deploy summary", mode_label)

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
    p = argparse.ArgumentParser(description="Generate/rollout/delete Dense embedder Kubernetes manifests.")
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
