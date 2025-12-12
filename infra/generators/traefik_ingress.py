#!/usr/bin/env python3
"""
gen_auth_and_ingress.py

Deterministic generator for Traefik values + Traefik middlewares (stripPrefix, cors).
Cluster-level rate-limiting is opt-in (TRAEFIK_ENABLE_CLUSTER_RATELIMIT).
This generator will automatically ensure Traefik (CRDs) is installed via Helm before applying middleware manifests.

Usage:
  python infra/generators/traefik_ingress.py --generate
  python infra/generators/traefik_ingress.py --apply --confirm
  python infra/generators/traefik_ingress.py --setup-traefik --confirm

Notes:
- Default behavior follows the platform plan: Front Door handles rate-limiting/WAF at the edge.
- Traefik is configured to use CRD provider only (IngressRoute CRDs).
- Idempotent: writes .inputs_hash and skips if unchanged (SECRET_VALUES excluded).
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
import subprocess
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

# -----------------------
# Helpers
# -----------------------
def die(msg: str):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)

def info(msg: str):
    print("INFO:", msg)

def atomic_write(path: Path, content: str, mode: int = 0o600):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    try:
        path.chmod(mode)
    except Exception:
        pass

def safe_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)

def canonicalize(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: canonicalize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [canonicalize(v) for v in o]
    return o

def canonical_inputs_hash(cfg: Dict[str, Any]) -> str:
    serial = {}
    for k in sorted(cfg.keys()):
        # skip files and secret / runtime-only keys
        if k in ("FILES", "SECRET_VALUES", "INPUTS_HASH_PATH"):
            continue
        serial[k] = canonicalize(cfg[k])
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

# -----------------------
# Config loader
# -----------------------
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["REPO_ROOT"] = Path(os.getenv("REPO_ROOT", Path.cwd()))
    cfg["KUBE_ENV"] = os.getenv("KUBE_ENV", os.getenv("ENV", "kind")).lower()
    if cfg["KUBE_ENV"] not in ("kind", "staging", "prod", "aks"):
        die("KUBE_ENV must be one of: kind, staging, prod, aks (or set ENV)")

    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", cfg["REPO_ROOT"] / "infra" / "manifests" / "auth"))
    cfg["TRAEFIK_RELEASE"] = os.getenv("TRAEFIK_RELEASE", "traefik")
    cfg["TRAEFIK_NAMESPACE"] = os.getenv("TRAEFIK_NAMESPACE", "kube-system")
    cfg["TRAEFIK_CHART_VERSION"] = os.getenv("TRAEFIK_CHART_VERSION", "20.3.1")
    cfg["TRAEFIK_REPLICAS"] = int(os.getenv("TRAEFIK_REPLICAS", "1" if cfg["KUBE_ENV"] == "kind" else "2"))
    cfg["TRAEFIK_SERVICE_TYPE_KIND"] = os.getenv("TRAEFIK_SERVICE_TYPE_KIND", "NodePort")
    cfg["TRAEFIK_SERVICE_TYPE_AKS"] = os.getenv("TRAEFIK_SERVICE_TYPE_AKS", "LoadBalancer")

    # middleware knobs (non-secret)
    # Rate-limit is opt-in; by default platform uses Front Door for rate-limiting.
    cfg["TRAEFIK_ENABLE_CLUSTER_RATELIMIT"] = os.getenv("TRAEFIK_ENABLE_CLUSTER_RATELIMIT", "false").lower() in ("1","true","yes")
    cfg["RATE_LIMIT_AVERAGE"] = int(os.getenv("RATE_LIMIT_AVERAGE", "100"))
    cfg["RATE_LIMIT_BURST"] = int(os.getenv("RATE_LIMIT_BURST", "200"))

    cfg["STRIPPREFIX_PATHS"] = [p for p in os.getenv("STRIPPREFIX_PATHS", "").split(",") if p]
    cfg["CORS_ALLOW_ORIGINS"] = os.getenv("CORS_ALLOW_ORIGINS", "")

    # Files emitted by this generator
    m = cfg["MANIFESTS_DIR"]
    cfg["FILES"] = {
        "namespace": m / "00-namespace.yaml",
        "traefik_values_kind": m / "01-traefik-values-kind.yaml",
        "traefik_values_aks": m / "02-traefik-values-aks.yaml",
        "mw_ratelimit": m / "10-middleware-ratelimit.yaml",
        "mw_stripprefix": m / "11-middleware-stripprefix.yaml",
        "mw_cors": m / "12-middleware-cors.yaml",
        "meta": m / "manifest.meta.json",
        "inputs_hash": m / ".inputs_hash",
    }

    # no secrets here — forward-auth and secrets are managed by a separate auth generator
    return cfg

# -----------------------
# Renderers
# -----------------------
def render_namespace(cfg: Dict[str, Any]) -> str:
    obj = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": cfg["TRAEFIK_NAMESPACE"]}}
    return safe_yaml(obj)

def render_traefik_values(kind: bool, cfg: Dict[str, Any]) -> str:
    """
    Produce Helm values for Traefik.
    Use CRD provider only (IngressRoute + Middleware CRDs). Do NOT enable classic KubernetesIngress provider.
    """
    svc_type = cfg["TRAEFIK_SERVICE_TYPE_KIND"] if kind else cfg["TRAEFIK_SERVICE_TYPE_AKS"]
    vals = {
        "replicas": cfg["TRAEFIK_REPLICAS"],
        "service": {"spec": {"type": svc_type}},
        # CRD provider only (preferred for IngressRoute)
        "additionalArguments": ["--providers.kubernetescrd"],
        "providers": {"kubernetesCRD": {"enabled": True}, "kubernetesIngress": {"enabled": False}},
        "ports": {"web": {"port": 80}, "websecure": {"port": 443}},
        "ingressClass": {"enabled": True},
        # ensure sensible resources defaults (operators can override)
        "resources": {
            "requests": {"cpu": "250m", "memory": "256Mi"},
            "limits": {"cpu": "1000m", "memory": "1Gi"}
        },
    }
    return safe_yaml(vals)

def render_mw_ratelimit(cfg: Dict[str, Any]) -> str:
    """
    Render cluster rate-limit middleware ONLY when explicitly enabled.
    Default: disabled (platform expects Front Door to rate-limit at edge).
    """
    mw = {
        "apiVersion": "traefik.containo.us/v1alpha1",
        "kind": "Middleware",
        "metadata": {"name": "global-ratelimit", "namespace": cfg["TRAEFIK_NAMESPACE"]},
        "spec": {"rateLimit": {"average": cfg["RATE_LIMIT_AVERAGE"], "burst": cfg["RATE_LIMIT_BURST"]}}
    }
    return safe_yaml(mw)

def render_mw_stripprefix(cfg: Dict[str, Any]) -> str:
    if not cfg["STRIPPREFIX_PATHS"]:
        return ""
    mw = {
        "apiVersion": "traefik.containo.us/v1alpha1",
        "kind": "Middleware",
        "metadata": {"name": "global-stripprefix", "namespace": cfg["TRAEFIK_NAMESPACE"]},
        "spec": {"stripPrefix": {"prefixes": cfg["STRIPPREFIX_PATHS"]}}
    }
    return safe_yaml(mw)

def render_mw_cors(cfg: Dict[str, Any]) -> str:
    if not cfg["CORS_ALLOW_ORIGINS"]:
        return ""
    origins = [o.strip() for o in cfg["CORS_ALLOW_ORIGINS"].split(",") if o.strip()]
    mw = {
        "apiVersion": "traefik.containo.us/v1alpha1",
        "kind": "Middleware",
        "metadata": {"name": "global-cors", "namespace": cfg["TRAEFIK_NAMESPACE"]},
        "spec": {
            "headers": {
                "accessControlAllowOriginList": origins,
                "accessControlAllowMethods": ["GET", "POST", "OPTIONS"],
                "accessControlAllowHeaders": ["Authorization", "Content-Type", "X-Requested-With"]
            }
        }
    }
    return safe_yaml(mw)

# -----------------------
# Generate / apply / validate / delete
# -----------------------
def ensure_dir(cfg: Dict[str, Any]):
    cfg["MANIFESTS_DIR"].mkdir(parents=True, exist_ok=True)

def generate(cfg: Dict[str, Any], dry_run: bool = False):
    ensure_dir(cfg)
    ihash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["FILES"]["inputs_hash"].exists():
        existing = cfg["FILES"]["inputs_hash"].read_text(encoding="utf-8").strip()
    if existing == ihash and not dry_run:
        info("No non-secret changes; skipping generation.")
        return

    written = []
    # Namespace
    atomic_write(cfg["FILES"]["namespace"], render_namespace(cfg)); written.append(cfg["FILES"]["namespace"])

    # Traefik Helm values (kind vs aks)
    atomic_write(cfg["FILES"]["traefik_values_kind"], render_traefik_values(True, cfg)); written.append(cfg["FILES"]["traefik_values_kind"])
    atomic_write(cfg["FILES"]["traefik_values_aks"], render_traefik_values(False, cfg)); written.append(cfg["FILES"]["traefik_values_aks"])

    # Rate limit middleware only if explicitly enabled
    if cfg["TRAEFIK_ENABLE_CLUSTER_RATELIMIT"]:
        atomic_write(cfg["FILES"]["mw_ratelimit"], render_mw_ratelimit(cfg)); written.append(cfg["FILES"]["mw_ratelimit"])
    else:
        try:
            cfg["FILES"]["mw_ratelimit"].unlink()
        except Exception:
            pass

    # stripPrefix middleware (optional)
    sp = render_mw_stripprefix(cfg)
    if sp:
        atomic_write(cfg["FILES"]["mw_stripprefix"], sp); written.append(cfg["FILES"]["mw_stripprefix"])
    else:
        try:
            cfg["FILES"]["mw_stripprefix"].unlink()
        except Exception:
            pass

    # cors middleware (optional)
    cp = render_mw_cors(cfg)
    if cp:
        atomic_write(cfg["FILES"]["mw_cors"], cp); written.append(cfg["FILES"]["mw_cors"])
    else:
        try:
            cfg["FILES"]["mw_cors"].unlink()
        except Exception:
            pass

    # manifest.meta.json describes middleware names and namespace.
    # forward_auth_middleware intentionally left null — separate auth generator provides it.
    meta = {
        "traefik_namespace": cfg["TRAEFIK_NAMESPACE"],
        "ratelimit_middleware": "global-ratelimit" if cfg["TRAEFIK_ENABLE_CLUSTER_RATELIMIT"] else None,
        "stripprefix_middleware": "global-stripprefix" if cfg["STRIPPREFIX_PATHS"] else None,
        "cors_middleware": "global-cors" if cfg["CORS_ALLOW_ORIGINS"] else None,
        "forward_auth_middleware": None,
        "files": {k: str(v) for k, v in cfg["FILES"].items()},
        "inputs_hash": ihash
    }
    atomic_write(cfg["FILES"]["meta"], json.dumps(meta, indent=2), mode=0o600)
    atomic_write(cfg["FILES"]["inputs_hash"], ihash)
    info(f"Wrote {len(written)} files: {', '.join([p.name for p in written])}")

# -----------------------
# Ensure Traefik CRDs / Helm install helper
# -----------------------
def setup_traefik(cfg: Dict[str, Any], upgrade: bool = False, timeout: int = 180):
    """Install or upgrade Traefik via Helm. This will create CRDs.
    Uses helm upgrade --install which is idempotent.
    """
    if not which("helm"):
        die("helm not found; install Helm to auto-install Traefik and CRDs or run --setup-traefik manually on a system with helm.")
    ensure_dir = cfg["MANIFESTS_DIR"]; ensure_dir.mkdir(parents=True, exist_ok=True)
    values_file = cfg["FILES"]["traefik_values_kind"] if cfg["KUBE_ENV"] == "kind" else cfg["FILES"]["traefik_values_aks"]
    if not values_file.exists():
        die("Traefik values not generated; run --generate first")
    info(f"Installing/upgrading Traefik release={cfg['TRAEFIK_RELEASE']} namespace={cfg['TRAEFIK_NAMESPACE']}")
    subprocess.run(["helm", "repo", "add", "traefik", "https://helm.traefik.io/traefik"], check=False)
    subprocess.run(["helm", "repo", "update"], check=False)
    cmd = ["helm", "upgrade", "--install", cfg["TRAEFIK_RELEASE"], "traefik/traefik",
           "--namespace", cfg["TRAEFIK_NAMESPACE"], "--create-namespace", "-f", str(values_file),
           "--version", cfg["TRAEFIK_CHART_VERSION"], "--wait", "--timeout", "180s"]
    # run the helm command; it's idempotent
    proc = subprocess.run(cmd, capture_output=True)
    sys.stdout.buffer.write(proc.stdout); sys.stderr.buffer.write(proc.stderr)
    if proc.returncode != 0:
        die("helm upgrade/install failed; see output above.")
    # wait for CRD presence (api-resources) up to timeout
    end = time.time() + timeout
    while time.time() < end:
        proc = subprocess.run(["kubectl", "api-resources"], capture_output=True)
        out = proc.stdout.decode() if proc.stdout else ""
        if "ingressroutes" in out and "middlewares" in out:
            info("Traefik CRDs present")
            break
        time.sleep(2)
    else:
        die("Timed out waiting for Traefik CRDs after Helm install")

# -----------------------
# Apply (auto-install CRDs if needed) and other operations
# -----------------------
def apply(cfg: Dict[str, Any], confirm: bool = False):
    if not confirm:
        die("Refusing to apply without --confirm")
    if not which("kubectl"):
        die("kubectl not found in PATH")

    # Generate files first (idempotent)
    generate(cfg, dry_run=False)

    # check CRDs presence (ingressroutes, middlewares)
    proc = subprocess.run(["kubectl", "api-resources"], capture_output=True)
    api_resources = proc.stdout.decode() if proc.stdout else ""
    need_crds = not ("ingressroutes" in api_resources and "middlewares" in api_resources)
    if need_crds:
        info("Traefik CRDs not found in cluster. Attempting to install Traefik via Helm (this will also create CRDs).")
        setup_traefik(cfg, upgrade=False, timeout=300)

    # Now apply the manifests (namespace + middlewares)
    order = [cfg["FILES"]["namespace"]]
    # include optional files when present; rate-limit only if file exists (enabled)
    if cfg["FILES"]["mw_ratelimit"].exists():
        order.append(cfg["FILES"]["mw_ratelimit"])
    if cfg["FILES"]["mw_stripprefix"].exists():
        order.append(cfg["FILES"]["mw_stripprefix"])
    if cfg["FILES"]["mw_cors"].exists():
        order.append(cfg["FILES"]["mw_cors"])

    combined = ""
    for p in order:
        combined += f"---\n# source: {p.name}\n" + p.read_text(encoding="utf-8") + "\n"

    info("Applying ingress/middleware manifests (kubectl apply -f -)")
    proc = subprocess.run(["kubectl", "apply", "-f", "-"], input=combined.encode(), capture_output=True)
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    if proc.returncode != 0:
        die("kubectl apply failed")
    info("Applied ingress/middleware manifests")

def validate(cfg: Dict[str, Any]):
    if not which("kubectl"):
        info("kubectl not found; skipping validation")
        return
    for k, p in cfg["FILES"].items():
        if not isinstance(p, Path) or not p.exists():
            continue
        info(f"Validating {p.name} (kubectl --dry-run=client)")
        proc = subprocess.run(["kubectl", "apply", "--dry-run=client", "-f", str(p)], capture_output=True)
        print(proc.stdout.decode(), proc.stderr.decode())
        if proc.returncode != 0:
            die(f"Validation failed for {p.name}")

def delete_manifests(cfg: Dict[str, Any], confirm: bool = False):
    if not confirm:
        die("Refusing to delete without --confirm")
    d = cfg["MANIFESTS_DIR"]
    if not d.exists():
        info("No manifests to delete")
        return
    for p in sorted(d.glob("*")):
        try:
            p.unlink()
        except IsADirectoryError:
            shutil.rmtree(p)
    info("Deleted generated ingress/middleware manifests")

# -----------------------
# CLI
# -----------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--setup-traefik", action="store_true")
    g.add_argument("--validate", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--upgrade", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    if args.generate:
        generate(cfg, dry_run=args.dry_run); return
    if args.setup_traefik:
        generate(cfg, dry_run=False)
        setup_traefik(cfg, upgrade=args.upgrade); return
    if args.apply:
        generate(cfg, dry_run=args.dry_run)
        apply(cfg, confirm=args.confirm); return
    if args.validate:
        generate(cfg, dry_run=args.dry_run)
        validate(cfg); return
    if args.delete:
        delete_manifests(cfg, confirm=args.confirm); return

if __name__ == "__main__":
    main()
