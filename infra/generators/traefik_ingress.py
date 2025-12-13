#!/usr/bin/env python3
"""
traefik_ingress.py

Generator for Traefik values + Traefik middlewares (rateLimit, stripPrefix, cors).
- Writes only non-secret manifests to disk.
- Applies secrets in-memory via kubectl (never writes secret YAML to disk).
- Ensures namespaces exist before applying secrets.
- Optionally installs Traefik via Helm if CRDs missing.
- Fail-fast, deterministic, idempotent.

Usage:
  python traefik_ingress.py --generate
  python traefik_ingress.py --apply --confirm
  python traefik_ingress.py --setup-traefik --confirm

Secrets input:
- SECRET_VALUES (JSON string) or SECRET_VALUES_FILE (path to file).
- JSON shape: list of objects or dict-of-objects:
  [
    {"name":"my-secret","namespace":"ns","type":"Opaque","data":{"KEY":"value"}}
  ]
"""

from __future__ import annotations
import os
import sys
import json
import base64
import hashlib
import subprocess
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import yaml

# -----------------------
# Utilities & helpers
# -----------------------
def die(msg: str) -> None:
    print("FATAL:", msg, file=sys.stderr)
    sys.exit(2)

def info(msg: str) -> None:
    print("INFO:", msg)

def warn(msg: str) -> None:
    print("WARN:", msg, file=sys.stderr)

def atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if k in ("FILES", "SECRET_VALUES", "INPUTS_HASH_PATH"):
            continue
        serial[k] = canonicalize(cfg[k])
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_kubectl(args: List[str], input_bytes: Optional[bytes] = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["kubectl"] + args
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True)
    # stream outputs for visibility
    if proc.stdout:
        sys.stdout.buffer.write(proc.stdout)
    if proc.stderr:
        sys.stderr.buffer.write(proc.stderr)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    return proc

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
    cfg["TRAEFIK_ENABLE_CLUSTER_RATELIMIT"] = os.getenv("TRAEFIK_ENABLE_CLUSTER_RATELIMIT", "false").lower() in ("1","true","yes")
    cfg["RATE_LIMIT_AVERAGE"] = int(os.getenv("RATE_LIMIT_AVERAGE", "100"))
    cfg["RATE_LIMIT_BURST"] = int(os.getenv("RATE_LIMIT_BURST", "200"))
    cfg["STRIPPREFIX_PATHS"] = [p for p in os.getenv("STRIPPREFIX_PATHS", "").split(",") if p]
    cfg["CORS_ALLOW_ORIGINS"] = os.getenv("CORS_ALLOW_ORIGINS", "")
    # Files emitted by this generator (non-secrets)
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
    # Secret inputs: either JSON string in SECRET_VALUES or a file path SECRET_VALUES_FILE
    cfg["SECRET_VALUES_RAW"] = os.getenv("SECRET_VALUES")
    cfg["SECRET_VALUES_FILE"] = os.getenv("SECRET_VALUES_FILE")
    cfg["SECRET_VALUES"] = []
    # parse secrets (fail-fast on bad JSON)
    raw = None
    if cfg["SECRET_VALUES_RAW"]:
        raw = cfg["SECRET_VALUES_RAW"]
    elif cfg["SECRET_VALUES_FILE"]:
        fp = Path(cfg["SECRET_VALUES_FILE"])
        if not fp.exists():
            die(f"SECRET_VALUES_FILE points to missing file: {fp}")
        raw = fp.read_text(encoding="utf-8")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            die(f"Invalid JSON in SECRET_VALUES / SECRET_VALUES_FILE: {e}")
        if isinstance(parsed, dict):
            secrets_list = []
            if "name" in parsed and "data" in parsed and "namespace" in parsed:
                secrets_list = [parsed]
            else:
                for k, v in parsed.items():
                    if not isinstance(v, dict):
                        die("SECRET_VALUES object must map secret-name -> object or be a list of secret objects")
                    new = {"name": k, **v}
                    secrets_list.append(new)
            cfg["SECRET_VALUES"] = secrets_list
        elif isinstance(parsed, list):
            cfg["SECRET_VALUES"] = parsed
        else:
            die("SECRET_VALUES JSON must be an object or an array of objects")
    # validate shape
    for s in cfg["SECRET_VALUES"]:
        if "name" not in s or "namespace" not in s or "data" not in s:
            die("Each secret must include at least 'name', 'namespace', and 'data' fields")
    return cfg

# -----------------------
# Renderers (non-secret)
# -----------------------
def render_namespace_obj(name: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}}

def render_traefik_values(kind: bool, cfg: Dict[str, Any]) -> str:
    svc_type = cfg["TRAEFIK_SERVICE_TYPE_KIND"] if kind else cfg["TRAEFIK_SERVICE_TYPE_AKS"]
    vals = {
        "replicas": cfg["TRAEFIK_REPLICAS"],
        "service": {"spec": {"type": svc_type}},
        "additionalArguments": ["--providers.kubernetescrd"],
        "providers": {"kubernetesCRD": {"enabled": True}, "kubernetesIngress": {"enabled": False}},
        "ports": {"web": {"port": 80}, "websecure": {"port": 443}},
        "ingressClass": {"enabled": True},
        "resources": {"requests": {"cpu": "250m", "memory": "256Mi"}, "limits": {"cpu": "1000m", "memory": "1Gi"}}
    }
    return safe_yaml(vals)

def render_mw_ratelimit(cfg: Dict[str, Any]) -> str:
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
# Secret helpers (in-memory apply)
# -----------------------
def build_secret_manifest(secret: Dict[str, Any]) -> Dict[str, Any]:
    name = secret["name"]
    ns = secret["namespace"]
    stype = secret.get("type", "Opaque")
    data = secret["data"]
    b64data: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, str):
            raw = v.encode("utf-8")
        elif isinstance(v, bytes):
            raw = v
        else:
            raw = json.dumps(v, separators=(",", ":"), sort_keys=True).encode("utf-8")
        b64data[k] = base64.b64encode(raw).decode("ascii")
    manifest = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": name, "namespace": ns}, "type": stype, "data": b64data}
    return manifest

def ensure_namespace_exists(name: str, timeout: int = 30):
    ns_obj = render_namespace_obj(name)
    ns_yaml = safe_yaml(ns_obj).encode("utf-8")
    try:
        run_kubectl(["apply", "-f", "-"], input_bytes=ns_yaml)
    except subprocess.CalledProcessError:
        die(f"Failed to apply namespace {name}")
    end = time.time() + timeout
    while time.time() < end:
        proc = subprocess.run(["kubectl", "get", "namespace", name], capture_output=True)
        if proc.returncode == 0:
            info(f"Namespace '{name}' present")
            return
        time.sleep(1)
    die(f"Timed out waiting for namespace '{name}' to become present")

def apply_secret_to_cluster(secret: Dict[str, Any]):
    name = secret["name"]
    ns = secret["namespace"]
    manifest = build_secret_manifest(secret)
    yaml_bytes = safe_yaml(manifest).encode("utf-8")
    try:
        run_kubectl(["apply", "-f", "-"], input_bytes=yaml_bytes)
        info(f"Applied secret '{name}' in namespace '{ns}'")
    except subprocess.CalledProcessError as e:
        die(f"kubectl apply for secret {name}/{ns} failed: {e}")

def apply_all_secrets(cfg: Dict[str, Any]):
    if not cfg["SECRET_VALUES"]:
        info("No secrets provided — skipping secret application.")
        return
    if not which("kubectl"):
        die("kubectl not found; secrets cannot be applied")
    namespaces = {s["namespace"] for s in cfg["SECRET_VALUES"]}
    for ns in sorted(namespaces):
        ensure_namespace_exists(ns)
    for s in cfg["SECRET_VALUES"]:
        info(f"Applying secret: name={s['name']} namespace={s['namespace']}")
        apply_secret_to_cluster(s)

# -----------------------
# Generation / apply / delete (non-secret)
# -----------------------
def ensure_dir(cfg: Dict[str, Any]) -> None:
    cfg["MANIFESTS_DIR"].mkdir(parents=True, exist_ok=True)

def generate(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    ensure_dir(cfg)
    ihash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["FILES"]["inputs_hash"].exists():
        existing = cfg["FILES"]["inputs_hash"].read_text(encoding="utf-8").strip()
    if existing == ihash and not dry_run:
        info("No non-secret changes; skipping generation.")
        return
    written: List[Path] = []
    ns_yaml = safe_yaml(render_namespace_obj(cfg["TRAEFIK_NAMESPACE"]))
    atomic_write(cfg["FILES"]["namespace"], ns_yaml); written.append(cfg["FILES"]["namespace"])
    atomic_write(cfg["FILES"]["traefik_values_kind"], render_traefik_values(True, cfg)); written.append(cfg["FILES"]["traefik_values_kind"])
    atomic_write(cfg["FILES"]["traefik_values_aks"], render_traefik_values(False, cfg)); written.append(cfg["FILES"]["traefik_values_aks"])
    if cfg["TRAEFIK_ENABLE_CLUSTER_RATELIMIT"]:
        atomic_write(cfg["FILES"]["mw_ratelimit"], render_mw_ratelimit(cfg)); written.append(cfg["FILES"]["mw_ratelimit"])
    else:
        try:
            cfg["FILES"]["mw_ratelimit"].unlink()
        except Exception:
            pass
    sp = render_mw_stripprefix(cfg)
    if sp:
        atomic_write(cfg["FILES"]["mw_stripprefix"], sp); written.append(cfg["FILES"]["mw_stripprefix"])
    else:
        try:
            cfg["FILES"]["mw_stripprefix"].unlink()
        except Exception:
            pass
    cp = render_mw_cors(cfg)
    if cp:
        atomic_write(cfg["FILES"]["mw_cors"], cp); written.append(cfg["FILES"]["mw_cors"])
    else:
        try:
            cfg["FILES"]["mw_cors"].unlink()
        except Exception:
            pass
    meta = {
        "traefik_namespace": cfg["TRAEFIK_NAMESPACE"],
        "ratelimit_middleware": "global-ratelimit" if cfg["TRAEFIK_ENABLE_CLUSTER_RATELIMIT"] else None,
        "stripprefix_middleware": "global-stripprefix" if cfg["STRIPPREFIX_PATHS"] else None,
        "cors_middleware": "global-cors" if cfg["CORS_ALLOW_ORIGINS"] else None,
        "files": {k: str(v) for k, v in cfg["FILES"].items()},
        "inputs_hash": ihash
    }
    atomic_write(cfg["FILES"]["meta"], json.dumps(meta, indent=2), mode=0o600)
    atomic_write(cfg["FILES"]["inputs_hash"], ihash)
    info(f"Wrote {len(written)} files: {', '.join([p.name for p in written])}")

# -----------------------
# Helm / Traefik setup
# -----------------------
def setup_traefik(cfg: Dict[str, Any], upgrade: bool = False, timeout: int = 180) -> None:
    if not which("helm"):
        die("helm not found; install Helm to auto-install Traefik and CRDs or run --setup-traefik manually.")
    ensure_dir(cfg)
    values_file = cfg["FILES"]["traefik_values_kind"] if cfg["KUBE_ENV"] == "kind" else cfg["FILES"]["traefik_values_aks"]
    if not values_file.exists():
        die("Traefik values not generated; run --generate first")
    info(f"Installing/upgrading Traefik release={cfg['TRAEFIK_RELEASE']} namespace={cfg['TRAEFIK_NAMESPACE']}")
    subprocess.run(["helm", "repo", "add", "traefik", "https://helm.traefik.io/traefik"], check=False)
    subprocess.run(["helm", "repo", "update"], check=False)
    cmd = ["helm", "upgrade", "--install", cfg["TRAEFIK_RELEASE"], "traefik/traefik",
           "--namespace", cfg["TRAEFIK_NAMESPACE"], "--create-namespace", "-f", str(values_file),
           "--version", cfg["TRAEFIK_CHART_VERSION"], "--wait", "--timeout", "180s"]
    proc = subprocess.run(cmd, capture_output=True)
    sys.stdout.buffer.write(proc.stdout); sys.stderr.buffer.write(proc.stderr)
    if proc.returncode != 0:
        die("helm upgrade/install failed; see output above.")
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
# Apply flow (secrets first, then manifests)
# -----------------------
def apply(cfg: Dict[str, Any], confirm: bool = False) -> None:
    if not confirm:
        die("Refusing to apply without --confirm")
    if not which("kubectl"):
        die("kubectl not found in PATH")
    generate(cfg, dry_run=False)
    ensure_namespace_exists(cfg["TRAEFIK_NAMESPACE"])
    apply_all_secrets(cfg)
    proc = subprocess.run(["kubectl", "api-resources"], capture_output=True)
    api_resources = proc.stdout.decode() if proc.stdout else ""
    need_crds = not ("ingressroutes" in api_resources and "middlewares" in api_resources)
    if need_crds:
        info("Traefik CRDs not found. Attempting Helm install for Traefik (will create CRDs).")
        setup_traefik(cfg, upgrade=False, timeout=300)
    order = [cfg["FILES"]["namespace"]]
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

# -----------------------
# Validate, delete
# -----------------------
def validate(cfg: Dict[str, Any]) -> None:
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

def delete_manifests(cfg: Dict[str, Any], confirm: bool = False) -> None:
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
def parse_args() -> Any:
    p = argparse = __import__("argparse")
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--setup-traefik", action="store_true")
    g.add_argument("--validate", action="store_true")
    g.add_argument("--delete", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.generate:
        generate(cfg, dry_run=args.dry_run)
        return
    if args.setup_traefik:
        generate(cfg, dry_run=False)
        setup_traefik(cfg, upgrade=args.upgrade)
        return
    if args.apply:
        generate(cfg, dry_run=args.dry_run)
        apply(cfg, confirm=args.confirm)
        return
    if args.validate:
        generate(cfg, dry_run=args.dry_run)
        validate(cfg)
        return
    if args.delete:
        delete_manifests(cfg, confirm=args.confirm)
        return

if __name__ == "__main__":
    main()
