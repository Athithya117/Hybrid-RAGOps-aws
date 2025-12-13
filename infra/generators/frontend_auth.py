from __future__ import annotations
import os
import sys
import json
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml
from urllib.parse import urlparse

# -----------------------------
# Utility & Logging
# -----------------------------
def die(msg: str):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)

def info(msg: str):
    print("INFO:", msg)

def warn(msg: str):
    print("WARN:", msg, file=sys.stderr)

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

def make_serializable(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: make_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [make_serializable(v) for v in o]
    return o

def canonical_inputs_hash(cfg: Dict[str, Any]) -> str:
    serial = {}
    for k in sorted(cfg.keys()):
        if k in ("SECRET_VALUES", "INPUTS_HASH_PATH"):
            continue
        serial[k] = make_serializable(cfg[k])
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run(cmd: List[str], input_bytes: Optional[bytes] = None, check: bool = False) -> subprocess.CompletedProcess:
    info(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True)
    if proc.stdout:
        sys.stdout.buffer.write(proc.stdout)
    if proc.stderr:
        sys.stderr.buffer.write(proc.stderr)
    if check and proc.returncode != 0:
        die(f"Command failed: {' '.join(cmd)} (exit {proc.returncode})")
    return proc

# -----------------------------
# Config loader
# -----------------------------
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["REPO_ROOT"] = Path(os.getenv("REPO_ROOT", Path.cwd()))
    cfg["KUBE_ENV"] = os.getenv("KUBE_ENV", os.getenv("ENV", "STAGING")).lower()
    if cfg["KUBE_ENV"] not in ("kind", "staging", "prod", "aks"):
        die("KUBE_ENV must be one of kind/staging/prod/aks")

    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", cfg["REPO_ROOT"] / "infra" / "manifests" / "frontend"))
    cfg["FRONTEND_NAMESPACE"] = os.getenv("FRONTEND_NAMESPACE", "inference")
    cfg["SERVICE_NAME"] = os.getenv("FRONTEND_SERVICE_NAME", "frontend")
    cfg["IMAGE"] = os.getenv("FRONTEND_IMAGE", "athithya5354/frontend-and-auth:v5")
    cfg["PORT"] = int(os.getenv("FRONTEND_PORT", "8000"))
    cfg["REPLICAS"] = int(os.getenv("FRONTEND_REPLICAS", "1" if cfg["KUBE_ENV"] == "kind" else "3"))
    cfg["CPU_REQUEST"] = os.getenv("FRONTEND_CPU_REQUEST", "100m")
    cfg["CPU_LIMIT"] = os.getenv("FRONTEND_CPU_LIMIT", "500m")
    cfg["MEMORY_REQUEST"] = os.getenv("FRONTEND_MEMORY_REQUEST", "128Mi")
    cfg["MEMORY_LIMIT"] = os.getenv("FRONTEND_MEMORY_LIMIT", "512Mi")
    cfg["HPA_ENABLED"] = os.getenv("FRONTEND_HPA_ENABLED", "false").lower() in ("1", "true", "yes")
    cfg["HPA_MIN"] = int(os.getenv("FRONTEND_HPA_MIN", "1"))
    cfg["HPA_MAX"] = int(os.getenv("FRONTEND_HPA_MAX", "5"))

    cfg["QUERY_URL"] = os.getenv("QUERY_URL", "http://retrieval-svc.inference.svc.cluster.local:8001")
    cfg["FRONTEND_URL"] = os.getenv("FRONTEND_URL", "https://frontend.local")
    parsed = urlparse(cfg["FRONTEND_URL"])
    cfg["FRONTEND_HOST"] = parsed.hostname or "frontend.local"
    cfg["FRONTEND_SCHEME"] = parsed.scheme or "https"
    cfg["REDIRECT_URI_ENTRA"] = cfg["FRONTEND_URL"].rstrip("/") + "/auth/callback/entra"
    cfg["REDIRECT_URI_EXTERNAL"] = cfg["FRONTEND_URL"].rstrip("/") + "/auth/callback/external-id"

    cfg["AUTH_MODE"] = os.getenv("AUTH_MODE", "external-id")
    cfg["OIDC_ISSUER"] = os.getenv("OIDC_ISSUER", "")
    cfg["OIDC_JWKS_URI"] = os.getenv("OIDC_JWKS_URI", "")
    cfg["OIDC_AUDIENCE"] = os.getenv("OIDC_AUDIENCE", "")
    cfg["SPA_CLIENT_ID"] = os.getenv("SPA_CLIENT_ID", "")
    cfg["AZURE_TENANT_ID"] = os.getenv("AZURE_TENANT_ID", "")
    cfg["JWKS_REFRESH_INTERVAL_SECONDS"] = int(os.getenv("JWKS_REFRESH_INTERVAL_SECONDS", "900"))

    cfg["ENABLE_CORS"] = os.getenv("ENABLE_CORS", "false").lower() in ("1", "true", "yes")
    cfg["CORS_ALLOWED_ORIGINS"] = os.getenv("CORS_ALLOWED_ORIGINS", cfg["FRONTEND_URL"])

    cfg["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")

    # Secret names mapping: env var -> key in k8s secret
    cfg["SECRET_NAMES"] = {"JWT_SECRET": "jwt_secret", "SESSION_SECRET": "session_secret"}
    secret_values: Dict[str, str] = {}
    for envk in cfg["SECRET_NAMES"].keys():
        v = os.getenv(envk)
        if v:
            secret_values[envk] = v
    cfg["SECRET_VALUES"] = secret_values
    cfg["ALLOW_MISSING_SECRETS"] = os.getenv("ALLOW_MISSING_SECRETS", "false").lower() in ("1", "true", "yes")

    m = cfg["MANIFESTS_DIR"]
    cfg["FILES"] = {
        "namespace": m / "00-namespace.yaml",
        "sa_role": m / "01-sa-role.yaml",
        "configmap": m / "02-configmap.yaml",
        # secret file intentionally omitted (we apply secrets directly)
        "deployment": m / "04-deployment.yaml",
        "service": m / "05-service.yaml",
        "hpa": m / "06-hpa.yaml",
        "ingressroute": m / "07-ingressroute.yaml",
        "inputs_hash": m / ".inputs_hash"
    }

    cfg["AUTH_META_PATH"] = Path(os.getenv("AUTH_META_PATH", cfg["REPO_ROOT"] / "infra" / "manifests" / "auth" / "manifest.meta.json"))

    # Runtime validation: if secrets are required but absent, fail fast unless allowed
    if not cfg["SECRET_VALUES"] and not cfg["ALLOW_MISSING_SECRETS"]:
        warn("No SECRET_VALUES found in environment and ALLOW_MISSING_SECRETS is false. If you intend to proceed without secrets, set ALLOW_MISSING_SECRETS=true.")
    return cfg

# -----------------------------
# Renderers
# -----------------------------
def render_namespace(cfg: Dict[str, Any]) -> str:
    ns = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": cfg["FRONTEND_NAMESPACE"], "labels": {"app": "frontend"}}}
    return safe_yaml(ns)

def render_sa_role(cfg: Dict[str, Any]) -> str:
    sa = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": f"{cfg['SERVICE_NAME']}-sa", "namespace": cfg["FRONTEND_NAMESPACE"]}}
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-role", "namespace": cfg["FRONTEND_NAMESPACE"]},
        "rules": [
            {"apiGroups": [""], "resources": ["pods", "services", "endpoints", "configmaps"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get"]}
        ],
    }
    rb = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-rb", "namespace": cfg["FRONTEND_NAMESPACE"]},
        "subjects": [{"kind": "ServiceAccount", "name": f"{cfg['SERVICE_NAME']}-sa", "namespace": cfg["FRONTEND_NAMESPACE"]}],
        "roleRef": {"kind": "Role", "name": f"{cfg['SERVICE_NAME']}-role", "apiGroup": "rbac.authorization.k8s.io"},
    }
    return "\n---\n".join([safe_yaml(x) for x in (sa, role, rb)])

def render_configmap(cfg: Dict[str, Any]) -> str:
    data = {
        "QUERY_URL": cfg["QUERY_URL"],
        "FRONTEND_URL": cfg["FRONTEND_URL"],
        "FRONTEND_HOST": cfg["FRONTEND_HOST"],
        "PORT": str(cfg["PORT"]),
        "ENV": cfg["KUBE_ENV"].upper(),
        "AUTH_MODE": cfg["AUTH_MODE"],
        "OIDC_ISSUER": cfg["OIDC_ISSUER"] or "",
        "OIDC_JWKS_URI": cfg["OIDC_JWKS_URI"] or "",
        "OIDC_AUDIENCE": cfg["OIDC_AUDIENCE"],
        "SPA_CLIENT_ID": cfg["SPA_CLIENT_ID"],
        "AZURE_TENANT_ID": cfg["AZURE_TENANT_ID"],
        "JWKS_REFRESH_INTERVAL_SECONDS": str(cfg["JWKS_REFRESH_INTERVAL_SECONDS"]),
        "ENABLE_CORS": "true" if cfg["ENABLE_CORS"] else "false",
        "CORS_ALLOWED_ORIGINS": cfg["CORS_ALLOWED_ORIGINS"],
        "LOG_LEVEL": cfg["LOG_LEVEL"],
    }
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": f"{cfg['SERVICE_NAME']}-config", "namespace": cfg["FRONTEND_NAMESPACE"]}, "data": data}
    return safe_yaml(cm)

def render_deployment(cfg: Dict[str, Any]) -> str:
    labels = {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}
    envFrom = [{"configMapRef": {"name": f"{cfg['SERVICE_NAME']}-config"}}]
    # secrets are provided via secretRef at runtime if present
    if cfg["SECRET_VALUES"]:
        envFrom.append({"secretRef": {"name": f"{cfg['SERVICE_NAME']}-secret"}})
    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["PORT"]}],
        "envFrom": envFrom,
        "resources": {
            "requests": {"cpu": cfg["CPU_REQUEST"], "memory": cfg["MEMORY_REQUEST"]},
            "limits": {"cpu": cfg["CPU_LIMIT"], "memory": cfg["MEMORY_LIMIT"]},
        },
        "livenessProbe": {"httpGet": {"path": "/orchestrator/health", "port": cfg["PORT"]}, "initialDelaySeconds": 10, "periodSeconds": 10},
        "readinessProbe": {"httpGet": {"path": "/orchestrator/health", "port": cfg["PORT"]}, "initialDelaySeconds": 3, "periodSeconds": 5},
    }
    pod = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-deployment", "namespace": cfg["FRONTEND_NAMESPACE"], "labels": labels},
        "spec": {"replicas": cfg["REPLICAS"], "selector": {"matchLabels": labels}, "template": {"metadata": {"labels": labels}, "spec": {"serviceAccountName": f"{cfg['SERVICE_NAME']}-sa", "containers": [container]}}},
    }
    return safe_yaml(pod)

def render_service(cfg: Dict[str, Any]) -> str:
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-svc", "namespace": cfg["FRONTEND_NAMESPACE"]},
        "spec": {
            "type": "ClusterIP",
            "ports": [{"port": cfg["PORT"], "targetPort": cfg["PORT"], "protocol": "TCP", "name": "http"}],
            "selector": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]},
        },
    }
    return safe_yaml(svc)

def render_hpa(cfg: Dict[str, Any]) -> str:
    if not cfg["HPA_ENABLED"]:
        return ""
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-hpa", "namespace": cfg["FRONTEND_NAMESPACE"]},
        "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": f"{cfg['SERVICE_NAME']}-deployment"}, "minReplicas": cfg["HPA_MIN"], "maxReplicas": cfg["HPA_MAX"], "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": 60}}}]} ,
    }
    return safe_yaml(hpa)

def load_auth_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = cfg["AUTH_META_PATH"]
    if not path.exists():
        info(f"Auth meta not found at {path} (set AUTH_META_PATH or run auth generator first).")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"Failed reading auth meta: {e}")

def render_ingressroute(cfg: Dict[str, Any], meta: Dict[str, Any]) -> str:
    middlewares = []
    ns = meta.get("traefik_namespace") or meta.get("auth_namespace")
    if meta.get("forward_auth_middleware"):
        middlewares.append({"name": meta["forward_auth_middleware"], "namespace": ns})
    if meta.get("stripprefix_middleware"):
        middlewares.append({"name": meta["stripprefix_middleware"], "namespace": ns})
    if meta.get("cors_middleware"):
        middlewares.append({"name": meta["cors_middleware"], "namespace": ns})
    if meta.get("ratelimit_middleware"):
        middlewares.append({"name": meta["ratelimit_middleware"], "namespace": ns})

    host = os.getenv("FRONTEND_HOST", cfg["FRONTEND_HOST"])
    ir = {
        "apiVersion": "traefik.containo.us/v1alpha1",
        "kind": "IngressRoute",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-ingress", "namespace": cfg["FRONTEND_NAMESPACE"]},
        "spec": {"entryPoints": ["websecure"], "routes": [{"match": f"Host(`{host}`) && PathPrefix(`/`)", "kind": "Rule", "services": [{"name": f"{cfg['SERVICE_NAME']}-svc", "port": cfg["PORT"]}] }]},
    }
    if middlewares:
        ir["spec"]["routes"][0]["middlewares"] = [{"name": m["name"], "namespace": m["namespace"]} for m in middlewares]
    return safe_yaml(ir)

# -----------------------------
# Apply secrets directly (no files)
# -----------------------------
def apply_namespace_to_cluster(cfg: Dict[str, Any]):
    # Ensure namespace exists first
    ns_yaml = render_namespace(cfg).encode("utf-8")
    proc = run(["kubectl", "apply", "-f", "-"], input_bytes=ns_yaml)
    if proc.returncode != 0:
        die("Failed to create namespace")

def apply_secret_to_cluster(cfg: Dict[str, Any]):
    # If no secrets and missing allowed, skip
    if not cfg["SECRET_VALUES"]:
        info("No secrets to apply (SECRET_VALUES empty). Skipping secret creation.")
        return
    # Build kubectl create secret generic ... --from-literal=key=value ... --dry-run=client -o yaml
    secret_name = f"{cfg['SERVICE_NAME']}-secret"
    ns = cfg["FRONTEND_NAMESPACE"]
    create_cmd = ["kubectl", "create", "secret", "generic", secret_name, "--namespace", ns, "--dry-run=client", "-o", "yaml"]
    # Add --from-literal flags for each mapping (k8s key = mapped name)
    for env_var, k8s_key in cfg["SECRET_NAMES"].items():
        val = cfg["SECRET_VALUES"].get(env_var)
        if val is None:
            # missing secret: if allowed, skip this key, otherwise die
            if cfg["ALLOW_MISSING_SECRETS"]:
                warn(f"Secret env {env_var} missing; skipping that key in secret")
                continue
            else:
                die(f"Secret env {env_var} missing and ALLOW_MISSING_SECRETS=false")
        # --from-literal expects literal form; ensure we pass as single arg
        create_cmd.append(f"--from-literal={k8s_key}={val}")
    # Generate YAML then apply it (idempotent)
    info("Generating secret YAML via kubectl create --dry-run and applying")
    proc = subprocess.run(create_cmd, capture_output=True)
    if proc.returncode != 0:
        sys.stdout.buffer.write(proc.stdout); sys.stderr.buffer.write(proc.stderr)
        die("Failed to build secret YAML with kubectl")
    secret_yaml = proc.stdout
    # Apply secret YAML
    proc2 = run(["kubectl", "apply", "-f", "-"], input_bytes=secret_yaml)
    if proc2.returncode != 0:
        die("Failed to apply secret to cluster")
    info(f"Secret '{secret_name}' applied to namespace '{ns}'")

# -----------------------------
# Generation / Apply / Validate / Delete
# -----------------------------
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

    meta = load_auth_meta(cfg)

    atomic_write(cfg["FILES"]["namespace"], render_namespace(cfg))
    atomic_write(cfg["FILES"]["sa_role"], render_sa_role(cfg))
    atomic_write(cfg["FILES"]["configmap"], render_configmap(cfg))
    # secret intentionally NOT written to disk
    atomic_write(cfg["FILES"]["deployment"], render_deployment(cfg))
    atomic_write(cfg["FILES"]["service"], render_service(cfg))
    if cfg["HPA_ENABLED"]:
        atomic_write(cfg["FILES"]["hpa"], render_hpa(cfg))
    else:
        try:
            cfg["FILES"]["hpa"].unlink()
        except Exception:
            pass

    allow_missing = os.getenv("FRONTEND_ALLOW_MISSING_AUTH_META", "false").lower() in ("1", "true", "yes")
    if meta or allow_missing:
        ing = render_ingressroute(cfg, meta or {})
        atomic_write(cfg["FILES"]["ingressroute"], ing)
    else:
        info("Auth meta not present; skipping ingressroute generation.")
        try:
            cfg["FILES"]["ingressroute"].unlink()
        except Exception:
            pass

    atomic_write(cfg["FILES"]["inputs_hash"], ihash)
    info("Wrote frontend manifests to %s" % str(cfg["MANIFESTS_DIR"]))

def apply(cfg: Dict[str, Any], confirm: bool = False):
    if not confirm:
        die("Refusing to apply without --confirm")
    if not which("kubectl"):
        die("kubectl not found in PATH")

    # 1) Ensure namespace in cluster first
    apply_namespace_to_cluster(cfg)

    # 2) Apply secrets directly (namespace must exist)
    apply_secret_to_cluster(cfg)

    # 3) Apply remaining manifests from disk (sa/role, configmap, deployment, service, hpa, ingressroute)
    files: List[Path] = [cfg["FILES"]["sa_role"], cfg["FILES"]["configmap"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    if cfg["FILES"]["ingressroute"].exists():
        files.append(cfg["FILES"]["ingressroute"])
    combined = ""
    for p in files:
        if not p.exists():
            warn(f"Manifest {p} missing; skipping")
            continue
        combined += f"---\n# source: {p.name}\n" + p.read_text(encoding="utf-8") + "\n"
    if not combined:
        info("No manifests to apply (after filtering)")
        return
    proc = subprocess.run(["kubectl", "apply", "-f", "-"], input=combined.encode(), capture_output=True)
    sys.stdout.buffer.write(proc.stdout); sys.stderr.buffer.write(proc.stderr)
    if proc.returncode != 0:
        die("kubectl apply failed for manifests")
    info("Applied frontend manifests (non-secret resources)")

def validate(cfg: Dict[str, Any]):
    if not which("kubectl"):
        info("kubectl not found; skipping validation")
        return
    for k, p in cfg["FILES"].items():
        if not isinstance(p, Path) or not p.exists():
            continue
        info(f"Validating {p.name}")
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
    info("Deleted frontend manifests from disk")

# -----------------------------
# CLI & Entry
# -----------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--validate", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    if args.generate:
        generate(cfg, dry_run=args.dry_run)
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
