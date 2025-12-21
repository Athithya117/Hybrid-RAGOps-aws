#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, hashlib, subprocess, shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import yaml
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DEFAULT = ROOT / "infra" / "manifests" / "frontend"

def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr); sys.exit(2)
def info(msg: str) -> None:
    print("INFO:", msg)
def warn(msg: str) -> None:
    print("WARN:", msg, file=sys.stderr)

def atomic_write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(content, encoding="utf-8"); tmp.replace(path)
    try: path.chmod(mode)
    except Exception: pass

def safe_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: int = 120) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def make_serializable(o: Any) -> Any:
    if isinstance(o, Path): return str(o)
    if isinstance(o, dict): return {k: make_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)): return [make_serializable(v) for v in o]
    return o

def canonical_inputs_hash(cfg: Dict[str, Any]) -> str:
    serial = {}
    for k in sorted(cfg.keys()):
        if k in ("SECRET_VALUES", "INPUTS_HASH_PATH"): continue
        serial[k] = make_serializable(cfg[k])
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def parse_bool_env(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes")

def norm_url(u: Optional[str], default: str) -> str:
    if not u:
        return default
    s = str(u).strip()
    if not s:
        return default
    if s.endswith("/"):
        s = s[:-1]
    if "://" not in s:
        if s.startswith("localhost") or s.startswith("127.") or (":" in s and s.split(":")[0].isdigit()):
            s = "http://" + s
        else:
            s = "https://" + s
    return s

def derive_redirects(base: str) -> Dict[str, str]:
    b = base.rstrip("/")
    if b.endswith("/auth/callback"):
        return {"google": f"{b}/google", "microsoft": f"{b}/microsoft", "github": f"{b}/github"}
    return {"google": f"{b}/auth/callback/google", "microsoft": f"{b}/auth/callback/microsoft", "github": f"{b}/auth/callback/github"}

def _normalize_csv(s: Optional[str]) -> str:
    if not s:
        return ""
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return ",".join(parts)

def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["REPO_ROOT"] = Path(os.getenv("REPO_ROOT", Path.cwd()))
    cfg["KUBE_ENV"] = os.getenv("KUBE_ENV", os.getenv("ENV", "staging")).lower()
    if cfg["KUBE_ENV"] not in ("kind", "staging", "prod", "aks"):
        die("KUBE_ENV must be one of kind/staging/prod/aks")
    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", MANIFESTS_DEFAULT))
    cfg["FRONTEND_NAMESPACE"] = os.getenv("FRONTEND_NAMESPACE", "inference")
    cfg["SERVICE_NAME"] = os.getenv("FRONTEND_SERVICE_NAME", "frontend")
    cfg["IMAGE"] = os.getenv("FRONTEND_AND_AUTH_IMAGE", "athithya5354/frontend-and-auth:v14")
    try: cfg["PORT"] = int(os.getenv("FRONTEND_PORT", "8000"))
    except Exception: cfg["PORT"] = 8000
    try: cfg["REPLICAS"] = int(os.getenv("FRONTEND_AND_AUTH_REPLICAS","1"))
    except Exception: cfg["REPLICAS"] = 1
    cfg["CPU_REQUEST"] = os.getenv("FRONTEND_CPU_REQUEST", "100m")
    cfg["CPU_LIMIT"] = os.getenv("FRONTEND_CPU_LIMIT", "500m")
    cfg["MEMORY_REQUEST"] = os.getenv("FRONTEND_MEMORY_REQUEST", "128Mi")
    cfg["MEMORY_LIMIT"] = os.getenv("FRONTEND_MEMORY_LIMIT", "512Mi")
    cfg["HPA_ENABLED"] = parse_bool_env(os.getenv("FRONTEND_HPA_ENABLED"), False)
    try: cfg["HPA_MIN"] = int(os.getenv("FRONTEND_HPA_MIN", "1"))
    except Exception: cfg["HPA_MIN"] = 1
    try: cfg["HPA_MAX"] = int(os.getenv("FRONTEND_HPA_MAX", "5"))
    except Exception: cfg["HPA_MAX"] = 5
    cfg["QUERY_URL"] = norm_url(os.getenv("QUERY_URL", "http://retrieval-svc.inference.svc.cluster.local:8001"), "http://retrieval-svc.inference.svc.cluster.local:8001")

    hostname = os.getenv("FRONTEND_HOSTNAME")
    if hostname: hostname = hostname.strip()
    if not hostname:
        if cfg["KUBE_ENV"] == "kind":
            hostname = "frontend.local"; warn("FRONTEND_HOSTNAME not set; using frontend.local (dev only).")
        else:
            die("FRONTEND_HOSTNAME is required for non-dev environments. Set FRONTEND_HOSTNAME to your public hostname (eg: ui.example.com).")
    hostname = hostname.rstrip(".")
    cfg["FRONTEND_HOSTNAME"] = hostname
    cfg["FRONTEND_URL"] = norm_url(os.getenv("FRONTEND_URL", f"https://{hostname}"), f"https://{hostname}")
    parsed = urlparse(cfg["FRONTEND_URL"])
    cfg["FRONTEND_HOST"] = parsed.hostname or hostname
    cfg["FRONTEND_SCHEME"] = parsed.scheme or "https"
    cfg["FRONTEND_BASE"] = norm_url(os.getenv("FRONTEND_BASE", cfg["FRONTEND_URL"]), cfg["FRONTEND_URL"])
    cfg["OAUTH_REDIRECT_BASE"] = norm_url(os.getenv("OAUTH_REDIRECT_BASE", cfg["FRONTEND_BASE"]), cfg["FRONTEND_BASE"])
    redirects = derive_redirects(cfg["OAUTH_REDIRECT_BASE"])
    cfg["REDIRECTS"] = redirects

    cfg["AUTH_MODE"] = os.getenv("AUTH_MODE", "external-id")
    cfg["OIDC_ISSUER"] = os.getenv("OIDC_ISSUER", "")
    cfg["OIDC_JWKS_URI"] = os.getenv("OIDC_JWKS_URI", "")
    cfg["OIDC_AUDIENCE"] = os.getenv("OIDC_AUDIENCE", "")
    cfg["SPA_CLIENT_ID"] = os.getenv("SPA_CLIENT_ID", "")

    cfg["AZURE_TENANT_ID"] = os.getenv("AZURE_TENANT_ID", "common")
    cfg["MS_TENANT_ID"] = os.getenv("MS_TENANT_ID", os.getenv("AZURE_TENANT_ID", "common"))

    try: cfg["JWKS_REFRESH_INTERVAL_SECONDS"] = int(os.getenv("JWKS_REFRESH_INTERVAL_SECONDS", "900"))
    except Exception: cfg["JWKS_REFRESH_INTERVAL_SECONDS"] = 900
    cfg["ENABLE_CORS"] = parse_bool_env(os.getenv("ENABLE_CORS"), False)
    cfg["CORS_ALLOWED_ORIGINS"] = os.getenv("CORS_ALLOWED_ORIGINS", cfg["FRONTEND_URL"])
    cfg["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")
    cfg["ENABLE_GOOGLE_AUTH"] = parse_bool_env(os.getenv("ENABLE_GOOGLE_AUTH"), False)
    cfg["ENABLE_MICROSOFT_AUTH"] = parse_bool_env(os.getenv("ENABLE_MICROSOFT_AUTH"), False)
    cfg["ENABLE_GITHUB_AUTH"] = parse_bool_env(os.getenv("ENABLE_GITHUB_AUTH"), False)

    cfg["GOOGLE_CLIENT_ID"] = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    cfg["GOOGLE_REDIRECT_URI"] = os.getenv("GOOGLE_REDIRECT_URI", "").strip() or redirects.get("google", "")

    cfg["MS_CLIENT_ID"] = os.getenv("MS_CLIENT_ID", "").strip()
    cfg["MS_REDIRECT_URI"] = os.getenv("MS_REDIRECT_URI", "").strip() or redirects.get("microsoft", "")
    cfg["MS_TENANT_ID"] = os.getenv("MS_TENANT_ID", cfg["MS_TENANT_ID"]).strip()

    cfg["GITHUB_CLIENT_ID"] = os.getenv("GITHUB_CLIENT_ID", "").strip()
    cfg["GITHUB_REDIRECT_URI"] = os.getenv("GITHUB_REDIRECT_URI", "").strip() or redirects.get("github", "")

    # Normalize allowed-lists into comma-separated strings (app reads them as CSVs)
    cfg["GOOGLE_ALLOWED_DOMAINS"] = _normalize_csv(os.getenv("GOOGLE_ALLOWED_DOMAINS", ""))
    cfg["MICROSOFT_ALLOWED_DOMAINS"] = _normalize_csv(os.getenv("MICROSOFT_ALLOWED_DOMAINS", ""))
    cfg["MICROSOFT_ALLOWED_TENANT_IDS"] = _normalize_csv(os.getenv("MICROSOFT_ALLOWED_TENANT_IDS", ""))
    cfg["GITHUB_ALLOWED_DOMAINS"] = _normalize_csv(os.getenv("GITHUB_ALLOWED_DOMAINS", ""))
    cfg["GITHUB_ALLOWED_ORGS"] = _normalize_csv(os.getenv("GITHUB_ALLOWED_ORGS", ""))

    cfg["JWT_EXP_SECONDS"] = os.getenv("JWT_EXP_SECONDS", "1800")
    cfg["JWT_ISS"] = os.getenv("JWT_ISS", "stateless-openid-auth")
    cfg["JWT_AUD"] = os.getenv("JWT_AUD", "rag-ui")
    cfg["COOKIE_NAME"] = os.getenv("COOKIE_NAME", "app_session")
    cfg["COOKIE_SAMESITE"] = os.getenv("COOKIE_SAMESITE", "lax")
    cfg["COOKIE_SECURE"] = "true" if parse_bool_env(os.getenv("COOKIE_SECURE"), True) else "false"
    cfg["TEMP_DOMAIN"] = parse_bool_env(os.getenv("TEMP_DOMAIN"), False)
    cfg["CLOUDFLARE_TUNNEL_NAME"] = os.getenv("CLOUDFLARE_TUNNEL_NAME", "rag-frontend")
    cfg["HOST"] = os.getenv("HOST", "0.0.0.0")
    cfg["PORT_OVERRIDE"] = os.getenv("PORT", str(cfg["PORT"]))
    cfg["DISPLAY_SOURCES_IN_UI"] = "true" if parse_bool_env(os.getenv("DISPLAY_SOURCES_IN_UI"), True) else "false"
    cfg["DISPLAY_TOPK_IN_UI"] = "true" if parse_bool_env(os.getenv("DISPLAY_TOPK_IN_UI"), True) else "false"

    # SECRET_NAMES maps expected environment variable name -> secret key name in k8s Secret.
    # We intentionally make secret key name equal to env var name for deterministic mapping.
    cfg["SECRET_NAMES"] = {
        "JWT_SECRET": "JWT_SECRET",
        "SESSION_SECRET": "SESSION_SECRET",
        "GOOGLE_CLIENT_SECRET": "GOOGLE_CLIENT_SECRET",
        "MS_CLIENT_SECRET": "MS_CLIENT_SECRET",
        "GITHUB_CLIENT_SECRET": "GITHUB_CLIENT_SECRET",
    }

    secret_values: Dict[str, str] = {}
    for envk in cfg["SECRET_NAMES"].keys():
        v = os.getenv(envk)
        if v:
            secret_values[envk] = v
    cfg["SECRET_VALUES"] = secret_values
    cfg["ALLOW_MISSING_SECRETS"] = parse_bool_env(os.getenv("ALLOW_MISSING_SECRETS"), True)

    # Conservative warnings: provider enabled but client id missing
    if cfg["ENABLE_GOOGLE_AUTH"] and not cfg["GOOGLE_CLIENT_ID"]:
        warn("ENABLE_GOOGLE_AUTH=true but GOOGLE_CLIENT_ID missing.")
    if cfg["ENABLE_MICROSOFT_AUTH"] and not cfg["MS_CLIENT_ID"]:
        warn("ENABLE_MICROSOFT_AUTH=true but MS_CLIENT_ID missing.")
    if cfg["ENABLE_GITHUB_AUTH"] and not cfg["GITHUB_CLIENT_ID"]:
        warn("ENABLE_GITHUB_AUTH=true but GITHUB_CLIENT_ID missing.")

    # Additional friendly warnings when allow lists are present but empty/likely misconfigured
    if cfg["ENABLE_MICROSOFT_AUTH"] and not (cfg["MICROSOFT_ALLOWED_DOMAINS"] or cfg["MICROSOFT_ALLOWED_TENANT_IDS"]):
        warn("MICROSOFT auth enabled but neither MICROSOFT_ALLOWED_DOMAINS nor MICROSOFT_ALLOWED_TENANT_IDS are set; this will allow any Microsoft account (depending on MS_TENANT_ID).")

    m = cfg["MANIFESTS_DIR"]
    cfg["FILES"] = {
        "namespace": m / "00-namespace.yaml",
        "sa_role": m / "01-sa-role.yaml",
        "configmap": m / "02-configmap.yaml",
        "deployment": m / "04-deployment.yaml",
        "service": m / "05-service.yaml",
        "hpa": m / "06-hpa.yaml",
        "ingressroute": m / "07-ingressroute.yaml",
        "inputs_hash": m / ".inputs_hash",
    }
    cfg["AUTH_META_PATH"] = Path(os.getenv("AUTH_META_PATH", cfg["REPO_ROOT"] / "infra" / "manifests" / "auth" / "manifest.meta.json"))
    return cfg

def render_namespace(cfg: Dict[str, Any]) -> str:
    ns = {"apiVersion":"v1","kind":"Namespace","metadata":{"name":cfg["FRONTEND_NAMESPACE"],"labels":{"app":"frontend"}}}
    return safe_yaml(ns)

def render_sa_role(cfg: Dict[str, Any]) -> str:
    sa = {"apiVersion":"v1","kind":"ServiceAccount","metadata":{"name":f"{cfg['SERVICE_NAME']}-sa","namespace":cfg["FRONTEND_NAMESPACE"]}}
    role = {"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role","metadata":{"name":f"{cfg['SERVICE_NAME']}-role","namespace":cfg['FRONTEND_NAMESPACE']},"rules":[{"apiGroups":[""],"resources":["pods","services","endpoints","configmaps"],"verbs":["get","list","watch"]},{"apiGroups":[""],"resources":["secrets"],"verbs":["get"]}]}
    rb = {"apiVersion":"rbac.authorization.k8s.io/v1","kind":"RoleBinding","metadata":{"name":f"{cfg['SERVICE_NAME']}-rb","namespace":cfg['FRONTEND_NAMESPACE']},"subjects":[{"kind":"ServiceAccount","name":f"{cfg['SERVICE_NAME']}-sa","namespace":cfg['FRONTEND_NAMESPACE']}],"roleRef":{"kind":"Role","name":f"{cfg['SERVICE_NAME']}-role","apiGroup":"rbac.authorization.k8s.io"}}
    return "\n---\n".join([safe_yaml(x) for x in (sa, role, rb)])

def render_configmap(cfg: Dict[str, Any]) -> str:
    data = {
        "QUERY_URL": str(cfg["QUERY_URL"]),
        "FRONTEND_URL": str(cfg["FRONTEND_URL"]),
        "FRONTEND_BASE": str(cfg["FRONTEND_BASE"]),
        "FRONTEND_HOST": str(cfg["FRONTEND_HOST"]),
        "FRONTEND_SCHEME": str(cfg["FRONTEND_SCHEME"]),
        "FRONTEND_HOSTNAME": str(cfg["FRONTEND_HOSTNAME"]),
        "PORT": str(cfg["PORT"]),
        "ENV": str(cfg["KUBE_ENV"]).upper(),
        "AUTH_MODE": str(cfg["AUTH_MODE"]),
        "OAUTH_REDIRECT_BASE": str(cfg["OAUTH_REDIRECT_BASE"]),
        "GOOGLE_REDIRECT_URI": str(cfg["GOOGLE_REDIRECT_URI"]),
        "MS_REDIRECT_URI": str(cfg["MS_REDIRECT_URI"]),
        "GITHUB_REDIRECT_URI": str(cfg["GITHUB_REDIRECT_URI"]),
        "OIDC_ISSUER": str(cfg["OIDC_ISSUER"] or ""),
        "OIDC_JWKS_URI": str(cfg["OIDC_JWKS_URI"] or ""),
        "OIDC_AUDIENCE": str(cfg["OIDC_AUDIENCE"] or ""),
        "SPA_CLIENT_ID": str(cfg["SPA_CLIENT_ID"] or ""),
        "AZURE_TENANT_ID": str(cfg["AZURE_TENANT_ID"] or ""),
        "MS_TENANT_ID": str(cfg.get("MS_TENANT_ID", "") or ""),
        "JWKS_REFRESH_INTERVAL_SECONDS": str(cfg["JWKS_REFRESH_INTERVAL_SECONDS"]),
        "ENABLE_CORS": "true" if cfg["ENABLE_CORS"] else "false",
        "CORS_ALLOWED_ORIGINS": str(cfg["CORS_ALLOWED_ORIGINS"]),
        "LOG_LEVEL": str(cfg["LOG_LEVEL"]),
        "ENABLE_GOOGLE_AUTH": "true" if cfg["ENABLE_GOOGLE_AUTH"] else "false",
        "ENABLE_MICROSOFT_AUTH": "true" if cfg["ENABLE_MICROSOFT_AUTH"] else "false",
        "ENABLE_GITHUB_AUTH": "true" if cfg["ENABLE_GITHUB_AUTH"] else "false",
        "GOOGLE_CLIENT_ID": str(cfg["GOOGLE_CLIENT_ID"]),
        "MS_CLIENT_ID": str(cfg["MS_CLIENT_ID"]),
        "GITHUB_CLIENT_ID": str(cfg["GITHUB_CLIENT_ID"]),
        "GOOGLE_ALLOWED_DOMAINS": str(cfg["GOOGLE_ALLOWED_DOMAINS"] or ""),
        "MICROSOFT_ALLOWED_DOMAINS": str(cfg["MICROSOFT_ALLOWED_DOMAINS"] or ""),
        "MICROSOFT_ALLOWED_TENANT_IDS": str(cfg["MICROSOFT_ALLOWED_TENANT_IDS"] or ""),
        "GITHUB_ALLOWED_DOMAINS": str(cfg["GITHUB_ALLOWED_DOMAINS"] or ""),
        "GITHUB_ALLOWED_ORGS": str(cfg["GITHUB_ALLOWED_ORGS"] or ""),
        "JWT_EXP_SECONDS": str(cfg["JWT_EXP_SECONDS"]),
        "JWT_ISS": str(cfg["JWT_ISS"]),
        "JWT_AUD": str(cfg["JWT_AUD"]),
        "COOKIE_NAME": str(cfg["COOKIE_NAME"]),
        "COOKIE_SAMESITE": str(cfg["COOKIE_SAMESITE"]),
        "COOKIE_SECURE": str(cfg["COOKIE_SECURE"]),
        "TEMP_DOMAIN": "true" if cfg["TEMP_DOMAIN"] else "false",
        "CLOUDFLARE_TUNNEL_NAME": str(cfg["CLOUDFLARE_TUNNEL_NAME"]),
        "HOST": str(cfg["HOST"]),
        "PORT_OVERRIDE": str(cfg["PORT_OVERRIDE"]),
        "DISPLAY_SOURCES_IN_UI": str(cfg["DISPLAY_SOURCES_IN_UI"]),
        "DISPLAY_TOPK_IN_UI": str(cfg["DISPLAY_TOPK_IN_UI"]),
    }
    cm = {"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":f"{cfg['SERVICE_NAME']}-config","namespace":cfg['FRONTEND_NAMESPACE']},"data":data}
    return safe_yaml(cm)

def render_deployment(cfg: Dict[str, Any]) -> str:
    labels = {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}
    env_from: List[Dict[str, Any]] = [{"configMapRef": {"name": f"{cfg['SERVICE_NAME']}-config"}}]
    secret_envs: List[Dict[str, Any]] = []
    # For each secret mapping: create env entry referencing secretKeyRef only for keys present (or fatal if missing and not allowed)
    for env_var, secret_key in cfg["SECRET_NAMES"].items():
        if env_var in cfg["SECRET_VALUES"]:
            secret_envs.append({"name": env_var, "valueFrom": {"secretKeyRef": {"name": f"{cfg['SERVICE_NAME']}-secret", "key": secret_key}}})
        else:
            if not cfg["ALLOW_MISSING_SECRETS"]:
                die(f"Required secret env {env_var} missing and ALLOW_MISSING_SECRETS=false")
            # if allowed to miss, do not create env var; app will fallback to warnings on startup
    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["PORT"]}],
        "envFrom": env_from,
        "env": secret_envs if secret_envs else None,
        "resources": {"requests": {"cpu": cfg["CPU_REQUEST"], "memory": cfg["MEMORY_REQUEST"]}, "limits": {"cpu": cfg["CPU_LIMIT"], "memory": cfg["MEMORY_LIMIT"]}},
        "livenessProbe": {"httpGet": {"path": "/orchestrator/health", "port": cfg["PORT"]}, "initialDelaySeconds": 10, "periodSeconds": 10, "timeoutSeconds": 1},
        "readinessProbe": {"httpGet": {"path": "/orchestrator/health", "port": cfg["PORT"]}, "initialDelaySeconds": 3, "periodSeconds": 5, "timeoutSeconds": 1}
    }
    # normalize None for env
    if container["env"] is None:
        del container["env"]
    pod = {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":f"{cfg['SERVICE_NAME']}-deployment","namespace":cfg['FRONTEND_NAMESPACE'],"labels":labels},"spec":{"replicas":cfg['REPLICAS'],"selector":{"matchLabels":labels},"template":{"metadata":{"labels":labels},"spec":{"serviceAccountName":f"{cfg['SERVICE_NAME']}-sa","containers":[container]}}}}
    return safe_yaml(pod)

def render_service(cfg: Dict[str, Any]) -> str:
    svc = {"apiVersion":"v1","kind":"Service","metadata":{"name":f"{cfg['SERVICE_NAME']}-svc","namespace":cfg['FRONTEND_NAMESPACE']},"spec":{"type":"ClusterIP","ports":[{"port":cfg['PORT'],"targetPort":cfg['PORT'],"protocol":"TCP","name":"http"}],"selector":{"app.kubernetes.io/name":cfg['SERVICE_NAME']}}}
    return safe_yaml(svc)

def render_hpa(cfg: Dict[str, Any]) -> str:
    if not cfg["HPA_ENABLED"]: return ""
    hpa = {"apiVersion":"autoscaling/v2","kind":"HorizontalPodAutoscaler","metadata":{"name":f"{cfg['SERVICE_NAME']}-hpa","namespace":cfg['FRONTEND_NAMESPACE']},"spec":{"scaleTargetRef":{"apiVersion":"apps/v1","kind":"Deployment","name":f"{cfg['SERVICE_NAME']}-deployment"},"minReplicas":cfg["HPA_MIN"],"maxReplicas":cfg["HPA_MAX"],"metrics":[{"type":"Resource","resource":{"name":"cpu","target":{"type":"Utilization","averageUtilization":60}}}]}}
    return safe_yaml(hpa)

def render_ingressroute(cfg: Dict[str, Any], meta: Dict[str, Any]) -> str:
    middlewares = []
    ns = meta.get("traefik_namespace") or meta.get("auth_namespace")
    if meta.get("forward_auth_middleware"): middlewares.append({"name": meta["forward_auth_middleware"], "namespace": ns})
    if meta.get("stripprefix_middleware"): middlewares.append({"name": meta["stripprefix_middleware"], "namespace": ns})
    if meta.get("cors_middleware"): middlewares.append({"name": meta["cors_middleware"], "namespace": ns})
    if meta.get("ratelimit_middleware"): middlewares.append({"name": meta["ratelimit_middleware"], "namespace": ns})
    host = cfg["FRONTEND_HOSTNAME"]
    ir = {"apiVersion":"traefik.containo.us/v1alpha1","kind":"IngressRoute","metadata":{"name":f"{cfg['SERVICE_NAME']}-ingress","namespace":cfg['FRONTEND_NAMESPACE']},"spec":{"entryPoints":["websecure"],"routes":[{"match":f"Host(`{host}`) && PathPrefix(`/`)","kind":"Rule","services":[{"name":f"{cfg['SERVICE_NAME']}-svc","port":cfg['PORT']}] }]} }
    if middlewares: ir["spec"]["routes"][0]["middlewares"] = [{"name": m["name"], "namespace": m["namespace"]} for m in middlewares]
    return safe_yaml(ir)

def apply_namespace_to_cluster(cfg: Dict[str, Any]) -> None:
    ns_yaml = render_namespace(cfg).encode("utf-8")
    rc, out, err = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=ns_yaml)
    if rc != 0: die(f"Failed to create namespace: {err or out}")

def apply_secret_to_cluster(cfg: Dict[str, Any]) -> None:
    # Create a Kubernetes Secret with keys matching SECRET_NAMES values (which equal env var names)
    secret_name = f"{cfg['SERVICE_NAME']}-secret"; ns = cfg["FRONTEND_NAMESPACE"]
    # prepare --from-literal entries only for provided values
    literals: List[str] = []
    for env_var, secret_key in cfg["SECRET_NAMES"].items():
        val = cfg["SECRET_VALUES"].get(env_var)
        if val is None:
            if cfg["ALLOW_MISSING_SECRETS"]:
                warn(f"{env_var} missing; skipping that key in secret")
                continue
            else:
                die(f"Secret env {env_var} missing and ALLOW_MISSING_SECRETS=false")
        literals.append(f"--from-literal={secret_key}={val}")
    if not literals:
        info("No secret key-values provided; skipping secret creation.")
        return
    create_cmd = ["kubectl", "create", "secret", "generic", secret_name, "--namespace", ns, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run_cmd(create_cmd, timeout=30)
    if rc != 0: die(f"Failed to generate secret YAML: {err or out}")
    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=(out.encode("utf-8")), timeout=30)
    if rc2 != 0: die(f"Failed to apply secret: {err2 or out2}")
    info(f"Secret '{secret_name}' applied to namespace '{ns}'")

def ensure_dir(cfg: Dict[str, Any]) -> None:
    cfg["MANIFESTS_DIR"].mkdir(parents=True, exist_ok=True)

def detect_secret_leak(rendered: str, secret_values: Dict[str, str]) -> Optional[str]:
    for envk, v in secret_values.items():
        if not v: continue
        if len(v) >= 8 and v in rendered: return envk
    return None

def load_auth_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = cfg["AUTH_META_PATH"]
    if not path.exists():
        info(f"Auth meta not found at {path} (set AUTH_META_PATH or run auth generator first).")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"Failed reading auth meta: {e}")

def generate(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    ensure_dir(cfg)
    ihash = canonical_inputs_hash(cfg)
    existing = None
    try:
        if cfg["FILES"]["inputs_hash"].exists(): existing = cfg["FILES"]["inputs_hash"].read_text(encoding="utf-8").strip()
    except Exception:
        existing = None
    if existing == ihash and not dry_run:
        info("No non-secret changes; skipping generation.")
        return
    meta = load_auth_meta(cfg)
    deployment_yaml = render_deployment(cfg)
    leak = detect_secret_leak(deployment_yaml, cfg["SECRET_VALUES"])
    if leak: die(f"Secret value for {leak} would be embedded in generated Deployment YAML; refuse to generate.")
    atomic_write(cfg["FILES"]["namespace"], render_namespace(cfg))
    atomic_write(cfg["FILES"]["sa_role"], render_sa_role(cfg))
    atomic_write(cfg["FILES"]["configmap"], render_configmap(cfg))
    atomic_write(cfg["FILES"]["deployment"], deployment_yaml)
    atomic_write(cfg["FILES"]["service"], render_service(cfg))
    if cfg["HPA_ENABLED"]: atomic_write(cfg["FILES"]["hpa"], render_hpa(cfg))
    else:
        try: cfg["FILES"]["hpa"].unlink()
        except Exception: pass
    allow_missing = parse_bool_env(os.getenv("FRONTEND_ALLOW_MISSING_AUTH_META"), False)
    if meta or allow_missing: atomic_write(cfg["FILES"]["ingressroute"], render_ingressroute(cfg, meta or {}))
    else:
        info("Auth meta not present; skipping ingressroute generation.")
        try: cfg["FILES"]["ingressroute"].unlink()
        except Exception: pass
    cfg["FILES"]["inputs_hash"].write_text(ihash, encoding="utf-8")
    info("Wrote frontend manifests to %s" % str(cfg["MANIFESTS_DIR"]))

def apply(cfg: Dict[str, Any], confirm: bool = False) -> None:
    if not confirm: die("Refusing to apply without --confirm")
    if not which("kubectl"): die("kubectl not found in PATH")
    apply_namespace_to_cluster(cfg)
    apply_secret_to_cluster(cfg)
    files: List[Path] = [cfg["FILES"]["sa_role"], cfg["FILES"]["configmap"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]: files.append(cfg["FILES"]["hpa"])
    if cfg["FILES"]["ingressroute"].exists(): files.append(cfg["FILES"]["ingressroute"])
    combined = ""
    for p in files:
        if not p.exists(): warn(f"Manifest {p} missing; skipping"); continue
        combined += f"---\n# source: {p.name}\n" + p.read_text(encoding="utf-8") + "\n"
    if not combined:
        info("No manifests to apply (after filtering)")
        return
    rc, out, err = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=(combined.encode("utf-8")), timeout=60)
    if rc != 0: die(f"kubectl apply failed: {err or out}")
    info("Applied frontend manifests (non-secret resources)")

def validate(cfg: Dict[str, Any]) -> None:
    if not which("kubectl"):
        info("kubectl not found; skipping validation")
        return
    for k, p in cfg["FILES"].items():
        if not isinstance(p, Path) or not p.exists(): continue
        info(f"Validating {p.name}")
        rc, out, err = run_cmd(["kubectl", "apply", "--dry-run=client", "-f", str(p)], timeout=20)
        if rc != 0: die(f"Validation failed for {p.name}: {err or out}")

def delete_manifests(cfg: Dict[str, Any], confirm: bool = False) -> None:
    if not confirm: die("Refusing to delete without --confirm")
    d = cfg["MANIFESTS_DIR"]
    if not d.exists(): info("No manifests to delete"); return
    for p in sorted(d.glob("*")):
        try:
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
        except Exception: pass
    info("Deleted frontend manifests from disk")

def parse_args() -> Any:
    import argparse
    p = argparse.ArgumentParser(description="Generate/apply frontend + auth manifests; secrets applied in-cluster (not written to disk).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true"); g.add_argument("--apply", action="store_true"); g.add_argument("--validate", action="store_true"); g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true", help="required for apply/delete"); p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args(); cfg = load_config()
    if args.generate: generate(cfg, dry_run=args.dry_run); return
    if args.apply: generate(cfg, dry_run=args.dry_run); apply(cfg, confirm=args.confirm); return
    if args.validate: generate(cfg, dry_run=args.dry_run); validate(cfg); return
    if args.delete: delete_manifests(cfg, confirm=args.confirm); return

if __name__ == "__main__":
    main()
