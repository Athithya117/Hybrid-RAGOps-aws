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
import base64
from typing import Dict, Any, Tuple

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

def run_cmd(cmd, capture=True, check=False, timeout=None, input_bytes=None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=capture, text=False, check=check, timeout=timeout, shell=False)
        return proc.returncode, (proc.stdout.decode() if proc.stdout else ""), (proc.stderr.decode() if proc.stderr else "")
    except subprocess.CalledProcessError as e:
        return e.returncode, (e.stdout.decode() if e.stdout else ""), (e.stderr.decode() if e.stderr else "")
    except subprocess.TimeoutExpired as e:
        return 124, (getattr(e, "stdout", b"") or b"").decode(errors="ignore"), f"timeout after {timeout}s"

def canonical_inputs_hash(cfg: dict) -> str:
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

def kubectl_apply_yaml(yaml_str: str, timeout: int = 120) -> dict:
    """Apply YAML to cluster via kubectl stdin. Returns dict with applied status and messages."""
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply", "-f", "-"]
    try:
        proc = subprocess.run(cmd, input=yaml_str.encode("utf-8"), capture_output=True, check=True, timeout=timeout)
        stdout = proc.stdout.decode() if proc.stdout else ""
        return {"applied": True, "stdout": stdout}
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        return {"applied": False, "stderr": stderr}
    except subprocess.TimeoutExpired as e:
        return {"applied": False, "stderr": f"timeout: {e}"}

# -------------------- config loader (DENSE_*) --------------------
def load_config() -> Dict[str, Any]:
    cfg = {}
    # environment context
    cfg["ENV"] = os.environ.get("DENSE_ENV", os.environ.get("ENV", "STAGING")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/dense"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    # image and runtime
    cfg["IMAGE"] = os.environ.get("DENSE_IMAGE", "athithya5354/dense:amd64-arm64-v1")
    cfg["NAMESPACE"] = os.environ.get("DENSE_NAMESPACE", "models")
    cfg["SERVICE_NAME"] = os.environ.get("DENSE_SERVICE_NAME", "dense")
    cfg["CONTAINER_PORT"] = int(os.environ.get("DENSE_PORT", "8200"))
    cfg["LOGLEVEL"] = os.environ.get("DENSE_LOGLEVEL", "INFO")
    # replicas & resources (defaults differ by ENV)
    if cfg["ENV"] == "PROD":
        cfg.update({
            "REPLICAS": int(os.environ.get("DENSE_REPLICAS", "3")),
            "CPU_REQUEST": os.environ.get("DENSE_CPU_REQUEST", "1000m"),
            "CPU_LIMIT": os.environ.get("DENSE_CPU_LIMIT", "4000m"),
            "MEMORY_REQUEST": os.environ.get("DENSE_MEMORY_REQUEST", "1Gi"),
            "MEMORY_LIMIT": os.environ.get("DENSE_MEMORY_LIMIT", "4Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("DENSE_STARTUP_FAILURE_THRESHOLD", "24")),
        })
    else:
        cfg.update({
            "REPLICAS": int(os.environ.get("DENSE_REPLICAS", "1")),
            "CPU_REQUEST": os.environ.get("DENSE_CPU_REQUEST", "250m"),
            "CPU_LIMIT": os.environ.get("DENSE_CPU_LIMIT", "1000m"),
            "MEMORY_REQUEST": os.environ.get("DENSE_MEMORY_REQUEST", "512Mi"),
            "MEMORY_LIMIT": os.environ.get("DENSE_MEMORY_LIMIT", "1Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("DENSE_STARTUP_FAILURE_THRESHOLD", "60")),
        })
    # probe timings
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
    cfg["SECRETS_FILE"] = Path(os.environ.get("DENSE_SECRETS_FILE", "infra/secrets/dense-secrets.yaml"))
    cfg["SECRETS_JSON"] = os.environ.get("DENSE_SECRETS_JSON", "")
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    return cfg

# -------------------- YAML renderers --------------------
def render_namespace(cfg: Dict[str, Any]) -> str:
    ns = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "gen_dense"}},
    }
    return yaml.safe_dump(ns, sort_keys=False)

def render_sa_role(cfg: Dict[str, Any]) -> str:
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
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get", "list"]},
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

def render_deployment(cfg: Dict[str, Any]) -> str:
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

    if cfg["ENABLE_GPU"] and cfg["GPU_NODE_SELECTOR"]:
        # GPU_NODE_SELECTOR example: "gpu=true" or "kubernetes.io/hostname=node1"
        if "=" in cfg["GPU_NODE_SELECTOR"]:
            k, v = cfg["GPU_NODE_SELECTOR"].split("=", 1)
            pod_spec["spec"]["template"]["spec"]["nodeSelector"] = {k: v}
        else:
            pod_spec["spec"]["template"]["spec"]["nodeSelector"] = {cfg["GPU_NODE_SELECTOR"]: "true"}

    pod_spec["spec"]["template"]["metadata"].setdefault("annotations", {})
    pod_spec["spec"]["template"]["metadata"]["annotations"].update({
        "prometheus.io/scrape": "true",
        "prometheus.io/port": str(cfg["CONTAINER_PORT"]),
        "prometheus.io/path": "/metrics",
    })

    return yaml.safe_dump(pod_spec, sort_keys=False)

def render_service(cfg: Dict[str, Any]) -> str:
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

def render_hpa(cfg: Dict[str, Any]) -> str:
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

# -------------------- Secrets collection & apply (direct-to-cluster) --------------------
def _collect_env_prefixed_secrets(prefix: str = "DENSE_SECRET_") -> Dict[str, Dict[str, str]]:
    """
    Parse env vars of pattern:
      DENSE_SECRET_<SECRETNAME>__<KEY>=value
    Produces: { "<SECRETNAME>": { "<KEY>": "value", ... }, ... }
    """
    secrets: Dict[str, Dict[str, str]] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        remainder = k[len(prefix):]
        if "__" not in remainder:
            # fallback to single secret 'default' mapping if no secretname specified
            secret_name = "default"
            key = remainder
        else:
            secret_name, key = remainder.split("__", 1)
        if not secret_name:
            secret_name = "default"
        secrets.setdefault(secret_name, {})[key] = v
    return secrets

def _collect_secrets_from_file(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load secrets file. Accepts either:
      - { secretname: { key: val, ... }, ... }
      - { key: val, ... } -> becomes a single secret named 'default'
    """
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as e:
        log.error("Failed to load secrets file %s: %s", path, e)
        sys.exit(3)
    if not isinstance(raw, dict):
        log.error("Secrets file %s must contain a YAML mapping at top level", path)
        sys.exit(3)
    # If top-level values are scalars, treat as default secret
    sample_value = next(iter(raw.values())) if raw else None
    if not raw:
        return {}
    if isinstance(sample_value, dict):
        # assume correct shape
        return {k: {str(kk): str(vv) for kk, vv in val.items()} for k, val in raw.items()}
    else:
        # scalar mapping -> default secret
        return {"default": {str(k): str(v) for k, v in raw.items()}}

def _collect_secrets_from_json(json_str: str) -> Dict[str, Dict[str, str]]:
    if not json_str:
        return {}
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        log.error("Invalid JSON in DENSE_SECRETS_JSON: %s", e)
        sys.exit(3)
    if not isinstance(parsed, dict):
        log.error("DENSE_SECRETS_JSON must be a JSON object mapping secretname->mapping")
        sys.exit(3)
    out: Dict[str, Dict[str, str]] = {}
    for k, v in parsed.items():
        if isinstance(v, dict):
            out[k] = {str(kk): str(vv) for kk, vv in v.items()}
        else:
            log.error("DENSE_SECRETS_JSON values must be objects (mapping keys->values). error at %s", k)
            sys.exit(3)
    return out

def collect_secrets(cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Aggregate secrets from (highest precedence first):
      1. env prefix DENSE_SECRET_*
      2. DENSE_SECRETS_JSON
      3. DENSE_SECRETS_FILE (on-disk)
    """
    secrets: Dict[str, Dict[str, str]] = {}
    # env prefixed secrets take precedence
    env_secrets = _collect_env_prefixed_secrets()
    if env_secrets:
        log.debug("Collected %d secret(s) from environment variables", len(env_secrets))
        secrets.update(env_secrets)
    # inline JSON
    json_secrets = _collect_secrets_from_json(cfg.get("SECRETS_JSON", "") or "")
    if json_secrets:
        # do not overwrite env-provided keys for the same secret (env precedence)
        for sname, mapping in json_secrets.items():
            secrets.setdefault(sname, {}).update(mapping)
    # file-based secrets (lowest precedence)
    file_secrets = _collect_secrets_from_file(cfg.get("SECRETS_FILE"))
    if file_secrets:
        for sname, mapping in file_secrets.items():
            secrets.setdefault(sname, {}).update(mapping)
    return secrets

def _secret_to_k8s_yaml(name: str, namespace: str, data: Dict[str, str], labels: Dict[str,str]) -> str:
    """
    Build a Secret YAML with base64-encoded data. We do not print values anywhere.
    """
    b64_data = {k: base64.b64encode(v.encode("utf-8")).decode("utf-8") for k, v in data.items()}
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace, "labels": {**labels, "app.kubernetes.io/managed-by": "gen_dense"}},
        "type": "Opaque",
        "data": b64_data,
    }
    return yaml.safe_dump(secret, sort_keys=False)

def apply_secrets_to_cluster(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Apply all collected secrets directly to the cluster.
    Ensures namespace exists first.
    """
    # pre-validate kubectl
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot apply secrets")
        sys.exit(2)

    # Ensure namespace is present (apply namespace YAML)
    ns_yaml = render_namespace(cfg)
    if dry_run:
        log.info("Dry-run: would ensure namespace exists (not applying)")
    else:
        res_ns = kubectl_apply_yaml(ns_yaml)
        if not res_ns.get("applied", False):
            log.error("Failed to ensure namespace: %s", res_ns.get("stderr") or res_ns.get("error"))
            sys.exit(2)
        log.info("Ensured namespace '%s' exists", cfg["NAMESPACE"])

    # collect secrets (do not log secret values)
    secrets = collect_secrets(cfg)
    if not secrets:
        log.info("No secrets found in env/JSON/file. Skipping secret apply.")
        return

    applied_count = 0
    for sname, mapping in secrets.items():
        if not mapping:
            log.warning("Secret '%s' has no keys; skipping", sname)
            continue
        secret_yaml = _secret_to_k8s_yaml(sname, cfg["NAMESPACE"], mapping, cfg["LABELS"])
        if dry_run:
            log.info("Dry-run: would apply secret '%s' (keys: %d)", sname, len(mapping))
        else:
            res = kubectl_apply_yaml(secret_yaml)
            if not res.get("applied", False):
                log.error("Failed to apply secret '%s': %s", sname, res.get("stderr") or res.get("error"))
                sys.exit(2)
            applied_count += 1
            log.info("Applied secret '%s' (keys: %d)", sname, len(mapping))
    log.info("Applied %d secret(s) to namespace '%s'", applied_count, cfg["NAMESPACE"])

# -------------------- generate / apply / delete --------------------
def generate_manifests(cfg: Dict[str, Any], dry_run: bool = False, verbose: bool = False):
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
        log.info("Namespace (head):\n%s", ns_yaml.splitlines()[:20])
        log.info("Deployment (head):\n%s", deploy_yaml.splitlines()[:60])
    return

def apply_to_cluster(cfg: Dict[str, Any], dry_run: bool = False, verbose: bool = False):
    # validate kubectl
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot apply")
        sys.exit(2)

    # generate non-secret manifests first
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)

    # 1) Ensure namespace exists and create it (no secrets written to disk)
    ns_yaml = render_namespace(cfg)
    if dry_run:
        log.info("Dry-run: would ensure namespace exists")
    else:
        res_ns = kubectl_apply_yaml(ns_yaml)
        if not res_ns.get("applied", False):
            log.error("Failed to ensure namespace exists: %s", res_ns.get("stderr") or res_ns.get("error"))
            sys.exit(2)
        log.info("Namespace '%s' ensured", cfg["NAMESPACE"])

    # 2) Apply secrets directly (namespace now exists)
    apply_secrets_to_cluster(cfg, dry_run=dry_run)

    # 3) Apply non-secret manifests in deterministic order
    files = [cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    if dry_run:
        log.info("Dry-run: skipping kubectl apply of non-secret manifests")
    else:
        res = kubectl_apply_yaml(combined)
        if not res.get("applied", False):
            log.error("kubectl apply of manifests failed: %s", res.get("stderr") or res.get("error"))
            sys.exit(2)
        summary = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "image": cfg["IMAGE"],
            "namespace": cfg["NAMESPACE"],
            "replicas": cfg["REPLICAS"],
            "files": {k: str(v) for k, v in cfg["FILES"].items()},
        }
        atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
        log.info("Applied manifests to cluster and wrote deploy summary")

def delete_manifests(cfg: Dict[str, Any]):
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
    p = argparse.ArgumentParser(description="Generate/apply Dense embedder Kubernetes manifests and create secrets directly.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--apply", action="store_true", help="Generate manifests and apply to cluster (requires kubectl). Defaults to creating namespace -> secrets -> rest.")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests.")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    # Basic validation
    if not isinstance(cfg["MANIFESTS_DIR"], Path):
        log.error("MANIFESTS_DIR invalid")
        sys.exit(3)

    if args.delete:
        delete_manifests(cfg)
        return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
    if args.apply:
        # The apply flow ensures namespace, applies secrets directly, then applies other manifests.
        if args.dry_run:
            generate_manifests(cfg, dry_run=True, verbose=args.verbose)
            # show what secrets would be applied (names only)
            secrets = collect_secrets(cfg)
            log.info("Dry-run: would create namespace '%s' and apply %d secret(s): %s", cfg["NAMESPACE"], len(secrets), ", ".join(sorted(secrets.keys())) if secrets else "none")
            return
        apply_to_cluster(cfg, dry_run=False, verbose=args.verbose)
        return

if __name__ == "__main__":
    main()
