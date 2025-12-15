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
from typing import Dict, Tuple, Optional

# -------------------- logging --------------------
LOGLEVEL = os.environ.get("GEN_RERANKER_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOGLEVEL, logging.INFO), format="%(levelname)s: %(message)s")
log = logging.getLogger("gen_reranker")

# -------------------- helpers --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def run_cmd(cmd: list, input_bytes: Optional[bytes] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """
    Minimal wrapper for deterministic subprocess runs.
    Returns (returncode, stdout, stderr) with stdout/stderr as strings.
    """
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=True, check=False, timeout=timeout)
        stdout = proc.stdout.decode() if proc.stdout else ""
        stderr = proc.stderr.decode() if proc.stderr else ""
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", b"").decode() if getattr(e, "stdout", None) else "", f"timeout after {timeout}s"

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

def kubectl_apply_yaml(yaml_str: str, timeout: int = 120) -> Dict:
    """
    Apply YAML to cluster using `kubectl apply -f -`.
    YAML is passed on stdin (so secrets are not written to disk).
    """
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply", "-f", "-"]
    rc, stdout, stderr = run_cmd(cmd, input_bytes=yaml_str.encode("utf-8"), timeout=timeout)
    if rc == 0:
        return {"applied": True, "stdout": stdout}
    else:
        return {"applied": False, "stderr": stderr or f"exit={rc}"}

def kubectl_delete_yaml(yaml_str: str, timeout: int = 120) -> Dict:
    """
    Delete YAML resources via `kubectl delete -f - --ignore-not-found`.
    """
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"deleted": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "delete", "-f", "-", "--ignore-not-found"]
    rc, stdout, stderr = run_cmd(cmd, input_bytes=yaml_str.encode("utf-8"), timeout=timeout)
    if rc == 0:
        return {"deleted": True, "stdout": stdout}
    else:
        return {"deleted": False, "stderr": stderr or f"exit={rc}"}

# -------------------- config loader (RERANKER_*) --------------------
def load_config():
    cfg = {}
    cfg["ENV"] = os.environ.get("RERANKER_ENV", os.environ.get("ENV", "STAGING")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/reranker"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["IMAGE"] = os.environ.get("RERANKER_IMAGE", "athithya5354/reranker:amd64-arm64-v1")
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
    cfg["FILES"] = {
        "namespace": cfg["MANIFESTS_DIR"] / "00-namespace.yaml",
        "sa_role": cfg["MANIFESTS_DIR"] / "01-sa-role.yaml",
        "deployment": cfg["MANIFESTS_DIR"] / "02-deployment.yaml",
        "service": cfg["MANIFESTS_DIR"] / "03-service.yaml",
        "hpa": cfg["MANIFESTS_DIR"] / "04-hpa.yaml",
    }
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    # secret handling
    cfg["SECRET_NAME"] = os.environ.get("RERANKER_SECRET_NAME", f"{cfg['SERVICE_NAME']}-secrets")
    cfg["SECRET_PREFIX"] = os.environ.get("RERANKER_SECRET_PREFIX", "RERANKER_SECRET_")
    cfg["SECRETS_JSON_ENV"] = os.environ.get("RERANKER_SECRETS_JSON", "")
    return cfg

# -------------------- YAML renderers (non-secret) --------------------
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

def render_deployment(cfg):
    labels = cfg["LABELS"].copy()
    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["CONTAINER_PORT"]}],
        "env": [
            {"name": "RERANKER_PORT", "value": str(cfg["CONTAINER_PORT"])},
            {"name": "ENV", "value": cfg["ENV"]},
            {"name": "RERANKER_LOGLEVEL", "value": cfg["LOGLEVEL"]},
            # NOTE: secrets will be mounted via envFrom in deployment referencing the secret object name
            {"name": "SECRET_PROVIDER", "value": cfg["SECRET_NAME"]},
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

    # mount secrets as envFrom so secret keys become env vars at runtime
    container["envFrom"] = [{"secretRef": {"name": cfg["SECRET_NAME"]}}]

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
        key, val = (cfg["GPU_NODE_SELECTOR"].split("=", 1) + ["true"])[:2]
        pod_spec["spec"]["template"]["spec"]["nodeSelector"] = {key: val}

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

# -------------------- secret handling (ENV only; never write to disk) --------------------
def collect_secrets_from_env(cfg) -> Dict[str, str]:
    """
    Collect secrets from environment without persisting them.
    Rules:
      - Any env var starting with cfg['SECRET_PREFIX'] (default RERANKER_SECRET_) becomes a key.
        e.g. RERANKER_SECRET_GROQ_API_KEY -> key GROQ_API_KEY
      - RERANKER_SECRETS_JSON with JSON object is merged in (overrides prefix vars on same key).
    """
    prefix = cfg["SECRET_PREFIX"]
    secrets: Dict[str, str] = {}

    # prefix-based secrets
    for k, v in os.environ.items():
        if k.startswith(prefix) and len(k) > len(prefix):
            key = k[len(prefix):]
            if v is not None and v != "":
                secrets[key] = v

    # json-based secrets (optional)
    sj = cfg.get("SECRETS_JSON_ENV", "") or os.environ.get("RERANKER_SECRETS_JSON", "")
    if sj:
        try:
            parsed = json.loads(sj)
            if isinstance(parsed, dict):
                for kk, vv in parsed.items():
                    if vv is None:
                        continue
                    # treat only primitive values as strings
                    if not isinstance(vv, (str, int, float, bool)):
                        vv = json.dumps(vv, separators=(",", ":"))
                    secrets[str(kk)] = str(vv)
            else:
                log.warning("RERANKER_SECRETS_JSON is not an object; ignoring it.")
        except json.JSONDecodeError:
            log.warning("RERANKER_SECRETS_JSON failed to parse as JSON; ignoring it.")

    return secrets

def build_secret_manifest(name: str, data: Dict[str, str], namespace: str, labels: Dict[str, str], secret_type: str = "Opaque") -> str:
    """
    Construct a secret manifest using stringData (plaintext). The manifest is returned as YAML string.
    Note: We intentionally use stringData so we avoid base64 encoding and still do not write to disk.
    """
    obj = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "type": secret_type,
        "stringData": data,
    }
    return yaml.safe_dump(obj, sort_keys=False)

def apply_secret_to_cluster(cfg, secret_name: str, data: Dict[str, str]) -> bool:
    """
    Apply secret manifest to cluster (stringData). Returns True on success.
    """
    if not data:
        log.info("No secret data found; skipping secret '%s'.", secret_name)
        return True

    # DO NOT LOG secret values. Only log safe metadata.
    yaml_str = build_secret_manifest(secret_name, data, cfg["NAMESPACE"], cfg["LABELS"])
    res = kubectl_apply_yaml(yaml_str)
    if not res.get("applied", False):
        log.error("Failed to apply secret '%s': %s", secret_name, res.get("stderr") or res.get("error"))
        return False
    log.info("Applied secret '%s' to namespace '%s' (keys: %d)", secret_name, cfg["NAMESPACE"], len(data))
    return True

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
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash)
    log.info("Wrote manifests to %s", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        log.info("Namespace (head):\n%s", ns_yaml.splitlines()[:20])
        log.info("Deployment (head):\n%s", deploy_yaml.splitlines()[:60])

def apply_to_cluster(cfg, dry_run=False, verbose=False):
    """
    Apply sequence:
      1. Generate non-secret manifests locally (namespace, sa_role, deployment, service, hpa).
      2. Apply namespace manifest immediately (idempotent).
      3. Apply secrets directly (stringData over stdin) -- namespace must exist.
      4. Apply remaining non-secret manifests to cluster.
    """
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot apply")
        sys.exit(2)

    # 1. generate files locally (non-secret)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    if dry_run:
        log.info("Dry-run: skipping kubectl apply")
        return

    # 2. apply namespace (must exist before secrets)
    ns_yaml = cfg["FILES"]["namespace"].read_text()
    res_ns = kubectl_apply_yaml(ns_yaml)
    if not res_ns.get("applied", False):
        log.error("Failed to apply namespace: %s", res_ns.get("stderr") or res_ns.get("error"))
        sys.exit(3)
    log.info("Namespace '%s' ensured.", cfg["NAMESPACE"])

    # 3. secrets from env (never persisted)
    secrets = collect_secrets_from_env(cfg)
    if secrets:
        # possible to have many keys; group them under single secret object
        ok = apply_secret_to_cluster(cfg, cfg["SECRET_NAME"], secrets)
        if not ok:
            log.error("Failed to apply secrets to cluster.")
            sys.exit(4)
    else:
        log.info("No secrets found in environment (prefix: %s or JSON).", cfg["SECRET_PREFIX"])

    # 4. apply other manifests together
    files = [cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    res = kubectl_apply_yaml(combined)
    if not res.get("applied", False):
        log.error("kubectl apply failed for non-secret manifests: %s", res.get("stderr") or res.get("error"))
        sys.exit(5)

    summary = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image": cfg["IMAGE"],
        "namespace": cfg["NAMESPACE"],
        "replicas": cfg["REPLICAS"],
        "files": {k: str(v) for k, v in cfg["FILES"].items()},
        "secrets_applied": bool(secrets),
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    log.info("Applied manifests to cluster and wrote deploy summary")

def delete_manifests(cfg):
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except Exception as e:
                log.debug("Failed to remove %s: %s", p, e)
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except FileNotFoundError:
            pass
        log.info("Deleted manifests at %s", str(cfg["MANIFESTS_DIR"]))
    else:
        log.info("Manifests dir not present: %s", str(cfg["MANIFESTS_DIR"]))

def delete_from_cluster(cfg):
    """
    Helper to delete the generated resources from cluster including the secret object.
    This is destructive. Use with care.
    """
    kubectl = shutil.which("kubectl")
    if not kubectl:
        log.error("kubectl not found in PATH; cannot delete")
        sys.exit(2)

    # delete non-secret manifests (ignore not-found)
    files = [cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        if p.exists():
            combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    if combined:
        res = kubectl_delete_yaml(combined)
        if not res.get("deleted", False):
            log.warning("kubectl delete returned: %s", res.get("stderr") or res.get("error"))
        else:
            log.info("Deleted non-secret resources from cluster (if existed).")

    # delete secret
    # build a minimal manifest for the secret (delete by name)
    secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": cfg["SECRET_NAME"], "namespace": cfg["NAMESPACE"]},
    }
    res = kubectl_delete_yaml(yaml.safe_dump(secret_manifest, sort_keys=False))
    if not res.get("deleted", False):
        log.warning("Secret delete returned: %s", res.get("stderr") or res.get("error"))
    else:
        log.info("Deleted secret '%s' (if it existed).", cfg["SECRET_NAME"])

    # optionally delete namespace? not doing by default to avoid accidental broad deletes
    log.info("Cluster delete sequence finished (did not delete namespace).")

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply Reranker Kubernetes manifests.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--apply", action="store_true", help="Generate manifests and apply to cluster (requires kubectl).")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests (local files).")
    p.add_argument("--delete-cluster", action="store_true", help="When used with --delete, also delete resources from the cluster (including secret).")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()

    if args.delete:
        if args.delete_cluster:
            delete_from_cluster(cfg)
        delete_manifests(cfg)
        return

    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return

    if args.apply:
        # ensure kubectl accessibility early
        if not shutil.which("kubectl"):
            log.error("kubectl not found in PATH. Please install/configure kubectl before applying.")
            sys.exit(2)
        apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return

if __name__ == "__main__":
    main()
