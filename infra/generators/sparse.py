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
from typing import Dict, Any, Tuple, Optional

# -------------------- logging --------------------
LOGLEVEL = os.environ.get("GEN_SPARSE_LOGLEVEL", "INFO")
logging.basicConfig(level=LOGLEVEL)
log = logging.getLogger("gen_sparse")

# -------------------- helpers --------------------
def fatal(msg: str, code: int = 1) -> None:
    log.error(msg)
    sys.exit(code)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def run_cmd(cmd, capture=True, check=False, timeout=None, input_bytes=None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=capture, text=False, check=check, timeout=timeout)
        stdout = proc.stdout.decode() if proc.stdout else ""
        stderr = proc.stderr.decode() if proc.stderr else ""
        return proc.returncode, stdout, stderr
    except subprocess.CalledProcessError as e:
        stdout = getattr(e, "stdout", b"")
        stderr = getattr(e, "stderr", b"")
        return e.returncode, (stdout.decode() if stdout else ""), (stderr.decode() if stderr else "")
    except subprocess.TimeoutExpired as e:
        stdout = getattr(e, "stdout", b"")
        stderr = getattr(e, "stderr", b"")
        return 124, (stdout.decode() if stdout else ""), (stderr.decode() if stderr else f"timeout after {timeout}s")

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
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply", "-f", "-"]
    code, out, err = run_cmd(cmd, input_bytes=yaml_str.encode("utf-8"), timeout=timeout)
    if code == 0:
        return {"applied": True, "stdout": out}
    return {"applied": False, "stderr": err or out or f"kubectl exited {code}"}

def kubectl_execute(cmd_args: list, timeout: int = 30) -> dict:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"ok": False, "error": "kubectl-not-found"}
    cmd = [kubectl] + cmd_args
    code, out, err = run_cmd(cmd, timeout=timeout)
    if code == 0:
        return {"ok": True, "stdout": out}
    return {"ok": False, "stderr": err or out or f"exit {code}"}

# -------------------- config loader (SPARSE_*) --------------------
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    # environment context
    cfg["ENV"] = os.environ.get("SPARSE_ENV", os.environ.get("ENV", "STAGING")).upper()
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/sparse"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    # image and runtime
    cfg["IMAGE"] = os.environ.get("SPARSE_IMAGE", "athithya5354/sparse:amd64-arm64-v2")
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
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("SPARSE_STARTUP_FAILURE_THRESHOLD", "24")),
        })
    else:
        cfg.update({
            "REPLICAS": int(os.environ.get("SPARSE_REPLICAS", "1")),
            "CPU_REQUEST": os.environ.get("SPARSE_CPU_REQUEST", "250m"),
            "CPU_LIMIT": os.environ.get("SPARSE_CPU_LIMIT", "1000m"),
            "MEMORY_REQUEST": os.environ.get("SPARSE_MEMORY_REQUEST", "512Mi"),
            "MEMORY_LIMIT": os.environ.get("SPARSE_MEMORY_LIMIT", "1Gi"),
            "STARTUP_FAILURE_THRESHOLD": int(os.environ.get("SPARSE_STARTUP_FAILURE_THRESHOLD", "60")),
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
    # secrets sources
    cfg["SECRETS_JSON"] = os.environ.get("SPARSE_SECRETS_JSON", "")  # JSON string: {"secretname": {"k":"v",...}, ...}
    cfg["SECRETS_DIR"] = os.environ.get("SPARSE_SECRETS_DIR", "")    # directory path; subdirs => secret names; files => keys
    cfg["SECRETS_ENV_PREFIX"] = os.environ.get("SPARSE_SECRETS_ENV_PREFIX", "")  # prefix for env-secret mapping; format: PREFIX<SECRET>__<KEY>
    cfg["APPLY_SECRETS"] = os.environ.get("SPARSE_APPLY_SECRETS", "true").lower() in ("1", "true", "yes")
    cfg["DELETE_SECRETS_ON_DELETE"] = os.environ.get("SPARSE_DELETE_SECRETS", "false").lower() in ("1", "true", "yes")
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

# -------------------- YAML renderers (non-secret) --------------------
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
    labels = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "embedder",
        "app.kubernetes.io/managed-by": "gen_sparse",
        "app.kubernetes.io/instance": cfg["SERVICE_NAME"],
        "env": cfg["ENV"].lower(),
    }
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
        ns = {}
        if "=" in cfg["GPU_NODE_SELECTOR"]:
            k, v = cfg["GPU_NODE_SELECTOR"].split("=", 1)
            ns[k] = v
        else:
            ns[cfg["GPU_NODE_SELECTOR"]] = "true"
        pod_spec["spec"]["template"]["spec"]["nodeSelector"] = ns

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
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-svc", "namespace": cfg["NAMESPACE"], "labels": {
            "app.kubernetes.io/name": cfg["SERVICE_NAME"],
            "app.kubernetes.io/managed-by": "gen_sparse",
            "env": cfg["ENV"].lower(),
        }},
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

# -------------------- secrets ingestion --------------------
def load_secrets_from_json(json_str: str) -> Dict[str, Dict[str, str]]:
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ValueError("SPARSE_SECRETS_JSON must be a JSON object mapping secretName -> {key: value}")
        # normalize to strings
        out: Dict[str, Dict[str, str]] = {}
        for sname, body in parsed.items():
            if not isinstance(body, dict):
                raise ValueError(f"secret {sname} must be an object")
            out[sname] = {str(k): str(v) for k, v in body.items()}
        return out
    except Exception as e:
        fatal(f"Invalid SPARSE_SECRETS_JSON: {e}")

def load_secrets_from_dir(path: str) -> Dict[str, Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        fatal(f"SPARSE_SECRETS_DIR path does not exist: {path}")
    out: Dict[str, Dict[str, str]] = {}
    # if path contains subdirectories -> each subdir is a secret
    subdirs = [x for x in p.iterdir() if x.is_dir()]
    if subdirs:
        for d in subdirs:
            secret_name = d.name
            keys = {}
            for f in d.iterdir():
                if f.is_file():
                    keys[f.name] = f.read_bytes().decode("utf-8")
            if keys:
                out[secret_name] = keys
    else:
        # no subdirs: treat the directory as single secret
        keys = {}
        for f in p.iterdir():
            if f.is_file():
                keys[f.name] = f.read_bytes().decode("utf-8")
        if keys:
            out[p.name] = keys
    return out

def load_secrets_from_env(prefix: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]
        # expected format: SECRETNAME__KEY (double underscore)
        if "__" not in suffix:
            log.warning("Ignoring env secret variable without secret/key format (expected SECRET__KEY): %s", k)
            continue
        secret_name, key = suffix.split("__", 1)
        secret_name = secret_name.lower()
        out.setdefault(secret_name, {})[key] = v
    return out

def collect_secrets(cfg: dict) -> Dict[str, Dict[str, str]]:
    secrets: Dict[str, Dict[str, str]] = {}
    if cfg.get("SECRETS_JSON"):
        log.info("Loading secrets from SPARSE_SECRETS_JSON")
        secrets.update(load_secrets_from_json(cfg["SECRETS_JSON"]))
    if cfg.get("SECRETS_DIR"):
        log.info("Loading secrets from SPARSE_SECRETS_DIR: %s", cfg["SECRETS_DIR"])
        secrets.update(load_secrets_from_dir(cfg["SECRETS_DIR"]))
    if cfg.get("SECRETS_ENV_PREFIX"):
        log.info("Loading secrets from env with prefix: %s", cfg["SECRETS_ENV_PREFIX"])
        secrets.update(load_secrets_from_env(cfg["SECRETS_ENV_PREFIX"]))
    # normalize secret names to safe dns-1123 label if necessary
    normalized = {}
    for name, kv in secrets.items():
        safe = name.lower().replace("_", "-")
        safe = "".join(c for c in safe if (c.isalnum() or c == "-"))
        if not safe:
            fatal(f"Invalid secret name derived from '{name}'")
        normalized[safe] = {str(k): str(v) for k, v in kv.items()}
    return normalized

# -------------------- cluster helpers for namespace & secrets --------------------
def ensure_namespace(cfg: dict) -> None:
    ns = cfg["NAMESPACE"]
    res = kubectl_execute(["get", "namespace", ns])
    if res.get("ok"):
        log.info("Namespace exists: %s", ns)
        # try to add label idempotently
        _patch = {
            "metadata": {"labels": {"app.kubernetes.io/managed-by": "gen_sparse"}}
        }
        _yaml = yaml.safe_dump(_patch, sort_keys=False)
        kubectl_execute(["patch", "namespace", ns, "--patch", _yaml])
        return
    # create namespace
    log.info("Creating namespace: %s", ns)
    ns_manifest = render_namespace(cfg)
    out = kubectl_apply_yaml(ns_manifest)
    if not out.get("applied", False):
        fatal(f"Failed to create namespace {ns}: {out.get('stderr') or out.get('error')}")
    log.info("Namespace created: %s", ns)

def apply_secret(cfg: dict, secret_name: str, data: Dict[str, str]) -> None:
    """
    Apply a Kubernetes secret using 'stringData' (so we never write base64 or secret to disk).
    This is idempotent.
    """
    if not secret_name or not data:
        fatal("secret_name and data are required")
    # validate keys/values to be strings and not empty
    for k, v in data.items():
        if v is None or v == "":
            fatal(f"Secret {secret_name} has empty value for key '{k}' (refusing to apply empty secret).")
    secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "gen_sparse"}},
        # use stringData to let the API encode server-side; avoids base64 handling locally
        "stringData": data,
        "type": "Opaque"
    }
    y = yaml.safe_dump(secret_manifest, sort_keys=False)
    out = kubectl_apply_yaml(y)
    if not out.get("applied", False):
        # surface error
        fatal(f"Failed to apply secret {secret_name}: {out.get('stderr') or out.get('error')}")
    log.info("Applied secret: %s (keys: %s)", secret_name, ", ".join(sorted(data.keys())))

def delete_secret(cfg: dict, secret_name: str) -> None:
    res = kubectl_execute(["delete", "secret", secret_name, "-n", cfg["NAMESPACE"]])
    if not res.get("ok"):
        log.warning("Failed to delete secret %s: %s", secret_name, res.get("stderr") or res.get("error"))
    else:
        log.info("Deleted secret %s", secret_name)

# -------------------- generate / apply / delete --------------------
def generate_manifests(cfg: dict, dry_run: bool = False, verbose: bool = False) -> None:
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

def apply_to_cluster(cfg: dict, dry_run: bool = False, verbose: bool = False) -> None:
    # ensure kubectl exists
    if not shutil.which("kubectl"):
        fatal("kubectl not found in PATH; cannot apply")

    # Generate non-secret manifests first (they are safe to write)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    if dry_run:
        log.info("Dry-run: skipping kubectl apply.")
        return

    # 1) Ensure namespace exists (idempotent)
    ensure_namespace(cfg)

    # 2) Collect and apply secrets (in-memory, never written)
    if cfg.get("APPLY_SECRETS"):
        secrets = collect_secrets(cfg)
        if secrets:
            log.info("Applying %d secret(s) to namespace %s", len(secrets), cfg["NAMESPACE"])
            for sname, kv in secrets.items():
                # secret names: if the user supplied arbitrary names, prefix them for clarity
                # but do not overwrite a conscious name if it already contains the service name
                if sname.startswith(f"{cfg['SERVICE_NAME']}-") or sname == cfg['SERVICE_NAME']:
                    final_name = sname
                else:
                    final_name = f"{cfg['SERVICE_NAME']}-{sname}"
                apply_secret(cfg, final_name, kv)
        else:
            log.info("No secrets found to apply (APPLY_SECRETS=%s)", cfg.get("APPLY_SECRETS"))
    else:
        log.info("APPLY_SECRETS disabled; skipping secret application.")

    # 3) Apply non-secret manifests in deterministic order
    files = [cfg["FILES"]["namespace"], cfg["FILES"]["sa_role"], cfg["FILES"]["deployment"], cfg["FILES"]["service"]]
    if cfg["HPA_ENABLED"]:
        files.append(cfg["FILES"]["hpa"])
    combined = ""
    for p in files:
        combined += f"---\n# source: {p.name}\n" + p.read_text() + "\n"
    res = kubectl_apply_yaml(combined)
    if not res.get("applied", False):
        fatal(f"kubectl apply failed: {res.get('stderr') or res.get('error')}")
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

def delete_manifests(cfg: dict) -> None:
    # optionally delete applied secrets (explicit opt-in via SPARSE_DELETE_SECRETS=1)
    if cfg.get("DELETE_SECRETS_ON_DELETE"):
        log.info("Deleting applied secrets (DELETE_SECRETS_ON_DELETE=true)")
        secrets = collect_secrets(cfg)
        for sname in secrets.keys():
            if sname.startswith(f"{cfg['SERVICE_NAME']}-") or sname == cfg['SERVICE_NAME']:
                final_name = sname
            else:
                final_name = f"{cfg['SERVICE_NAME']}-{sname}"
            delete_secret(cfg, final_name)
    # remove generated manifests
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except Exception:
                log.exception("Failed to remove %s", p)
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except FileNotFoundError:
            pass
        log.info("Deleted manifests at %s", str(cfg["MANIFESTS_DIR"]))
    else:
        log.info("Manifests dir not present: %s", str(cfg["MANIFESTS_DIR"]))

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply Sparse embedder Kubernetes manifests.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--apply", action="store_true", help="Generate manifests and apply to cluster (requires kubectl).")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests.")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    # Fail-fast validations
    if not shutil.which("kubectl"):
        log.debug("kubectl not found; will only be able to generate files if --generate used.")
    # Ensure manifests dir exists for generation/deletion
    ensure_dir(cfg["MANIFESTS_DIR"])
    if args.delete:
        delete_manifests(cfg)
        return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return
    if args.apply:
        # apply will ensure namespace, secrets, then deployment
        apply_to_cluster(cfg, dry_run=args.dry_run, verbose=args.verbose)
        return

if __name__ == "__main__":
    main()
