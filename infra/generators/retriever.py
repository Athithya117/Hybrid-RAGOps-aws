from __future__ import annotations
import os
import sys
import json
import yaml
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import datetime

# ---------- small helpers ----------
def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)

def info(msg: str) -> None:
    print("INFO:", msg)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    try:
        path.chmod(mode)
    except Exception:
        pass

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
        if k in ("INPUTS_HASH_PATH", "SECRET_VALUES"):
            continue
        serial[k] = canonicalize(cfg[k])
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd_capture(cmd: list[str], input_bytes: Optional[bytes] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)

# ---------------------------
# Config loader
# ---------------------------
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    # generator basics
    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", "infra/manifests/retriever"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["ENV"] = os.getenv("ENV", "STAGING").upper()
    cfg["USE_AZURE_KEYVAULT"] = os.getenv("USE_AZURE_KEYVAULT", "false").lower() in ("1", "true", "yes")
    cfg["ALLOW_MISSING_SECRETS"] = os.getenv("ALLOW_MISSING_SECRETS", "false").lower() in ("1", "true", "yes")

    # core service
    cfg["IMAGE"] = os.getenv("QUERY_IMAGE", "athithya5354/retrieval:amd64-arm64-v2")
    cfg["NAMESPACE"] = os.getenv("QUERY_NAMESPACE", "inference")
    cfg["SERVICE_NAME"] = os.getenv("QUERY_SERVICE_NAME", "retrieval")
    cfg["PORT"] = int(os.getenv("QUERY_PORT", "8001"))
    cfg["REPLICAS"] = int(os.getenv("REPLICAS", "1")) if cfg["ENV"] != "PROD" else int(os.getenv("REPLICAS", os.getenv("QUERY_REPLICAS", "3")))
    cfg["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")

    # resources
    cfg["CPU_REQUEST"] = os.getenv("CPU_REQUEST", "250m")
    cfg["CPU_LIMIT"] = os.getenv("CPU_LIMIT", "1000m")
    cfg["MEMORY_REQUEST"] = os.getenv("MEMORY_REQUEST", "512Mi")
    cfg["MEMORY_LIMIT"] = os.getenv("MEMORY_LIMIT", "1Gi")

    # probes
    cfg["READINESS_INITIAL_DELAY"] = int(os.getenv("READINESS_INITIAL_DELAY", "5"))
    cfg["LIVENESS_INITIAL_DELAY"] = int(os.getenv("LIVENESS_INITIAL_DELAY", "10"))
    cfg["PROBE_PERIOD_SECONDS"] = int(os.getenv("PROBE_PERIOD_SECONDS", "5"))
    cfg["PROBE_TIMEOUT_SECONDS"] = int(os.getenv("PROBE_TIMEOUT_SECONDS", "3"))
    cfg["STARTUP_FAILURE_THRESHOLD"] = int(os.getenv("STARTUP_FAILURE_THRESHOLD", "60"))

    # HPA
    cfg["HPA_ENABLED"] = os.getenv("HPA_ENABLED", "false").lower() in ("1", "true", "yes")
    cfg["HPA_MIN"] = int(os.getenv("HPA_MIN", "1"))
    cfg["HPA_MAX"] = int(os.getenv("HPA_MAX", "5"))
    cfg["HPA_TARGET_CPU"] = int(os.getenv("HPA_TARGET_CPU", "60"))

    # endpoints / service discovery
    cfg["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://qdrant.inference.svc.cluster.local:6333")
    cfg["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY", "")
    cfg["COLLECTION_NAME"] = os.getenv("COLLECTION_NAME", "default_rag_collection1")
    cfg["DENSE_URL"] = os.getenv("DENSE_URL", "http://dense.inference.svc.cluster.local:8200")
    cfg["SPARSE_URL"] = os.getenv("SPARSE_URL", "http://sparse.inference.svc.cluster.local:8201")
    cfg["RERANKER_URL"] = os.getenv("RERANKER_URL", "http://reranker.inference.svc.cluster.local:8202")

    # azure storage (for presign)
    cfg["AZURE_STORAGE_ACCOUNT_NAME"] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    cfg["AZURE_STORAGE_ACCOUNT_KEY"] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
    cfg["AZURE_ENDPOINT_SUFFIX"] = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
    cfg["AZURE_USE_MANAGED_IDENTITY"] = os.getenv("AZURE_USE_MANAGED_IDENTITY", "false").lower() in ("1", "true", "yes")

    # node selector
    cfg["NODE_SELECTOR"] = os.getenv("NODE_SELECTOR", "")

    # secret values (do not include in inputs hash, and will not be written to disk)
    cfg["SECRET_VALUES"] = {}
    if cfg["QDRANT_API_KEY"]:
        cfg["SECRET_VALUES"]["QDRANT_API_KEY"] = cfg["QDRANT_API_KEY"]
    if cfg["AZURE_STORAGE_ACCOUNT_KEY"]:
        cfg["SECRET_VALUES"]["AZURE_STORAGE_ACCOUNT_KEY"] = cfg["AZURE_STORAGE_ACCOUNT_KEY"]

    # RBAC names
    cfg["SA_NAME"] = os.getenv("QUERY_SA_NAME", f"{cfg['SERVICE_NAME']}-sa")
    cfg["ROLE_NAME"] = os.getenv("QUERY_ROLE_NAME", f"{cfg['SERVICE_NAME']}-role")
    cfg["ROLEBIND_NAME"] = os.getenv("QUERY_ROLEBIND_NAME", f"{cfg['SERVICE_NAME']}-rb")

    # files (we will not write secret manifests)
    m = cfg["MANIFESTS_DIR"]
    cfg["FILES"] = {
        "namespace": m / "00-namespace.yaml",
        "sa_role": m / "01-sa-role.yaml",
        "deployment": m / "02-deployment.yaml",
        "service": m / "03-service.yaml",
        "hpa": m / "04-hpa.yaml",
        # secret paths retained in mapping for compatibility, but generator will NOT write them
        "secret": m / "05-secret.yaml",
        "externalsecret": m / "05-externalsecret.yaml",
    }
    cfg["LABELS"] = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "inference",
        "app.kubernetes.io/managed-by": "retriever-generator",
        "app.kubernetes.io/instance": cfg["SERVICE_NAME"],
        "env": cfg["ENV"].lower(),
    }
    return cfg

# ---------------------------
# Renderers (unchanged except secrets are returned for in-memory apply)
# ---------------------------
def render_namespace(cfg: Dict[str, Any]) -> str:
    obj = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": cfg["NAMESPACE"], "labels": {"app.kubernetes.io/managed-by": "retriever-generator"}}}
    return yaml.safe_dump(obj, sort_keys=False)

def render_sa_role(cfg: Dict[str, Any]) -> str:
    sa = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": cfg["SA_NAME"], "namespace": cfg["NAMESPACE"]}}
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

def render_deployment(cfg: Dict[str, Any]) -> str:
    labels = cfg["LABELS"].copy()
    env_list = [
        {"name": "QDRANT_URL", "value": cfg["QDRANT_URL"]},
        {"name": "COLLECTION_NAME", "value": cfg["COLLECTION_NAME"]},
        {"name": "DENSE_URL", "value": cfg["DENSE_URL"]},
        {"name": "SPARSE_URL", "value": cfg["SPARSE_URL"]},
        {"name": "RERANKER_URL", "value": cfg["RERANKER_URL"]},
        {"name": "DENSE_DIM", "value": os.getenv("DENSE_DIM", "384")},
        {"name": "HTTP_TIMEOUT", "value": os.getenv("HTTP_TIMEOUT", "10.0")},
        {"name": "SPARSE_BATCH_FALLBACK", "value": os.getenv("SPARSE_BATCH_FALLBACK", "8")},
        {"name": "LLM_MODEL", "value": os.getenv("LLM_MODEL", "llama-3.1-8b-instant")},
        {"name": "LLM_MAX_TOKENS", "value": str(os.getenv("LLM_MAX_TOKENS", "512"))},
        {"name": "LLM_TEMPERATURE", "value": str(os.getenv("LLM_TEMPERATURE", "0.0"))},
        {"name": "ENV", "value": cfg["ENV"]},
        {"name": "LOG_LEVEL", "value": cfg["LOG_LEVEL"]},
        # Azure storage envs used by app for presign; AZURE_STORAGE_ACCOUNT_KEY is secret and will be mounted via Secret if provided
        {"name": "AZURE_STORAGE_ACCOUNT_NAME", "value": cfg["AZURE_STORAGE_ACCOUNT_NAME"]},
        {"name": "AZURE_ENDPOINT_SUFFIX", "value": cfg["AZURE_ENDPOINT_SUFFIX"]},
        {"name": "AZURE_USE_MANAGED_IDENTITY", "value": str(cfg["AZURE_USE_MANAGED_IDENTITY"]).lower()},
    ]

    # if secret present then reference via secretKeyRef (QDRANT_API_KEY / AZURE_STORAGE_ACCOUNT_KEY)
    if "QDRANT_API_KEY" in cfg["SECRET_VALUES"]:
        env_list.append({"name": "QDRANT_API_KEY", "valueFrom": {"secretKeyRef": {"name": f"{cfg['SERVICE_NAME']}-secret", "key": "qdrant_api_key"}}})
    if "AZURE_STORAGE_ACCOUNT_KEY" in cfg["SECRET_VALUES"]:
        env_list.append({"name": "AZURE_STORAGE_ACCOUNT_KEY", "valueFrom": {"secretKeyRef": {"name": f"{cfg['SERVICE_NAME']}-secret", "key": "azure_storage_account_key"}}})

    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": [{"containerPort": cfg["PORT"]}],
        "env": env_list,
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": cfg["PORT"]},
            "initialDelaySeconds": cfg["LIVENESS_INITIAL_DELAY"],
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": 3,
        },
        "readinessProbe": {
            "httpGet": {"path": "/readyz", "port": cfg["PORT"]},
            "initialDelaySeconds": cfg["READINESS_INITIAL_DELAY"],
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": 3,
        },
        "startupProbe": {
            "httpGet": {"path": "/healthz", "port": cfg["PORT"]},
            "periodSeconds": cfg["PROBE_PERIOD_SECONDS"],
            "timeoutSeconds": cfg["PROBE_TIMEOUT_SECONDS"],
            "failureThreshold": cfg["STARTUP_FAILURE_THRESHOLD"],
        },
        "resources": {
            "requests": {"cpu": cfg["CPU_REQUEST"], "memory": cfg["MEMORY_REQUEST"]},
            "limits": {"cpu": cfg["CPU_LIMIT"], "memory": cfg["MEMORY_LIMIT"]},
        },
    }

    pod_spec = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-deployment", "namespace": cfg["NAMESPACE"], "labels": labels},
        "spec": {
            "replicas": cfg["REPLICAS"],
            "selector": {"matchLabels": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}},
            "template": {
                "metadata": {"labels": labels, "annotations": {"prometheus.io/scrape": "true", "prometheus.io/port": str(cfg["PORT"]), "prometheus.io/path": "/metrics"}},
                "spec": {"serviceAccountName": cfg["SA_NAME"], "containers": [container]},
            },
        },
    }

    # optional node selector
    if cfg["NODE_SELECTOR"]:
        sel = {}
        for part in cfg["NODE_SELECTOR"].split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                sel[k.strip()] = v.strip()
        if sel:
            pod_spec["spec"]["template"]["spec"]["nodeSelector"] = sel

    return yaml.safe_dump(pod_spec, sort_keys=False)

def render_service(cfg: Dict[str, Any]) -> str:
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-svc", "namespace": cfg["NAMESPACE"], "labels": cfg["LABELS"]},
        "spec": {"type": "ClusterIP", "ports": [{"port": cfg["PORT"], "targetPort": cfg["PORT"], "protocol": "TCP", "name": "http"}], "selector": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}},
    }
    return yaml.safe_dump(svc, sort_keys=False)

def render_hpa(cfg: Dict[str, Any]) -> str:
    if not cfg["HPA_ENABLED"]:
        return ""
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-hpa", "namespace": cfg["NAMESPACE"]},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": f"{cfg['SERVICE_NAME']}-deployment"},
            "minReplicas": cfg["HPA_MIN"],
            "maxReplicas": cfg["HPA_MAX"],
            "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": cfg["HPA_TARGET_CPU"]}}}],
        },
    }
    return yaml.safe_dump(hpa, sort_keys=False)

def render_secret_yaml_string(cfg: Dict[str, Any]) -> Optional[str]:
    # Returns YAML string for secret (stringData) for in-memory apply; does NOT write to disk
    if cfg["USE_AZURE_KEYVAULT"]:
        return None
    if not cfg["SECRET_VALUES"]:
        return None
    data = {}
    if "QDRANT_API_KEY" in cfg["SECRET_VALUES"]:
        data["qdrant_api_key"] = cfg["SECRET_VALUES"]["QDRANT_API_KEY"]
    if "AZURE_STORAGE_ACCOUNT_KEY" in cfg["SECRET_VALUES"]:
        data["azure_storage_account_key"] = cfg["SECRET_VALUES"]["AZURE_STORAGE_ACCOUNT_KEY"]
    sec = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": f"{cfg['SERVICE_NAME']}-secret", "namespace": cfg["NAMESPACE"]}, "type": "Opaque", "stringData": data}
    return yaml.safe_dump(sec, sort_keys=False)

def render_external_secret(cfg: Dict[str, Any]) -> Optional[str]:
    if not cfg["USE_AZURE_KEYVAULT"]:
        return None
    kv = os.getenv("AZURE_KEY_VAULT_NAME")
    if not kv and not cfg["ALLOW_MISSING_SECRETS"]:
        die("USE_AZURE_KEYVAULT=true requires AZURE_KEY_VAULT_NAME env (or set ALLOW_MISSING_SECRETS=true)")
    keys = []
    q_name = os.getenv("AZ_KEYVAULT_QDRANT_SECRET", "retriever-qdrant-api-key")
    s_name = os.getenv("AZ_KEYVAULT_STORAGE_SECRET", "retriever-storage-account-key")
    if q_name:
        keys.append({"secretKey": "qdrant_api_key", "remoteRef": {"key": q_name}})
    if s_name:
        keys.append({"secretKey": "azure_storage_account_key", "remoteRef": {"key": s_name}})
    if not keys and cfg["ALLOW_MISSING_SECRETS"]:
        return None
    es = {
        "apiVersion": "external-secrets.io/v1beta1",
        "kind": "ExternalSecret",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-externalsecret", "namespace": cfg["NAMESPACE"]},
        "spec": {
            "refreshInterval": "1h",
            "secretStoreRef": {"name": os.getenv("AZURE_KEYVAULT_SECRETSTORE", "azure-keyvault-store"), "kind": "ClusterSecretStore"},
            "target": {"name": f"{cfg['SERVICE_NAME']}-secret", "creationPolicy": "Merge"},
            "data": keys,
        },
    }
    return yaml.safe_dump(es, sort_keys=False)

# ---------------------------
# K8s apply helpers
# ---------------------------
def kubectl_apply_stream(yaml_docs: str, dry_run: bool = False) -> Dict[str, Any]:
    kubectl = which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply", "-f", "-"]
    if dry_run:
        cmd = [kubectl, "apply", "--dry-run=client", "-f", "-"]
    try:
        proc = run_cmd_capture(cmd, input_bytes=yaml_docs.encode("utf-8"), timeout=120)
        if proc.returncode != 0:
            return {"applied": False, "stderr": (proc.stderr.decode() if proc.stderr else "unknown error")}
        out = proc.stdout.decode() if proc.stdout else ""
        return {"applied": True, "stdout": out}
    except Exception as e:
        return {"applied": False, "stderr": str(e)}

def ensure_namespace_exists_in_cluster(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Ensure the namespace exists in-cluster. Use kubectl apply on the namespace manifest for idempotency.
    Returns dict with 'ok' boolean.
    """
    ns_yaml = render_namespace(cfg)
    res = kubectl_apply_stream(ns_yaml, dry_run=dry_run)
    if not res.get("applied", False):
        return {"ok": False, "stderr": res.get("stderr", "failed to ensure namespace")}
    return {"ok": True, "stdout": res.get("stdout", "")}

def create_or_update_secret_in_cluster(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Create/update k8s secret from env values using kubectl pipeline.
    Does NOT write secrets to disk.
    """
    if cfg["USE_AZURE_KEYVAULT"]:
        return {"created": False, "reason": "using_azure_keyvault"}
    if not cfg["SECRET_VALUES"]:
        return {"created": False, "reason": "no_secrets"}
    kubectl = which("kubectl")
    if not kubectl:
        return {"created": False, "reason": "kubectl_missing"}

    # Ensure namespace exists before creating secret
    ns_check = ensure_namespace_exists_in_cluster(cfg, dry_run=dry_run)
    if not ns_check.get("ok", False):
        return {"created": False, "reason": "namespace_ensure_failed", "stderr": ns_check.get("stderr")}

    # Build create secret dry-run yaml using kubectl create secret generic ... --dry-run=client -o yaml
    name = f"{cfg['SERVICE_NAME']}-secret"
    ns = cfg["NAMESPACE"]
    cmd_parts = [kubectl, "create", "secret", "generic", name, "--namespace", ns]
    for k, v in cfg["SECRET_VALUES"].items():
        if k == "QDRANT_API_KEY":
            cmd_parts += ["--from-literal", f"qdrant_api_key={v}"]
        elif k == "AZURE_STORAGE_ACCOUNT_KEY":
            cmd_parts += ["--from-literal", f"azure_storage_account_key={v}"]
    cmd_parts += ["--dry-run=client", "-o", "yaml"]

    if dry_run:
        # Return what would be run
        return {"created": False, "reason": "dry_run", "cmd": " ".join(cmd_parts)}

    try:
        p1 = run_cmd_capture(cmd_parts, timeout=30)
        if p1.returncode != 0:
            return {"created": False, "stderr": (p1.stderr.decode() if p1.stderr else "failed to render secret yaml")}
        yaml_out = p1.stdout.decode()
        # Apply the generated YAML (namespace guaranteed earlier)
        res = kubectl_apply_stream(yaml_out, dry_run=False)
        if not res.get("applied", False):
            return {"created": False, "stderr": res.get("stderr")}
        return {"created": True, "stdout": res.get("stdout")}
    except subprocess.CalledProcessError as e:
        return {"created": False, "stderr": e.stderr.decode() if e.stderr else str(e)}
    except Exception as e:
        return {"created": False, "stderr": str(e)}

# ---------------------------
# Generation / apply / delete (secrets not written to disk)
# ---------------------------
def generate(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    ensure_dir(cfg["MANIFESTS_DIR"])
    ihash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing = cfg["INPUTS_HASH_PATH"].read_text(encoding="utf-8").strip()
    # If only secrets changed, do not rewrite non-secret manifests
    if existing == ihash and not dry_run:
        info("No non-secret changes; skipping generation.")
        return

    # render files (namespace, sa_role, deployment, service, hpa); secrets NOT written to disk
    atomic_write(cfg["FILES"]["namespace"], render_namespace(cfg))
    atomic_write(cfg["FILES"]["sa_role"], render_sa_role(cfg))
    atomic_write(cfg["FILES"]["deployment"], render_deployment(cfg))
    atomic_write(cfg["FILES"]["service"], render_service(cfg))
    if cfg["HPA_ENABLED"]:
        atomic_write(cfg["FILES"]["hpa"], render_hpa(cfg))
    else:
        try:
            cfg["FILES"]["hpa"].unlink()
        except Exception:
            pass

    # Ensure we DO NOT write secret/externalsecret files to disk (safety)
    try:
        if cfg["FILES"]["secret"].exists():
            cfg["FILES"]["secret"].unlink()
    except Exception:
        pass
    try:
        if cfg["FILES"]["externalsecret"].exists():
            cfg["FILES"]["externalsecret"].unlink()
    except Exception:
        pass

    cfg["INPUTS_HASH_PATH"].write_text(ihash, encoding="utf-8")
    info(f"Wrote manifests to {cfg['MANIFESTS_DIR']} (secrets are applied directly; not written to disk)")

def apply(cfg: Dict[str, Any], confirm: bool = False, dry_run: bool = False) -> None:
    if not confirm:
        die("Refusing to apply without --confirm")
    if not which("kubectl"):
        die("kubectl not found in PATH")

    # generate non-secret manifests first (writes non-secret manifests)
    generate(cfg, dry_run=dry_run)

    # ensure namespace exists before anything else (secrets, etc.)
    ns_ok = ensure_namespace_exists_in_cluster(cfg, dry_run=dry_run)
    if not ns_ok.get("ok", False):
        die(f"Failed to ensure namespace: {ns_ok.get('stderr')}")

    # create/update secret in-cluster if provided and not using KeyVault
    if cfg["USE_AZURE_KEYVAULT"]:
        # apply ExternalSecret directly into the cluster (no on-disk secrets)
        es = render_external_secret(cfg)
        if es:
            res_es = kubectl_apply_stream(es, dry_run=dry_run)
            if not res_es.get("applied", False):
                die(f"Applying ExternalSecret failed: {res_es.get('stderr')}")
            info("Applied ExternalSecret manifest into cluster (KeyVault mode).")
        else:
            if not cfg["ALLOW_MISSING_SECRETS"]:
                die("Key Vault mode enabled but no ExternalSecret data rendered and ALLOW_MISSING_SECRETS=false")
            info("No ExternalSecret rendered (ALLOW_MISSING_SECRETS=true).")
    else:
        secret_res = create_or_update_secret_in_cluster(cfg, dry_run=dry_run)
        if secret_res.get("created") is False and secret_res.get("reason") == "no_secrets":
            if not cfg["ALLOW_MISSING_SECRETS"]:
                die("No secrets present. Provide secrets as envs or set ALLOW_MISSING_SECRETS=true")
            info("No secrets present, but ALLOW_MISSING_SECRETS=true — continuing.")
        elif secret_res.get("created") is False and secret_res.get("reason") == "kubectl_missing":
            die("kubectl required to create secrets in cluster; install kubectl or run --generate and create secrets manually.")
        elif secret_res.get("created") is False and secret_res.get("reason") == "namespace_ensure_failed":
            die(f"Failed to ensure namespace before secret creation: {secret_res.get('stderr')}")
        elif secret_res.get("created"):
            info("Created/updated in-cluster secret from environment variables.")

    # Build combined YAML stream for non-secret manifests and apply
    parts = []
    for key in ("namespace", "sa_role", "service", "deployment", "hpa"):
        p = cfg["FILES"].get(key)
        if p and p.exists():
            parts.append(p.read_text(encoding="utf-8"))

    if not parts:
        die("No manifests to apply (generate first)")

    combined = "\n---\n".join(parts)
    res = kubectl_apply_stream(combined, dry_run=dry_run)
    if not res.get("applied", False):
        die(f"kubectl apply failed: {res.get('stderr')}")
    info("Applied manifests to cluster (non-secret resources).")

    # write last deploy summary
    summary = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "image": cfg["IMAGE"], "namespace": cfg["NAMESPACE"], "replicas": cfg["REPLICAS"]}
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    info("Deployment completed.")

def delete(cfg: Dict[str, Any], confirm: bool = False) -> None:
    if not confirm:
        die("Refusing to delete without --confirm")
    # remove manifest files locally (non-secret)
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                p.unlink()
            except Exception:
                pass
        info(f"Deleted local manifests in {cfg['MANIFESTS_DIR']}")
    # attempt to delete from cluster (best-effort)
    if which("kubectl"):
        ns = cfg["NAMESPACE"]
        name = cfg["SERVICE_NAME"]
        cmds = [
            ["kubectl", "delete", "deployment", f"{name}-deployment", "-n", ns, "--ignore-not-found"],
            ["kubectl", "delete", "service", f"{name}-svc", "-n", ns, "--ignore-not-found"],
            ["kubectl", "delete", "hpa", f"{name}-hpa", "-n", ns, "--ignore-not-found"],
            ["kubectl", "delete", "sa", cfg["SA_NAME"], "-n", ns, "--ignore-not-found"],
            ["kubectl", "delete", "role", cfg["ROLE_NAME"], "-n", ns, "--ignore-not-found"],
            ["kubectl", "delete", "rolebinding", cfg["ROLEBIND_NAME"], "-n", ns, "--ignore-not-found"],
        ]
        for c in cmds:
            try:
                subprocess.run(c, check=False, capture_output=True, timeout=30)
            except Exception:
                pass
        info("Requested deletion of k8s resources (ignore-not-found).")
    else:
        info("kubectl not present; local manifests removed but cluster resources remain.")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generator for retrieval/query manifests (Azure-native).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--confirm", action="store_true", help="required to apply/delete")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()

    # Basic validation (fail-fast)
    if not cfg["IMAGE"] and not (args.generate and os.getenv("ALLOW_EMPTY_IMAGE", "false").lower() in ("1", "true", "yes")):
        die("QUERY_IMAGE is required (set QUERY_IMAGE env) unless ALLOW_EMPTY_IMAGE=true for dry generation.")
    if not cfg["NAMESPACE"]:
        die("QUERY_NAMESPACE is required (set QUERY_NAMESPACE env).")
    if cfg["USE_AZURE_KEYVAULT"] and not os.getenv("AZURE_KEY_VAULT_NAME") and not cfg["ALLOW_MISSING_SECRETS"]:
        die("AZURE_KEY_VAULT_NAME required when USE_AZURE_KEYVAULT=true (or set ALLOW_MISSING_SECRETS=true)")

    if args.generate:
        generate(cfg, dry_run=args.dry_run)
        return
    if args.apply:
        apply(cfg, confirm=args.confirm, dry_run=args.dry_run)
        return
    if args.delete:
        delete(cfg, confirm=args.confirm)
        return

if __name__ == "__main__":
    main()
