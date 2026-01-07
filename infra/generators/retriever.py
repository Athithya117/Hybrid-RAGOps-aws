#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import yaml
import hashlib
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List
import datetime

# ---- helpers ----
def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)

def info(msg: str) -> None:
    print("INFO:", msg)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str, mode: int = 0o644) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    try:
        path.chmod(mode)
    except Exception:
        pass

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd_capture(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)

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

def ensure_scheme(url: str, default_scheme: str = "http://") -> str:
    if not url:
        return url
    if "://" in url:
        return url
    return default_scheme + url

# ---- config loader ----
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg["MANIFESTS_DIR"] = Path(os.getenv("MANIFESTS_DIR", "infra/manifests/retriever")).resolve()
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["ENV"] = os.getenv("ENV", "STAGING").upper()
    cfg["USE_AZURE_KEYVAULT"] = os.getenv("USE_AZURE_KEYVAULT", "false").lower() in ("1", "true", "yes")
    cfg["ALLOW_MISSING_SECRETS"] = os.getenv("ALLOW_MISSING_SECRETS", "false").lower() in ("1", "true", "yes")

    # core
    cfg["IMAGE"] = os.getenv("QUERY_IMAGE", os.getenv("RETRIEVAL_IMAGE", "docker.io/athithya5354/retrieval:v21"))
    cfg["NAMESPACE"] = os.getenv("QUERY_NAMESPACE", os.getenv("RETRIEVAL_NAMESPACE", "inference"))
    cfg["SERVICE_NAME"] = os.getenv("QUERY_SERVICE_NAME", os.getenv("RETRIEVAL_NAME", "retrieval"))
    cfg["PORT"] = int(os.getenv("QUERY_PORT", os.getenv("RETRIEVAL_HTTP_PORT", "8001")))
    cfg["METRICS_PORT"] = int(os.getenv("RETRIEVAL_METRICS_PORT", str(cfg["PORT"])))
    cfg["REPLICAS"] = int(os.getenv("REPLICAS", os.getenv("RETRIEVER_REPLICAS", "1")))
    cfg["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")

    # resources
    cfg["CPU_REQUEST"] = os.getenv("RETRIEVAL_RES_CPU", "250m")
    cfg["CPU_LIMIT"] = os.getenv("RETRIEVAL_RES_CPU", cfg["CPU_REQUEST"])
    cfg["MEMORY_REQUEST"] = os.getenv("RETRIEVAL_RES_MEM", "512Mi")
    cfg["MEMORY_LIMIT"] = os.getenv("RETRIEVAL_RES_MEM", cfg["MEMORY_REQUEST"])

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

    # endpoints + defaults (kept in effective config but NOT written into configmap unless opt-in)
    defaults = {
        "QDRANT_URL": "http://qdrant.qdrant.svc.cluster.local:6333",
        "DENSE_URL": "http://dense.models.svc.cluster.local:8200",
        "SPARSE_URL": "http://sparse.models.svc.cluster.local:8201",
        "RERANKER_URL": "http://reranker.models.svc.cluster.local:8202",
    }
    for k, v in defaults.items():
        ev = os.environ.get(k)
        if ev and ev.strip():
            cfg[k] = ensure_scheme(ev.strip())
        else:
            cfg[k] = v

    # opt-in: whether to write the endpoints into the ConfigMap. Default = False (safe).
    cfg["USE_WRITE_ENDPOINTS"] = os.getenv("USE_WRITE_ENDPOINTS", "false").lower() in ("1", "true", "yes")

    # app / LLM / retrieval envs
    cfg["COLLECTION_NAME"] = os.getenv("COLLECTION_NAME", "default_rag_collection1")
    cfg["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    cfg["HTTP_TIMEOUT"] = os.getenv("HTTP_TIMEOUT", "10.0")
    cfg["DENSE_DIM"] = os.getenv("DENSE_DIM", "384")
    cfg["MAX_CHUNKS_TO_LLM"] = os.getenv("MAX_CHUNKS_TO_LLM", "6")
    cfg["LLM_MODEL"] = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    cfg["LLM_MAX_TOKENS"] = os.getenv("LLM_MAX_TOKENS", "512")
    cfg["LLM_TEMPERATURE"] = os.getenv("LLM_TEMPERATURE", "0.0")
    cfg["RERANKER_MODE"] = os.getenv("RERANKER_MODE", "AUTO")
    cfg["RERANK_TOPK"] = os.getenv("RERANK_TOPK", "20")
    cfg["QUERY_TOPK_DENSE"] = os.getenv("QUERY_TOPK_DENSE", "")
    cfg["QUERY_TOPK_SPARSE"] = os.getenv("QUERY_TOPK_SPARSE", "")
    cfg["RRF_TOP_N"] = os.getenv("RRF_TOP_N", "")

    # azure storage (connection string to be treated as secret)
    cfg["AZURE_STORAGE_CONNECTION_STRING"] = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    cfg["AZURE_STORAGE_ACCOUNT_NAME"] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    cfg["AZURE_ENDPOINT_SUFFIX"] = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
    cfg["AZURE_USE_MANAGED_IDENTITY"] = os.getenv("AZURE_USE_MANAGED_IDENTITY", "false").lower() in ("1", "true", "yes")

    # node selector
    cfg["NODE_SELECTOR"] = os.getenv("NODE_SELECTOR", "")

    # secret values (in-memory only) - include AZURE_STORAGE_CONNECTION_STRING and LLM/other API keys
    cfg["SECRET_VALUES"] = {}
    for k in ("QDRANT_API_KEY", "AZURE_STORAGE_CONNECTION_STRING", "GROQ_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "AZURE_STORAGE_ACCOUNT_KEY", "AZURE_SAS_TOKEN"):
        v = os.getenv(k)
        if v:
            cfg["SECRET_VALUES"][k] = v

    # RBAC names
    cfg["SA_NAME"] = os.getenv("QUERY_SA_NAME", f"{cfg['SERVICE_NAME']}-sa")
    cfg["ROLE_NAME"] = os.getenv("QUERY_ROLE_NAME", f"{cfg['SERVICE_NAME']}-role")
    cfg["ROLEBIND_NAME"] = os.getenv("QUERY_ROLEBIND_NAME", f"{cfg['SERVICE_NAME']}-rb")

    # files
    m = cfg["MANIFESTS_DIR"]
    cfg["FILES"] = {
        "namespace": m / "00-namespace.yaml",
        "sa_role": m / "01-sa-role.yaml",
        "deployment": m / "02-deployment.yaml",
        "service": m / "03-service.yaml",
        "configmap": m / "04-configmap.yaml",
        "hpa": m / "05-hpa.yaml",
        "externalsecret": m / "06-externalsecret.yaml",
        "inputs_hash": m / ".inputs_hash",
    }

    cfg["LABELS"] = {
        "app.kubernetes.io/name": cfg["SERVICE_NAME"],
        "app.kubernetes.io/component": "inference",
        "app.kubernetes.io/managed-by": "retriever-generator",
        "app.kubernetes.io/instance": cfg["SERVICE_NAME"],
        "env": cfg["ENV"].lower(),
    }
    return cfg

# ---- renderers ----
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

def _cm_data_entries(cfg: Dict[str, Any]) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    candidates = [
        ("COLLECTION_NAME", cfg.get("COLLECTION_NAME", "")),
        ("HTTP_TIMEOUT", cfg.get("HTTP_TIMEOUT", "")),
        ("DENSE_DIM", cfg.get("DENSE_DIM", "")),
        ("MAX_CHUNKS_TO_LLM", cfg.get("MAX_CHUNKS_TO_LLM", "")),
        ("LLM_MODEL", cfg.get("LLM_MODEL", "")),
        ("LLM_MAX_TOKENS", str(cfg.get("LLM_MAX_TOKENS", ""))),
        ("LLM_TEMPERATURE", str(cfg.get("LLM_TEMPERATURE", ""))),
        ("RERANKER_MODE", str(cfg.get("RERANKER_MODE", ""))),
        ("RERANK_TOPK", str(cfg.get("RERANK_TOPK", ""))),
        ("QUERY_TOPK_DENSE", cfg.get("QUERY_TOPK_DENSE", "")),
        ("QUERY_TOPK_SPARSE", cfg.get("QUERY_TOPK_SPARSE", "")),
        ("RRF_TOP_N", cfg.get("RRF_TOP_N", "")),
    ]
    for k, v in candidates:
        if v is not None and str(v).strip() != "":
            entries[k] = str(v)

    if cfg.get("USE_WRITE_ENDPOINTS"):
        for k in ("QDRANT_URL", "DENSE_URL", "SPARSE_URL", "RERANKER_URL"):
            val = cfg.get(k)
            if val and str(val).strip() != "":
                entries[k] = val

    if cfg.get("AZURE_STORAGE_ACCOUNT_NAME"):
        entries["AZURE_STORAGE_ACCOUNT_NAME"] = cfg["AZURE_STORAGE_ACCOUNT_NAME"]
    if cfg.get("AZURE_ENDPOINT_SUFFIX"):
        entries["AZURE_ENDPOINT_SUFFIX"] = cfg["AZURE_ENDPOINT_SUFFIX"]

    sys_prompt = os.getenv("LLM_SYSTEM_PROMPT")
    user_prompt = os.getenv("LLM_USER_PROMPT_TEMPLATE")
    if sys_prompt and sys_prompt.strip():
        entries["LLM_SYSTEM_PROMPT"] = sys_prompt
    if user_prompt and user_prompt.strip():
        entries["LLM_USER_PROMPT_TEMPLATE"] = user_prompt
    return entries

def render_configmap(cfg: Dict[str, Any]) -> str:
    data = _cm_data_entries(cfg)
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-config", "namespace": cfg["NAMESPACE"], "labels": cfg["LABELS"]},
        "data": data,
    }
    return yaml.safe_dump(cm, sort_keys=False)

def render_deployment(cfg: Dict[str, Any]) -> str:
    labels = cfg["LABELS"].copy()
    env_from = [{"configMapRef": {"name": f"{cfg['SERVICE_NAME']}-config"}}]
    secret_name = f"{cfg['SERVICE_NAME']}-secret"
    if cfg["SECRET_VALUES"] and not cfg["USE_AZURE_KEYVAULT"]:
        env_from.append({"secretRef": {"name": secret_name}})

    env_list = [
        {"name": "PORT", "value": str(cfg["PORT"])},
        {"name": "ENV", "value": cfg["ENV"]},
        {"name": "LOG_LEVEL", "value": cfg["LOG_LEVEL"]},
    ]

    container_ports = [{"containerPort": cfg["PORT"]}]
    if cfg["METRICS_PORT"] != cfg["PORT"]:
        container_ports.append({"containerPort": cfg["METRICS_PORT"], "name": "metrics"})

    container = {
        "name": cfg["SERVICE_NAME"],
        "image": cfg["IMAGE"],
        "ports": container_ports,
        "envFrom": env_from,
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

    # NOTE: changed annotations to monitoring.io/* so vmagent's relabel rules will pick up the pod.
    pod_annotations = {
        "monitoring.io/scrape": "true",
        "monitoring.io/port": str(cfg["METRICS_PORT"]),
        "monitoring.io/path": "/metrics",
    }

    pod_template = {
        "metadata": {"labels": labels, "annotations": pod_annotations},
        "spec": {"serviceAccountName": cfg["SA_NAME"], "containers": [container]},
    }
    if cfg["NODE_SELECTOR"]:
        sel = {}
        for part in cfg["NODE_SELECTOR"].split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                sel[k.strip()] = v.strip()
        if sel:
            pod_template["spec"]["nodeSelector"] = sel

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-deployment", "namespace": cfg["NAMESPACE"], "labels": labels},
        "spec": {"replicas": cfg["REPLICAS"], "selector": {"matchLabels": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}}, "template": pod_template},
    }
    return yaml.safe_dump(deployment, sort_keys=False)

def render_service(cfg: Dict[str, Any]) -> str:
    ports = [{"port": cfg["PORT"], "targetPort": cfg["PORT"], "protocol": "TCP", "name": "http"}]
    if cfg["METRICS_PORT"] != cfg["PORT"]:
        ports.append({"port": cfg["METRICS_PORT"], "targetPort": cfg["METRICS_PORT"], "protocol": "TCP", "name": "metrics"})
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{cfg['SERVICE_NAME']}-svc", "namespace": cfg["NAMESPACE"], "labels": cfg["LABELS"]},
        "spec": {"type": "ClusterIP", "ports": ports, "selector": {"app.kubernetes.io/name": cfg["SERVICE_NAME"]}},
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

def render_external_secret(cfg: Dict[str, Any]) -> Optional[str]:
    if not cfg["USE_AZURE_KEYVAULT"]:
        return None
    kv = os.getenv("AZURE_KEY_VAULT_NAME")
    if not kv and not cfg["ALLOW_MISSING_SECRETS"]:
        die("USE_AZURE_KEYVAULT=true requires AZURE_KEY_VAULT_NAME env (or set ALLOW_MISSING_SECRETS=true)")
    keys: List[Dict[str, Any]] = []
    # map remote secret names to both uppercase and legacy lowercase keys
    mapping = {
        "retriever-qdrant-api-key": "QDRANT_API_KEY",
        "retriever-storage-account-connstr": "AZURE_STORAGE_CONNECTION_STRING",
        "retriever-groq-api-key": "GROQ_API_KEY",
        "retriever-openai-api-key": "OPENAI_API_KEY",
        "retriever-llm-api-key": "LLM_API_KEY",
        "retriever-azure-sas-token": "AZURE_SAS_TOKEN",
        "retriever-azure-account-key": "AZURE_STORAGE_ACCOUNT_KEY",
    }
    for remote, upper_key in mapping.items():
        name = os.getenv(f"AZ_KEYVAULT_{remote.upper().replace('-', '_')}", remote)
        # add uppercase entry
        keys.append({"secretKey": upper_key, "remoteRef": {"key": name}})
        # add lowercase legacy variant for backward compatibility
        lower_key = upper_key.lower()
        if lower_key != upper_key:
            keys.append({"secretKey": lower_key, "remoteRef": {"key": name}})
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

# ---- secrets in-cluster (no disk files) ----
def kubectl_apply_stream(yaml_docs: str, dry_run: bool = False) -> Dict[str, Any]:
    kubectl = which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply", "-f", "-"]
    if dry_run:
        cmd = [kubectl, "apply", "--dry-run=client", "-f", "-"]
    try:
        proc = subprocess.run(cmd, input=yaml_docs.encode("utf-8"), capture_output=True, check=True, timeout=120)
        out = proc.stdout.decode() if proc.stdout else ""
        return {"applied": True, "stdout": out}
    except subprocess.CalledProcessError as e:
        return {"applied": False, "stderr": e.stderr.decode() if e.stderr else str(e)}
    except Exception as e:
        return {"applied": False, "stderr": str(e)}

def create_or_update_secret_in_cluster(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    if cfg["USE_AZURE_KEYVAULT"]:
        return {"created": False, "reason": "using_azure_keyvault"}
    if not cfg["SECRET_VALUES"]:
        return {"created": False, "reason": "no_secrets"}
    kubectl = which("kubectl")
    if not kubectl:
        return {"created": False, "reason": "kubectl_missing"}
    name = f"{cfg['SERVICE_NAME']}-secret"
    ns = cfg["NAMESPACE"]
    # create secret with literal keys matching app environment variable names (UPPERCASE) and legacy lowercase keys
    args = [kubectl, "create", "secret", "generic", name, "--namespace", ns]
    seen_keys = set()
    for k, v in cfg["SECRET_VALUES"].items():
        # primary (uppercase) key
        if k not in seen_keys:
            args += ["--from-literal", f"{k}={v}"]
            seen_keys.add(k)
        # legacy lowercase variant
        lower = k.lower()
        if lower not in seen_keys:
            args += ["--from-literal", f"{lower}={v}"]
            seen_keys.add(lower)
    args += ["--dry-run=client", "-o", "yaml"]
    if dry_run:
        return {"created": False, "reason": "dry_run", "cmd": " ".join(args)}
    try:
        p1 = subprocess.run(args, capture_output=True, check=True, timeout=30)
        yaml_out = p1.stdout.decode()
        res = kubectl_apply_stream(yaml_out, dry_run=False)
        if not res.get("applied"):
            return {"created": False, "stderr": res.get("stderr")}
        return {"created": True, "stdout": res.get("stdout")}
    except subprocess.CalledProcessError as e:
        return {"created": False, "stderr": e.stderr.decode() if e.stderr else str(e)}
    except Exception as e:
        return {"created": False, "stderr": str(e)}

# ---- leak detection ----
def detect_secret_leak(rendered: str, secret_map: Dict[str, str]) -> Optional[str]:
    for k, v in secret_map.items():
        if not v:
            continue
        if len(v) >= 8 and v in rendered:
            return k
    return None

# ---- generate / apply / delete ----
def generate(cfg: Dict[str, Any]) -> None:
    ensure_dir(cfg["MANIFESTS_DIR"])
    ihash = canonical_inputs_hash(cfg)
    existing = None
    try:
        if cfg["INPUTS_HASH_PATH"].exists():
            existing = cfg["INPUTS_HASH_PATH"].read_text(encoding="utf-8").strip()
    except Exception:
        existing = None

    # If hash matches AND core non-secret files exist, skip. Otherwise regenerate.
    core_keys = ("sa_role", "configmap", "deployment", "service")
    core_files_exist = all(cfg["FILES"].get(k) and cfg["FILES"][k].exists() for k in core_keys)
    if existing == ihash and core_files_exist:
        info("No non-secret changes; skipping generation.")
        return

    ns_yaml = render_namespace(cfg)
    sa_yaml = render_sa_role(cfg)
    cm_yaml = render_configmap(cfg)
    dep_yaml = render_deployment(cfg)
    svc_yaml = render_service(cfg)
    hpa_yaml = render_hpa(cfg)
    es_yaml = render_external_secret(cfg)

    leak = detect_secret_leak(dep_yaml, cfg["SECRET_VALUES"])
    if leak:
        die(f"Secret value for {leak} would be embedded in deployment; refusing to generate.")

    atomic_write(cfg["FILES"]["namespace"], ns_yaml)
    atomic_write(cfg["FILES"]["sa_role"], sa_yaml)
    atomic_write(cfg["FILES"]["configmap"], cm_yaml)
    atomic_write(cfg["FILES"]["deployment"], dep_yaml)
    atomic_write(cfg["FILES"]["service"], svc_yaml)
    if hpa_yaml:
        atomic_write(cfg["FILES"]["hpa"], hpa_yaml)
    else:
        try:
            cfg["FILES"]["hpa"].unlink()
        except Exception:
            pass
    if es_yaml:
        atomic_write(cfg["FILES"]["externalsecret"], es_yaml)
    else:
        try:
            cfg["FILES"]["externalsecret"].unlink()
        except Exception:
            pass

    cfg["INPUTS_HASH_PATH"].write_text(ihash, encoding="utf-8")
    info(f"Wrote manifests to {cfg['MANIFESTS_DIR']} (secrets are NOT written to disk)")

def apply(cfg: Dict[str, Any]) -> None:
    if not which("kubectl"):
        die("kubectl not found in PATH")

    # generate non-secret manifests (writes namespace yaml to disk)
    # ensure generate will run if files are missing even when hash matches
    generate(cfg)

    # ensure namespace exists in cluster first
    ns_yaml = render_namespace(cfg)
    ns_res = kubectl_apply_stream(ns_yaml, dry_run=False)
    if not ns_res.get("applied", False):
        die(f"kubectl apply namespace failed: {ns_res.get('stderr')}")
    info(f"Ensured namespace {cfg['NAMESPACE']} exists.")

    # create/update secret in-cluster if provided and not using KeyVault
    secret_res = create_or_update_secret_in_cluster(cfg, dry_run=False)
    if secret_res.get("created") is False and secret_res.get("reason") == "no_secrets":
        if not cfg["ALLOW_MISSING_SECRETS"]:
            info("No secrets present. If your app requires secrets (qdrant key, storage connection string, API keys) provide them as envs or enable KeyVault.")
    elif secret_res.get("created") is False and secret_res.get("reason") == "kubectl_missing":
        die("kubectl required to create secrets in cluster; install kubectl or run generate and create secrets manually.")
    elif secret_res.get("created") is False and secret_res.get("reason") == "using_azure_keyvault":
        info("Using Azure KeyVault mode; ensure ExternalSecrets operator and SecretStore exist before applying.")
    elif secret_res.get("created"):
        info("Created/updated in-cluster secret from environment variables. Secret keys include both uppercase and legacy lowercase variants.")

    # Build combined YAML stream, excluding namespace which was applied already
    parts: List[str] = []
    for key in ("sa_role", "configmap", "service", "deployment", "hpa"):
        p = cfg["FILES"].get(key)
        if p and p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    if cfg["USE_AZURE_KEYVAULT"]:
        p = cfg["FILES"].get("externalsecret")
        if p and p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    if not parts:
        die("No manifests to apply (generate first)")
    combined = "\n---\n".join(parts)
    res = kubectl_apply_stream(combined, dry_run=False)
    if not res.get("applied", False):
        die(f"kubectl apply failed: {res.get('stderr')}")
    info("Applied manifests to cluster.")
    summary = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "image": cfg["IMAGE"], "namespace": cfg["NAMESPACE"], "replicas": cfg["REPLICAS"]}
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))

def delete(cfg: Dict[str, Any]) -> None:
    # remove local manifests but preserve namespace manifest and inputs hash
    if cfg["MANIFESTS_DIR"].exists():
        preserve = {cfg["FILES"]["namespace"].name, cfg["FILES"]["inputs_hash"].name}
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            if p.name in preserve:
                continue
            try:
                p.unlink()
            except Exception:
                pass
        info(f"Deleted local manifests in {cfg['MANIFESTS_DIR']} (namespace manifest and inputs hash preserved)")
    # delete cluster resources except namespace
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
        if cfg["USE_AZURE_KEYVAULT"]:
            cmds.append(["kubectl", "delete", "externalsecret", f"{cfg['SERVICE_NAME']}-externalsecret", "-n", ns, "--ignore-not-found"])
        # do NOT delete namespace
        for c in cmds:
            try:
                subprocess.run(c, check=False, capture_output=True, timeout=30)
            except Exception:
                pass
        info("Requested deletion of k8s resources (ignore-not-found). Namespace not deleted.")
    else:
        info("kubectl not present; local manifests removed (namespace preserved) but cluster resources remain.")

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Generator for retrieval/query manifests (secret-safe).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="generate + apply manifests (creates namespace if needed)")
    g.add_argument("--delete", action="store_true", help="delete manifests/resources (preserves namespace)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()

    if not cfg["IMAGE"]:
        die("QUERY_IMAGE / RETRIEVAL_IMAGE is required (set env).")
    if not cfg["NAMESPACE"]:
        die("QUERY_NAMESPACE required")
    if cfg["USE_AZURE_KEYVAULT"] and not os.getenv("AZURE_KEY_VAULT_NAME") and not cfg["ALLOW_MISSING_SECRETS"]:
        die("AZURE_KEY_VAULT_NAME required when USE_AZURE_KEYVAULT=true (or set ALLOW_MISSING_SECRETS=true)")

    if args.apply:
        apply(cfg)
        return
    if args.delete:
        delete(cfg)
        return

if __name__ == "__main__":
    main()
