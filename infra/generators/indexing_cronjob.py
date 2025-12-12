#!/usr/bin/env python3
"""
indexing_cronjob.py (final, kubectl + pyyaml)

- Uses PyYAML to render Kubernetes manifests and `kubectl` to apply/delete them.
- Autodetects Managed Identity / UAI mode and does NOT require storage keys when UAI present.
- Injects a deterministic, exhaustive set of runtime envs into the CronJob container.
- Sensitive values are mounted from Kubernetes Secrets only if explicitly provided.
- CLI: --apply, --delete, --dry-run, --delete-secrets
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
import argparse
import traceback
from typing import Dict, List
import yaml
from pathlib import Path

# -------------------- Comprehensive defaults (app + infra) -------------------- #
DEFAULTS = {
    "NAMESPACE": "indexing",
    "CRONJOB_NAME": "indexing-backup-cronjob",
    "INDEXING_BACKUP_CRON_EXPRESSION": "0 */6 * * *",
    "CRON_SCHEDULE": "0 */6 * * *",
    "CRONJOB_CONCURRENCY": "Allow",
    "CRONJOB_BACKOFF_LIMIT": "1",
    "CRONJOB_PARALLELISM": "1",
    "CRONJOB_COMPLETIONS": "1",
    "CRONJOB_DEBUG_KEEP_POD": "false",
    "CRONJOB_TIMEZONE": "",
    "SUCCESSFUL_JOBS_HISTORY_LIMIT": "3",
    "FAILED_JOBS_HISTORY_LIMIT": "1",
    "SERVICE_ACCOUNT_NAME": "indexer-cron-sa",
    "ROLE_NAME": "indexer-cron-role",
    "ROLEBINDING_NAME": "indexer-cron-rb",
    "INDEXING_PIPELINE_CPU_IMAGE_REPO": "athithya5354/indexing_pipeline_cpu",
    "INDEXING_PIPELINE_CPU_IMAGE_TAG": "v11",
    "INDEXING_BACKUP_CRONJOB_CPU_REQUEST": "2",
    "INDEXING_BACKUP_CRONJOB_CPU_LIMIT": "4",
    "INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST": "1Gi",
    "INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT": "2Gi",
    "AZURE_CHUNKED_PREFIX": "data/chunked/",
    "AZURE_CONTAINER": "rag-data-prod",
    "AZURE_ENDPOINT_SUFFIX": "core.windows.net",
    "OVERWRITE_DOC_DOCX_TO_PDF": "true",
    "OVERWRITE_ALL_AUDIO_FILES": "true",
    "OVERWRITE_SPREADSHEETS_WITH_CSV": "true",
    "OVERWRITE_PPT_WITH_PPTS": "true",
    "MAX_TOKENS_PER_CHUNK": "320",
    "MIN_TOKENS_PER_CHUNK": "100",
    "NUMBER_OF_OVERLAPPING_SENTENCES": "2",
    "PDF_DISABLE_OCR": "false",
    "PDF_OCR_ENGINE": "rapidocr",
    "PDF_TESSERACT_LANG": "eng",
    "IMAGE_TESSERACT_LANG": "eng",
    "TESSERACT_CONFIG": "--oem 1 --psm 6",
    "PDF_FORCE_OCR": "false",
    "PDF_OCR_RENDER_DPI": "400",
    "PDF_MIN_IMG_SIZE_BYTES": "3072",
    "IMAGE_OCR_ENGINE": "rapidocr",
    "IMAGE_MIN_IMG_SIZE_BYTES": "3072",
    "IMAGE_RENDER_DPI": "600",
    "IMAGE_UPSCALE_FACTOR": "2.0",
    "CSV_TARGET_TOKENS_PER_CHUNK": "600",
    "JSONL_TARGET_TOKENS_PER_CHUNK": "600",
    "PPTX_SLIDES_PER_CHUNK": "4",
    "PPTX_OCR_ENGINE": "rapidocr",
    "QDRANT_URL": "http://qdrant.qdrant.svc.cluster.local:6333",
    "DENSE_URL": "http://dense-svc.models.svc.cluster.local:8200",
    "SPARSE_URL": "http://sparse-svc.models.svc.cluster.local:8201",
    "RERANKER_URL": "http://reranker-svc.models.svc.cluster.local:8300",
    "LLM_PROVIDER_URL": "http://llm-svc.models.svc.cluster.local:8500",
    "LOG_LEVEL": "INFO",
    "HTTP_TIMEOUT": "60",
    "BATCH_SIZE": "16",
    "UPSERT_CHUNK": "500",
    "DENSE_DIM": "384",
    "SPARSE_BATCH_FALLBACK": "8",
    "QDRANT_HNSW_EF_CONSTRUCT": "128",
    "QDRANT_HNSW_M": "32",
    "QDRANT_HNSW_FULL_SCAN_THRESHOLD": "10000",
    "QDRANT_ONDISK": "TRUE",
    "PYTHONUNBUFFERED": "1",
    "MANIFESTS_DIR": "infra/manifests/jobs",
}

# Sensitive keys map
SENSITIVE_KEYS = {
    "QDRANT_API_KEY",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "AZURE_SAS_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "EMBEDDER_API_KEY",
    "SENTRY_DSN",
    "DATABASE_URL",
}

NAMED_SECRET_MAP = {
    "QDRANT_API_KEY": "QDRANT",
    "AZURE_STORAGE_ACCOUNT_KEY": "AZURE",
    "AZURE_SAS_TOKEN": "AZURE",
    "AZURE_STORAGE_CONNECTION_STRING": "AZURE",
}

CRON_KEYS_PATTERNS = ("CRON", "CRONJOB", "INDEXING_BACKUP_CRON", "SUCCESSFUL_JOBS_HISTORY_LIMIT", "FAILED_JOBS_HISTORY_LIMIT")

# -------------------- utilities -------------------- #
def run_cmd(cmd: List[str], input_bytes: bytes = None, timeout: int = 120) -> tuple:
    try:
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

def must_env(keys: List[str]):
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        print("ERROR: missing required environment variables: " + ", ".join(missing), file=sys.stderr)
        raise SystemExit(2)

def is_cron_key(k: str) -> bool:
    up = k.upper()
    for p in CRON_KEYS_PATTERNS:
        if p in up:
            return True
    return False

# -------------------- env collection -------------------- #
def collect_runtime_env_map() -> Dict[str,str]:
    out: Dict[str,str] = {}
    # include defaults first
    for k in sorted(DEFAULTS.keys()):
        if is_cron_key(k):
            continue
        v = os.environ.get(k, DEFAULTS.get(k, ""))
        out[k] = "" if v is None else str(v)
    # include all current envs (overrides defaults or adds new)
    for k in sorted(os.environ.keys()):
        if is_cron_key(k):
            continue
        out[k] = str(os.environ.get(k, out.get(k, "")))
    if out.get("INDEXING_BACKUP_CRON_EXPRESSION") and not out.get("CRON_SCHEDULE"):
        out["CRON_SCHEDULE"] = out["INDEXING_BACKUP_CRON_EXPRESSION"]
    return out

def validate_required_runtime_envs(env_map: Dict[str,str]):
    """
    Optional strict validation to ensure critical runtime envs exist.
    Enable by setting REQUIRE_ALL_RUNTIME_ENVS=true in the environment.
    """
    if os.environ.get("REQUIRE_ALL_RUNTIME_ENVS", "").lower() in ("1", "true", "yes"):
        required = ["QDRANT_URL", "DENSE_URL", "SPARSE_URL", "AZURE_STORAGE_ACCOUNT_NAME"]
        missing = [k for k in required if not env_map.get(k) and not os.environ.get(k)]
        if missing:
            print("ERROR: missing required runtime envs:", ", ".join(missing), file=sys.stderr)
            raise SystemExit(2)

# -------------------- manifest builders (plain dicts) -------------------- #
def ns_manifest(ns: str) -> Dict:
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns}}

def serviceaccount_manifest(ns: str, name: str, annotate_use_wi: bool) -> Dict:
    meta = {"name": name, "namespace": ns}
    annotations = {"azure.workload.identity/use": "true"} if annotate_use_wi else None
    if annotations:
        meta["annotations"] = annotations
    return {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": meta}

def role_manifest(ns: str, name: str) -> Dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": name, "namespace": ns},
        "rules": [
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get", "list", "watch", "create", "update", "patch"]},
        ],
    }

def rolebinding_manifest(ns: str, name: str, role_name: str, sa_name: str) -> Dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": name, "namespace": ns},
        "subjects": [{"kind": "ServiceAccount", "name": sa_name, "namespace": ns}],
        "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": role_name},
    }

def secret_manifest(ns: str, name: str, literals: Dict[str,str]) -> Dict:
    return {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": name, "namespace": ns}, "type": "Opaque", "stringData": literals}

def cronjob_manifest(cfg: Dict[str,str], env_map: Dict[str,str]) -> Dict:
    ns = cfg["NAMESPACE"]
    cron_name = cfg["CRONJOB_NAME"]
    image = f"{cfg.get('INDEXING_PIPELINE_CPU_IMAGE_REPO')}:{cfg.get('INDEXING_PIPELINE_CPU_IMAGE_TAG')}"
    sa_name = cfg["SERVICE_ACCOUNT_NAME"]

    env_list = []
    azure_secret_name = cfg.get("AZURE_SECRET_NAME", "indexer-azure-creds")
    qdrant_secret_name = cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key")
    extra_secret_name = cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets")

    use_mi = cfg.get("USE_MANAGED_IDENTITY", "0") == "1"
    if use_mi and cfg.get("UAI_RAG_RW_CLIENT_ID"):
        env_map.setdefault("AZURE_CLIENT_ID", cfg["UAI_RAG_RW_CLIENT_ID"])
    if use_mi and cfg.get("AZURE_TENANT_ID"):
        env_map.setdefault("AZURE_TENANT_ID", cfg["AZURE_TENANT_ID"])

    for k in sorted(env_map.keys()):
        if is_cron_key(k):
            continue
        v = env_map[k]
        if k in SENSITIVE_KEYS:
            if k == "QDRANT_API_KEY" and os.environ.get("QDRANT_API_KEY"):
                env_list.append({"name": k, "valueFrom": {"secretKeyRef": {"name": qdrant_secret_name, "key": "QDRANT_API_KEY"}}})
                continue
            if k in ("AZURE_STORAGE_ACCOUNT_KEY", "AZURE_SAS_TOKEN", "AZURE_STORAGE_CONNECTION_STRING") and os.environ.get(k):
                env_list.append({"name": k, "valueFrom": {"secretKeyRef": {"name": azure_secret_name, "key": k}}})
                continue
            if os.environ.get(k):
                env_list.append({"name": k, "valueFrom": {"secretKeyRef": {"name": extra_secret_name, "key": k}}})
                continue
            env_list.append({"name": k, "value": v or ""})
            continue
        env_list.append({"name": k, "value": v or ""})

    if not any(e.get("name") == "HTTP_TIMEOUT" for e in env_list):
        env_list.append({"name": "HTTP_TIMEOUT", "value": os.environ.get("HTTP_TIMEOUT", DEFAULTS["HTTP_TIMEOUT"])})

    pod_annotations = {}
    if cfg.get("USE_MANAGED_IDENTITY", "0") == "1":
        if cfg.get("UAI_RAG_RW_CLIENT_ID"):
            pod_annotations["azure.workload.identity/client-id"] = cfg["UAI_RAG_RW_CLIENT_ID"]
        if cfg.get("AZURE_TENANT_ID"):
            pod_annotations["azure.workload.identity/tenant-id"] = cfg["AZURE_TENANT_ID"]

    cron = {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": cron_name, "namespace": ns},
        "spec": {
            "schedule": cfg.get("CRON_SCHEDULE", DEFAULTS["INDEXING_BACKUP_CRON_EXPRESSION"]),
            "concurrencyPolicy": cfg.get("CRONJOB_CONCURRENCY", "Allow"),
            "successfulJobsHistoryLimit": int(cfg.get("SUCCESSFUL_JOBS_HISTORY_LIMIT", 3)),
            "failedJobsHistoryLimit": int(cfg.get("FAILED_JOBS_HISTORY_LIMIT", 1)),
            "jobTemplate": {
                "spec": {
                    "template": {
                        "metadata": {"labels": {"app": cron_name}, "annotations": pod_annotations or None},
                        "spec": {
                            "serviceAccountName": sa_name,
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "indexer",
                                    "image": image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["/bin/sh", "-c", "/opt/venv/bin/python /indexing_pipeline/indexing_pipeline.py"],
                                    "env": env_list,
                                    "resources": {
                                        "requests": {
                                            "cpu": cfg.get("INDEXING_BACKUP_CRONJOB_CPU_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"]),
                                            "memory": cfg.get("INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"]),
                                        },
                                        "limits": {
                                            "cpu": cfg.get("INDEXING_BACKUP_CRONJOB_CPU_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"]),
                                            "memory": cfg.get("INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"]),
                                        },
                                    },
                                }
                            ],
                        },
                    },
                    "backoffLimit": int(cfg.get("CRONJOB_BACKOFF_LIMIT", 1)),
                }
            },
        },
    }
    if cfg.get("CRONJOB_TIMEZONE"):
        cron["spec"]["timeZone"] = cfg["CRONJOB_TIMEZONE"]
    return cron

# -------------------- apply/delete helpers (kubectl + yaml) -------------------- #
def kubectl_apply_multi(manifests: List[Dict]) -> None:
    yaml_docs = "\n---\n".join(yaml.safe_dump(m, sort_keys=False, default_flow_style=False, allow_unicode=True) for m in manifests)
    rc, out, err = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=yaml_docs.encode("utf-8"), timeout=60)
    if rc != 0:
        print("ERROR: kubectl apply failed:", err or out, file=sys.stderr)
        raise SystemExit(3)
    print(out)

def kubectl_delete(kind: str, name: str, ns: str, ignore_not_found: bool = True) -> None:
    cmd = ["kubectl", "delete", kind, name, "-n", ns]
    if ignore_not_found:
        cmd += ["--ignore-not-found"]
    rc, out, err = run_cmd(cmd, timeout=30)
    if rc != 0:
        print(f"Warning: kubectl delete {kind}/{name} failed:", err or out, file=sys.stderr)
    else:
        print(out)

# -------------------- main lifecycle: load cfg, validate, apply/delete -------------------- #
def load_cfg() -> Dict[str,str]:
    cfg: Dict[str,str] = {}
    cfg["NAMESPACE"] = os.environ.get("NAMESPACE", DEFAULTS["NAMESPACE"])
    cfg["CRONJOB_NAME"] = os.environ.get("CRONJOB_NAME", DEFAULTS["CRONJOB_NAME"]).lower()
    cfg["CRON_SCHEDULE"] = os.environ.get("INDEXING_BACKUP_CRON_EXPRESSION", os.environ.get("CRON_SCHEDULE", DEFAULTS["INDEXING_BACKUP_CRON_EXPRESSION"]))
    cfg["SERVICE_ACCOUNT_NAME"] = os.environ.get("SERVICE_ACCOUNT_NAME", DEFAULTS["SERVICE_ACCOUNT_NAME"])
    cfg["ROLE_NAME"] = os.environ.get("ROLE_NAME", DEFAULTS["ROLE_NAME"])
    cfg["ROLEBINDING_NAME"] = os.environ.get("ROLEBINDING_NAME", DEFAULTS["ROLEBINDING_NAME"])
    cfg["QDRANT_SECRET_NAME"] = os.environ.get("QDRANT_SECRET_NAME", "qdrant-api-key")
    cfg["AZURE_SECRET_NAME"] = os.environ.get("AZURE_SECRET_NAME", "indexer-azure-creds")
    cfg["EXTRA_SECRET_NAME"] = os.environ.get("EXTRA_SECRET_NAME", "indexer-extra-secrets")
    cfg["INDEXING_PIPELINE_CPU_IMAGE_REPO"] = os.environ.get("INDEXING_PIPELINE_CPU_IMAGE_REPO", DEFAULTS["INDEXING_PIPELINE_CPU_IMAGE_REPO"])
    cfg["INDEXING_PIPELINE_CPU_IMAGE_TAG"] = os.environ.get("INDEXING_PIPELINE_CPU_IMAGE_TAG", DEFAULTS["INDEXING_PIPELINE_CPU_IMAGE_TAG"])
    cfg["CRONJOB_BACKOFF_LIMIT"] = os.environ.get("CRONJOB_BACKOFF_LIMIT", DEFAULTS["CRONJOB_BACKOFF_LIMIT"])
    cfg["CRONJOB_CONCURRENCY"] = os.environ.get("CRONJOB_CONCURRENCY", DEFAULTS["CRONJOB_CONCURRENCY"])
    cfg["SUCCESSFUL_JOBS_HISTORY_LIMIT"] = os.environ.get("SUCCESSFUL_JOBS_HISTORY_LIMIT", DEFAULTS["SUCCESSFUL_JOBS_HISTORY_LIMIT"])
    cfg["FAILED_JOBS_HISTORY_LIMIT"] = os.environ.get("FAILED_JOBS_HISTORY_LIMIT", DEFAULTS["FAILED_JOBS_HISTORY_LIMIT"])
    cfg["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"] = os.environ.get("INDEXING_BACKUP_CRONJOB_CPU_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"])
    cfg["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"] = os.environ.get("INDEXING_BACKUP_CRONJOB_CPU_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"])
    cfg["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"] = os.environ.get("INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"])
    cfg["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"] = os.environ.get("INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"])

    # UAI autodetect
    env = os.environ.get("ENV", "STAGING").upper()
    use_mi_env = os.environ.get("AZURE_USE_MANAGED_IDENTITY", "").strip().lower() in ("1", "true", "yes")
    uai_present = bool(os.environ.get("UAI_RAG_RW_CLIENT_ID"))
    use_mi = True if (env == "PROD") else (use_mi_env or uai_present)
    cfg["USE_MANAGED_IDENTITY"] = "1" if use_mi else "0"

    cfg["UAI_RAG_RW_CLIENT_ID"] = os.environ.get("UAI_RAG_RW_CLIENT_ID", "")
    cfg["AZURE_TENANT_ID"] = os.environ.get("AZURE_TENANT_ID", "")

    env_map = collect_runtime_env_map()
    for k, v in env_map.items():
        if k not in cfg:
            cfg[k] = v
    return cfg

def validate_cfg(cfg: Dict[str,str]):
    if cfg.get("USE_MANAGED_IDENTITY", "0") == "1":
        must_env(["UAI_RAG_RW_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_CONTAINER"])
    else:
        if not (os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or os.environ.get("AZURE_SAS_TOKEN")):
            print("ERROR: non-managed identity mode requires AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_CONNECTION_STRING or AZURE_SAS_TOKEN", file=sys.stderr)
            raise SystemExit(2)

def apply(cfg: Dict[str,str], dry_run: bool = False):
    ns = cfg["NAMESPACE"]
    sa_name = cfg["SERVICE_ACCOUNT_NAME"]
    role_name = cfg["ROLE_NAME"]
    rb_name = cfg["ROLEBINDING_NAME"]

    env_map = {k: v for k, v in cfg.items()}
    validate_required_runtime_envs(env_map)

    manifests = []
    manifests.append(ns_manifest(ns))
    manifests.append(serviceaccount_manifest(ns, sa_name, annotate_use_wi=(cfg.get("USE_MANAGED_IDENTITY", "0") == "1")))
    manifests.append(role_manifest(ns, role_name))
    manifests.append(rolebinding_manifest(ns, rb_name, role_name, sa_name))

    created_secret_names = []
    if os.environ.get("QDRANT_API_KEY"):
        manifests.append(secret_manifest(ns, cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key"), {"QDRANT_API_KEY": os.environ["QDRANT_API_KEY"]}))
        created_secret_names.append(cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key"))
    azure_literals = {}
    if os.environ.get("AZURE_STORAGE_ACCOUNT_NAME"):
        azure_literals["AZURE_STORAGE_ACCOUNT_NAME"] = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    if os.environ.get("AZURE_STORAGE_ACCOUNT_KEY"):
        azure_literals["AZURE_STORAGE_ACCOUNT_KEY"] = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    if os.environ.get("AZURE_SAS_TOKEN"):
        azure_literals["AZURE_SAS_TOKEN"] = os.environ["AZURE_SAS_TOKEN"]
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        azure_literals["AZURE_STORAGE_CONNECTION_STRING"] = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    if azure_literals:
        manifests.append(secret_manifest(ns, cfg.get("AZURE_SECRET_NAME", "indexer-azure-creds"), azure_literals))
        created_secret_names.append(cfg.get("AZURE_SECRET_NAME", "indexer-azure-creds"))
    extras = {k: os.environ[k] for k in sorted(SENSITIVE_KEYS) if k not in ("QDRANT_API_KEY","AZURE_STORAGE_ACCOUNT_KEY","AZURE_SAS_TOKEN","AZURE_STORAGE_CONNECTION_STRING") and os.environ.get(k)}
    if extras:
        manifests.append(secret_manifest(ns, cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets"), extras))
        created_secret_names.append(cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets"))

    cron = cronjob_manifest(cfg, {k: v for k, v in cfg.items()})
    manifests.append(cron)

    if dry_run:
        print("--- DRY RUN: would apply manifests ---")
        for m in manifests:
            print("---")
            print(yaml.safe_dump(m, sort_keys=False, default_flow_style=False, allow_unicode=True))
        return

    try:
        kubectl_apply_multi(manifests)
        print("Applied all manifests.")
    except SystemExit:
        raise
    except Exception as e:
        print("ERROR applying manifests:", e, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(4)

def delete(cfg: Dict[str,str], dry_run: bool = False, delete_secrets: bool = False):
    ns = cfg["NAMESPACE"]
    sa_name = cfg["SERVICE_ACCOUNT_NAME"]
    role_name = cfg["ROLE_NAME"]
    rb_name = cfg["ROLEBINDING_NAME"]
    cron_name = cfg["CRONJOB_NAME"]

    if dry_run:
        print("--- DRY RUN delete ---")
        print("Would delete:", f"CronJob {cron_name}", f"RoleBinding {rb_name}", f"Role {role_name}", f"ServiceAccount {sa_name}")
        if delete_secrets:
            print("Would delete secrets if present")
        return

    kubectl_delete("cronjob", cron_name, ns)
    kubectl_delete("rolebinding", rb_name, ns)
    kubectl_delete("role", role_name, ns)
    kubectl_delete("serviceaccount", sa_name, ns)

    if delete_secrets:
        if os.environ.get("QDRANT_API_KEY"):
            kubectl_delete("secret", cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key"), ns)
        if os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_SAS_TOKEN") or os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            kubectl_delete("secret", cfg.get("AZURE_SECRET_NAME", "indexer-azure-creds"), ns)
        extras_present = [k for k in SENSITIVE_KEYS if k not in ("QDRANT_API_KEY","AZURE_STORAGE_ACCOUNT_KEY","AZURE_SAS_TOKEN","AZURE_STORAGE_CONNECTION_STRING") and os.environ.get(k)]
        if extras_present:
            kubectl_delete("secret", cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets"), ns)

# -------------------- CLI -------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Apply indexing CronJob (UAI-enabled) using kubectl + pyyaml.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="Create/replace CronJob + RBAC + secrets (secrets only if env provided)")
    g.add_argument("--delete", action="store_true", help="Delete CronJob + RBAC; use --delete-secrets to remove created secrets")
    p.add_argument("--dry-run", action="store_true", help="Show actions without making changes")
    p.add_argument("--delete-secrets", action="store_true", help="With --delete remove secrets created by this script")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg()
    cfg["UAI_RAG_RW_CLIENT_ID"] = os.environ.get("UAI_RAG_RW_CLIENT_ID", "")
    cfg["AZURE_TENANT_ID"] = os.environ.get("AZURE_TENANT_ID", "")
    try:
        validate_cfg(cfg)
    except SystemExit:
        raise
    except Exception as e:
        print("ERROR validating config:", e, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2) from e

    try:
        if args.apply:
            apply(cfg, dry_run=args.dry_run)
        elif args.delete:
            delete(cfg, dry_run=args.dry_run, delete_secrets=args.delete_secrets)
    except SystemExit:
        raise
    except Exception as e:
        print("Operation failed:", e, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(4) from e

if __name__ == "__main__":
    main()
