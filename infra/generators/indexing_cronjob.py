#!/usr/bin/env python3
# infra/generators/indexing_cronjob.py
# Generates CronJob + RBAC + supporting manifests for the indexing pipeline.
# The generated CronJob overrides the image entrypoint by setting a safe,
# portable command that works in kind and AKS without rebuilding images.

from __future__ import annotations

import os
import sys
import subprocess
import traceback
import time
import shutil
from typing import Dict, List, Tuple, Any
from pathlib import Path
import yaml

DEFAULTS: Dict[str, str] = {
    "NAMESPACE": "indexing",
    "CRONJOB_NAME": "indexing-backup-cronjob",
    "INDEXING_BACKUP_CRON_EXPRESSION": "0 */6 * * *",
    "CRON_SCHEDULE": "0 */6 * * *",
    "CRONJOB_CONCURRENCY": "Allow",
    "CRONJOB_BACKOFF_LIMIT": "1",
    "CRONJOB_PARALLELISM": "3",
    "CRONJOB_COMPLETIONS": "1",
    "CRONJOB_DEBUG_KEEP_POD": "false",
    "CRONJOB_TIMEZONE": "",
    "SUCCESSFUL_JOBS_HISTORY_LIMIT": "3",
    "FAILED_JOBS_HISTORY_LIMIT": "1",
    "SERVICE_ACCOUNT_NAME": "indexer-cron-sa",
    "ROLE_NAME": "indexer-cron-role",
    "ROLEBINDING_NAME": "indexer-cron-rb",
    "INDEXING_PIPELINE_CPU_IMAGE_REPO": "athithya5354/indexing_pipeline_cpu",
    "INDEXING_PIPELINE_CPU_IMAGE_TAG": "v12",
    "INDEXING_BACKUP_CRONJOB_CPU_REQUEST": "2",
    "INDEXING_BACKUP_CRONJOB_CPU_LIMIT": "4",
    "INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST": "1Gi",
    "INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT": "2Gi",
    "AZURE_CHUNKED_PREFIX": "data/chunked/",
    "AZURE_CONTAINER": "rag-data-prod",
    "AZURE_ENDPOINT_SUFFIX": "core.windows.net",
    "LOG_LEVEL": "INFO",
    "HTTP_TIMEOUT": "60",
    "QDRANT_URL": "http://qdrant.qdrant.svc.cluster.local:6333",
    "DENSE_URL": "http://dense-svc.models.svc.cluster.local:8200",
    "SPARSE_URL": "http://sparse-svc.models.svc.cluster.local:8201",
    "PYTHONUNBUFFERED": "1",
    "MANIFESTS_DIR": "infra/manifests/jobs",
}

SENSITIVE_KEYS = {
    "QDRANT_API_KEY",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "AZURE_SAS_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
}

NAMED_SECRET_MAP = {
    "QDRANT_API_KEY": "qdrant-api-key",
    "AZURE_STORAGE_ACCOUNT_KEY": "indexer-azure-creds",
    "AZURE_SAS_TOKEN": "indexer-azure-creds",
    "AZURE_STORAGE_CONNECTION_STRING": "indexer-azure-creds",
}

RUNTIME_KEYS = set(DEFAULTS.keys()).union(
    {
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_SAS_TOKEN",
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP_NAME",
        "AZURE_LOCATION",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CONTAINER",
        "AZURE_CHUNKED_PREFIX",
        "AZURE_ENDPOINT_SUFFIX",
        "AZURE_STORAGE_API_VERSION",
        "UAI_RAG_RW_CLIENT_ID",
        "UAI_RAG_RO_CLIENT_ID",
        "UAI_RAG_RW_PRINCIPAL_ID",
        "UAI_RAG_RO_PRINCIPAL_ID",
        "UAI_RAG_RW_NAME",
        "UAI_RAG_RO_NAME",
        "QDRANT_API_KEY",
        "QDRANT_SECRET_NAME",
        "BATCH_SIZE",
        "MAX_TOKENS_PER_CHUNK",
        "MIN_TOKENS_PER_CHUNK",
        "CSV_TARGET_TOKENS_PER_CHUNK",
        "JSONL_TARGET_TOKENS_PER_CHUNK",
        "UPSERT_CHUNK",
        "DENSE_DIM",
        "SPARSE_BATCH_FALLBACK",
        "OVERWRITE_DOC_DOCX_TO_PDF",
        "OVERWRITE_ALL_AUDIO_FILES",
        "OVERWRITE_SPREADSHEETS_WITH_CSV",
        "OVERWRITE_PPT_WITH_PPTS",
        "USE_MANAGED_IDENTITY",
        "ENV",
    }
)


def run_cmd(cmd: List[str],
            input_bytes: bytes | None = None,
            timeout: int = 120) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr",
                                                             "") or f"timeout after {timeout}s"


def ensure_kubectl_available():
    rc, out, err = run_cmd(["kubectl", "version", "--client=true"])
    if rc != 0:
        print("ERROR: kubectl not available or not in PATH. details:",
              err or out,
              file=sys.stderr)
        raise SystemExit(2)


def is_cron_key(k: str) -> bool:
    up = k.upper()
    for p in ("CRON", "CRONJOB", "INDEXING_BACKUP_CRON",
              "SUCCESSFUL_JOBS_HISTORY_LIMIT",
              "FAILED_JOBS_HISTORY_LIMIT"):
        if p in up:
            return True
    return False


def collect_runtime_env_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    keys = sorted(RUNTIME_KEYS.union(DEFAULTS.keys()))
    for k in keys:
        if is_cron_key(k):
            continue
        v = os.environ.get(k)
        if v is None:
            v = DEFAULTS.get(k, "")
        out[k] = "" if v is None else str(v)
    if out.get("INDEXING_BACKUP_CRON_EXPRESSION") and not out.get(
            "CRON_SCHEDULE"):
        out["CRON_SCHEDULE"] = out["INDEXING_BACKUP_CRON_EXPRESSION"]
    return out


def ns_manifest(ns: str) -> Dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Namespace",
            "metadata": {"name": ns}}


def serviceaccount_manifest(ns: str, name: str,
                            annotate_use_wi: bool) -> Dict[str, Any]:
    meta = {"name": name, "namespace": ns}
    if annotate_use_wi:
        meta.setdefault("annotations", {})[
            "azure.workload.identity/use"] = "true"
    return {"apiVersion": "v1", "kind": "ServiceAccount",
            "metadata": meta}


def role_manifest(ns: str, name: str) -> Dict[str, Any]:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": name, "namespace": ns},
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["secrets"],
                "verbs": ["get", "list", "watch"],
            },
            {
                "apiGroups": [""],
                "resources": ["configmaps"],
                "verbs": ["get", "list", "watch", "create", "update", "patch"],
            },
        ],
    }


def rolebinding_manifest(ns: str, name: str, role_name: str,
                         sa_name: str) -> Dict[str, Any]:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": name, "namespace": ns},
        "subjects": [{
            "kind": "ServiceAccount",
            "name": sa_name,
            "namespace": ns,
        }],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": role_name,
        },
    }


def cronjob_manifest(cfg: Dict[str, str],
                     env_map: Dict[str, str]) -> Dict[str, Any]:
    ns = cfg["NAMESPACE"]
    cron_name = cfg["CRONJOB_NAME"]
    image = (
        f"{cfg.get('INDEXING_PIPELINE_CPU_IMAGE_REPO')}:"
        f"{cfg.get('INDEXING_PIPELINE_CPU_IMAGE_TAG')}"
    )
    sa_name = cfg["SERVICE_ACCOUNT_NAME"]
    azure_secret = cfg.get("AZURE_SECRET_NAME",
                           NAMED_SECRET_MAP.get("AZURE_STORAGE_ACCOUNT_KEY",
                                                "indexer-azure-creds"))
    qdrant_secret = cfg.get("QDRANT_SECRET_NAME",
                            NAMED_SECRET_MAP.get("QDRANT_API_KEY",
                                                 "qdrant-api-key"))
    extra_secret = cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets")

    env_list: List[Dict[str, Any]] = []
    use_mi = cfg.get("USE_MANAGED_IDENTITY", "0") in ("1", "true", "yes")

    if use_mi and cfg.get("UAI_RAG_RW_CLIENT_ID"):
        env_map.setdefault("AZURE_CLIENT_ID", cfg["UAI_RAG_RW_CLIENT_ID"])
    if use_mi and cfg.get("AZURE_TENANT_ID"):
        env_map.setdefault("AZURE_TENANT_ID", cfg["AZURE_TENANT_ID"])

    for k in sorted(env_map.keys()):
        if is_cron_key(k):
            continue
        v = env_map[k] or ""
        if k in SENSITIVE_KEYS:
            if os.environ.get(k):
                if k == "QDRANT_API_KEY":
                    env_list.append({
                        "name": k,
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": qdrant_secret,
                                "key": "QDRANT_API_KEY",
                            }
                        }
                    })
                elif k in ("AZURE_STORAGE_ACCOUNT_KEY", "AZURE_SAS_TOKEN",
                           "AZURE_STORAGE_CONNECTION_STRING"):
                    env_list.append({
                        "name": k,
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": azure_secret,
                                "key": k,
                            }
                        }
                    })
                else:
                    env_list.append({
                        "name": k,
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": extra_secret,
                                "key": k,
                            }
                        }
                    })
                continue
            env_list.append({"name": k, "value": v})
            continue
        env_list.append({"name": k, "value": v})

    if not any(e.get("name") == "HTTP_TIMEOUT" for e in env_list):
        env_list.append({
            "name": "HTTP_TIMEOUT",
            "value": cfg.get("HTTP_TIMEOUT", DEFAULTS["HTTP_TIMEOUT"]),
        })

    pod_annotations: Dict[str, str] = {}
    if use_mi:
        if cfg.get("UAI_RAG_RW_CLIENT_ID"):
            pod_annotations["azure.workload.identity/client-id"] = cfg[
                "UAI_RAG_RW_CLIENT_ID"]
        if cfg.get("AZURE_TENANT_ID"):
            pod_annotations["azure.workload.identity/tenant-id"] = cfg[
                "AZURE_TENANT_ID"]

    # New: override container command/args to skip image ENTRYPOINT
    # Use a short, robust shell wrapper that:
    #  - tries ulimit -n but never references $SHELL
    #  - execs the Python interpreter inside the venv directly
    wrapper_lines = [
        "set -e",
        "DESIRED=\"${DESIRED_NOFILE:-262144}\"",
        "ulimit -n \"$DESIRED\" 2>/dev/null || true",
        "echo \"nofile limit: $(ulimit -n 2>/dev/null || echo unknown)\"",
        "exec /opt/venv/bin/python indexing_pipeline.py",
    ]
    wrapper_script = "\n".join(wrapper_lines)

    container_spec: Dict[str, Any] = {
        "name": "indexer",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        # override ENTRYPOINT: use portable /bin/sh -c wrapper
        "command": ["/bin/sh", "-c"],
        "args": [wrapper_script],
        "env": env_list,
        "resources": {
            "requests": {
                "cpu": cfg.get("INDEXING_BACKUP_CRONJOB_CPU_REQUEST",
                               DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"]),
                "memory": cfg.get("INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST",
                                  DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"]),
            },
            "limits": {
                "cpu": cfg.get("INDEXING_BACKUP_CRONJOB_CPU_LIMIT",
                               DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"]),
                "memory": cfg.get("INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT",
                                  DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"]),
            },
        },
    }

    job_spec: Dict[str, Any] = {
        "backoffLimit": int(cfg.get("CRONJOB_BACKOFF_LIMIT",
                                    DEFAULTS["CRONJOB_BACKOFF_LIMIT"])),
        "parallelism": int(cfg.get("CRONJOB_PARALLELISM",
                                   DEFAULTS["CRONJOB_PARALLELISM"])),
        "completions": int(cfg.get("CRONJOB_COMPLETIONS",
                                   DEFAULTS["CRONJOB_COMPLETIONS"])),
        "template": {
            "metadata": {
                "labels": {"app": cron_name},
                **(
                    {"annotations": pod_annotations}
                    if pod_annotations else {}
                ),
            },
            "spec": {
                "serviceAccountName": sa_name,
                "restartPolicy": "Never",
                "containers": [container_spec],
            },
        },
    }

    cron: Dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": cron_name, "namespace": ns},
        "spec": {
            "schedule": cfg.get("CRON_SCHEDULE",
                                DEFAULTS["INDEXING_BACKUP_CRON_EXPRESSION"]),
            "concurrencyPolicy": cfg.get("CRONJOB_CONCURRENCY",
                                        DEFAULTS["CRONJOB_CONCURRENCY"]),
            "successfulJobsHistoryLimit": int(
                cfg.get("SUCCESSFUL_JOBS_HISTORY_LIMIT",
                        DEFAULTS["SUCCESSFUL_JOBS_HISTORY_LIMIT"])),
            "failedJobsHistoryLimit": int(
                cfg.get("FAILED_JOBS_HISTORY_LIMIT",
                        DEFAULTS["FAILED_JOBS_HISTORY_LIMIT"])),
            "jobTemplate": {
                "spec": job_spec
            },
        },
    }

    if cfg.get("CRONJOB_TIMEZONE"):
        cron["spec"]["timeZone"] = cfg["CRONJOB_TIMEZONE"]

    return cron


def kubectl_create_secret_inline(name: str, namespace: str,
                                 literals: Dict[str, str]) -> Tuple[bool, str]:
    if not literals:
        return False, "no-literals"
    cmd = [
        "kubectl", "create", "secret", "generic", name, "-n", namespace,
        "--dry-run=client", "-o", "yaml"
    ]
    for k, v in literals.items():
        if not k:
            continue
        cmd += ["--from-literal", f"{k}={v}"]
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        return False, err or out
    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"],
                              input_bytes=(out.encode("utf-8")),
                              timeout=20)
    if rc2 != 0:
        return False, err2 or out2
    return True, ""


def validate_azure_connstr(connstr: str) -> bool:
    if not connstr or not isinstance(connstr, str):
        return False
    s = connstr.strip()
    if "\n" in s or "\r" in s:
        return False
    if "AccountName=" not in s:
        return False
    if ("AccountKey=" in s) or ("SharedAccessSignature=" in s) or ("sv=" in s):
        return True
    return False


def load_cfg() -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    cfg["NAMESPACE"] = os.environ.get("NAMESPACE", DEFAULTS["NAMESPACE"])
    cfg["CRONJOB_NAME"] = os.environ.get("CRONJOB_NAME",
                                        DEFAULTS["CRONJOB_NAME"]).lower()
    cfg["CRON_SCHEDULE"] = os.environ.get(
        "INDEXING_BACKUP_CRON_EXPRESSION",
        os.environ.get("CRON_SCHEDULE",
                       DEFAULTS["INDEXING_BACKUP_CRON_EXPRESSION"]))
    cfg["SERVICE_ACCOUNT_NAME"] = os.environ.get("SERVICE_ACCOUNT_NAME",
                                                DEFAULTS["SERVICE_ACCOUNT_NAME"])
    cfg["ROLE_NAME"] = os.environ.get("ROLE_NAME", DEFAULTS["ROLE_NAME"])
    cfg["ROLEBINDING_NAME"] = os.environ.get("ROLEBINDING_NAME",
                                            DEFAULTS["ROLEBINDING_NAME"])
    cfg["QDRANT_SECRET_NAME"] = os.environ.get(
        "QDRANT_SECRET_NAME",
        NAMED_SECRET_MAP.get("QDRANT_API_KEY", "qdrant-api-key"))
    cfg["AZURE_SECRET_NAME"] = os.environ.get(
        "AZURE_SECRET_NAME",
        NAMED_SECRET_MAP.get("AZURE_STORAGE_ACCOUNT_KEY",
                             "indexer-azure-creds"))
    cfg["EXTRA_SECRET_NAME"] = os.environ.get("EXTRA_SECRET_NAME",
                                             "indexer-extra-secrets")
    cfg["INDEXING_PIPELINE_CPU_IMAGE_REPO"] = os.environ.get(
        "INDEXING_PIPELINE_CPU_IMAGE_REPO",
        DEFAULTS["INDEXING_PIPELINE_CPU_IMAGE_REPO"])
    cfg["INDEXING_PIPELINE_CPU_IMAGE_TAG"] = os.environ.get(
        "INDEXING_PIPELINE_CPU_IMAGE_TAG",
        DEFAULTS["INDEXING_PIPELINE_CPU_IMAGE_TAG"])
    cfg["CRONJOB_BACKOFF_LIMIT"] = os.environ.get(
        "CRONJOB_BACKOFF_LIMIT", DEFAULTS["CRONJOB_BACKOFF_LIMIT"])
    cfg["CRONJOB_CONCURRENCY"] = os.environ.get(
        "CRONJOB_CONCURRENCY", DEFAULTS["CRONJOB_CONCURRENCY"])
    cfg["SUCCESSFUL_JOBS_HISTORY_LIMIT"] = os.environ.get(
        "SUCCESSFUL_JOBS_HISTORY_LIMIT",
        DEFAULTS["SUCCESSFUL_JOBS_HISTORY_LIMIT"])
    cfg["FAILED_JOBS_HISTORY_LIMIT"] = os.environ.get(
        "FAILED_JOBS_HISTORY_LIMIT", DEFAULTS["FAILED_JOBS_HISTORY_LIMIT"])
    cfg["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"] = os.environ.get(
        "INDEXING_BACKUP_CRONJOB_CPU_REQUEST",
        DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"])
    cfg["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"] = os.environ.get(
        "INDEXING_BACKUP_CRONJOB_CPU_LIMIT",
        DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"])
    cfg["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"] = os.environ.get(
        "INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST",
        DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"])
    cfg["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"] = os.environ.get(
        "INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT",
        DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"])
    cfg["MANIFESTS_DIR"] = os.environ.get("MANIFESTS_DIR",
                                         DEFAULTS["MANIFESTS_DIR"])
    use_mi_env = os.environ.get("AZURE_USE_MANAGED_IDENTITY",
                                os.environ.get("USE_MANAGED_IDENTITY",
                                               "")).strip().lower() in (
                                                   "1", "true", "yes")
    cfg["USE_MANAGED_IDENTITY"] = "1" if use_mi_env else "0"
    cfg["UAI_RAG_RW_CLIENT_ID"] = os.environ.get("UAI_RAG_RW_CLIENT_ID", "")
    cfg["AZURE_TENANT_ID"] = os.environ.get("AZURE_TENANT_ID", "")
    env_map = collect_runtime_env_map()
    for k, v in env_map.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def validate_cfg(cfg: Dict[str, str]):
    if cfg.get("USE_MANAGED_IDENTITY", "0") in ("1", "true", "yes"):
        required: List[str] = []
        if not cfg.get("UAI_RAG_RW_CLIENT_ID"):
            required.append("UAI_RAG_RW_CLIENT_ID")
        if not cfg.get("AZURE_TENANT_ID"):
            required.append("AZURE_TENANT_ID")
        if not cfg.get("AZURE_STORAGE_ACCOUNT_NAME"):
            required.append("AZURE_STORAGE_ACCOUNT_NAME")
        if required:
            print("ERROR: When USE_MANAGED_IDENTITY enabled, the following "
                  "envs are required:", ", ".join(required), file=sys.stderr)
            raise SystemExit(2)
    else:
        if not (
            os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            or os.environ.get("AZURE_SAS_TOKEN")
            or os.environ.get("QDRANT_API_KEY")
        ):
            print(
                "ERROR: non-managed identity mode requires "
                "AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_CONNECTION_STRING "
                "or AZURE_SAS_TOKEN or QDRANT_API_KEY",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            cs = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
            if not validate_azure_connstr(cs):
                print(
                    "ERROR: AZURE_STORAGE_CONNECTION_STRING looks malformed.",
                    file=sys.stderr,
                )
                raise SystemExit(2)

    for k in ("CRONJOB_PARALLELISM", "CRONJOB_COMPLETIONS"):
        v = cfg.get(k, DEFAULTS.get(k, ""))
        if v is None or v == "":
            continue
        try:
            ival = int(str(v))
            if ival < 1:
                print(f"ERROR: {k} must be a positive integer",
                      file=sys.stderr)
                raise SystemExit(2)
        except Exception:
            print(f"ERROR: {k} must be an integer", file=sys.stderr)
            raise SystemExit(2)


def write_manifest_file(manifests_dir: Path, filename: str,
                        manifest: Dict[str, Any]) -> Path:
    path = manifests_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest,
                       fh,
                       sort_keys=False,
                       default_flow_style=False,
                       allow_unicode=True)
    return path


def recreate_manifests_dir(manifests_dir: Path):
    if manifests_dir.exists():
        shutil.rmtree(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)


def apply(cfg: Dict[str, str], dry_run: bool = False):
    ensure_kubectl_available()
    ns = cfg["NAMESPACE"]
    sa_name = cfg["SERVICE_ACCOUNT_NAME"]
    role_name = cfg["ROLE_NAME"]
    rb_name = cfg["ROLEBINDING_NAME"]
    manifests_dir = Path(cfg.get("MANIFESTS_DIR", DEFAULTS["MANIFESTS_DIR"]))
    env_map = {k: v for k, v in cfg.items()}

    if os.environ.get("REQUIRE_ALL_RUNTIME_ENVS", "").lower() in (
            "1", "true", "yes"):
        required_runtime = [
            "QDRANT_URL", "DENSE_URL", "SPARSE_URL",
            "AZURE_STORAGE_ACCOUNT_NAME"
        ]
        missing = [k for k in required_runtime
                   if not env_map.get(k) and not os.environ.get(k)]
        if missing:
            print("ERROR: missing required runtime envs:", ", ".join(missing),
                  file=sys.stderr)
            raise SystemExit(2)

    manifests: List[Tuple[str, Dict[str, Any]]] = []
    manifests.append(("00-namespace.yaml", ns_manifest(ns)))
    manifests.append(
        ("10-serviceaccount.yaml",
         serviceaccount_manifest(ns, sa_name,
                                 annotate_use_wi=(cfg.get("USE_MANAGED_IDENTITY",
                                                          "0") == "1"))))
    manifests.append(("20-role.yaml", role_manifest(ns, role_name)))
    manifests.append(("30-rolebinding.yaml",
                      rolebinding_manifest(ns, rb_name, role_name, sa_name)))

    if os.environ.get("QDRANT_API_KEY"):
        qname = cfg.get("QDRANT_SECRET_NAME",
                        NAMED_SECRET_MAP.get("QDRANT_API_KEY",
                                             "qdrant-api-key"))
        manifests.append(
            ("41-secret-qdrant-placeholder.yaml",
             {
                 "apiVersion": "v1",
                 "kind": "Secret",
                 "metadata": {"name": qname, "namespace": ns},
                 "type": "Opaque",
                 "stringData": {"QDRANT_API_KEY": "REPLACE_WITH_REAL_KEY"},
             }))

    azure_placeholders: Dict[str, str] = {}
    if os.environ.get("AZURE_STORAGE_ACCOUNT_NAME"):
        azure_placeholders["AZURE_STORAGE_ACCOUNT_NAME"] = os.environ[
            "AZURE_STORAGE_ACCOUNT_NAME"]
    if os.environ.get("AZURE_STORAGE_ACCOUNT_KEY"):
        azure_placeholders["AZURE_STORAGE_ACCOUNT_KEY"] = "REPLACE_WITH_REAL_VALUE"
    if os.environ.get("AZURE_SAS_TOKEN"):
        azure_placeholders["AZURE_SAS_TOKEN"] = "REPLACE_WITH_REAL_VALUE"
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        azure_placeholders["AZURE_STORAGE_CONNECTION_STRING"] = (
            "REPLACE_WITH_REAL_VALUE"
        )
    if azure_placeholders:
        aname = cfg.get("AZURE_SECRET_NAME",
                        NAMED_SECRET_MAP.get("AZURE_STORAGE_ACCOUNT_KEY",
                                             "indexer-azure-creds"))
        manifests.append(
            ("40-secret-azure-placeholder.yaml",
             {
                 "apiVersion": "v1",
                 "kind": "Secret",
                 "metadata": {"name": aname, "namespace": ns},
                 "type": "Opaque",
                 "stringData": azure_placeholders,
             }))

    extras: Dict[str, str] = {}
    for k in sorted(SENSITIVE_KEYS):
        if k in ("QDRANT_API_KEY", "AZURE_STORAGE_ACCOUNT_KEY",
                 "AZURE_SAS_TOKEN", "AZURE_STORAGE_CONNECTION_STRING"):
            continue
        if os.environ.get(k):
            extras[k] = "REPLACE_WITH_REAL_VALUE"
    if extras:
        ename = cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets")
        manifests.append(
            ("42-secret-extra-placeholder.yaml",
             {
                 "apiVersion": "v1",
                 "kind": "Secret",
                 "metadata": {"name": ename, "namespace": ns},
                 "type": "Opaque",
                 "stringData": extras,
             }))

    cron = cronjob_manifest(cfg, {k: v for k, v in cfg.items()})
    manifests.append(("50-cronjob.yaml", cron))

    recreate_manifests_dir(Path(manifests_dir))
    written_files: List[Path] = []
    for fname, m in manifests:
        p = write_manifest_file(manifests_dir, fname, m)
        written_files.append(p)

    if dry_run:
        print("--- DRY RUN: wrote placeholders to", str(manifests_dir))
        for p in written_files:
            print(p)
        return

    ns_file = manifests_dir / "00-namespace.yaml"
    if ns_file.exists():
        rc, out, err = run_cmd(["kubectl", "apply", "-f", str(ns_file)], timeout=20)
    else:
        ns_yaml = yaml.safe_dump(ns_manifest(ns), sort_keys=False)
        rc, out, err = run_cmd(["kubectl", "apply", "-f", "-"],
                               input_bytes=ns_yaml.encode("utf-8"),
                               timeout=20)
    if rc != 0:
        print("ERROR: applying namespace failed:", err or out, file=sys.stderr)
        raise SystemExit(4)

    waited = 0
    max_wait = 30
    while True:
        rc2, out2, err2 = run_cmd(["kubectl", "get", "namespace", ns])
        if rc2 == 0:
            break
        time.sleep(1)
        waited += 1
        if waited >= max_wait:
            print(
                f"ERROR: namespace '{ns}' not ready after {max_wait}s. "
                f"kubectl get ns returned: {err2 or out2}",
                file=sys.stderr,
            )
            raise SystemExit(5)

    created_secret_names: List[str] = []
    if os.environ.get("QDRANT_API_KEY"):
        qname = cfg.get("QDRANT_SECRET_NAME",
                        NAMED_SECRET_MAP.get("QDRANT_API_KEY",
                                             "qdrant-api-key"))
        ok, err = kubectl_create_secret_inline(
            qname, ns, {"QDRANT_API_KEY": os.environ["QDRANT_API_KEY"]})
        if not ok:
            print("ERROR creating qdrant secret:", err, file=sys.stderr)
            raise SystemExit(3)
        created_secret_names.append(qname)

    azure_literals_live: Dict[str, str] = {}
    if os.environ.get("AZURE_STORAGE_ACCOUNT_NAME"):
        azure_literals_live["AZURE_STORAGE_ACCOUNT_NAME"] = os.environ[
            "AZURE_STORAGE_ACCOUNT_NAME"]
    if os.environ.get("AZURE_STORAGE_ACCOUNT_KEY"):
        azure_literals_live["AZURE_STORAGE_ACCOUNT_KEY"] = os.environ[
            "AZURE_STORAGE_ACCOUNT_KEY"]
    if os.environ.get("AZURE_SAS_TOKEN"):
        azure_literals_live["AZURE_SAS_TOKEN"] = os.environ["AZURE_SAS_TOKEN"]
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        cs = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        if not validate_azure_connstr(cs):
            print(
                "ERROR: AZURE_STORAGE_CONNECTION_STRING appears malformed; "
                "aborting to avoid runtime crash.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        azure_literals_live["AZURE_STORAGE_CONNECTION_STRING"] = cs
    if azure_literals_live:
        aname = cfg.get("AZURE_SECRET_NAME",
                        NAMED_SECRET_MAP.get("AZURE_STORAGE_ACCOUNT_KEY",
                                             "indexer-azure-creds"))
        ok, err = kubectl_create_secret_inline(aname, ns, azure_literals_live)
        if not ok:
            print("ERROR creating azure secret:", err, file=sys.stderr)
            raise SystemExit(3)
        created_secret_names.append(aname)

    extras_live: Dict[str, str] = {}
    for k in sorted(SENSITIVE_KEYS):
        if k in ("QDRANT_API_KEY", "AZURE_STORAGE_ACCOUNT_KEY", "AZURE_SAS_TOKEN",
                 "AZURE_STORAGE_CONNECTION_STRING"):
            continue
        if os.environ.get(k):
            extras_live[k] = os.environ[k]
    if extras_live:
        ename = cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets")
        ok, err = kubectl_create_secret_inline(ename, ns, extras_live)
        if not ok:
            print("ERROR creating extra secret:", err, file=sys.stderr)
            raise SystemExit(3)
        created_secret_names.append(ename)

    to_apply_docs = []
    for fname, _ in manifests:
        if fname == "00-namespace.yaml":
            continue
        p = manifests_dir / fname
        if not p.exists():
            continue
        to_apply_docs.append(p.read_text(encoding="utf-8"))
    if to_apply_docs:
        yaml_docs = "\n---\n".join(to_apply_docs)
        rc3, out3, err3 = run_cmd(["kubectl", "apply", "-f", "-"],
                                  input_bytes=yaml_docs.encode("utf-8"),
                                  timeout=60)
        if rc3 != 0:
            print("ERROR: kubectl apply failed for manifests:", err3 or out3,
                  file=sys.stderr)
            raise SystemExit(4)
        print(out3)

    if created_secret_names:
        print("Created/updated secrets in-cluster (names):",
              ", ".join(created_secret_names))
    print("Wrote non-secret manifest files to:", str(manifests_dir))


def delete(cfg: Dict[str, str], dry_run: bool = False,
           delete_secrets: bool = False):
    ensure_kubectl_available()
    ns = cfg["NAMESPACE"]
    sa_name = cfg["SERVICE_ACCOUNT_NAME"]
    role_name = cfg["ROLE_NAME"]
    rb_name = cfg["ROLEBINDING_NAME"]
    cron_name = cfg["CRONJOB_NAME"]
    manifests_dir = Path(cfg.get("MANIFESTS_DIR", DEFAULTS["MANIFESTS_DIR"]))
    if dry_run:
        print("--- DRY RUN delete ---")
        print("Would delete:", f"CronJob {cron_name}",
              f"RoleBinding {rb_name}", f"Role {role_name}",
              f"ServiceAccount {sa_name}")
        if delete_secrets:
            print("Would delete secrets if present")
        if manifests_dir.exists():
            print("Would remove manifests dir:", manifests_dir)
        return

    run_cmd(["kubectl", "delete", "cronjob", cron_name, "-n", ns,
             "--ignore-not-found"],
            timeout=30)
    run_cmd(["kubectl", "delete", "rolebinding", rb_name, "-n", ns,
             "--ignore-not-found"],
            timeout=30)
    run_cmd(["kubectl", "delete", "role", role_name, "-n", ns,
             "--ignore-not-found"],
            timeout=30)
    run_cmd(["kubectl", "delete", "serviceaccount", sa_name, "-n", ns,
             "--ignore-not-found"],
            timeout=30)

    if delete_secrets:
        if os.environ.get("QDRANT_API_KEY"):
            run_cmd([
                "kubectl", "delete", "secret",
                cfg.get("QDRANT_SECRET_NAME",
                        NAMED_SECRET_MAP.get("QDRANT_API_KEY",
                                             "qdrant-api-key")), "-n", ns,
                "--ignore-not-found"
            ])
        if (os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
                or os.environ.get("AZURE_SAS_TOKEN")
                or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")):
            run_cmd([
                "kubectl", "delete", "secret",
                cfg.get("AZURE_SECRET_NAME",
                        NAMED_SECRET_MAP.get("AZURE_STORAGE_ACCOUNT_KEY",
                                             "indexer-azure-creds")), "-n", ns,
                "--ignore-not-found"
            ])
        extras_present = [
            k for k in SENSITIVE_KEYS
            if k not in ("QDRANT_API_KEY", "AZURE_STORAGE_ACCOUNT_KEY",
                         "AZURE_SAS_TOKEN", "AZURE_STORAGE_CONNECTION_STRING")
            and os.environ.get(k)
        ]
        if extras_present:
            run_cmd([
                "kubectl", "delete", "secret",
                cfg.get("EXTRA_SECRET_NAME", "indexer-extra-secrets"), "-n", ns,
                "--ignore-not-found"
            ])

    manifests_dir = Path(cfg.get("MANIFESTS_DIR", DEFAULTS["MANIFESTS_DIR"]))
    if manifests_dir.exists():
        shutil.rmtree(manifests_dir)
    print("Delete completed.")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Apply indexing CronJob (non-UAI by default).")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delete-secrets", action="store_true")
    p.add_argument("--print-envs", action="store_true")
    return p.parse_args()


def print_env_examples():
    non_uai = [
        "# Non-UAI (default) — provide a valid connection string",
        "export AZURE_USE_MANAGED_IDENTITY=0",
        "export AZURE_STORAGE_CONNECTION_STRING='DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;"
        "AccountName=storeragprod42;AccountKey=...;BlobEndpoint=https://storeragprod42.blob.core.windows.net/;'",
        "export AZURE_CONTAINER=rag-data-prod",
        "export QDRANT_API_KEY='REDACTED_IF_USED'",
    ]
    uai = [
        "# UAI / Managed Identity (explicit) — must set USE_MANAGED_IDENTITY=1",
        "export AZURE_USE_MANAGED_IDENTITY=1",
        "export USE_MANAGED_IDENTITY=1",
        "export AZURE_STORAGE_ACCOUNT_NAME=storeragprod42",
        "export AZURE_TENANT_ID='YOUR_TENANT_ID'",
        "export UAI_RAG_RW_CLIENT_ID='6a687dad-cd44-4fcf-99b5-b596cd2e3c77'",
        "export AZURE_CONTAINER=rag-data-prod",
        "export QDRANT_API_KEY='REDACTED_IF_USED'",
    ]
    print("\n".join(non_uai + ["# ---"] + uai))


def main():
    args = parse_args()
    if args.print_envs:
        print_env_examples()
        return
    cfg = load_cfg()
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
            delete(cfg, dry_run=args.dry_run,
                   delete_secrets=args.delete_secrets)
        else:
            print("No action specified. Use --apply or --delete or --print-envs",
                  file=sys.stderr)
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        print("Operation failed:", e, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(4) from e

if __name__ == "__main__":
    main()
