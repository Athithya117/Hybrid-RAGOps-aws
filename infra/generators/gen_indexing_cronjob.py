#!/usr/bin/env python3
"""
gen_indexing_cronjob.py

Generate / apply an indexing CronJob + minimal RBAC + sample secrets.

Usage:
  python3 infra/generators/gen_indexing_cronjob.py --generate [--manifests-dir PATH] [--dry-run]
  python3 infra/generators/gen_indexing_cronjob.py --apply [--manifests-dir PATH] [--dry-run]
  python3 infra/generators/gen_indexing_cronjob.py --delete [--manifests-dir PATH] [--delete-secrets]

Defaults are chosen for a development-kind cluster; override via environment variables.
"""


from pathlib import Path
import os
import sys
import argparse
import subprocess
import json
import yaml
from typing import Dict, Tuple, List

# -------------------- Defaults --------------------
DEFAULTS = {
    # default output directory for generated manifests (user requested)
    "MANIFESTS_DIR": "infra/manifests/jobs",
    "NAMESPACE": "indexing",
    "CRONJOB_NAME": "indexing-backup-cronjob",
    "CRON_SCHEDULE": "0 */6 * * *",
    "CRONJOB_CONCURRENCY": "Allow",
    "CRONJOB_BACKOFF_LIMIT": "1",
    "CRONJOB_PARALLELISM": "1",
    "CRONJOB_COMPLETIONS": "1",
    "CRONJOB_DEBUG_KEEP_POD": "false",
    "INDEXING_PIPELINE_CPU_IMAGE_REPO": "athithya5354/indexing_pipeline_cpu",
    "INDEXING_PIPELINE_CPU_IMAGE_TAG": "amd64-arm64-v7",
    "PLATFORMS": "linux/amd64,linux/arm64",
    "INDEXING_BACKUP_CRONJOB_CPU_REQUEST": "2",
    "INDEXING_BACKUP_CRONJOB_CPU_LIMIT": "4",
    "INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST": "1Gi",
    "INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT": "2Gi",
    "SERVICE_ACCOUNT_NAME": "indexer-cron-sa",
    "ROLE_NAME": "indexer-cron-role",
    "ROLEBINDING_NAME": "indexer-cron-rb",
}

# canonical explicit defaults for many pipeline envs you listed
EXPLICIT_DEFAULTS = {
    "LOG_LEVEL": "INFO",
    "COLLECTION_NAME": "default_rag_collection1",
    "DENSE_DIM": "384",
    "BATCH_SIZE": "16",
    "UPSERT_CHUNK": "500",
    "SPARSE_BATCH_FALLBACK": "8",
    "QDRANT_HNSW_EF_CONSTRUCT": "128",
    "QDRANT_HNSW_M": "32",
    "QDRANT_HNSW_FULL_SCAN_THRESHOLD": "10000",
    "QDRANT_ONDISK": "TRUE",
    "INDEXING_BACKUP_CRON_EXPRESSION": DEFAULTS["CRON_SCHEDULE"],
    "CRONJOB_CONCURRENCY": DEFAULTS["CRONJOB_CONCURRENCY"],
    "CRONJOB_BACKOFF_LIMIT": DEFAULTS["CRONJOB_BACKOFF_LIMIT"],
    "CRONJOB_PARALLELISM": DEFAULTS["CRONJOB_PARALLELISM"],
    "CRONJOB_COMPLETIONS": DEFAULTS["CRONJOB_COMPLETIONS"],
    "CRONJOB_DEBUG_KEEP_POD": DEFAULTS["CRONJOB_DEBUG_KEEP_POD"],
    "PLATFORMS": DEFAULTS["PLATFORMS"],
    "AWS_REGION": "ap-south-1",
    "S3_RAW_PREFIX": "data/raw/",
    "S3_CHUNKED_PREFIX": "data/chunked/",
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
    "PYTHONUNBUFFERED": "1",
}

# Keys to export into CronJob env block (all the user-provided vars)
ENV_KEYS = [
    # control & identification
    "LOG_LEVEL", "QDRANT_API_KEY", "COLLECTION_NAME", "DENSE_DIM", "BATCH_SIZE", "UPSERT_CHUNK", "SPARSE_BATCH_FALLBACK",
    "QDRANT_HNSW_EF_CONSTRUCT", "QDRANT_HNSW_M", "QDRANT_HNSW_FULL_SCAN_THRESHOLD", "QDRANT_ONDISK",

    # cron control
    "INDEXING_BACKUP_CRON_EXPRESSION", "CRON_SCHEDULE", "CRONJOB_CONCURRENCY", "CRONJOB_BACKOFF_LIMIT",
    "CRONJOB_PARALLELISM", "CRONJOB_COMPLETIONS", "CRONJOB_DEBUG_KEEP_POD", "CRONJOB_TIMEZONE",

    # image & platforms
    "INDEXING_PIPELINE_CPU_IMAGE_REPO", "INDEXING_PIPELINE_CPU_IMAGE_TAG", "PLATFORMS",

    # AWS / S3
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET",
    "S3_RAW_PREFIX", "S3_CHUNKED_PREFIX",

    # pipeline behavior / conversions
    "OVERWRITE_DOC_DOCX_TO_PDF", "OVERWRITE_ALL_AUDIO_FILES", "OVERWRITE_SPREADSHEETS_WITH_CSV",
    "OVERWRITE_PPT_WITH_PPTS",

    # chunking & tokens
    "MAX_TOKENS_PER_CHUNK", "MIN_TOKENS_PER_CHUNK", "NUMBER_OF_OVERLAPPING_SENTENCES",

    # pdf / image / ocr
    "PDF_DISABLE_OCR", "PDF_OCR_ENGINE", "PDF_TESSERACT_LANG", "IMAGE_TESSERACT_LANG",
    "TESSERACT_CONFIG", "PDF_FORCE_OCR", "PDF_OCR_RENDER_DPI", "PDF_MIN_IMG_SIZE_BYTES",
    "IMAGE_OCR_ENGINE", "IMAGE_MIN_IMG_SIZE_BYTES", "IMAGE_RENDER_DPI", "IMAGE_UPSCALE_FACTOR",

    # csv/jsonl/pptx
    "CSV_TARGET_TOKENS_PER_CHUNK", "JSONL_TARGET_TOKENS_PER_CHUNK", "PPTX_SLIDES_PER_CHUNK", "PPTX_OCR_ENGINE",

    # runtime flags
    "PYTHONUNBUFFERED", "HTTP_TIMEOUT",

    # resource requests/limits (exposed as envs too)
    "INDEXING_BACKUP_CRONJOB_CPU_REQUEST", "INDEXING_BACKUP_CRONJOB_CPU_LIMIT",
    "INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST", "INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT",
]

# -------------------- small helpers --------------------
def getenv(key: str) -> str:
    """Return environment variable with explicit/default fallback or empty string."""
    if key in os.environ:
        return os.environ[key]
    if key in EXPLICIT_DEFAULTS:
        return EXPLICIT_DEFAULTS[key]
    if key in DEFAULTS:
        return DEFAULTS[key]
    return ""

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str], input_bytes: bytes = None, timeout: int = 60) -> Tuple[int, str, str]:
    """
    Run a subprocess and return (rc, stdout, stderr).
    Accepts input_bytes (bytes) or text input (str). Works with text=True.
    """
    try:
        if input_bytes is not None and isinstance(input_bytes, bytes):
            inp = input_bytes.decode("utf-8")
        else:
            inp = input_bytes
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=inp)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"
    except Exception as e:
        return 1, "", str(e)

# -------------------- renderers --------------------
def collect_pipeline_envs() -> Dict[str,str]:
    """
    Collect all ENV_KEYS into a dict. For backward compatibility we map
    INDEXING_BACKUP_CRON_EXPRESSION -> CRON_SCHEDULE as canonical.
    """
    out = {}
    for k in ENV_KEYS:
        v = getenv(k)
        # only include ones with defined values (non-empty)
        if v != "":
            out[k] = str(v)

    # canonical schedule name
    if out.get("INDEXING_BACKUP_CRON_EXPRESSION"):
        out["CRON_SCHEDULE"] = out["INDEXING_BACKUP_CRON_EXPRESSION"]
    if not out.get("CRON_SCHEDULE"):
        out["CRON_SCHEDULE"] = getenv("CRON_SCHEDULE") or DEFAULTS["CRON_SCHEDULE"]

    return out

def render_serviceaccount(cfg: Dict) -> Dict:
    return {"apiVersion": "v1", "kind": "ServiceAccount",
            "metadata": {"name": cfg["SERVICE_ACCOUNT_NAME"], "namespace": cfg["NAMESPACE"]}}

def render_role(cfg: Dict) -> Dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": cfg["ROLE_NAME"], "namespace": cfg["NAMESPACE"]},
        "rules": [
            {"apiGroups": [""], "resources": ["secrets"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get", "list", "watch", "create", "update", "patch"]},
        ],
    }

def render_rolebinding(cfg: Dict) -> Dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": cfg["ROLEBINDING_NAME"], "namespace": cfg["NAMESPACE"]},
        "subjects": [{"kind": "ServiceAccount", "name": cfg["SERVICE_ACCOUNT_NAME"], "namespace": cfg["NAMESPACE"]}],
        "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": cfg["ROLE_NAME"]},
    }

def render_secret_sample(cfg: Dict) -> Dict:
    return {
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {"name": cfg.get("QDRANT_SECRET_NAME","qdrant-api-key"), "namespace": cfg["NAMESPACE"]},
        "type": "Opaque",
        "stringData": {"QDRANT_API_KEY": "REPLACE_WITH_REAL_KEY"},
    }

def render_aws_secret_sample(cfg: Dict) -> Dict:
    return {
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {"name": cfg.get("AWS_SECRET_NAME","indexer-aws-creds"), "namespace": cfg["NAMESPACE"]},
        "type": "Opaque",
        "stringData": {"AWS_ACCESS_KEY_ID": "AKIA...", "AWS_SECRET_ACCESS_KEY": "..."},
    }

def render_cronjob(cfg: Dict, env_map: Dict[str,str]) -> Dict:
    """
    Return a Kubernetes batch/v1 CronJob manifest as a Python dict.
    If CRONJOB_TIMEZONE is provided it will set spec.timeZone (string must be valid IANA timezone).
    Sensitive values (QDRANT_API_KEY, AWS creds) are referenced with secretKeyRef (if present in environment).
    """
    image = f"{cfg['INDEXING_PIPELINE_CPU_IMAGE_REPO']}:{cfg['INDEXING_PIPELINE_CPU_IMAGE_TAG']}"
    secret_name = cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key")
    aws_secret_name = cfg.get("AWS_SECRET_NAME", "indexer-aws-creds")

    env_list = []
    # inject envs from env_map except sensitive ones -> sensitive handled via secretKeyRef
    for k, v in env_map.items():
        if k in ("QDRANT_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            continue
        if k in ("INDEXING_BACKUP_CRON_EXPRESSION", "CRON_SCHEDULE"):
            continue
        env_list.append({"name": k, "value": str(v)})

    # attach secretKeyRef if env present in user environment
    if os.getenv("QDRANT_API_KEY"):
        env_list.append({"name": "QDRANT_API_KEY", "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "QDRANT_API_KEY"}}})
    if os.getenv("AWS_ACCESS_KEY_ID"):
        env_list.append({"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": aws_secret_name, "key": "AWS_ACCESS_KEY_ID"}}})
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        env_list.append({"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": aws_secret_name, "key": "AWS_SECRET_ACCESS_KEY"}}})

    # add HTTP_TIMEOUT default if not set
    if not any(e["name"] == "HTTP_TIMEOUT" for e in env_list):
        env_list.append({"name":"HTTP_TIMEOUT", "value": os.getenv("HTTP_TIMEOUT", "60")})

    base_cmd = "/opt/venv/bin/python /indexing_pipeline/indexing_pipeline.py"
    debug_keep = os.getenv("CRONJOB_DEBUG_KEEP_POD", getenv("CRONJOB_DEBUG_KEEP_POD")).lower() in ("1","true","yes")
    if debug_keep:
        cmd = f"sh -c '{base_cmd}; echo debug-sleep; sleep 3600'"
    else:
        cmd = base_cmd

    # resource requests/limits
    req_cpu = str(cfg.get("INDEXING_BACKUP_CRONJOB_CPU_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"]))
    req_mem = str(cfg.get("INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"]))
    lim_cpu = str(cfg.get("INDEXING_BACKUP_CRONJOB_CPU_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"]))
    lim_mem = str(cfg.get("INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"]))

    cj_spec = {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": cfg["CRONJOB_NAME"], "namespace": cfg["NAMESPACE"]},
        "spec": {
            "schedule": cfg["CRON_SCHEDULE"],
            "concurrencyPolicy": cfg.get("CRONJOB_CONCURRENCY", DEFAULTS["CRONJOB_CONCURRENCY"]),
            "successfulJobsHistoryLimit": int(cfg.get("SUCCESSFUL_JOBS_HISTORY_LIMIT", 3)),
            "failedJobsHistoryLimit": int(cfg.get("FAILED_JOBS_HISTORY_LIMIT", 1)),
            "jobTemplate": {
                "spec": {
                    "parallelism": int(cfg.get("CRONJOB_PARALLELISM", DEFAULTS["CRONJOB_PARALLELISM"])),
                    "completions": int(cfg.get("CRONJOB_COMPLETIONS", DEFAULTS["CRONJOB_COMPLETIONS"])),
                    "backoffLimit": int(cfg.get("CRONJOB_BACKOFF_LIMIT", DEFAULTS["CRONJOB_BACKOFF_LIMIT"])),
                    "template": {
                        "metadata": {"labels": {"app": cfg["CRONJOB_NAME"]}},
                        "spec": {
                            "serviceAccountName": cfg["SERVICE_ACCOUNT_NAME"],
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "indexer",
                                    "image": image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "env": env_list,
                                    "command": ["sh", "-c", cmd],
                                    "resources": {
                                        "requests": {"cpu": req_cpu, "memory": req_mem},
                                        "limits": {"cpu": lim_cpu, "memory": lim_mem},
                                    },
                                }
                            ],
                        }
                    }
                }
            }
        }
    }

    # optional timezone support: set spec.timeZone if provided (string must be IANA tz name)
    tz = os.getenv("CRONJOB_TIMEZONE", "").strip() or os.getenv("CRONJOB_TIMEZONE") or getenv("CRONJOB_TIMEZONE")
    if tz:
        cj_spec["spec"]["timeZone"] = tz

    return cj_spec

# -------------------- kubectl helpers --------------------
def kubectl_apply_yaml(yaml_bytes: bytes) -> Tuple[int,str,str]:
    return run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=yaml_bytes, timeout=60)

def kubectl_delete(kind: str, name: str, namespace: str) -> Tuple[int,str,str]:
    return run_cmd(["kubectl", "delete", kind, name, "-n", namespace, "--ignore-not-found"], timeout=30)

def kubectl_create_secret_inline(name: str, namespace: str, literals: Dict[str,str]) -> Tuple[bool,str]:
    if not literals:
        return False, "no-literals"
    cmd = ["kubectl", "create", "secret", "generic", name, "-n", namespace, "--dry-run=client", "-o", "yaml"]
    for k,v in literals.items():
        cmd += ["--from-literal", f"{k}={v}"]
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        return False, err or out
    rc2, out2, err2 = run_cmd(["kubectl","apply","-f","-"], input_bytes=(out.encode("utf-8")), timeout=20)
    if rc2 != 0:
        return False, err2 or out2
    return True, None

def ensure_namespace(ns: str) -> Tuple[bool,str]:
    rc, out, err = run_cmd(["kubectl", "create", "namespace", ns, "--dry-run=client", "-o", "yaml"], timeout=10)
    if rc != 0:
        rcg, outg, errg = run_cmd(["kubectl","get","namespace", ns], timeout=10)
        return (rcg == 0, errg or outg or err)
    rc2, out2, err2 = run_cmd(["kubectl","apply","-f","-"], input_bytes=(out.encode("utf-8")), timeout=10)
    return (rc2 == 0, err2 if rc2 != 0 else None)

# -------------------- generate / apply / delete --------------------
def generate_manifests(cfg: Dict, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    sa = render_serviceaccount(cfg)
    role = render_role(cfg)
    rb = render_rolebinding(cfg)
    env_map = collect_pipeline_envs()
    cron = render_cronjob(cfg, env_map)
    sample_q = render_secret_sample(cfg)
    sample_aws = render_aws_secret_sample(cfg)

    sa_p = out_dir / "serviceaccount.yaml"
    role_p = out_dir / "role.yaml"
    rb_p = out_dir / "rolebinding.yaml"
    cron_p = out_dir / f"{cfg['CRONJOB_NAME']}.cronjob.yaml"
    samples_dir = out_dir / "_samples"
    ensure_dir(samples_dir)
    sample_q_p = samples_dir / "qdrant-secret-sample.yaml"
    sample_aws_p = samples_dir / "aws-secret-sample.yaml"

    sa_p.write_text(yaml.safe_dump(sa, sort_keys=False))
    role_p.write_text(yaml.safe_dump(role, sort_keys=False))
    rb_p.write_text(yaml.safe_dump(rb, sort_keys=False))
    cron_p.write_text(yaml.safe_dump(cron, sort_keys=False))
    sample_q_p.write_text(yaml.safe_dump(sample_q, sort_keys=False))
    sample_aws_p.write_text(yaml.safe_dump(sample_aws, sort_keys=False))
    print("Wrote manifests to:", out_dir)
    print(" -", sa_p.name, role_p.name, rb_p.name, cron_p.name)
    print(" - sample secrets in _samples/ (do NOT commit real secrets)")
    return [sa_p, role_p, rb_p, cron_p, sample_q_p, sample_aws_p]

def apply_to_cluster(cfg: Dict, out_dir: Path, dry_run: bool=False):
    ok, err = ensure_namespace(cfg["NAMESPACE"])
    if not ok:
        print("ERROR: ensure namespace:", err, file=sys.stderr); sys.exit(2)
    print("Namespace exists:", cfg["NAMESPACE"])

    # create in-cluster secrets if env provided (do not write real secret YAML to disk)
    q_key = os.getenv("QDRANT_API_KEY", "")
    if q_key:
        ok, err = kubectl_create_secret_inline(cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key"), cfg["NAMESPACE"], {"QDRANT_API_KEY": q_key})
        if not ok:
            print("ERROR: failed creating qdrant secret:", err, file=sys.stderr); sys.exit(2)
        print("Created/updated secret:", cfg.get("QDRANT_SECRET_NAME", "qdrant-api-key"))
    else:
        print("QDRANT_API_KEY not set -> skipping secret creation (ensure auth available to CronJob).")

    aws_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    if aws_id and aws_secret:
        ok, err = kubectl_create_secret_inline(cfg.get("AWS_SECRET_NAME", "indexer-aws-creds"), cfg["NAMESPACE"], {"AWS_ACCESS_KEY_ID": aws_id, "AWS_SECRET_ACCESS_KEY": aws_secret})
        if not ok:
            print("ERROR: failed creating aws secret:", err, file=sys.stderr); sys.exit(2)
        print("Created/updated AWS secret:", cfg.get("AWS_SECRET_NAME", "indexer-aws-creds"))
    else:
        print("AWS creds not set or incomplete -> skipping AWS secret creation.")

    # apply SA / Role / RoleBinding
    sa = render_serviceaccount(cfg)
    role = render_role(cfg)
    rb = render_rolebinding(cfg)
    for obj in (sa, role, rb):
        if dry_run:
            print("--- DRY RUN apply ---")
            print(yaml.safe_dump(obj, sort_keys=False))
            continue
        rc, out, err = kubectl_apply_yaml(yaml.safe_dump(obj, sort_keys=False).encode("utf-8"))
        if rc != 0:
            print("ERROR: kubectl apply failed:", err or out, file=sys.stderr); sys.exit(2)
    print("Applied ServiceAccount + Role + RoleBinding")

    # apply CronJob
    env_map = collect_pipeline_envs()
    cron = render_cronjob(cfg, env_map)
    if dry_run:
        print("--- DRY RUN CronJob ---")
        print(yaml.safe_dump(cron, sort_keys=False))
        return
    rc, out, err = kubectl_apply_yaml(yaml.safe_dump(cron, sort_keys=False).encode("utf-8"))
    if rc != 0:
        print("ERROR: kubectl apply CronJob failed:", err or out, file=sys.stderr); sys.exit(2)
    print("Applied CronJob:", cfg["CRONJOB_NAME"])

def delete_from_cluster(cfg: Dict, delete_secrets: bool=False):
    rc, out, err = kubectl_delete("cronjob", cfg["CRONJOB_NAME"], cfg["NAMESPACE"])
    print("Deleted cronjob:", cfg["CRONJOB_NAME"])
    kubectl_delete("rolebinding", cfg["ROLEBINDING_NAME"], cfg["NAMESPACE"])
    kubectl_delete("role", cfg["ROLE_NAME"], cfg["NAMESPACE"])
    kubectl_delete("serviceaccount", cfg["SERVICE_ACCOUNT_NAME"], cfg["NAMESPACE"])
    print("Deleted RBAC resources and ServiceAccount")
    if delete_secrets:
        kubectl_delete("secret", cfg.get("QDRANT_SECRET_NAME","qdrant-api-key"), cfg["NAMESPACE"])
        kubectl_delete("secret", cfg.get("AWS_SECRET_NAME","indexer-aws-creds"), cfg["NAMESPACE"])
        print("Deleted secrets (Qdrant + AWS)")

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--delete", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manifests-dir", default=None, help="Override default manifests output dir")
    p.add_argument("--delete-secrets", action="store_true", help="When used with --delete remove secrets created by this script")
    return p.parse_args()

def load_cfg() -> Dict:
    cfg = {}
    # base values (allow env override)
    cfg["MANIFESTS_DIR"] = os.environ.get("MANIFESTS_DIR", DEFAULTS["MANIFESTS_DIR"])
    cfg["NAMESPACE"] = os.environ.get("NAMESPACE", DEFAULTS["NAMESPACE"])
    cfg["CRONJOB_NAME"] = os.environ.get("CRONJOB_NAME", DEFAULTS["CRONJOB_NAME"]).lower()
    # schedule precedence: explicit env names
    cfg["CRON_SCHEDULE"] = os.environ.get("INDEXING_BACKUP_CRON_EXPRESSION", os.environ.get("CRON_SCHEDULE", DEFAULTS["CRON_SCHEDULE"]))
    cfg["CRONJOB_CONCURRENCY"] = os.environ.get("CRONJOB_CONCURRENCY", DEFAULTS["CRONJOB_CONCURRENCY"])
    cfg["CRONJOB_BACKOFF_LIMIT"] = os.environ.get("CRONJOB_BACKOFF_LIMIT", DEFAULTS["CRONJOB_BACKOFF_LIMIT"])
    cfg["CRONJOB_PARALLELISM"] = os.environ.get("CRONJOB_PARALLELISM", DEFAULTS["CRONJOB_PARALLELISM"])
    cfg["CRONJOB_COMPLETIONS"] = os.environ.get("CRONJOB_COMPLETIONS", DEFAULTS["CRONJOB_COMPLETIONS"])
    cfg["CRONJOB_DEBUG_KEEP_POD"] = os.environ.get("CRONJOB_DEBUG_KEEP_POD", DEFAULTS["CRONJOB_DEBUG_KEEP_POD"])
    cfg["CRONJOB_TIMEZONE"] = os.environ.get("CRONJOB_TIMEZONE", "")  # optional IANA timezone string

    # image + resources
    cfg["INDEXING_PIPELINE_CPU_IMAGE_REPO"] = os.environ.get("INDEXING_PIPELINE_CPU_IMAGE_REPO", DEFAULTS["INDEXING_PIPELINE_CPU_IMAGE_REPO"])
    cfg["INDEXING_PIPELINE_CPU_IMAGE_TAG"] = os.environ.get("INDEXING_PIPELINE_CPU_IMAGE_TAG", DEFAULTS["INDEXING_PIPELINE_CPU_IMAGE_TAG"])
    cfg["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"] = os.environ.get("INDEXING_BACKUP_CRONJOB_CPU_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_REQUEST"])
    cfg["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"] = os.environ.get("INDEXING_BACKUP_CRONJOB_CPU_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_CPU_LIMIT"])
    cfg["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"] = os.environ.get("INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST"])
    cfg["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"] = os.environ.get("INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT", DEFAULTS["INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT"])

    # RBAC names
    cfg["SERVICE_ACCOUNT_NAME"] = os.environ.get("SERVICE_ACCOUNT_NAME", DEFAULTS["SERVICE_ACCOUNT_NAME"])
    cfg["ROLE_NAME"] = os.environ.get("ROLE_NAME", DEFAULTS["ROLE_NAME"])
    cfg["ROLEBINDING_NAME"] = os.environ.get("ROLEBINDING_NAME", DEFAULTS["ROLEBINDING_NAME"])

    # secret names
    cfg["QDRANT_SECRET_NAME"] = os.environ.get("QDRANT_SECRET_NAME", "qdrant-api-key")
    cfg["AWS_SECRET_NAME"] = os.environ.get("AWS_SECRET_NAME", "indexer-aws-creds")

    # history limits
    cfg["SUCCESSFUL_JOBS_HISTORY_LIMIT"] = os.environ.get("SUCCESSFUL_JOBS_HISTORY_LIMIT", "3")
    cfg["FAILED_JOBS_HISTORY_LIMIT"] = os.environ.get("FAILED_JOBS_HISTORY_LIMIT", "1")

    # pass-through pipeline envs
    env_map = collect_pipeline_envs()
    # include raw creds & sensitive values into env_map (they will be secret references when present)
    env_map["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
    env_map["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    env_map["QDRANT_API_KEY"] = os.environ.get("QDRANT_API_KEY", "")
    env_map["CRONJOB_TIMEZONE"] = os.environ.get("CRONJOB_TIMEZONE", "")
    cfg.update(env_map)
    return cfg

def main():
    args = parse_args()
    cfg = load_cfg()
    if args.manifests_dir:
        cfg["MANIFESTS_DIR"] = args.manifests_dir
    out_dir = Path(cfg["MANIFESTS_DIR"])

    if args.generate:
        generate_manifests(cfg, out_dir)
        if args.dry_run:
            print("--- DRY RUN: preview CronJob ---")
            env_map = collect_pipeline_envs()
            print(yaml.safe_dump(render_cronjob(cfg, env_map), sort_keys=False))
        return

    if args.delete:
        if args.dry_run:
            print("DRY RUN delete: would remove CronJob + RBAC; --delete-secrets would remove secrets.")
        else:
            delete_from_cluster(cfg, delete_secrets=args.delete_secrets)
        # remove local manifests produced earlier
        try:
            if out_dir.exists():
                for p in out_dir.iterdir():
                    if p.is_file():
                        p.unlink()
                    else:
                        import shutil
                        shutil.rmtree(p)
                print("Removed local manifests at", out_dir)
        except Exception:
            pass
        return

    if args.apply:
        # generate manifests, then apply; --dry-run will print but not change cluster
        generate_manifests(cfg, out_dir)
        apply_to_cluster(cfg, out_dir, dry_run=args.dry_run)
        return

if __name__ == "__main__":
    main()
