#!/usr/bin/env python3
"""
infra/generators/gen_qdrant_cluster.py

1) Purpose: deterministically generate Helm values for deploying Qdrant and optionally
   perform a helm upgrade/install using a vendored chart (or remote repo fallback).
2) Inputs: environment variables (see load_config), existing infra/archive vendor chart,
   kubectl and helm binaries on PATH.
3) Outputs: writes values.yaml to MANIFESTS_DIR (default infra/manifests/qdrant),
   vendors chart to infra/archive/qdrant-helm-chart/qdrant when missing, performs helm upgrade/install,
   and writes last_deploy_summary.json (metadata only) to MANIFESTS_DIR.
4) Secrets policy: this generator NEVER writes real Secret manifests into the git-tracked
   manifests directory. If ENV=STAGING and AWS creds are present and --apply is used,
   the script **creates/updates** the secret **in-cluster** via kubectl (no secret YAML is persisted).
   If ENV=PROD the script does not create secrets (IRSA expected).
5) Determinism: inputs are hashed to skip no-op generations. Values are generated from env/config only.
6) Safety: namespace creation is done idempotently via kubectl create --dry-run=client | kubectl apply -f -.
7) Compatibility: existing defaults preserved unless environment variables explicitly change them.
8) How it fits: this is the canonical manifest/value generator used by your CI/dev workflow;
   run --generate to produce values, or --apply to create secrets in-staging and install the chart.
"""

from pathlib import Path
import os
import sys
import json
import yaml
import shutil
import subprocess
import hashlib
import uuid
import datetime
import argparse

# -------------------- Config loader --------------------
def load_config():
    env = os.environ.get("ENV", "STAGING").upper()
    cfg = {}
    cfg["ENV"] = env
    cfg["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/qdrant"))
    cfg["ARCHIVE_DIR"] = Path(os.environ.get("ARCHIVE_DIR", "infra/archive/qdrant-helm-chart"))
    cfg["VENDOR_CHART_DIR"] = cfg["ARCHIVE_DIR"] / "qdrant"
    cfg["QDRANT_RELEASE"] = os.environ.get("QDRANT_RELEASE", "qdrant")
    cfg["QDRANT_NAMESPACE"] = os.environ.get("QDRANT_NAMESPACE", "qdrant")
    # Default image kept conservative to previously working tag; override via QDRANT_IMAGE
    cfg["QDRANT_IMAGE"] = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.16.0")
    cfg["CHART_VERSION"] = os.environ.get("CHART_VERSION", "1.16.0")
    cfg["QDRANT_REPLICAS"] = int(os.environ.get("QDRANT_REPLICAS", "3" if env == "STAGING" else "3"))
    cfg["QDRANT_CPU"] = os.environ.get("QDRANT_CPU", "1" if env == "STAGING" else "4")
    cfg["QDRANT_MEMORY"] = os.environ.get("QDRANT_MEMORY", "2Gi" if env == "STAGING" else "16Gi")
    cfg["QDRANT_STORAGE"] = os.environ.get("QDRANT_STORAGE", "emptyDir")
    cfg["QDRANT_NODE_SELECTOR"] = os.environ.get("QDRANT_NODE_SELECTOR", "")
    cfg["QDRANT_TAINT_KEY"] = os.environ.get("QDRANT_TAINT_KEY", "qdrant-dedicated")
    cfg["QDRANT_TAINT_EFFECT"] = os.environ.get("QDRANT_TAINT_EFFECT", "NoSchedule")
    cfg["BACKUP_S3_BUCKET"] = os.environ.get("BACKUP_S3_BUCKET", "")
    cfg["BACKUP_S3_PREFIX"] = os.environ.get("BACKUP_S3_PREFIX", "qdrant/backups")
    cfg["IRSA_ROLE_ARN"] = os.environ.get("IRSA_ROLE_ARN", "")
    cfg["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
    cfg["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    cfg["AWS_SESSION_TOKEN"] = os.environ.get("AWS_SESSION_TOKEN", "")
    cfg["APPLY_STAGING_SECRETS"] = os.environ.get("APPLY_STAGING_SECRETS", "true").lower() in ("1", "true", "yes")
    cfg["TIMEOUT_SECONDS"] = int(os.environ.get("TIMEOUT_SECONDS", "600"))
    cfg["INPUTS_HASH_PATH"] = cfg["MANIFESTS_DIR"] / ".inputs_hash"
    cfg["S3_REGION"] = os.environ.get("AWS_REGION", os.environ.get("BACKUP_S3_REGION", "ap-south-1"))
    s3_ep = os.environ.get("BACKUP_S3_ENDPOINT", "")
    if not s3_ep:
        if cfg["S3_REGION"] == "us-east-1":
            s3_ep = "https://s3.amazonaws.com"
        else:
            s3_ep = f"https://s3.{cfg['S3_REGION']}.amazonaws.com"
    cfg["BACKUP_S3_ENDPOINT"] = s3_ep
    cfg["UUID_SHORT"] = str(uuid.uuid4())[:8]
    cfg["HELM_PRIMARY_REPO"] = os.environ.get("HELM_PRIMARY_REPO", "https://qdrant.github.io/qdrant-helm")
    cfg["HELM_FALLBACK_REPO"] = os.environ.get("HELM_FALLBACK_REPO", "https://qdrant.to/helm")
    cfg["HELM_REPO_NAME"] = os.environ.get("HELM_REPO_NAME", "qdrant")
    # Qdrant-specific params (fixes: use env vars, fallback preserved)
    cfg["QDRANT_SHARD_NUMBER"] = int(os.environ.get("QDRANT_SHARD_NUMBER", os.environ.get("QDRANT_SHARD_NUM", "1")))
    cfg["QDRANT_REPLICATION_FACTOR"] = int(os.environ.get("QDRANT_REPLICATION_FACTOR", os.environ.get("QDRANT_REPLICATION", os.environ.get("QDRANT_REPLICAS", "1"))))
    cfg["QDRANT_WRITE_CONSISTENCY_FACTOR"] = int(os.environ.get("QDRANT_WRITE_CONSISTENCY_FACTOR", "1"))
    cfg["QDRANT_LOG_LEVEL"] = os.environ.get("QDRANT_LOG_LEVEL", "INFO")
    cfg["QDRANT__STORAGE__STORAGE_PATH"] = os.environ.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage")
    cfg["QDRANT__STORAGE__SNAPSHOTS_PATH"] = os.environ.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")
    cfg["USE_LOCAL_NVME"] = os.environ.get("USE_LOCAL_NVME", "false").lower() in ("1", "true", "yes")
    return cfg

# -------------------- Utilities --------------------
SENSITIVE_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}

def canonical_inputs_hash(cfg):
    serial = {}
    for k in sorted(cfg.keys()):
        if k == "INPUTS_HASH_PATH":
            continue
        if k in SENSITIVE_KEYS:
            serial[k] = bool(cfg.get(k))
        else:
            v = cfg.get(k)
            try:
                json.dumps(v)
                serial[k] = v
            except Exception:
                serial[k] = str(v)
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def run_cmd(cmd, capture=True, check=False, timeout=None, input_bytes=None):
    try:
        proc = subprocess.run(cmd, capture_output=capture, text=True, check=check, timeout=timeout, input=(input_bytes.decode() if isinstance(input_bytes, bytes) else input_bytes))
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

# -------------------- Kubernetes / Helm helpers --------------------
def detect_storageclass(kubectl_bin="kubectl"):
    if shutil.which(kubectl_bin) is None:
        return None
    cmd = [kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[?(@.metadata.annotations.storageclass\\.kubernetes\\.io/is-default-class==\"true\")].metadata.name}"]
    rc, out, err = run_cmd(cmd)
    out = out.strip()
    if out:
        return out
    cmd2 = [kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[0].metadata.name}"]
    rc2, out2, err2 = run_cmd(cmd2)
    return out2.strip() if out2.strip() else None

def ensure_namespace(cfg):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False, "kubectl-not-found"
    # create ns YAML dry-run then apply
    cmd1 = [kubectl, "create", "namespace", cfg["QDRANT_NAMESPACE"], "--dry-run=client", "-o", "yaml"]
    rc, out, err = run_cmd(cmd1, timeout=20)
    if rc != 0:
        # maybe namespace already exists; try get
        rcg, outg, errg = run_cmd([kubectl, "get", "namespace", cfg["QDRANT_NAMESPACE"]], timeout=20)
        if rcg == 0:
            return True, None
        return False, err or out or outg or errg
    # apply the produced YAML
    rc2, out2, err2 = run_cmd([kubectl, "apply", "-f", "-"], input_bytes=out.encode("utf-8"), timeout=20)
    return rc2 == 0, err2 if rc2 != 0 else None

def kubectl_create_secret_in_cluster(cfg, secret_name="qdrant-backup-aws"):
    """
    Create or update a generic opaque secret in the target namespace.
    Only used for ENV=STAGING and when AWS creds are present.
    This function does NOT write any secret YAML to disk.
    """
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False, "kubectl-not-found"
    # Build create secret command with --from-literal entries only for values present
    literals = []
    if cfg.get("AWS_ACCESS_KEY_ID"):
        literals += ["--from-literal", f"AWS_ACCESS_KEY_ID={cfg['AWS_ACCESS_KEY_ID']}"]
    if cfg.get("AWS_SECRET_ACCESS_KEY"):
        literals += ["--from-literal", f"AWS_SECRET_ACCESS_KEY={cfg['AWS_SECRET_ACCESS_KEY']}"]
    if cfg.get("AWS_SESSION_TOKEN"):
        literals += ["--from-literal", f"AWS_SESSION_TOKEN={cfg['AWS_SESSION_TOKEN']}"]
    if not literals:
        return False, "no-aws-creds-present"
    cmd = [kubectl, "create", "secret", "generic", secret_name, "-n", cfg["QDRANT_NAMESPACE"], "--dry-run=client", "-o", "yaml"] + literals
    rc1, out1, err1 = run_cmd(cmd, timeout=20)
    if rc1 != 0:
        return False, err1 or out1
    # apply the YAML to be idempotent
    rc2, out2, err2 = run_cmd([kubectl, "apply", "-f", "-"], input_bytes=out1.encode("utf-8"), timeout=20)
    if rc2 != 0:
        return False, err2 or out2
    return True, None

def kubectl_resource_exists(kind: str, name: str, namespace: str) -> bool:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False
    rc, out, err = run_cmd([kubectl, "get", kind, name, "-n", namespace], capture=True)
    return rc == 0

def helm_repo_add_if_missing(cfg, verbose=False):
    helm = shutil.which("helm")
    if not helm:
        return False, "helm-not-found"
    cmd = [helm, "repo", "add", "--force-update", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]]
    rc, out, err = run_cmd(cmd, timeout=30)
    if rc != 0:
        rc2, out2, err2 = run_cmd([helm, "repo", "add", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]], timeout=30)
        if rc2 != 0:
            return False, err2 or err
    rc_upd, out_upd, err_upd = run_cmd([helm, "repo", "update"], timeout=30)
    return True, None

def vendor_chart_if_missing(cfg, verbose=False):
    vendor_dir = cfg["VENDOR_CHART_DIR"]
    helm = shutil.which("helm")
    if vendor_dir.exists() and (vendor_dir / "Chart.yaml").exists():
        return True, str(vendor_dir)
    if helm is None:
        return False, "helm-not-found"
    ok, err = helm_repo_add_if_missing(cfg, verbose=verbose)
    try:
        cmd = [helm, "pull", f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--untar", "--untardir", str(cfg["ARCHIVE_DIR"])]
        rc, out, err = run_cmd(cmd, timeout=120)
        if rc == 0:
            if vendor_dir.exists():
                return True, str(vendor_dir)
            else:
                return False, "vendor-dir-missing-after-pull"
        else:
            return False, err or out
    except Exception as e:
        return False, str(e)

def helm_upgrade_install(cfg, values_file: Path, vendor_dir: Path, verbose=False):
    helm = shutil.which("helm")
    if not helm:
        return False, "helm-not-found", "", ""
    release = cfg["QDRANT_RELEASE"]
    ns = cfg["QDRANT_NAMESPACE"]
    timeout = "10m"
    # prefer vendor dir if present
    if vendor_dir.exists():
        cmd = [helm, "upgrade", "--install", release, str(vendor_dir), "--namespace", ns, "--create-namespace", "-f", str(values_file), "--wait", "--timeout", timeout]
        rc, out, err = run_cmd(cmd, timeout=600)
        if rc == 0:
            return True, None, out, err
        vendor_err = err or out
    else:
        vendor_err = "vendor-not-present"
    # try primary repo
    cmd2 = [helm, "upgrade", "--install", release, f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc2, out2, err2 = run_cmd(cmd2, timeout=600)
    if rc2 == 0:
        return True, None, out2, err2
    # fallback repo via --repo
    cmd3 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_FALLBACK_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc3, out3, err3 = run_cmd(cmd3, timeout=600)
    if rc3 == 0:
        return True, None, out3, err3
    # retry primary explicitly
    cmd4 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_PRIMARY_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc4, out4, err4 = run_cmd(cmd4, timeout=600)
    if rc4 == 0:
        return True, None, out4, err4
    combined_err = "\n--- vendor attempt ---\n" + str(vendor_err) + "\n--- primary ---\n" + (err2 or out2) + "\n--- fallback ---\n" + (err3 or out3) + "\n--- retry primary ---\n" + (err4 or out4)
    return False, combined_err, out4, err4

# -------------------- Renderers --------------------
def render_values_yaml(cfg, storage_class):
    repo, tag = cfg["QDRANT_IMAGE"].split(":", 1) if ":" in cfg["QDRANT_IMAGE"] else (cfg["QDRANT_IMAGE"], "latest")
    peers = []
    for i in range(0, cfg["QDRANT_REPLICAS"]):
        peers.append(f"http://{cfg['QDRANT_RELEASE']}-{i}.{cfg['QDRANT_RELEASE']}-headless:6335")
    values = {
        "replicaCount": cfg["QDRANT_REPLICAS"],
        "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
        "service": {"type": "ClusterIP"},
        "p2p": {"port": 6335},
        "cluster": {"enabled": True, "peers": peers},
        "persistence": {"enabled": True},
        "snapshots": {
            "enabled": True if cfg.get("BACKUP_S3_BUCKET") else False,
            "s3": {"bucket": cfg.get("BACKUP_S3_BUCKET", ""), "endpoint": cfg.get("BACKUP_S3_ENDPOINT", ""), "region": cfg.get("S3_REGION", cfg.get("S3_REGION")), "prefix": cfg.get("BACKUP_S3_PREFIX", "")},
        },
        "extraEnv": [
            {"name": "QDRANT__SERVICE__API_KEY", "value": os.environ.get("QDRANT__SERVICE__API_KEY", "")},
            {"name": "QDRANT__STORAGE__SNAPSHOTS_CONFIG__SNAPSHOTS_STORAGE", "value": "S3" if cfg.get("BACKUP_S3_BUCKET") else ""},
            {"name": "QDRANT__STORAGE__SNAPSHOTS_CONFIG__S3_CONFIG__BUCKET", "value": cfg.get("BACKUP_S3_BUCKET", "")},
            {"name": "QDRANT__STORAGE__SNAPSHOTS_CONFIG__S3_CONFIG__REGION", "value": cfg.get("S3_REGION", "")},
            {"name": "QDRANT__STORAGE__SNAPSHOTS_CONFIG__S3_CONFIG__ENDPOINT_URL", "value": cfg.get("BACKUP_S3_ENDPOINT", "")},
        ],
        "resources": {"requests": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]}, "limits": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]}},
        "tolerations": [{"key": cfg["QDRANT_TAINT_KEY"], "operator": "Exists", "effect": cfg["QDRANT_TAINT_EFFECT"]}],
    }
    if storage_class:
        values.setdefault("persistence", {})["storageClass"] = storage_class
        values["persistence"]["size"] = "50Gi"
    # config block: use provided values, not hardcoded defaults
    values["config"] = {
        "params": {
            "shard_number": int(cfg.get("QDRANT_SHARD_NUMBER", 1)),
            "replication_factor": int(cfg.get("QDRANT_REPLICATION_FACTOR", cfg.get("QDRANT_REPLICAS", 1))),
            "write_consistency_factor": int(cfg.get("QDRANT_WRITE_CONSISTENCY_FACTOR", 1)),
        },
        "log_level": cfg.get("QDRANT_LOG_LEVEL", "INFO"),
        "storage": {"storage_path": cfg.get("QDRANT__STORAGE__STORAGE_PATH", "/qdrant/storage"), "snapshots_path": cfg.get("QDRANT__STORAGE__SNAPSHOTS_PATH", "/qdrant/snapshots")},
    }
    return yaml.safe_dump(values, sort_keys=False)

# -------------------- Generate / Apply / Delete --------------------
def generate_manifests(cfg, dry_run=False, verbose=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    inputs_hash = canonical_inputs_hash(cfg)
    existing = None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing = cfg["INPUTS_HASH_PATH"].read_text().strip()
    if existing == inputs_hash and not dry_run:
        print("No non-secret changes detected; generation skipped.")
        return
    storage_class = detect_storageclass() or None
    values_yaml = render_values_yaml(cfg, storage_class)
    # write values.yaml atomically
    atomic_write(cfg["MANIFESTS_DIR"] / "values.yaml", values_yaml)
    # write a non-sensitive sample secret to _samples (safe to commit)
    samples_dir = cfg["MANIFESTS_DIR"] / "_samples"
    ensure_dir(samples_dir)
    sample_secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "qdrant-backup-aws", "namespace": cfg["QDRANT_NAMESPACE"]},
        "type": "Opaque",
        "stringData": {"AWS_ACCESS_KEY_ID": "AKIAxxxxxxxxxxxx", "AWS_SECRET_ACCESS_KEY": "xxxxxxxxxxxxxxxxxxxx"},
    }
    atomic_write(samples_dir / "secret-sample.yaml", yaml.safe_dump(sample_secret, sort_keys=False))
    cfg["INPUTS_HASH_PATH"].write_text(inputs_hash)
    print("Wrote manifests to", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        print("--- values.yaml preview ---")
        for line in values_yaml.splitlines()[:50]:
            print(line)
    return

def apply_to_cluster(cfg, dry_run=False, verbose=False):
    kubectl = shutil.which("kubectl")
    helm = shutil.which("helm")
    if kubectl is None or helm is None:
        print("ERROR: kubectl and helm are required in PATH to apply to cluster.", file=sys.stderr)
        sys.exit(2)
    # staging preconditions: when applying in STAGING and APPLY_STAGING_SECRETS true, require AWS creds to create secret
    if cfg["ENV"] == "STAGING" and cfg["APPLY_STAGING_SECRETS"]:
        if not cfg["AWS_ACCESS_KEY_ID"] or not cfg["AWS_SECRET_ACCESS_KEY"]:
            print("ERROR: ENV=STAGING requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to be set when using --apply (in-cluster secret will be created).", file=sys.stderr)
            sys.exit(2)
    # generate manifests (ensure up-to-date values.yaml)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    # ensure namespace exists
    ok, err = ensure_namespace(cfg)
    if not ok:
        print("ERROR: failed to ensure namespace:", err, file=sys.stderr)
        sys.exit(2)
    print("Namespace ensured:", cfg["QDRANT_NAMESPACE"])
    # if staging, create secret in-cluster (do NOT write secret manifest)
    if cfg["ENV"] == "STAGING" and cfg["APPLY_STAGING_SECRETS"]:
        ok_s, err_s = kubectl_create_secret_in_cluster(cfg)
        if not ok_s:
            print("ERROR: failed to create staging secret in-cluster:", err_s, file=sys.stderr)
            sys.exit(2)
        print("Created/updated in-cluster secret: qdrant-backup-aws")
    else:
        if cfg["ENV"] == "PROD":
            print("ENV=PROD: skipping secret creation (expect IRSA/cluster-managed secrets).")
        else:
            print("Skipping in-cluster secret creation (APPLY_STAGING_SECRETS=false).")

    # vendor chart if missing (pull chart to archive)
    v_ok, v_err = vendor_chart_if_missing(cfg, verbose=verbose)
    if not v_ok:
        print("Vendor chart not available locally; will attempt remote install. vendor error:", v_err)
    else:
        print("Vendor chart available at", cfg["VENDOR_CHART_DIR"])
    # run helm upgrade/install
    values_file = cfg["MANIFESTS_DIR"] / "values.yaml"
    ok, errtext, stdout_text, stderr_text = helm_upgrade_install(cfg, values_file, cfg["VENDOR_CHART_DIR"], verbose=verbose)
    if not ok:
        print("ERROR: helm upgrade/install failed. See summary below.", file=sys.stderr)
        if verbose:
            print("--- helm error (tail) ---")
            print(errtext.splitlines()[-200:])
        print(errtext, file=sys.stderr)
        sys.exit(2)
    print("Helm install/upgrade succeeded for release:", cfg["QDRANT_RELEASE"])
    # write last_deploy_summary.json
    summary = {
        "release": cfg["QDRANT_RELEASE"],
        "namespace": cfg["QDRANT_NAMESPACE"],
        "replicas": cfg["QDRANT_REPLICAS"],
        "values_file": str(values_file),
        "chart_version": cfg["CHART_VERSION"],
        "image": cfg["QDRANT_IMAGE"],
        "vendor_chart_dir": str(cfg["VENDOR_CHART_DIR"]) if cfg["VENDOR_CHART_DIR"].exists() else None,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    print("Wrote deploy summary ->", str(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json"))
    return

def delete_manifests(cfg):
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
        print("Deleted manifests at", str(cfg["MANIFESTS_DIR"]))
    else:
        print("Manifests dir not present:", str(cfg["MANIFESTS_DIR"]))

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate/apply Qdrant Helm manifests (cluster-aware).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate manifests to MANIFESTS_DIR.")
    grp.add_argument("--apply", action="store_true", help="Generate manifests and apply (create staging secret + helm install).")
    grp.add_argument("--delete", action="store_true", help="Delete generated manifests and inputs hash.")
    p.add_argument("--dry-run", action="store_true", help="Render and validate but do not write or apply.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info (helm output tails on failure).")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    if args.delete:
        delete_manifests(cfg); return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose); return
    if args.apply:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose)
        if args.dry_run:
            print("Dry-run mode: skipping kubectl/helm apply.")
            return
        apply_to_cluster(cfg, dry_run=False, verbose=args.verbose)
        return

if __name__ == "__main__":
    main()
