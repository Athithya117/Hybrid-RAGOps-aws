#!/usr/bin/env python3
"""
Fixed gen_qdrant_cluster.py
- Robust kubectl/helm subprocess usage
- Fixed ensure_namespace (no hanging kubectl apply -f -)
- More defensive vendor/helm logic
- Default image set to a known working tag; override with QDRANT_IMAGE
"""
from pathlib import Path
import os, sys, json, yaml, shutil, subprocess, argparse, hashlib, uuid, datetime

# ---------- config loader ----------
def load_config():
    env = os.environ.get("ENV", "STAGING").upper()
    m = {}
    m["ENV"] = env
    m["MANIFESTS_DIR"] = Path(os.environ.get("MANIFESTS_DIR", "infra/manifests/qdrant"))
    m["ARCHIVE_DIR"] = Path(os.environ.get("ARCHIVE_DIR", "infra/archive/qdrant-helm-chart"))
    m["VENDOR_CHART_DIR"] = m["ARCHIVE_DIR"] / "qdrant"
    m["QDRANT_RELEASE"] = os.environ.get("QDRANT_RELEASE", "qdrant")
    m["QDRANT_NAMESPACE"] = os.environ.get("QDRANT_NAMESPACE", "qdrant")
    # default image: use chart-compatible, fallback to v1.16.0 (existing). Override with QDRANT_IMAGE env var.
    m["QDRANT_IMAGE"] = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.16.0")
    m["CHART_VERSION"] = os.environ.get("CHART_VERSION", "1.16.0")
    m["QDRANT_REPLICAS"] = int(os.environ.get("QDRANT_REPLICAS", "3" if env=="STAGING" else "3"))
    m["QDRANT_CPU"] = os.environ.get("QDRANT_CPU", "1" if env=="STAGING" else "4")
    m["QDRANT_MEMORY"] = os.environ.get("QDRANT_MEMORY", "4Gi" if env=="STAGING" else "16Gi")
    m["BACKUP_S3_BUCKET"] = os.environ.get("BACKUP_S3_BUCKET", "")
    m["BACKUP_S3_PREFIX"] = os.environ.get("BACKUP_S3_PREFIX", "qdrant/backups")
    m["IRSA_ROLE_ARN"] = os.environ.get("IRSA_ROLE_ARN", "")
    m["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
    m["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    m["TIMEOUT_SECONDS"] = int(os.environ.get("TIMEOUT_SECONDS", "600"))
    m["INPUTS_HASH_PATH"] = m["MANIFESTS_DIR"] / ".inputs_hash"
    m["S3_REGION"] = os.environ.get("AWS_REGION", os.environ.get("BACKUP_S3_REGION", "ap-south-1"))
    s3_ep = os.environ.get("BACKUP_S3_ENDPOINT", "")
    if not s3_ep:
        s3_ep = "https://s3.amazonaws.com" if m["S3_REGION"]=="us-east-1" else f"https://s3.{m['S3_REGION']}.amazonaws.com"
    m["BACKUP_S3_ENDPOINT"] = s3_ep
    m["UUID_SHORT"] = str(uuid.uuid4())[:8]
    m["HELM_PRIMARY_REPO"] = os.environ.get("HELM_PRIMARY_REPO", "https://qdrant.github.io/qdrant-helm")
    m["HELM_FALLBACK_REPO"] = os.environ.get("HELM_FALLBACK_REPO", "https://qdrant.to/helm")
    m["HELM_REPO_NAME"] = os.environ.get("HELM_REPO_NAME", "qdrant")
    return m

# ---------- utilities ----------
SENSITIVE_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}

def canonical_inputs_hash(cfg):
    serial = {}
    for k in sorted(cfg):
        if k == "INPUTS_HASH_PATH": continue
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
    return hashlib.sha256(j.encode()).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def run_cmd(cmd, input_text=None, timeout=None, check=False):
    """Return (rc, stdout, stderr). Uses text mode for simplicity."""
    try:
        proc = subprocess.run(cmd, input=input_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=check)
        return proc.returncode, (proc.stdout or ""), (proc.stderr or "")
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or f"timeout after {timeout}s"

# ---------- kubernetes/helm helpers ----------
def detect_storageclass(kubectl_bin="kubectl"):
    if shutil.which(kubectl_bin) is None:
        return None
    cmd = [kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[?(@.metadata.annotations.storageclass\\.kubernetes\\.io/is-default-class==\"true\")].metadata.name}"]
    rc, out, err = run_cmd(cmd)
    out = out.strip()
    if out:
        return out
    rc2, out2, err2 = run_cmd([kubectl_bin, "get", "storageclass", "-o", "jsonpath={.items[0].metadata.name}"])
    return out2.strip() if out2.strip() else None

def ensure_namespace(cfg):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False, "kubectl-not-found"
    # produce YAML for namespace via dry-run, then pipe it to apply (single roundtrip)
    create_cmd = [kubectl, "create", "namespace", cfg["QDRANT_NAMESPACE"], "--dry-run=client", "-o", "yaml"]
    rc, out, err = run_cmd(create_cmd, timeout=20)
    if rc != 0:
        # if namespace exists, treat as OK
        rcg, outg, errg = run_cmd([kubectl, "get", "namespace", cfg["QDRANT_NAMESPACE"]], timeout=10)
        if rcg == 0:
            return True, None
        return False, f"create-namespace-dryrun-failed: {err or out}"
    # apply the produced YAML
    rc_app, out_app, err_app = run_cmd([kubectl, "apply", "-f", "-"], input_text=out, timeout=20)
    if rc_app != 0:
        # if it failed but namespace now exists, treat ok
        rcg, outg, errg = run_cmd([kubectl, "get", "namespace", cfg["QDRANT_NAMESPACE"]], timeout=10)
        if rcg == 0:
            return True, None
        return False, f"apply-namespace-failed: {err_app or out_app}"
    return True, None

def kubectl_resource_exists(kind: str, name: str, namespace: str) -> bool:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return False
    rc, out, err = run_cmd([kubectl, "get", kind, name, "-n", namespace], timeout=15)
    return rc == 0

def kubectl_apply_yaml(yaml_str: str, dry_run=False):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return {"applied": False, "error": "kubectl-not-found"}
    cmd = [kubectl, "apply"]
    if dry_run:
        cmd += ["--dry-run=client", "-f", "-"]
    else:
        cmd += ["-f", "-"]
    rc, out, err = run_cmd(cmd, input_text=yaml_str, timeout=60)
    return {"applied": rc == 0, "stdout": out, "stderr": err}

def helm_repo_add_if_missing(cfg):
    helm = shutil.which("helm")
    if not helm:
        return False, "helm-not-found"
    # force-update add (ok if already present)
    rc, out, err = run_cmd([helm, "repo", "add", "--force-update", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]], timeout=30)
    if rc != 0:
        # try without force-update
        rc2, out2, err2 = run_cmd([helm, "repo", "add", cfg["HELM_REPO_NAME"], cfg["HELM_PRIMARY_REPO"]], timeout=30)
        if rc2 != 0:
            return False, err2 or err
    rcu, outu, erru = run_cmd([helm, "repo", "update"], timeout=30)
    return True, None

def vendor_chart_if_missing(cfg):
    vendor_dir = cfg["VENDOR_CHART_DIR"]
    helm = shutil.which("helm")
    if vendor_dir.exists() and (vendor_dir / "Chart.yaml").exists():
        return True, str(vendor_dir)
    if helm is None:
        return False, "helm-not-found"
    ok, err = helm_repo_add_if_missing(cfg)
    if not ok:
        return False, err
    rc, out, err = run_cmd([helm, "pull", f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--untar", "--untardir", str(cfg["ARCHIVE_DIR"])], timeout=120)
    if rc == 0 and vendor_dir.exists():
        return True, str(vendor_dir)
    return False, err or out or "helm-pull-failed"

def helm_upgrade_install(cfg, values_file: Path, vendor_dir: Path):
    helm = shutil.which("helm")
    if not helm:
        return False, "helm-not-found", "", ""
    release = cfg["QDRANT_RELEASE"]
    ns = cfg["QDRANT_NAMESPACE"]
    timeout = "10m"
    attempts = []
    # try vendor
    if vendor_dir.exists():
        cmd = [helm, "upgrade", "--install", release, str(vendor_dir), "--namespace", ns, "--create-namespace", "-f", str(values_file), "--wait", "--timeout", timeout]
        rc, out, err = run_cmd(cmd, timeout=600)
        attempts.append(("vendor", rc, out, err))
        if rc == 0:
            return True, None, out, err
    # try repo name
    cmd2 = [helm, "upgrade", "--install", release, f"{cfg['HELM_REPO_NAME']}/qdrant", "--version", cfg["CHART_VERSION"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc2, out2, err2 = run_cmd(cmd2, timeout=600)
    attempts.append(("primary", rc2, out2, err2))
    if rc2 == 0:
        return True, None, out2, err2
    # try fallback with --repo
    cmd3 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_FALLBACK_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc3, out3, err3 = run_cmd(cmd3, timeout=600)
    attempts.append(("fallback", rc3, out3, err3))
    if rc3 == 0:
        return True, None, out3, err3
    # final attempt: retry primary with explicit repo URL
    cmd4 = [helm, "upgrade", "--install", release, "qdrant/qdrant", "--version", cfg["CHART_VERSION"], "--repo", cfg["HELM_PRIMARY_REPO"], "--namespace", ns, "-f", str(values_file), "--wait", "--timeout", timeout]
    rc4, out4, err4 = run_cmd(cmd4, timeout=600)
    attempts.append(("retry_primary", rc4, out4, err4))
    combined = "\n".join([f"--- {t[0]} rc={t[1]} stderr={t[3]}\nstdout={t[2]}" for t in attempts])
    return False, combined, out4, err4

# ---------- renderers ----------
def render_values_yaml(cfg, storage_class):
    repo_tag = cfg["QDRANT_IMAGE"]
    if ":" in repo_tag:
        repo, tag = repo_tag.split(":", 1)
    else:
        repo, tag = repo_tag, "latest"
    peers = [f"http://{cfg['QDRANT_RELEASE']}-{i}.{cfg['QDRANT_RELEASE']}-headless:6335" for i in range(cfg["QDRANT_REPLICAS"])]
    values = {
        "replicaCount": cfg["QDRANT_REPLICAS"],
        "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
        "service": {"type": "ClusterIP"},
        "p2p": {"port": 6335},
        "cluster": {"enabled": True, "peers": peers},
        "persistence": {"enabled": True, "size": "50Gi"},
        "snapshots": {
            "enabled": bool(cfg.get("BACKUP_S3_BUCKET")),
            "s3": {"bucket": cfg.get("BACKUP_S3_BUCKET", ""), "endpoint": cfg.get("BACKUP_S3_ENDPOINT", ""), "region": cfg.get("S3_REGION", cfg.get("S3_REGION")), "prefix": cfg.get("BACKUP_S3_PREFIX", "")},
        },
        "resources": {"requests": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]}, "limits": {"cpu": cfg["QDRANT_CPU"], "memory": cfg["QDRANT_MEMORY"]}},
        "tolerations": [{"key": "qdrant-dedicated", "operator": "Exists", "effect": "NoSchedule"}],
        "config": {"params": {"shard_number": 1, "replication_factor": cfg["QDRANT_REPLICAS"], "write_consistency_factor": 1},
                   "log_level": "INFO", "storage": {"storage_path": "/qdrant/storage", "snapshots_path": "/qdrant/snapshots"}},
    }
    if storage_class:
        values["persistence"]["storageClass"] = storage_class
    return yaml.safe_dump(values, sort_keys=False)

def render_serviceaccount_yaml(cfg):
    sa_name = f"{cfg['QDRANT_RELEASE']}-backup-sa"
    sa = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": sa_name, "namespace": cfg["QDRANT_NAMESPACE"]}}
    if cfg["ENV"] == "PROD" and cfg["IRSA_ROLE_ARN"]:
        sa["metadata"]["annotations"] = {"eks.amazonaws.com/role-arn": cfg["IRSA_ROLE_ARN"]}
    return yaml.safe_dump(sa, sort_keys=False)

def render_secret_sample_yaml(cfg):
    secret_name = "qdrant-backup-aws"
    string_data = {"AWS_ACCESS_KEY_ID": "AKIAxxxxxxxxxxxx", "AWS_SECRET_ACCESS_KEY": "xxxxxxxxxxxxxxxxxxxx"}
    if cfg["AWS_ACCESS_KEY_ID"]:
        string_data["AWS_ACCESS_KEY_ID"] = cfg["AWS_ACCESS_KEY_ID"]
    if cfg["AWS_SECRET_ACCESS_KEY"]:
        string_data["AWS_SECRET_ACCESS_KEY"] = cfg["AWS_SECRET_ACCESS_KEY"]
    if cfg.get("AWS_SESSION_TOKEN"):
        string_data["AWS_SESSION_TOKEN"] = cfg["AWS_SESSION_TOKEN"]
    secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": secret_name, "namespace": cfg["QDRANT_NAMESPACE"]}, "type": "Opaque", "stringData": string_data}
    return yaml.safe_dump(secret, sort_keys=False)

# ---------- generate / apply / delete ----------
def generate_manifests(cfg, dry_run=False, verbose=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    ih = canonical_inputs_hash(cfg)
    if cfg["INPUTS_HASH_PATH"].exists() and cfg["INPUTS_HASH_PATH"].read_text().strip() == ih and not dry_run:
        print("No non-secret changes detected; generation skipped.")
        return
    storage_class = detect_storageclass() or None
    values_yaml = render_values_yaml(cfg, storage_class)
    sa_yaml = render_serviceaccount_yaml(cfg)
    secret_yaml = render_secret_sample_yaml(cfg)
    atomic_write(cfg["MANIFESTS_DIR"] / "values.yaml", values_yaml)
    atomic_write(cfg["MANIFESTS_DIR"] / "serviceaccount-backup.yaml", sa_yaml)
    atomic_write(cfg["MANIFESTS_DIR"] / "secret-sample.yaml", secret_yaml)
    cfg["INPUTS_HASH_PATH"].write_text(ih)
    print("Wrote manifests to", str(cfg["MANIFESTS_DIR"]))
    if verbose:
        print(values_yaml)
    return

def apply_to_cluster(cfg, dry_run=False, verbose=False):
    kubectl = shutil.which("kubectl")
    helm = shutil.which("helm")
    if not kubectl or not helm:
        print("ERROR: kubectl and helm required", file=sys.stderr); sys.exit(2)
    if cfg["ENV"] == "STAGING":
        if not cfg["AWS_ACCESS_KEY_ID"] or not cfg["AWS_SECRET_ACCESS_KEY"]:
            print("ERROR: ENV=STAGING requires AWS creds for --apply", file=sys.stderr); sys.exit(2)
    if cfg["ENV"] == "PROD" and not cfg["IRSA_ROLE_ARN"]:
        print("ERROR: ENV=PROD expects IRSA_ROLE_ARN for --apply", file=sys.stderr); sys.exit(2)
    generate_manifests(cfg, dry_run=dry_run, verbose=verbose)
    ok, err = ensure_namespace(cfg)
    if not ok:
        print("ERROR: failed to ensure namespace:", err, file=sys.stderr); sys.exit(2)
    print("Namespace ensured:", cfg["QDRANT_NAMESPACE"])
    sa_yaml = (cfg["MANIFESTS_DIR"] / "serviceaccount-backup.yaml").read_text()
    res_sa = kubectl_apply_yaml(sa_yaml, dry_run=dry_run)
    if not res_sa.get("applied"):
        print("ERROR: failed to apply ServiceAccount:", res_sa.get("stderr", res_sa.get("error", "<unknown>")), file=sys.stderr); sys.exit(2)
    print("Applied ServiceAccount")
    if cfg["ENV"] == "STAGING":
        secret_name = "qdrant-backup-aws"
        if kubectl_resource_exists("secret", secret_name, cfg["QDRANT_NAMESPACE"]):
            print("Secret exists -> skipping")
        else:
            secret_yaml = (cfg["MANIFESTS_DIR"] / "secret-sample.yaml").read_text()
            res_sec = kubectl_apply_yaml(secret_yaml, dry_run=dry_run)
            if not res_sec.get("applied"):
                print("ERROR: failed to apply Secret:", res_sec.get("stderr", res_sec.get("error", "<unknown>")), file=sys.stderr); sys.exit(2)
            print("Applied Secret")
    else:
        print("PROD: not applying secrets (use IRSA)")

    v_ok, v_err = vendor_chart_if_missing(cfg)
    if not v_ok:
        print("Vendor chart not available locally; will attempt remote install. vendor error:", v_err)
    else:
        print("Vendor chart available at", cfg["VENDOR_CHART_DIR"])

    values_file = cfg["MANIFESTS_DIR"] / "values.yaml"
    ok, errtext, out, err = helm_upgrade_install(cfg, values_file, cfg["VENDOR_CHART_DIR"])
    if not ok:
        print("ERROR: helm upgrade/install failed.", file=sys.stderr)
        if verbose:
            print(errtext)
        print(errtext, file=sys.stderr)
        sys.exit(2)
    print("Helm install/upgrade succeeded for release:", cfg["QDRANT_RELEASE"])
    summary = {
        "release": cfg["QDRANT_RELEASE"], "namespace": cfg["QDRANT_NAMESPACE"], "replicas": cfg["QDRANT_REPLICAS"],
        "values_file": str(values_file), "chart_version": cfg["CHART_VERSION"], "image": cfg["QDRANT_IMAGE"],
        "vendor_chart_dir": str(cfg["VENDOR_CHART_DIR"]) if cfg["VENDOR_CHART_DIR"].exists() else None,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    atomic_write(cfg["MANIFESTS_DIR"] / "last_deploy_summary.json", json.dumps(summary, indent=2))
    print("Wrote deploy summary")

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

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true")
    grp.add_argument("--apply", action="store_true")
    grp.add_argument("--delete", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    if args.delete:
        delete_manifests(cfg); return
    if args.generate:
        generate_manifests(cfg, dry_run=args.dry_run, verbose=args.verbose); return
    if args.apply:
        if args.dry_run:
            generate_manifests(cfg, dry_run=True, verbose=args.verbose)
            print("Dry-run: skipping apply"); return
        apply_to_cluster(cfg, dry_run=False, verbose=args.verbose); return

if __name__ == "__main__":
    main()
