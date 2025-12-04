#!/usr/bin/env python3
"""
infra/tests/qdrant_full_restore.py

Purpose:
  Restore a Qdrant cluster from a previously-created S3-hosted backup manifest.
  This script downloads the manifest (and referenced snapshot artifacts), copies
  those snapshot files into the correct pods (via kubectl cp), optionally restarts
  the Qdrant StatefulSet, and waits for pod readiness.

Key behaviours & constraints:
  - Does NOT write any Secret manifests to disk. If ENV=STAGING and AWS creds are
    present, the script will create/update an in-cluster secret named
    `qdrant-backup-aws` (kubectl create --dry-run=client | kubectl apply -f -).
    If ENV=PROD, no secret is created (IRSA expected).
  - Uses local boto3 (AWS env or profile) to download S3 objects. Does not depend
    on an in-cluster backup image.
  - Copies snapshot files into pods using `kubectl cp` and creates remote dirs
    via `kubectl exec`. After files are copied, the statefulset may be restarted
    (if --restart) to cause Qdrant to pick up restored snapshots.
  - Deterministic: manifest format expected:
      { "backup_id": "...", "namespace": "...", "pods": { "<pod>": { "collections": { "<col>": { "s3_key": "...", "sha256": "...", "size": N }}}}, "collections": [...] }
  - Safety: verifies local file SHA256 matches manifest before copying to pods.
  - Requires: Python 3.8+, boto3, requests, kubectl in PATH for pod operations.
"""

from pathlib import Path
import argparse
import os
import sys
import json
import hashlib
import datetime
import tempfile
import shutil
import subprocess
import time
import boto3

# ---------- Helpers ----------

def run(cmd, input_bytes=None, timeout=None):
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=True, check=True, timeout=timeout)
        return proc.returncode, proc.stdout.decode() if proc.stdout else "", proc.stderr.decode() if proc.stderr else ""
    except subprocess.CalledProcessError as e:
        return e.returncode, getattr(e, "stdout", b"").decode() if getattr(e, "stdout", None) else "", getattr(e, "stderr", b"").decode() if getattr(e, "stderr", None) else str(e)
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", b"").decode() if getattr(e, "stdout", None) else "", getattr(e, "stderr", b"").decode() if getattr(e, "stderr", None) else f"timeout after {timeout}s"

def sha256_of_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def kubectl_available():
    return shutil.which("kubectl") is not None

# ---------- Kubernetes helpers ----------

def kubectl_apply_yaml_from_string(yaml_str: str):
    rc, out, err = run(["kubectl", "apply", "-f", "-"], input_bytes=yaml_str.encode("utf-8"))
    return rc == 0, out if out else err

def create_staging_secret_in_cluster(namespace: str):
    """
    Create/update qdrant-backup-aws secret in-cluster from env AWS_* vars.
    Returns (ok:bool, message:str).
    """
    if not kubectl_available():
        return False, "kubectl-not-found"
    literals = []
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        literals += ["--from-literal", f"AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']}"]
    if os.environ.get("AWS_SECRET_ACCESS_KEY"):
        literals += ["--from-literal", f"AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}"]
    if os.environ.get("AWS_SESSION_TOKEN"):
        literals += ["--from-literal", f"AWS_SESSION_TOKEN={os.environ['AWS_SESSION_TOKEN']}"]
    if not literals:
        return False, "no-aws-creds-present"
    cmd = ["kubectl", "create", "secret", "generic", "qdrant-backup-aws", "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run(cmd, timeout=20)
    if rc != 0:
        return False, err or out
    rc2, out2, err2 = run(["kubectl", "apply", "-f", "-"], input_bytes=out.encode("utf-8"), timeout=20)
    if rc2 != 0:
        return False, err2 or out2
    return True, "secret-created"

def list_qdrant_pods(namespace: str):
    """Return sorted list of qdrant-* pods in namespace. Requires kubectl."""
    if not kubectl_available():
        return []
    rc, out, err = run(["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"])
    if rc != 0:
        return []
    names = [n for n in out.strip().split() if n.startswith("qdrant-")]
    names.sort()
    return names

def kubectl_exec(namespace: str, pod: str, args: list):
    cmd = ["kubectl", "exec", "-n", namespace, pod, "--"] + args
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(f"kubectl exec failed: {err or out}")
    return out

def kubectl_cp_to_pod(namespace: str, pod: str, local_path: str, remote_path: str):
    cmd = ["kubectl", "cp", str(local_path), f"{namespace}/{pod}:{remote_path}"]
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(f"kubectl cp failed: {err or out}")

def kubectl_rollout_restart(namespace: str, sts_name: str):
    rc, out, err = run(["kubectl", "rollout", "restart", f"statefulset/{sts_name}", "-n", namespace], timeout=60)
    if rc != 0:
        raise RuntimeError(f"rollout restart failed: {err or out}")

def wait_for_statefulset(namespace: str, name: str, timeout: int = 300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out, err = run(["kubectl", "rollout", "status", f"statefulset/{name}", "-n", namespace], timeout=30)
        if rc == 0:
            return True
        time.sleep(2)
    raise RuntimeError("timed out waiting for rollout status")

# ---------- S3 helpers ----------

def s3_client(region: str = None, endpoint_url: str = None):
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        return boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
    return boto3.client("s3", **kwargs)

def s3_download(bucket: str, key: str, dest: Path, region: str = None, endpoint_url: str = None):
    client = s3_client(region=region, endpoint_url=endpoint_url)
    ensure_dir(dest.parent)
    client.download_file(Bucket=bucket, Key=key, Filename=str(dest))

# ---------- Main restore flow ----------

def restore_from_s3(args):
    """
    Steps:
      - Optionally create staging secret in-cluster (if ENV=STAGING and AWS creds present).
      - Download manifest.json from S3 (s3_prefix/backup_id/manifest.json).
      - For each pod/collection entry referenced in manifest, download artifact to local tmpdir.
      - Verify sha256 (if provided).
      - Copy artifact into pod at /qdrant/snapshots/<filename> using kubectl cp.
      - Optionally restart statefulset and wait for readiness.
    """
    tmpdir = Path(args.local_dir or tempfile.gettempdir()) / f"restore-{args.backup_id}"
    ensure_dir(tmpdir)

    # create staging secret in-cluster if applicable
    env = os.environ.get("ENV", "STAGING").upper()
    if env == "STAGING":
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            ok, msg = create_staging_secret_in_cluster(args.namespace)
            if ok:
                print("Created/updated in-cluster secret: qdrant-backup-aws")
            else:
                print("Warning: could not create in-cluster secret:", msg)
        else:
            print("ENV=STAGING but AWS creds not present in environment; proceeding with local boto3 credentials (if available).")
    else:
        print("ENV=PROD: not creating secrets (IRSA expected).")

    # download manifest
    manifest_key = f"{args.s3_prefix.rstrip('/')}/{args.backup_id}/manifest.json"
    manifest_local = tmpdir / "manifest.json"
    print(f"Downloading manifest s3://{args.s3_bucket}/{manifest_key} -> {manifest_local}")
    s3_download(args.s3_bucket, manifest_key, manifest_local, region=args.aws_region, endpoint_url=(args.s3_endpoint or None))
    manifest = json.loads(manifest_local.read_text())

    # Build download list
    downloads = []  # list of tuples (s3_key, local_path, expected_sha)
    for pod, podinfo in manifest.get("pods", {}).items():
        cols = podinfo.get("collections", {}) or {}
        for col, meta in cols.items():
            s3_key = meta.get("s3_key")
            sha = meta.get("sha256")
            if not s3_key:
                raise RuntimeError(f"manifest missing s3_key for pod {pod} collection {col}")
            fname = Path(s3_key).name
            local_target = tmpdir / fname
            downloads.append((s3_key, local_target, sha, pod, col))

    # Download artifacts
    print(f"Downloading {len(downloads)} snapshot artifact(s) from S3 to {tmpdir}...")
    for key, local_path, sha, pod, col in downloads:
        print(f"  s3://{args.s3_bucket}/{key} -> {local_path}")
        s3_download(args.s3_bucket, key, local_path, region=args.aws_region, endpoint_url=(args.s3_endpoint or None))

    # Verify checksums
    print("Verifying checksums...")
    for key, local_path, expected_sha, pod, col in downloads:
        if expected_sha:
            actual = sha256_of_file(local_path)
            if actual != expected_sha:
                raise RuntimeError(f"Checksum mismatch for {local_path}: expected {expected_sha} actual {actual}")
            else:
                print(f"  OK {local_path} (sha256 matches)")
        else:
            print(f"  Warning: no sha256 in manifest for {local_path}; skipping checksum check")

    # Determine target pods list
    pods_in_manifest = list(manifest.get("pods", {}).keys())
    if pods_in_manifest == ["service"]:
        # service-mode backup: map to actual statefulset pods in namespace
        pods = list_qdrant_pods(args.namespace)
        if not pods:
            raise RuntimeError("No qdrant pods found in namespace to restore into")
    else:
        pods = pods_in_manifest

    # Copy files into pods
    print("Copying snapshots into pods...")
    # For mapping: when manifest contained per-pod files, each download tuple has pod name.
    for key, local_path, expected_sha, pod, col in downloads:
        # if manifest used 'service' mode then pod in tuple might be 'service'; handle mapping by distributing files across pods
        target_pod = pod
        if pod == "service":
            # choose a pod for this collection deterministically: use first pod (or better: preserve mapping by suffix if stored)
            pod_list = list_qdrant_pods(args.namespace)
            if not pod_list:
                raise RuntimeError("No qdrant pods discovered for service-mode restore")
            # Choose a deterministic pod based on collection name hash to spread files: simple mod
            target_pod = pod_list[hash(col) % len(pod_list)]
        print(f"  -> Copying {local_path.name} -> pod {target_pod} /qdrant/snapshots/")
        # create remote dir
        rc, out, err = run(["kubectl", "exec", "-n", args.namespace, target_pod, "--", "mkdir", "-p", "/qdrant/snapshots"])
        if rc != 0:
            raise RuntimeError(f"failed to mkdir on pod {target_pod}: {err or out}")
        # copy
        kubectl_cp_to_pod(args.namespace, target_pod, str(local_path), f"/qdrant/snapshots/{local_path.name}")
        print(f"    uploaded -> {target_pod} : /qdrant/snapshots/{local_path.name}")

    # Optionally restart statefulset to ensure qdrant picks up snapshots
    if args.restart:
        print("Restarting statefulset/qdrant...")
        kubectl_rollout_restart(args.namespace, "qdrant")
        print("Waiting for rollout to finish...")
        wait_for_statefulset(args.namespace, "qdrant", timeout=args.timeout)
        print("Rollout finished; waiting a few seconds for readiness checks...")
        time.sleep(5)

    print("Restore process completed successfully. Local artifacts retained at:", tmpdir)
    return tmpdir

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Restore Qdrant cluster from S3-hosted snapshot manifest.")
    p.add_argument("--s3-bucket", required=True, help="S3 bucket containing backups.")
    p.add_argument("--s3-prefix", default="qdrant/backups", help="S3 prefix where backups stored.")
    p.add_argument("--backup-id", required=True, help="Backup ID (folder under prefix) to restore.")
    p.add_argument("--local-dir", default=None, help="Local directory to stage download (default tmpdir).")
    p.add_argument("--namespace", default=os.environ.get("QDRANT_NAMESPACE", "qdrant"), help="Kubernetes namespace for qdrant pods.")
    p.add_argument("--restart", action="store_true", help="Restart statefulset/qdrant after copying snapshots.")
    p.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region for S3.")
    p.add_argument("--s3-endpoint", default=os.environ.get("BACKUP_S3_ENDPOINT", ""), help="Optional S3 endpoint URL.")
    p.add_argument("--timeout", type=int, default=300, help="Timeouts in seconds.")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        # quick S3 access check (non-fatal)
        try:
            client = s3_client = boto3.client("s3", region_name=args.aws_region) if not args.s3_endpoint else boto3.client("s3", region_name=args.aws_region, endpoint_url=args.s3_endpoint)
            client.head_bucket(Bucket=args.s3_bucket)
        except Exception as e:
            print("Warning: S3 head_bucket failed (still attempting):", e)
        restored_dir = restore_from_s3(args)
        print("Restore finished. Local staging dir:", restored_dir)
    except Exception as e:
        print("Restore failed:", str(e), file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
