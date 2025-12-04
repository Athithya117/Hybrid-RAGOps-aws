#!/usr/bin/env python3
"""
run_qdrant_backup.py

Backup & restore utility for Qdrant clusters supporting both:
 - cluster-level (service-mode) snapshots (default)
 - per-pod snapshots (--per-pod) for node-local storage (local NVMe)

Behavior summary:
 - BACKUP (default): create snapshots (service or per-pod), retrieve snapshot files,
   upload to S3, and write manifest.json + latest.manifest.json under S3 prefix.
 - RESTORE: download manifest and referenced snapshots from S3, copy into pods
   (/qdrant/snapshots/) with kubectl cp, optionally restart statefulset.
 - STAGING: if AWS env creds present, create/update in-cluster secret qdrant-backup-aws
   (applied directly via kubectl; no secret YAML written to repo).
 - PROD: expect IRSA / instance role; no in-cluster secret creation.

Requirements:
 - Python 3.8+, requests, boto3, kubectl in PATH for per-pod and restore flows.

Notes:
 - Per-pod mode is recommended if data lives on node-local NVMe (pod-local files).
 - Cluster-level is simpler and works if service can serve consistent snapshots.
"""

from pathlib import Path
import argparse
import os
import sys
import json
import hashlib
import time
import datetime
import uuid
import shutil
import subprocess
import tempfile
from typing import List, Tuple, Optional

import boto3
import requests

# ---------- Defaults ----------
DEFAULT_S3_PREFIX = "qdrant/backups"
DEFAULT_TMPDIR = Path("tmp")
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_NAMESPACE = os.environ.get("QDRANT_NAMESPACE", "qdrant")

# ---------- Utilities ----------
SENSITIVE = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}

def canonical_inputs_hash(obj: dict) -> str:
    serial = {}
    for k in sorted(obj.keys()):
        if k == "INPUTS_HASH_PATH":
            continue
        if k in SENSITIVE:
            serial[k] = bool(obj.get(k))
        else:
            v = obj.get(k)
            try:
                json.dumps(v)
                serial[k] = v
            except Exception:
                serial[k] = str(v)
    j = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def now_isoutc():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def run_cmd(cmd: List[str], timeout: Optional[int] = None, input_bytes: bytes = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout, input=input_bytes)
        return proc.returncode, proc.stdout.decode() if proc.stdout else "", proc.stderr.decode() if proc.stderr else ""
    except subprocess.CalledProcessError as e:
        return e.returncode, (getattr(e, "stdout", b"") or b"").decode(errors="ignore"), (getattr(e, "stderr", b"") or b"").decode(errors="ignore")
    except subprocess.TimeoutExpired as e:
        return 124, (getattr(e, "stdout", b"") or b"").decode(errors="ignore"), (getattr(e, "stderr", b"") or b"").decode(errors="ignore")

def kubectl_exists() -> bool:
    return shutil.which("kubectl") is not None

# ---------- S3 helpers ----------
def s3_client(region: str = None, endpoint_url: str = None):
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        return boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
    return boto3.client("s3", **kwargs)

def s3_upload_file(bucket: str, key: str, filename: str, region: str = None, endpoint_url: str = None):
    client = s3_client(region=region, endpoint_url=endpoint_url)
    client.upload_file(Filename=filename, Bucket=bucket, Key=key)

def s3_download_file(bucket: str, key: str, filename: str, region: str = None, endpoint_url: str = None):
    client = s3_client(region=region, endpoint_url=endpoint_url)
    client.download_file(Bucket=bucket, Key=key, Filename=filename)

# ---------- Qdrant helpers (cluster-level API) ----------
def list_collections(qdrant_url: str, timeout: int = 10) -> List[str]:
    try:
        r = requests.get(f"{qdrant_url}/collections", timeout=timeout)
        r.raise_for_status()
        j = r.json()
        result = j.get("result", {})
        cols = []
        if isinstance(result, dict) and "collections" in result:
            for c in result["collections"]:
                if isinstance(c, dict) and "name" in c:
                    cols.append(c["name"])
                elif isinstance(c, str):
                    cols.append(c)
        elif isinstance(result, list):
            for c in result:
                if isinstance(c, dict) and "name" in c:
                    cols.append(c["name"])
                elif isinstance(c, str):
                    cols.append(c)
        return cols
    except Exception as e:
        raise RuntimeError(f"Failed to list collections from Qdrant at {qdrant_url}: {e}")

def request_snapshot_and_get_name(qdrant_url: str, collection: str, wait: bool = True, timeout: int = 120) -> str:
    params = {}
    if wait:
        params["wait"] = "true"
    url = f"{qdrant_url}/collections/{collection}/snapshots"
    r = requests.post(url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    result = j.get("result")
    if isinstance(result, dict):
        for key in ("name", "snapshot", "snapshot_name"):
            if key in result:
                return result[key]
    if isinstance(result, str):
        return result
    for key in ("snapshot", "snapshot_name", "name"):
        if key in j:
            return j[key]
    raise RuntimeError(f"Unable to determine snapshot name from response: {j}")

def download_snapshot_via_service(qdrant_url: str, collection: str, snapshot_name: str, dest_path: Path, timeout: int = 300):
    """Try candidate URLs exposed by qdrant service/pod to download snapshot content."""
    candidates = [
        f"{qdrant_url}/collections/{collection}/snapshots/{snapshot_name}/download",
        f"{qdrant_url}/collections/{collection}/snapshots/{snapshot_name}",
    ]
    last_err = None
    for url in candidates:
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 200:
                    with dest_path.open("wb") as w:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                w.write(chunk)
                    return
                else:
                    last_err = f"{r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Failed to download snapshot from service; last error: {last_err}")

# ---------- Kubernetes helpers (per-pod mode) ----------
def list_qdrant_pods(namespace: str) -> List[str]:
    if not kubectl_exists():
        return []
    selectors = [
        "app.kubernetes.io/name=qdrant",
        "app=qdrant",
        "app.kubernetes.io/instance=qdrant",
    ]
    for sel in selectors:
        rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-l", sel, "-o", "jsonpath={.items[*].metadata.name}"])
        if rc == 0 and out.strip():
            return out.strip().split()
    rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"])
    if rc == 0 and out.strip():
        return [n for n in out.strip().split() if n.startswith("qdrant-")]
    return []

def start_port_forward(pod: str, namespace: str, local_port: int, remote_port: int = 6333, timeout: int = 10) -> subprocess.Popen:
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    cmd = ["kubectl", "port-forward", f"pod/{pod}", f"{local_port}:{remote_port}", "-n", namespace]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    probe = f"http://127.0.0.1:{local_port}/collections"
    deadline = time.time() + timeout
    ok = False
    while time.time() < deadline:
        try:
            r = requests.get(probe, timeout=1)
            if r.status_code == 200:
                ok = True
                break
        except Exception:
            pass
        time.sleep(0.25)
    if not ok:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"port-forward to pod/{pod} on local port {local_port} failed to respond")
    return proc

def stop_port_forward(proc: subprocess.Popen):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def kubectl_list_files_in_pod(namespace: str, pod: str, path: str) -> List[str]:
    rc, out, err = run_cmd(["kubectl", "exec", "-n", namespace, pod, "--", "sh", "-c", f"ls -1 {path} 2>/dev/null || true"], timeout=20)
    if rc != 0 and not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]

def kubectl_cp_from_pod(namespace: str, pod: str, pod_path: str, dst_path: Path):
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    full_src = f"{namespace}/{pod}:{pod_path}"
    rc, out, err = run_cmd(["kubectl", "cp", full_src, str(dst_path)], timeout=120)
    if rc != 0:
        raise RuntimeError(f"kubectl cp failed: {err or out}")

# ---------- Staging secret helper ----------
def create_staging_secret_in_cluster(namespace: str) -> bool:
    if not kubectl_exists():
        print("kubectl not found; cannot create in-cluster secret. Proceeding without creating secret.", file=sys.stderr)
        return False
    aws_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_token = os.environ.get("AWS_SESSION_TOKEN", "")
    if not (aws_id and aws_secret):
        print("AWS credentials not present in environment; skipping staging secret creation.", file=sys.stderr)
        return False
    literals = ["--from-literal", f"AWS_ACCESS_KEY_ID={aws_id}", "--from-literal", f"AWS_SECRET_ACCESS_KEY={aws_secret}"]
    if aws_token:
        literals += ["--from-literal", f"AWS_SESSION_TOKEN={aws_token}"]
    cmd = ["kubectl", "create", "secret", "generic", "qdrant-backup-aws", "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        print("Failed to render secret YAML for in-cluster apply:", err or out, file=sys.stderr)
        return False
    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=out.encode("utf-8"), timeout=20)
    if rc2 != 0:
        print("Failed to apply staging secret in-cluster:", err2 or out2, file=sys.stderr)
        return False
    return True

# ---------- Backup flows ----------
def backup_cluster_service_mode(args, cfg):
    """Cluster-level: snapshot via service and download via service."""
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_id = f"{timestamp}-{cfg['UUID_SHORT']}"
    local_tmp = Path(args.local_dir or DEFAULT_TMPDIR) / backup_id
    ensure_dir(local_tmp)
    s3_prefix = args.s3_prefix.rstrip("/")
    aws_region = args.aws_region
    s3_endpoint = args.s3_endpoint or None

    collections = args.collections or list_collections(cfg["QDRANT_URL"], timeout=args.timeout)
    if not collections:
        raise RuntimeError("No collections found to backup.")

    inputs = {
        "collections": collections,
        "qdrant_url": cfg["QDRANT_URL"],
        "s3_bucket": args.s3_bucket,
        "s3_prefix": s3_prefix,
        "mode": "service",
        "namespace": args.namespace,
    }
    inputs_hash = canonical_inputs_hash(inputs)
    inputs_hash_path = Path(args.local_dir or DEFAULT_TMPDIR) / ".inputs_hash"
    existing = inputs_hash_path.read_text().strip() if inputs_hash_path.exists() else None
    if existing == inputs_hash:
        print("No non-secret changes detected (inputs hash unchanged); proceeding (still performs backup).")
    else:
        inputs_hash_path.write_text(inputs_hash)

    manifest = {
        "backup_id": backup_id,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "namespace": args.namespace,
        "pods": {"service": {"collections": {}}},
        "collections": collections,
        "mode": "service",
    }

    for col in collections:
        print(f"Creating snapshot for collection '{col}' via service {cfg['QDRANT_URL']} ...")
        snap_name = request_snapshot_and_get_name(cfg["QDRANT_URL"], col, wait=True, timeout=args.timeout)
        print(f"Snapshot created: {snap_name}")
        fname = f"{col}-{snap_name}.snapshot"
        target = local_tmp / fname
        print(f"Downloading snapshot {snap_name} -> {target}")
        download_snapshot_via_service(cfg["QDRANT_URL"], col, snap_name, target, timeout=args.timeout)
        sha = sha256_of_file(target)
        size = target.stat().st_size
        s3_key = f"{s3_prefix}/{backup_id}/{target.name}"
        print(f"Uploading to s3: s3://{args.s3_bucket}/{s3_key}")
        s3_upload_file(args.s3_bucket, s3_key, str(target), region=aws_region, endpoint_url=s3_endpoint)
        manifest["pods"]["service"]["collections"][col] = {
            "snapshot_name": snap_name,
            "s3_key": s3_key,
            "sha256": sha,
            "size": size,
            "local_path": str(target),
        }

    # upload manifests
    manifest_json = json.dumps(manifest, indent=2)
    manifest_local = local_tmp / "manifest.json"
    manifest_local.write_text(manifest_json)
    latest_local = local_tmp / "latest.manifest.json"
    latest_local.write_text(manifest_json)
    manifest_s3_key = f"{s3_prefix}/{backup_id}/manifest.json"
    latest_s3_key = f"{s3_prefix}/latest.manifest.json"
    s3_upload_file(args.s3_bucket, manifest_s3_key, str(manifest_local), region=aws_region, endpoint_url=s3_endpoint)
    s3_upload_file(args.s3_bucket, latest_s3_key, str(latest_local), region=aws_region, endpoint_url=s3_endpoint)

    print("\n=== BACKUP MANIFEST ===")
    print(manifest_json)
    print("=== END MANIFEST ===\n")
    print("Backup finished. backup_id:", backup_id, "local:", str(local_tmp))
    return backup_id, str(local_tmp)

def backup_cluster_per_pod(args, cfg):
    """Per-pod: create snapshot via each pod and copy snapshot files from pod fs (kubectl cp)."""
    if not kubectl_exists():
        raise RuntimeError("--per-pod requires kubectl in PATH.")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_id = f"{timestamp}-{cfg['UUID_SHORT']}"
    local_tmp = Path(args.local_dir or DEFAULT_TMPDIR) / backup_id
    ensure_dir(local_tmp)
    s3_prefix = args.s3_prefix.rstrip("/")
    aws_region = args.aws_region
    s3_endpoint = args.s3_endpoint or None

    pods = list_qdrant_pods(args.namespace)
    if not pods:
        raise RuntimeError("No qdrant pods found in namespace: " + args.namespace)
    print("Per-pod mode: discovered pods:", pods)

    collections = args.collections or list_collections(cfg["QDRANT_URL"], timeout=args.timeout)
    if not collections:
        raise RuntimeError("No collections found to backup.")

    inputs = {
        "collections": collections,
        "qdrant_url": cfg["QDRANT_URL"],
        "s3_bucket": args.s3_bucket,
        "s3_prefix": s3_prefix,
        "mode": "per-pod",
        "namespace": args.namespace,
        "pods": pods,
    }
    inputs_hash = canonical_inputs_hash(inputs)
    inputs_hash_path = Path(args.local_dir or DEFAULT_TMPDIR) / ".inputs_hash"
    existing = inputs_hash_path.read_text().strip() if inputs_hash_path.exists() else None
    if existing == inputs_hash:
        print("No non-secret changes detected (inputs hash unchanged); proceeding (still performs backup).")
    else:
        inputs_hash_path.write_text(inputs_hash)

    manifest = {
        "backup_id": backup_id,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "namespace": args.namespace,
        "pods": {},
        "collections": collections,
        "mode": "per-pod",
    }

    port_forwards = []
    try:
        # start port forwards
        for idx, pod in enumerate(pods):
            local_port = args.port_base + idx
            print(f"Starting port-forward for pod {pod} -> localhost:{local_port}")
            pf = start_port_forward(pod, args.namespace, local_port, remote_port=6333, timeout=10)
            port_forwards.append((pod, pf, local_port))

        # for each pod, create snapshots (via forwarded pod endpoint) and fetch files
        for pod, pf, local_port in port_forwards:
            pod_dir = local_tmp / pod
            ensure_dir(pod_dir)
            qurl = f"http://127.0.0.1:{local_port}"
            manifest["pods"].setdefault(pod, {"collections": {}})
            for col in collections:
                print(f"[{pod}] requesting snapshot for collection '{col}' ...")
                snap_name = request_snapshot_and_get_name(qurl, col, wait=True, timeout=args.timeout)
                print(f"[{pod}] snapshot created: {snap_name}")
                # Try direct pod HTTP download first
                fname = f"{pod}-{col}-{snap_name}.snapshot"
                target = pod_dir / fname
                try:
                    print(f"[{pod}] attempting HTTP download via {qurl} ...")
                    download_snapshot_via_service(qurl, col, snap_name, target, timeout=args.timeout)
                    print(f"[{pod}] downloaded snapshot via HTTP -> {target}")
                except Exception as e_http:
                    # fallback to copying from pod filesystem
                    print(f"[{pod}] HTTP download failed: {e_http!s}; attempting kubectl cp from pod filesystem...")
                    pod_snap_dir = f"/qdrant/snapshots/{col}"
                    files = kubectl_list_files_in_pod(args.namespace, pod, pod_snap_dir)
                    # find file matching snapshot name
                    candidate = None
                    for f in files:
                        if snap_name in f:
                            candidate = f
                            break
                    if not candidate:
                        # maybe snapshot file not inside per-collection dir; try listing top-level /qdrant/snapshots
                        files2 = kubectl_list_files_in_pod(args.namespace, pod, "/qdrant/snapshots")
                        for f in files2:
                            if snap_name in f:
                                candidate = f
                                # adjust path
                                pod_snap_dir = "/qdrant/snapshots"
                                break
                    if not candidate:
                        raise RuntimeError(f"[{pod}] Unable to locate snapshot file in pod filesystem for snapshot {snap_name}. ls {pod_snap_dir} -> {files}")
                    pod_file_path = f"{pod_snap_dir}/{candidate}"
                    print(f"[{pod}] copying snapshot from pod path {pod_file_path} -> {target}")
                    kubectl_cp_from_pod(args.namespace, pod, pod_file_path, target)
                sha = sha256_of_file(target)
                size = target.stat().st_size
                s3_key = f"{s3_prefix}/{backup_id}/{target.name}"
                print(f"[{pod}] uploading to s3: s3://{args.s3_bucket}/{s3_key}")
                s3_upload_file(args.s3_bucket, s3_key, str(target), region=aws_region, endpoint_url=s3_endpoint)
                manifest["pods"][pod]["collections"][col] = {
                    "snapshot_name": snap_name,
                    "s3_key": s3_key,
                    "sha256": sha,
                    "size": size,
                    "local_path": str(target),
                    "pod_path": pod_file_path if 'pod_file_path' in locals() else None,
                }
    finally:
        for pod, pf, local_port in port_forwards:
            try:
                stop_port_forward(pf)
            except Exception:
                pass

    # upload manifest(s)
    manifest_json = json.dumps(manifest, indent=2)
    manifest_local = local_tmp / "manifest.json"
    manifest_local.write_text(manifest_json)
    latest_local = local_tmp / "latest.manifest.json"
    latest_local.write_text(manifest_json)
    manifest_s3_key = f"{s3_prefix}/{backup_id}/manifest.json"
    latest_s3_key = f"{s3_prefix}/latest.manifest.json"
    s3_upload_file(args.s3_bucket, manifest_s3_key, str(manifest_local), region=aws_region, endpoint_url=s3_endpoint)
    s3_upload_file(args.s3_bucket, latest_s3_key, str(latest_local), region=aws_region, endpoint_url=s3_endpoint)

    print("\n=== BACKUP MANIFEST ===")
    print(manifest_json)
    print("=== END MANIFEST ===\n")
    print("Backup finished. backup_id:", backup_id, "local:", str(local_tmp))
    return backup_id, str(local_tmp)

# ---------- Restore (cluster-level) ----------
def restore_cluster(args, cfg):
    aws_region = args.aws_region
    s3_endpoint = args.s3_endpoint or None
    tmpdir = Path(args.local_dir or DEFAULT_TMPDIR) / f"restore-{args.backup_id}"
    ensure_dir(tmpdir)

    manifest_s3_key = f"{args.s3_prefix.rstrip('/')}/{args.backup_id}/manifest.json"
    local_manifest = tmpdir / "manifest.json"
    print(f"Downloading manifest s3://{args.s3_bucket}/{manifest_s3_key} -> {local_manifest}")
    s3_download_file(args.s3_bucket, manifest_s3_key, str(local_manifest), region=aws_region, endpoint_url=s3_endpoint)
    manifest = json.loads(local_manifest.read_text())

    # build files list
    files_to_download = []
    for pod_key, podinfo in manifest.get("pods", {}).items():
        for col, meta in (podinfo.get("collections") or {}).items():
            s3_key = meta.get("s3_key")
            if s3_key:
                fname = Path(s3_key).name
                local_target = tmpdir / fname
                files_to_download.append((s3_key, local_target))
    print(f"Downloading {len(files_to_download)} snapshot files referenced in manifest...")
    for key, target in files_to_download:
        print(" ", f"s3://{args.s3_bucket}/{key}", "->", str(target))
        ensure_dir(target.parent)
        s3_download_file(args.s3_bucket, key, str(target), region=aws_region, endpoint_url=s3_endpoint)

    # verify checksums
    print("Verifying snapshot files checksums...")
    for pod_key, podinfo in manifest.get("pods", {}).items():
        for col, meta in (podinfo.get("collections") or {}).items():
            s3_key = meta.get("s3_key")
            if not s3_key:
                continue
            fname = Path(s3_key).name
            localf = tmpdir / fname
            if not localf.exists():
                raise RuntimeError(f"Missing expected snapshot file: {localf}")
            actual = sha256_of_file(localf)
            expected = meta.get("sha256")
            if expected and actual != expected:
                raise RuntimeError(f"Checksum mismatch for {localf}: expected {expected} actual {actual}")
    print("Checksums OK.")

    # determine pods to restore into
    pod_keys = list(manifest.get("pods", {}).keys())
    if pod_keys == ["service"]:
        pods = list_qdrant_pods(args.namespace)
    else:
        pods = pod_keys
    if not pods:
        raise RuntimeError("No qdrant pods found to restore into.")

    print("Pods to restore:", pods)
    # copy files into pods
    for pod in pods:
        pod_collections = manifest["pods"].get(pod) or manifest["pods"].get("service") or {}
        if not pod_collections:
            continue
        for col, meta in pod_collections.get("collections", {}).items():
            s3_key = meta.get("s3_key")
            fname = Path(s3_key).name
            localf = tmpdir / fname
            remote_path = f"/qdrant/snapshots/{fname}"
            print(f"Uploading {fname} -> pod {pod} : {remote_path}")
            rc, out, err = run_cmd(["kubectl", "exec", "-n", args.namespace, pod, "--", "mkdir", "-p", "/qdrant/snapshots"], timeout=20)
            if rc != 0:
                raise RuntimeError(f"mkdir in pod failed: {err or out}")
            rc2, out2, err2 = run_cmd(["kubectl", "cp", str(localf), f"{args.namespace}/{pod}:/qdrant/snapshots/{fname}"], timeout=120)
            if rc2 != 0:
                raise RuntimeError(f"kubectl cp failed: {err2 or out2}")
            print(f"  Uploaded {fname} to pod {pod}")

    if args.restart:
        print("Restarting statefulset/qdrant...")
        rc, out, err = run_cmd(["kubectl", "rollout", "restart", f"statefulset/qdrant", "-n", args.namespace], timeout=60)
        if rc != 0:
            raise RuntimeError(f"rollout restart failed: {err or out}")
        # wait for readiness
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            rc, out, err = run_cmd(["kubectl", "rollout", "status", f"statefulset/qdrant", "-n", args.namespace], timeout=30)
            if rc == 0:
                break
            time.sleep(2)
        print("Waiting a short while for pods to stabilize...")
        time.sleep(5)

    print("Restore completed successfully.")
    return

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Qdrant backup & restore utility (service-mode and per-pod).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--backup", action="store_true", help="Create snapshots and upload to S3.")
    grp.add_argument("--restore", metavar="BACKUP_ID", help="Restore backup ID from S3.")
    p.add_argument("--s3-bucket", required=True, help="S3 bucket to store backups.")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help="S3 prefix for backups (default qdrant/backups).")
    p.add_argument("--local-dir", default=str(DEFAULT_TMPDIR), help="Local directory to store temporary backup files.")
    p.add_argument("--collections", nargs="+", help="Collections to backup. If omitted, auto-detected.")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant service URL (default http://127.0.0.1:6333).")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace where qdrant runs.")
    p.add_argument("--timeout", type=int, default=300, help="Timeout seconds for HTTP/S3 operations.")
    p.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region (boto3).")
    p.add_argument("--s3-endpoint", default=os.environ.get("BACKUP_S3_ENDPOINT", ""), help="Optional S3 endpoint URL.")
    p.add_argument("--env", default=os.environ.get("ENV", "STAGING"), help="ENV (STAGING/PROD).")
    p.add_argument("--restart", action="store_true", help="When restoring, rollout restart the statefulset/qdrant after copying files.")
    # per-pod options
    p.add_argument("--per-pod", action="store_true", help="Use per-pod snapshot & filesystem copy (required for node-local NVMe).")
    p.add_argument("--port-base", type=int, default=7000, help="Base port for per-pod port-forwarding (7000).")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = {
        "ENV": args.env.upper(),
        "QDRANT_URL": args.qdrant_url,
        "UUID_SHORT": str(uuid.uuid4())[:8],
    }

    if cfg["ENV"] == "STAGING":
        print("ENV=STAGING: will create/update in-cluster secret qdrant-backup-aws if AWS creds present.")
        created = create_staging_secret_in_cluster(args.namespace)
        if created:
            print("Staging secret qdrant-backup-aws created/updated in-cluster.")
        else:
            print("Staging secret not created/updated (check messages).")
    else:
        print("ENV=PROD: no in-cluster secret created (expect IRSA / instance role).")

    # quick s3 check (warn only)
    try:
        client = s3_client(region=args.aws_region, endpoint_url=(args.s3_endpoint or None))
        client.head_bucket(Bucket=args.s3_bucket)
    except Exception as e:
        print("WARNING: cannot access S3 bucket (will attempt anyway):", e)

    try:
        if args.backup:
            if args.per_pod:
                bid, localdir = backup_cluster_per_pod(args, cfg)
            else:
                bid, localdir = backup_cluster_service_mode(args, cfg)
            if not bid:
                print("Backup failed.", file=sys.stderr)
                sys.exit(2)
            print("Backup complete:", bid, localdir)
            sys.exit(0)
        else:
            args.backup_id = args.restore
            restore_cluster(args, cfg)
            print("Restore completed.")
            sys.exit(0)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
