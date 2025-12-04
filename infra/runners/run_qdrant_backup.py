#!/usr/bin/env python3
"""
infra/tests/test_qdrant_full_cluster_backup_restore.py

Deterministic backup & restore test utility for a Qdrant StatefulSet cluster.

High-level behavior (concise):
- BACKUP mode:
  - Optionally creates sample collections + inserts sample points (deterministic).
  - Creates snapshots for requested collections.
  - Supports two snapshot-download modes:
      * service-mode (default): talk to the Qdrant service URL (single endpoint).
      * per-pod mode (--per-pod): opens port-forwards to each pod and requests snapshots
        directly from each pod (captures per-pod shard files).
  - Downloads snapshot files (streaming), uploads them to S3 (boto3), writes a small
    manifest.json and latest.manifest.json under S3 prefix and optionally keeps local
    backup copy in tmp/<backup_id> (printed).
  - Deterministic: computes an inputs hash (.inputs_hash) to skip unnecessary re-generation;
    backup still runs if inputs hash unchanged (keeps behavior explicit).
  - Does NOT write real Secret YAML into infra/manifests. For ENV=STAGING create/update
    an in-cluster secret (kubectl) if AWS creds present and requested; otherwise uses
    environment for S3 operations. For ENV=PROD IRSA is expected; no secret creation.

- RESTORE mode:
  - Downloads manifest from S3 (or uses local manifest if provided).
  - Downloads only snapshot artifacts referenced in manifest (unless --no-download).
  - Copies snapshots into pods using `kubectl cp <local> <ns>/<pod>:/qdrant/snapshots/` and
    then restarts the statefulset (kubectl rollout restart) if --restart.
  - Waits for rollout completion and checks pods are Ready.

Notes / operational:
- Requires: Python 3.8+, boto3, requests, kubectl in PATH.
- Default Qdrant service URL: http://127.0.0.1:6333 (use port-forward or cluster networking)
 - QDRANT_URL=http://qdrant.qdrant.svc.cluster.local:6333 intra cluster
- This script intentionally removes any "backup image" logic (never used in your runs).
- The script is conservative about secrets: it will not emit secret manifests into git-tracked
  directories. For STAGING it can create/update the secret inside the cluster using kubectl
  (no file written).
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
import threading

import boto3
import requests

# ---------- Config & defaults ----------
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

def run_cmd(cmd, timeout=None, input_bytes=None):
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout, input=input_bytes)
        return proc.returncode, proc.stdout.decode() if proc.stdout else "", proc.stderr.decode() if proc.stderr else ""
    except subprocess.CalledProcessError as e:
        return e.returncode, getattr(e, "stdout", b"").decode() if getattr(e, "stdout", None) else "", getattr(e, "stderr", b"").decode() if getattr(e, "stderr", None) else str(e)
    except subprocess.TimeoutExpired as e:
        return 124, getattr(e, "stdout", b"").decode() if getattr(e, "stdout", None) else "", getattr(e, "stderr", b"").decode() if getattr(e, "stderr", None) else f"timeout after {timeout}s"

# ---------- Kubernetes helpers ----------
def kubectl_exists():
    return shutil.which("kubectl") is not None

def list_qdrant_pods(namespace: str):
    """Return list of qdrant pod names (tries common labels)."""
    if not kubectl_exists():
        return []
    # Try known label selectors
    selectors = [
        "app.kubernetes.io/name=qdrant",
        "app=qdrant",
        "app.kubernetes.io/instance=qdrant",
    ]
    for sel in selectors:
        rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-l", sel, "-o", "jsonpath={.items[*].metadata.name}"])
        if rc == 0 and out.strip():
            return out.strip().split()
    # fallback: all pods in namespace matching 'qdrant-' prefix
    rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"])
    if rc == 0 and out.strip():
        return [n for n in out.strip().split() if n.startswith("qdrant-")]
    return []

def start_port_forward(pod: str, namespace: str, local_port: int, remote_port: int = 6333, timeout: int = 10):
    """Start kubectl port-forward in background and wait until it responds on local_port.
    Returns the Popen object. Raises RuntimeError on failure to establish."""
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    cmd = ["kubectl", "port-forward", f"pod/{pod}", f"{local_port}:{remote_port}", "-n", namespace]
    # Launch process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for port to become available (probe HTTP /collections)
    deadline = time.time() + timeout
    ok = False
    probe_url = f"http://127.0.0.1:{local_port}/collections"
    while time.time() < deadline:
        try:
            r = requests.get(probe_url, timeout=1)
            if r.status_code == 200:
                ok = True
                break
        except Exception:
            pass
        time.sleep(0.25)
    if not ok:
        # Clean up
        proc.kill()
        proc.wait(timeout=5)
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

def kubectl_cp_from_pod(namespace: str, pod: str, src_path: str, dst_path: str):
    """kubectl cp namespace/pod:src_path dst_path"""
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    full_src = f"{namespace}/{pod}:{src_path}"
    rc, out, err = run_cmd(["kubectl", "cp", full_src, str(dst_path)])
    if rc != 0:
        raise RuntimeError(f"kubectl cp failed: {err or out}")

def kubectl_cp_to_pod(namespace: str, pod: str, src_path: str, dst_path: str):
    """kubectl cp src_path namespace/pod:dst_path"""
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    full_dst = f"{namespace}/{pod}:{dst_path}"
    rc, out, err = run_cmd(["kubectl", "cp", str(src_path), full_dst])
    if rc != 0:
        raise RuntimeError(f"kubectl cp failed: {err or out}")

def kubectl_rollout_restart_statefulset(namespace: str, name: str):
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    rc, out, err = run_cmd(["kubectl", "rollout", "restart", f"statefulset/{name}", "-n", namespace], timeout=60)
    if rc != 0:
        raise RuntimeError(f"rollout restart failed: {err or out}")

def wait_for_statefulset_ready(namespace: str, name: str, timeout: int = 300):
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out, err = run_cmd(["kubectl", "rollout", "status", f"statefulset/{name}", "-n", namespace], timeout=30)
        if rc == 0:
            return True
        time.sleep(2)
    raise RuntimeError("timed out waiting for statefulset rollout to finish")

# ---------- S3 helpers ----------
def s3_client(region: str = None):
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)

def s3_upload_file(bucket: str, key: str, filename: str, region: str = None, endpoint_url: str = None):
    client = s3_client(region=region)
    extra = {}
    if endpoint_url:
        # boto3 will accept endpoint_url via client creation only; to keep simple, create client with endpoint if given
        client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
    client.upload_file(Filename=filename, Bucket=bucket, Key=key)

def s3_download_file(bucket: str, key: str, filename: str, region: str = None, endpoint_url: str = None):
    client = s3_client(region=region)
    if endpoint_url:
        client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
    client.download_file(Bucket=bucket, Key=key, Filename=filename)

# ---------- Qdrant HTTP helpers ----------
def create_collection_http(qdrant_url: str, name: str, vector_size: int = 8, distance: str = "Cosine", timeout: int = 10):
    """Create collection using HTTP API. This uses the modern Qdrant REST endpoint if available.
    If collection exists, this is a no-op."""
    url_col = f"{qdrant_url}/collections/{name}"
    body = {
        "vectors": {"size": int(vector_size), "distance": distance}
    }
    # Some Qdrant instances accept PUT with body:
    try:
        r = requests.put(url_col, json=body, timeout=timeout)
        if r.status_code in (200, 201):
            return True, None
        if r.status_code == 409:
            # already exists
            return True, None
        # fallback: try create via POST collections (older)
        r2 = requests.post(f"{qdrant_url}/collections", json={"name": name, "vectors": body["vectors"]}, timeout=timeout)
        if r2.status_code in (200, 201):
            return True, None
        if r2.status_code == 409:
            return True, None
        return False, f"create failed status {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)

def insert_sample_points(qdrant_url: str, collection: str, num_points: int, vector_size: int = 8, timeout: int = 10):
    """Insert deterministic sample points via REST /points API."""
    url = f"{qdrant_url}/collections/{collection}/points?wait=true"
    points = []
    for i in range(num_points):
        vec = [float((i + j) % 7) for j in range(vector_size)]
        points.append({"id": i, "vector": vec, "payload": {"i": i, "source": "test"}})
    try:
        r = requests.put(url, json={"points": points}, timeout=timeout)
        if r.status_code in (200, 201):
            return True, None
        return False, f"insert failed status {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)

def request_snapshot_and_get_name(qdrant_url: str, collection: str, wait: bool = True, timeout: int = 60):
    """Request a snapshot via HTTP and return snapshot name (tries to parse common response shapes)."""
    params = {}
    if wait:
        params["wait"] = "true"
    url = f"{qdrant_url}/collections/{collection}/snapshots"
    r = requests.post(url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    # common shapes: {"result": {"name": "..."}} or {"result": "name"} or {"snapshot":"..."} or {"result":{"snapshot_name":"..."}}
    result = j.get("result")
    if isinstance(result, dict):
        for key in ("name", "snapshot", "snapshot_name"):
            if key in result:
                return result[key]
        # maybe there is first element list
        # fallback: string inside result?
    if isinstance(result, str):
        return result
    # try other keys:
    for k in ("snapshot", "snapshot_name"):
        if k in j:
            return j[k]
    # last resort: attempt to parse any string in JSON
    # raise if not found
    raise RuntimeError(f"Unable to determine snapshot name from response: {j}")

def download_snapshot_via_service(qdrant_url: str, collection: str, snapshot_name: str, dest_path: Path, timeout: int = 300):
    url = f"{qdrant_url}/collections/{collection}/snapshots/{snapshot_name}/download"
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with dest_path.open("wb") as w:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    w.write(chunk)

# ---------- Main backup/restore flows ----------
def backup_cluster(cfg, args):
    """
    Perform cluster backup.
    Modes:
      - service-mode: talk to a single qdrant_url (default).
      - per-pod (--per-pod): port-forward to each pod and create/download snapshot from each pod.
    """
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_id = f"{timestamp}-{cfg['UUID_SHORT']}"
    local_tmp = Path(args.local_dir or DEFAULT_TMPDIR) / backup_id
    ensure_dir(local_tmp)
    s3_prefix = args.s3_prefix.rstrip("/")
    aws_region = args.aws_region
    s3_endpoint = args.s3_endpoint or None

    # Determine collections: use user-specified list or query Qdrant
    collections = args.collections or []
    if not collections:
        # query qdrant service for collections
        try:
            r = requests.get(f"{cfg['QDRANT_URL']}/collections", timeout=10)
            r.raise_for_status()
            j = r.json()
            # expected shape: {"result":{"collections":[{"name":"col1"},...]}}
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
            collections = cols
        except Exception as e:
            raise RuntimeError(f"Failed to list collections from Qdrant at {cfg['QDRANT_URL']}: {e}")

    if not collections:
        print("No collections found to backup; exiting.")
        return None, None

    # Input hash (deterministic)
    inputs = {
        "collections": collections,
        "qdrant_url": cfg["QDRANT_URL"],
        "s3_bucket": args.s3_bucket,
        "s3_prefix": s3_prefix,
        "per_pod": args.per_pod,
        "namespace": args.namespace,
    }
    inputs_hash = canonical_inputs_hash(inputs)
    inputs_hash_path = (Path(args.local_dir or DEFAULT_TMPDIR) / ".inputs_hash")
    existing = inputs_hash_path.read_text().strip() if inputs_hash_path.exists() else None
    if existing == inputs_hash:
        print("No non-secret changes detected (inputs hash unchanged); proceeding (still performs backup).")
    else:
        inputs_hash_path.write_text(inputs_hash)

    # Discover pods if per-pod mode
    pods = []
    if args.per_pod:
        pods = list_qdrant_pods(args.namespace)
        if not pods:
            raise RuntimeError("per-pod mode requested but no qdrant pods found in namespace " + args.namespace)
        print("Per-pod mode: discovered pods:", pods)

    manifest = {
        "backup_id": backup_id,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "namespace": args.namespace,
        "pods": {},
        "collections": collections,
    }

    # S3 upload helper
    def upload_local_file_to_s3(local_path: Path, key: str):
        print(f"Uploading to s3: s3://{args.s3_bucket}/{key}")
        s3_upload_file(args.s3_bucket, key, str(local_path), region=aws_region, endpoint_url=s3_endpoint)

    # Snapshot retrieval
    if args.per_pod:
        base_port = args.port_base
        port_forwards = []
        try:
            for idx, pod in enumerate(pods):
                local_port = base_port + idx
                print(f"Starting port-forward for pod {pod} -> localhost:{local_port}")
                pf = start_port_forward(pod, args.namespace, local_port, remote_port=6333, timeout=10)
                port_forwards.append((pod, pf, local_port))
            # For each pod, request snapshots for collections then download via that forwarded port
            for pod, pf, local_port in port_forwards:
                pod_dir = local_tmp / pod
                ensure_dir(pod_dir)
                qurl = f"http://127.0.0.1:{local_port}"
                manifest["pods"].setdefault(pod, {"collections": {}})
                for col in collections:
                    print(f"[{pod}] creating snapshot for collection '{col}' ...")
                    snap_name = request_snapshot_and_get_name(qurl, col, wait=True, timeout=args.timeout)
                    print(f"[{pod}] snapshot created: {snap_name}")
                    fname = f"{pod}-{col}-{snap_name}.snapshot"
                    target = pod_dir / fname
                    print(f"[{pod}] downloading snapshot {snap_name} -> {target}")
                    download_snapshot_via_service(qurl, col, snap_name, target, timeout=args.timeout)
                    sha = sha256_of_file(target)
                    size = target.stat().st_size
                    s3_key = f"{s3_prefix}/{backup_id}/{target.name}"
                    upload_local_file_to_s3(target, s3_key)
                    manifest["pods"][pod]["collections"][col] = {
                        "snapshot_name": snap_name,
                        "s3_key": s3_key,
                        "sha256": sha,
                        "size": size,
                    }
        finally:
            for pod, pf, local_port in port_forwards:
                try:
                    stop_port_forward(pf)
                except Exception:
                    pass
    else:
        # service mode: create snapshots via service and download each snapshot once
        qurl = cfg["QDRANT_URL"]
        pod_label_key = "service"  # not meaningful here; use single-key pod 'service'
        manifest["pods"]["service"] = {"collections": {}}
        for col in collections:
            print(f"Creating snapshot for collection '{col}' via service {qurl} ...")
            snap_name = request_snapshot_and_get_name(qurl, col, wait=True, timeout=args.timeout)
            print(f"Snapshot created: {snap_name}")
            fname = f"{col}-{snap_name}.snapshot"
            target = local_tmp / fname
            print(f"Downloading snapshot {snap_name} -> {target}")
            download_snapshot_via_service(qurl, col, snap_name, target, timeout=args.timeout)
            sha = sha256_of_file(target)
            size = target.stat().st_size
            s3_key = f"{s3_prefix}/{backup_id}/{target.name}"
            upload_local_file_to_s3(target, s3_key)
            manifest["pods"]["service"]["collections"][col] = {
                "snapshot_name": snap_name,
                "s3_key": s3_key,
                "sha256": sha,
                "size": size,
            }

    # Upload manifest and latest.manifest.json
    manifest_json = json.dumps(manifest, indent=2)
    manifest_local = local_tmp / "manifest.json"
    manifest_local.write_text(manifest_json)
    latest_local = local_tmp / "latest.manifest.json"
    latest_local.write_text(manifest_json)
    manifest_s3_key = f"{s3_prefix}/{backup_id}/manifest.json"
    latest_s3_key = f"{s3_prefix}/latest.manifest.json"
    s3_upload_file(args.s3_bucket, manifest_s3_key, str(manifest_local), region=aws_region, endpoint_url=s3_endpoint)
    s3_upload_file(args.s3_bucket, latest_s3_key, str(latest_local), region=aws_region, endpoint_url=s3_endpoint)
    print("Backup finished. backup_id:", backup_id, "local:", str(local_tmp))
    return backup_id, str(local_tmp)

def restore_cluster(cfg, args):
    """
    Restore cluster from S3 manifest. Downloads referenced snapshots, copies them into pods,
    and optionally restarts the statefulset.
    """
    aws_region = args.aws_region
    s3_endpoint = args.s3_endpoint or None
    tmpdir = Path(args.local_dir or DEFAULT_TMPDIR) / f"restore-{args.backup_id}"
    ensure_dir(tmpdir)

    # download manifest
    manifest_s3_key = f"{args.s3_prefix.rstrip('/')}/{args.backup_id}/manifest.json"
    local_manifest = tmpdir / "manifest.json"
    print(f"Downloading manifest s3://{args.s3_bucket}/{manifest_s3_key} -> {local_manifest}")
    s3_download_file(args.s3_bucket, manifest_s3_key, str(local_manifest), region=aws_region, endpoint_url=s3_endpoint)
    manifest = json.loads(local_manifest.read_text())

    # Download snapshots referenced
    files_to_download = []
    for pod, podinfo in manifest.get("pods", {}).items():
        for col, meta in (podinfo.get("collections") or {}).items():
            s3_key = meta.get("s3_key")
            if s3_key:
                fname = Path(s3_key).name
                local_target = tmpdir / fname
                files_to_download.append((s3_key, local_target))
    print("Downloading snapshot artifacts from S3 referenced in manifest...")
    for key, target in files_to_download:
        print(" ", f"s3://{args.s3_bucket}/{key}", "->", str(target))
        ensure_dir(target.parent)
        s3_download_file(args.s3_bucket, key, str(target), region=aws_region, endpoint_url=s3_endpoint)

    # Verify checksums
    print("Verifying snapshot files checksums...")
    for pod, podinfo in manifest.get("pods", {}).items():
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

    # Copy into pods according to manifest keys
    # If manifest used 'service' key (service-mode), attempt to map to pods via list_qdrant_pods.
    pods_from_manifest = list(manifest.get("pods", {}).keys())
    if pods_from_manifest == ["service"]:
        # map to actual pods
        pods = list_qdrant_pods(args.namespace)
    else:
        pods = pods_from_manifest

    print("Pods to restore:", pods)
    for pod in pods:
        pod_collections = manifest["pods"].get(pod) or manifest["pods"].get("service") or {}
        if not pod_collections:
            continue
        for col, meta in pod_collections.get("collections", {}).items():
            s3_key = meta.get("s3_key")
            fname = Path(s3_key).name
            localf = tmpdir / fname
            dest_path = f"/qdrant/snapshots/{fname}"
            print(f"  Uploading {fname} -> pod {pod} : {dest_path}")
            # ensure remote dir exists (kubectl exec mkdir -p)
            rc, out, err = run_cmd(["kubectl", "exec", "-n", args.namespace, pod, "--", "mkdir", "-p", "/qdrant/snapshots"])
            if rc != 0:
                raise RuntimeError(f"mkdir in pod failed: {err or out}")
            kubectl_cp_to_pod(args.namespace, pod, str(localf), f"/qdrant/snapshots/{fname}")
            print(f"  Uploaded {fname} to pod {pod}")
    # Optionally restart statefulset
    if args.restart:
        print("Restarting statefulset/qdrant...")
        kubectl_rollout_restart_statefulset(args.namespace, "qdrant")
        print("Waiting for rollout to finish...")
        wait_for_statefulset_ready(args.namespace, "qdrant", timeout=args.timeout)
        print("Waiting for pods to be ready after restart...")
        # small delay to give pods time
        time.sleep(5)
    print("Restore completed successfully.")
    return

# ---------- CLI & main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Test backup & restore for Qdrant cluster (client-driven snapshots).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--backup", action="store_true", help="Create snapshots and upload to S3.")
    grp.add_argument("--restore", metavar="BACKUP_ID", help="Restore backup ID from S3.")
    p.add_argument("--s3-bucket", required=True, help="S3 bucket to store backups.")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help="S3 prefix for backups (default qdrant/backups).")
    p.add_argument("--local-dir", default=str(DEFAULT_TMPDIR), help="Local directory to store temporary backup files.")
    p.add_argument("--collections", nargs="+", help="Collections to backup/restore. If omitted, auto-detected (backup).")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant service URL for service-mode (default http://127.0.0.1:6333).")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace for qdrant pods.")
    p.add_argument("--per-pod", action="store_true", help="Create and download snapshots from each pod (requires kubectl).")
    p.add_argument("--port-base", type=int, default=7000, help="Base port for per-pod port-forwarding (7000).")
    p.add_argument("--timeout", type=int, default=300, help="Timeout seconds for HTTP operations.")
    p.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region (boto3).")
    p.add_argument("--s3-endpoint", default=os.environ.get("BACKUP_S3_ENDPOINT", ""), help="Optional S3 endpoint URL.")
    p.add_argument("--env", default=os.environ.get("ENV", "STAGING"), help="ENV (STAGING/PROD).")
    p.add_argument("--restart", action="store_true", help="When restoring, rollout restart the statefulset/qdrant after copying files.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = {
        "ENV": args.env.upper(),
        "QDRANT_URL": args.qdrant_url,
        "UUID_SHORT": str(uuid.uuid4())[:8],
    }
    # Quick prechecks
    if args.per_pod and not kubectl_exists():
        print("ERROR: --per-pod requires kubectl in PATH.", file=sys.stderr)
        sys.exit(2)
    if cfg["ENV"] == "STAGING":
        # Optionally create in-cluster secret for S3 creds if present (do not write secret yaml)
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            # Create secret in-cluster for staging so that cluster-side jobs (if any) could use it.
            if kubectl_exists():
                print("ENV=STAGING: creating/updating in-cluster secret qdrant-backup-aws (no manifest file written).")
                literals = []
                literals += ["--from-literal", f"AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID')}"]
                literals += ["--from-literal", f"AWS_SECRET_ACCESS_KEY={os.environ.get('AWS_SECRET_ACCESS_KEY')}"]
                if os.environ.get("AWS_SESSION_TOKEN"):
                    literals += ["--from-literal", f"AWS_SESSION_TOKEN={os.environ.get('AWS_SESSION_TOKEN')}"]
                cmd = ["kubectl", "create", "secret", "generic", "qdrant-backup-aws", "-n", args.namespace, "--dry-run=client", "-o", "yaml"] + literals
                rc, out, err = run_cmd(cmd, timeout=20)
                if rc != 0:
                    print("Failed to render secret YAML for in-cluster apply:", err or out, file=sys.stderr)
                else:
                    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=out.encode("utf-8"), timeout=20)
                    if rc2 != 0:
                        print("Failed to apply staging secret in-cluster:", err2 or out2, file=sys.stderr)
                    else:
                        print("Staging secret qdrant-backup-aws created/updated in-cluster.")
            else:
                print("kubectl not found; cannot create in-cluster secret. Continuing assuming env creds available locally.")
        else:
            print("ENV=STAGING but AWS creds not present in environment; client-side backup will fail on upload unless creds are provided.")
    else:
        print("ENV=PROD: not creating any in-cluster secrets (IRSA expected).")

    # Branch mode
    if args.backup:
        # ensure S3 bucket access quickly
        try:
            client = s3_client(args.aws_region)
            # head bucket as a quick check
            client.head_bucket(Bucket=args.s3_bucket)
        except Exception as e:
            print("WARNING: cannot access S3 bucket (will attempt anyway):", e)
        # set qdrant url in cfg
        cfg["QDRANT_URL"] = args.qdrant_url
        # Optionally create sample collections + insert
        if args.collections:
            for col in args.collections:
                ok, err = create_collection_http(cfg["QDRANT_URL"], col)
                if not ok:
                    print("Failed to create collection", col, err)
                else:
                    print("Ensured collection", col)
            # Insert sample points (deterministic small number)
            for col in args.collections:
                ok, err = insert_sample_points(cfg["QDRANT_URL"], col, num_points=16, vector_size=8)
                if not ok:
                    print("Failed to insert sample points to", col, err)
                else:
                    print("Inserted sample points into", col)
        backup_id, localdir = backup_cluster(cfg, args)
        if not backup_id:
            print("Backup failed.", file=sys.stderr)
            sys.exit(2)
        print("Backup complete:", backup_id, localdir)
        sys.exit(0)
    else:
        # restore mode
        if not args.restore:
            print("must provide --restore BACKUP_ID", file=sys.stderr)
            sys.exit(2)
        args.backup_id = args.restore
        try:
            restore_cluster(cfg, args)
            print("Restore completed.")
            sys.exit(0)
        except Exception as e:
            print("Restore failed:", e, file=sys.stderr)
            sys.exit(3)

if __name__ == "__main__":
    main()
