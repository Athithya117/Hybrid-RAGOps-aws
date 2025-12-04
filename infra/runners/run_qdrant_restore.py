#!/usr/bin/env python3
"""
qdrant_restore.py

Restore Qdrant cluster from backups stored in S3. Supports two modes:
  - service/cluster-mode (default): call the Qdrant service endpoint (single URL)
    to recover each collection from a presigned S3 URL (preferred) or fall back
    to streaming upload.
  - per-pod mode (--per-pod): port-forward each qdrant pod and target the node
    directly (useful when using local NVMe / per-node snapshots).

Behavior summary
  * If --backup-id omitted, the script fetches <s3_prefix>/latest.manifest.json
    and uses its backup_id, then downloads <s3_prefix>/<backup_id>/manifest.json.
  * Manifest schema expected (same as backup tool):
      manifest["pods"] -> dict where keys are either "service" (cluster snapshot)
      or pod names (qdrant-0, qdrant-1, ...). Each pod entry contains 'collections'
      mapping collection -> metadata including "s3_key".
  * For each collection entry the script:
      - tries presigned GET URL + PUT /collections/{collection}/snapshots/recover
      - if that fails, streams S3 object to POST /collections/{collection}/snapshots/upload
  * If --per-pod is false, uses a single QDRANT_URL to run recover calls (cluster mode).
  * If --per-pod is true, discovers pods and port-forwards each pod to localhost
    and targets the node endpoints individually.
  * Optionally restarts statefulset/qdrant (--restart) and waits for rollout.

Requirements: Python 3.8+, boto3, requests, kubectl in PATH for port-forward and restart.
"""

from pathlib import Path
import argparse
import os
import sys
import json
import time
import datetime
import tempfile
import subprocess
import shutil
from typing import Optional, Tuple, List

import boto3
import requests

# ---------- Defaults ----------
DEFAULT_S3_PREFIX = "qdrant/backups"
DEFAULT_AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_NAMESPACE = os.environ.get("QDRANT_NAMESPACE", "qdrant")
DEFAULT_PORT_BASE = 7000
PORT_FORWARD_PROBE_PATH = "/collections"
KUBECTL_PROBE_TIMEOUT = 12

# ---------- Utilities ----------
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def run_cmd(cmd, input_bytes: bytes = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, input=input_bytes, timeout=timeout, check=True)
        out = proc.stdout.decode() if proc.stdout else ""
        err = proc.stderr.decode() if proc.stderr else ""
        return proc.returncode, out, err
    except subprocess.CalledProcessError as e:
        out = e.stdout.decode() if getattr(e, "stdout", None) else ""
        err = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
        return e.returncode, out, err
    except Exception as e:
        return 254, "", str(e)

def kubectl_exists() -> bool:
    return shutil.which("kubectl") is not None

# ---------- S3 helpers ----------
def s3_client(region: Optional[str] = None, endpoint_url: Optional[str] = None):
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        # boto3 expects endpoint_url as parameter, not in kwargs for client creation
        return boto3.client("s3", region_name=kwargs.get("region_name"), endpoint_url=endpoint_url)
    return boto3.client("s3", **kwargs)

def download_s3_object_to_file(bucket: str, key: str, target_path: Path, region: Optional[str] = None, endpoint_url: Optional[str] = None):
    client = s3_client(region, endpoint_url)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(Bucket=bucket, Key=key, Filename=str(target_path))

def generate_presigned_get(bucket: str, key: str, expires_in: int = 3600, region: Optional[str] = None, endpoint_url: Optional[str] = None) -> str:
    client = s3_client(region, endpoint_url)
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in)

# ---------- Port-forward helpers ----------
def start_port_forward(pod: str, namespace: str, local_port: int, remote_port: int = 6333, timeout: int = KUBECTL_PROBE_TIMEOUT) -> subprocess.Popen:
    if not kubectl_exists():
        raise RuntimeError("kubectl not found in PATH; required for port-forwarding to pods")
    cmd = ["kubectl", "port-forward", f"pod/{pod}", f"{local_port}:{remote_port}", "-n", namespace]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    probe_url = f"http://127.0.0.1:{local_port}{PORT_FORWARD_PROBE_PATH}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(probe_url, timeout=1)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.25)
    # didn't respond; cleanup and raise
    try:
        proc.kill()
    except Exception:
        pass
    raise RuntimeError(f"port-forward to pod/{pod} on local port {local_port} failed to respond")

def stop_port_forward(proc: subprocess.Popen):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

# ---------- Qdrant HTTP helpers ----------
def qdrant_recover_from_url(base_url: str, collection: str, presigned_url: str, api_key: Optional[str] = None, timeout: int = 600) -> Tuple[bool, str]:
    endpoint = f"{base_url}/collections/{collection}/snapshots/recover"
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    try:
        r = requests.put(endpoint, json={"location": presigned_url}, headers=headers, timeout=timeout)
        if 200 <= r.status_code < 300:
            return True, f"recovered from URL via {endpoint}"
        return False, f"{r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)

def qdrant_upload_snapshot_stream(base_url: str, collection: str, filename: str, fileobj, api_key: Optional[str] = None, timeout: int = 1800) -> Tuple[bool, str]:
    endpoint = f"{base_url}/collections/{collection}/snapshots/upload"
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    files = {"snapshot": (filename, fileobj)}
    try:
        r = requests.post(endpoint, files=files, headers=headers, timeout=timeout)
        if 200 <= r.status_code < 300:
            return True, f"uploaded snapshot via {endpoint}"
        return False, f"{r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)

# ---------- Manifest & discovery ----------
def discover_qdrant_pods(namespace: str) -> List[str]:
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

# ---------- Restore flow ----------
def restore_from_manifest_s3(
    bucket: str,
    s3_prefix: str,
    backup_id: Optional[str],
    namespace: str,
    aws_region: Optional[str],
    s3_endpoint: Optional[str],
    api_key: Optional[str],
    restart: bool,
    per_pod: bool,
    port_base: int = DEFAULT_PORT_BASE,
    wait_seconds: int = 20,
):
    s3_prefix = s3_prefix.rstrip("/")
    client = s3_client(aws_region, s3_endpoint or None)

    # determine backup_id (latest if omitted)
    if not backup_id:
        latest_key = f"{s3_prefix}/latest.manifest.json"
        print(f"Fetching latest manifest to determine backup_id: s3://{bucket}/{latest_key}")
        tmp_latest = Path(tempfile.mkdtemp(prefix="qdrant-restore-")) / "latest.manifest.json"
        try:
            download_s3_object_to_file(bucket, latest_key, tmp_latest, aws_region, s3_endpoint or None)
        except Exception as e:
            raise RuntimeError(f"Failed to download latest.manifest.json from s3://{bucket}/{latest_key}: {e}")
        j = json.loads(tmp_latest.read_text())
        if "backup_id" not in j:
            raise RuntimeError("latest.manifest.json missing backup_id")
        backup_id = j["backup_id"]
        print(f"Selected backup_id = {backup_id} (from latest.manifest.json)")

    # fetch manifest.json
    manifest_key = f"{s3_prefix}/{backup_id}/manifest.json"
    print(f"Downloading manifest s3://{bucket}/{manifest_key}")
    tmpdir = Path(tempfile.mkdtemp(prefix=f"qdrant-restore-{backup_id}-"))
    manifest_local = tmpdir / "manifest.json"
    try:
        download_s3_object_to_file(bucket, manifest_key, manifest_local, aws_region, s3_endpoint or None)
    except Exception as e:
        raise RuntimeError(f"Failed to download manifest {manifest_key}: {e}")

    manifest = json.loads(manifest_local.read_text())
    print(f"Manifest loaded. backup_id={manifest.get('backup_id')} created_at={manifest.get('created_at')}")

    # Determine target pods / mode
    pods_in_manifest = list(manifest.get("pods", {}).keys())
    if pods_in_manifest == ["service"]:
        if per_pod:
            # map to actual pods
            print("Manifest is 'service' (cluster-level). Per-pod restore requested -> discovering pods to target.")
            pods = discover_qdrant_pods(namespace)
            if not pods:
                raise RuntimeError("No qdrant pods discovered to target for per-pod restore")
        else:
            # cluster/service-mode restore: we'll target the provided QDRANT_URL (single endpoint)
            pods = ["service"]
    else:
        # manifest contains explicit pod entries
        pods = pods_in_manifest
        if not per_pod:
            # If manifest has per-pod entries but user asked cluster-mode, we can attempt to map collections
            # into single service restore by treating each collection once (de-duplicate).
            pods = ["service"]

    print("Restore target mode:", "per-pod" if per_pod else "service/cluster")
    print("Pods/targets:", pods)

    # Prepare results storage
    results = {"backup_id": backup_id, "created_at": manifest.get("created_at"), "pods": {}}

    port_forwards = []
    try:
        if per_pod:
            # per-pod: for each discovered pod open a port-forward and restore that pod's listed collections
            discovered_pods = discover_qdrant_pods(namespace)
            if not discovered_pods:
                raise RuntimeError("No qdrant pods discovered for per-pod restore")
            print("Discovered pods:", discovered_pods)
            for idx, pod in enumerate(discovered_pods):
                local_port = port_base + idx
                print(f"Starting port-forward for pod {pod} -> localhost:{local_port}")
                pf = start_port_forward(pod, namespace, local_port, remote_port=6333, timeout=KUBECTL_PROBE_TIMEOUT)
                port_forwards.append(pf)
                base_url = f"http://127.0.0.1:{local_port}"
                # manifest might have per-pod entries or 'service' entry; prefer per-pod metadata if present
                pod_entry = manifest.get("pods", {}).get(pod) or manifest.get("pods", {}).get("service") or {}
                collections = pod_entry.get("collections", {})
                results["pods"].setdefault(pod, {"collections": {}})
                if not collections:
                    print(f"  [skip] no collections metadata for pod {pod} in manifest")
                    continue
                for col, meta in collections.items():
                    s3_key = meta.get("s3_key")
                    if not s3_key:
                        print(f"  [skip] no s3_key for {pod}/{col}")
                        continue
                    print(f"  [{pod}] Restoring collection '{col}' from s3://{bucket}/{s3_key}")
                    ok, detail = _attempt_restore_to_target(
                        bucket, s3_key, aws_region, s3_endpoint, base_url, col, api_key
                    )
                    results["pods"][pod]["collections"][col] = {**meta, "_restore_status": {"ok": ok, "detail": detail}}
                    print(f"    [{ 'ok' if ok else 'fail' }] {detail}")
                print(f"  Finished restore attempts for pod {pod}")
        else:
            # service/cluster mode: target single service endpoint (args.qdrant_url)
            # pick base_url from env or allow user to pass QDRANT_URL via env var QDRANT_URL
            base_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
            print("Service-mode restore will target Qdrant service URL:", base_url)
            # determine collections to restore: union of all collections in manifest
            collections_map = {}
            for pod_key, podinfo in manifest.get("pods", {}).items():
                for col, meta in (podinfo.get("collections") or {}).items():
                    # if collection already present, keep first (manifest may be per-pod duplicates)
                    if col not in collections_map:
                        collections_map[col] = meta
            results["pods"].setdefault("service", {"collections": {}})
            for col, meta in collections_map.items():
                s3_key = meta.get("s3_key")
                if not s3_key:
                    print(f"  [skip] no s3_key for service/{col}")
                    continue
                print(f"  [service] Restoring collection '{col}' from s3://{bucket}/{s3_key}")
                ok, detail = _attempt_restore_to_target(bucket, s3_key, aws_region, s3_endpoint, base_url, col, api_key)
                results["pods"]["service"]["collections"][col] = {**meta, "_restore_status": {"ok": ok, "detail": detail}}
                print(f"    [{ 'ok' if ok else 'fail' }] {detail}")
    finally:
        for pf in port_forwards:
            try:
                stop_port_forward(pf)
            except Exception:
                pass

    # Optionally restart statefulset
    if restart:
        if not kubectl_exists():
            print("kubectl not found; cannot restart statefulset. Skipping restart.", file=sys.stderr)
        else:
            print("Restarting statefulset/qdrant...")
            rc, out, err = run_cmd(["kubectl", "rollout", "restart", "statefulset/qdrant", "-n", namespace], timeout=60)
            if rc != 0:
                raise RuntimeError(f"Failed to restart statefulset/qdrant: {err or out}")
            # wait for rollout
            deadline = time.time() + 600
            while time.time() < deadline:
                rc2, out2, err2 = run_cmd(["kubectl", "rollout", "status", "statefulset/qdrant", "-n", namespace], timeout=30)
                if rc2 == 0:
                    print("Rollout finished.")
                    break
                time.sleep(3)
            else:
                raise RuntimeError("Timeout waiting for statefulset rollout to finish")
            print("Waiting additional seconds for restored data to be available...")
            time.sleep(wait_seconds)

    print("\nRestore summary:")
    print(json.dumps(results, indent=2))
    print("Restore completed at", now_iso())
    return results

# helper that tries presigned URL first, falls back to streaming
def _attempt_restore_to_target(bucket: str, s3_key: str, aws_region: Optional[str], s3_endpoint: Optional[str],
                               base_url: str, collection: str, api_key: Optional[str]) -> Tuple[bool, str]:
    presigned = None
    try:
        presigned = generate_presigned_get(bucket, s3_key, expires_in=3600, region=aws_region, endpoint_url=s3_endpoint or None)
    except Exception as e:
        print(f"    [warn] Failed to generate presigned URL for s3://{bucket}/{s3_key}: {e}")
        presigned = None

    if presigned:
        ok, detail = qdrant_recover_from_url(base_url, collection, presigned, api_key=api_key, timeout=600)
        if ok:
            return True, detail
        else:
            print(f"    [warn] recover-from-url failed: {detail}")

    # fallback: stream S3 object to node upload endpoint
    print("    [info] Falling back to streaming S3 object to node's upload endpoint (may be slower).")
    s3 = s3_client(aws_region, s3_endpoint or None)
    try:
        resp = s3.get_object(Bucket=bucket, Key=s3_key)
        body = resp["Body"]  # StreamingBody
        fname = Path(s3_key).name
        ok2, detail2 = qdrant_upload_snapshot_stream(base_url, collection, fname, body, api_key=api_key, timeout=1800)
        return ok2, detail2
    except Exception as e:
        return False, f"streaming-from-s3-failed: {e}"

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Restore Qdrant cluster from S3 manifest. Supports --per-pod or service/cluster mode.")
    p.add_argument("--s3-bucket", required=True, help="S3 bucket where backups are stored.")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help="S3 prefix for backups (default qdrant/backups).")
    p.add_argument("--backup-id", default="", help="Backup ID to restore (if omitted, uses latest.manifest.json).")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace for qdrant.")
    p.add_argument("--aws-region", default=DEFAULT_AWS_REGION, help="AWS region for S3 client.")
    p.add_argument("--s3-endpoint", default=os.environ.get("BACKUP_S3_ENDPOINT", ""), help="Optional S3 endpoint URL.")
    p.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT__SERVICE__API_KEY", ""), help="Optional Qdrant API key for auth (header 'api-key').")
    p.add_argument("--restart", action="store_true", help="Rollout restart statefulset/qdrant after restore and wait for readiness.")
    p.add_argument("--per-pod", action="store_true", help="Use per-pod port-forwarded restore (target pods individually).")
    p.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE, help="Base local port used for port-forwarding (7000).")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        res = restore_from_manifest_s3(
            bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            backup_id=args.backup_id or None,
            namespace=args.namespace,
            aws_region=args.aws_region,
            s3_endpoint=args.s3_endpoint or None,
            api_key=args.qdrant_api_key or None,
            restart=args.restart,
            per_pod=args.per_pod,
            port_base=args.port_base,
        )
        print("Restore finished successfully.")
        sys.exit(0)
    except Exception as e:
        print("Restore failed:", e, file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
