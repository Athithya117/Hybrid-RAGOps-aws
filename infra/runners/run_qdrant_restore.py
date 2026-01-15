#!/usr/bin/env python3
"""
Restore Qdrant cluster from Azure Blob backup manifest.

Environment variables (used when CLI args are not provided):
  BACKUP_AZ_CONTAINER   - container name holding backups (required unless --azure-container supplied)
  BACKUP_PREFIX         - prefix inside container where backups live (default "qdrant/backup")
  BACKUP_ID             - specific backup id to restore (optional; if omitted uses latest.manifest.json)
  PER_POD               - "true"/"1" to enable per-pod port-forward restore (otherwise service mode)
  QDRANT__SERVICE__API_KEY - optional Qdrant API key for auth
  AZURE_*               - standard Azure auth environment (connection string or account/key or DefaultAzureCredential)
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

import requests

# Azure optional imports
try:
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    from azure.core.exceptions import ResourceNotFoundError
except Exception:
    BlobServiceClient = None
    generate_blob_sas = None
    BlobSasPermissions = None

DEFAULT_AZURE_PREFIX = os.environ.get("BACKUP_PREFIX", "qdrant/backup")
DEFAULT_NAMESPACE = os.environ.get("QDRANT_NAMESPACE", "qdrant")
DEFAULT_PORT_BASE = 7000
PORT_FORWARD_PROBE_PATH = "/collections"
KUBECTL_PROBE_TIMEOUT = 12


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


def azure_client_from_env():
    if BlobServiceClient is None:
        raise RuntimeError("azure-storage-blob not installed; pip install azure-storage-blob azure-identity")
    # Preferred: connection string
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "") or os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
    if conn:
        return BlobServiceClient.from_connection_string(conn)
    if account and key:
        url = f"https://{account}.blob.core.windows.net"
        return BlobServiceClient(account_url=url, credential=key)
    # Attempt DefaultAzureCredential flow
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        cred = DefaultAzureCredential()
        if not account:
            raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME required when using DefaultAzureCredential")
        url = f"https://{account}.blob.core.windows.net"
        return BlobServiceClient(account_url=url, credential=cred)
    except Exception as e:
        raise RuntimeError(
            "Azure credentials not found; set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME+AZURE_STORAGE_ACCOUNT_KEY or configure DefaultAzureCredential"
        ) from e


def azure_download_blob_to_file(container: str, blob_key: str, target_path: Path):
    client = azure_client_from_env()
    blob = client.get_blob_client(container=container, blob=blob_key)
    try:
        downloader = blob.download_blob()
    except Exception as e:
        raise RuntimeError(f"blob not found: azure://{container}/{blob_key}") from e
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as f:
        for chunk in downloader.chunks():
            if chunk:
                f.write(chunk)


def azure_generate_blob_sas_url(container: str, blob_key: str, expiry_seconds: int = 3600) -> str:
    if generate_blob_sas is None or BlobSasPermissions is None:
        raise RuntimeError("azure-storage-blob SAS helpers not available; update package")
    account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "") or os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
    start = datetime.datetime.utcnow() - datetime.timedelta(minutes=5)
    expiry = start + datetime.timedelta(seconds=expiry_seconds)
    # If account + key available, create account SAS
    if account and key:
        sas = generate_blob_sas(
            account_name=account,
            container_name=container,
            blob_name=blob_key,
            account_key=key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            start=start,
        )
        endpoint = os.environ.get("AZURE_STORAGE_BLOB_ENDPOINT", f"https://{account}.blob.core.windows.net")
        return f"{endpoint}/{container}/{blob_key}?{sas}"
    # Otherwise use user delegation (requires DefaultAzureCredential)
    client = azure_client_from_env()
    udeleg = client.get_user_delegation_key(start, expiry)
    sas = generate_blob_sas(
        account_name=client.account_name,
        container_name=container,
        blob_name=blob_key,
        user_delegation_key=udeleg,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
        start=start,
    )
    endpoint = os.environ.get("AZURE_STORAGE_BLOB_ENDPOINT", f"https://{client.account_name}.blob.core.windows.net")
    return f"{endpoint}/{container}/{blob_key}?{sas}"


def start_port_forward(pod: str, namespace: str, local_port: int, remote_port: int = 6333, timeout: int = KUBECTL_PROBE_TIMEOUT) -> subprocess.Popen:
    if not kubectl_exists():
        raise RuntimeError("kubectl not found")
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


def qdrant_upload_snapshot_file(base_url: str, collection: str, filename: str, file_path: Path, api_key: Optional[str] = None, timeout: int = 1800) -> Tuple[bool, str]:
    endpoint = f"{base_url}/collections/{collection}/snapshots/upload"
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    try:
        with file_path.open("rb") as fh:
            files = {"snapshot": (filename, fh)}
            r = requests.post(endpoint, files=files, headers=headers, timeout=timeout)
            if 200 <= r.status_code < 300:
                return True, f"uploaded snapshot via {endpoint}"
            return False, f"{r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)


def discover_qdrant_pods(namespace: str) -> List[str]:
    if not kubectl_exists():
        return []
    selectors = ["app.kubernetes.io/name=qdrant", "app=qdrant", "app.kubernetes.io/instance=qdrant"]
    for sel in selectors:
        rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-l", sel, "-o", "jsonpath={.items[*].metadata.name}"])
        if rc == 0 and out.strip():
            return out.strip().split()
    rc, out, err = run_cmd(["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"])
    if rc == 0 and out.strip():
        return [n for n in out.strip().split() if n.startswith("qdrant-")]
    return []


def parse_manifest_blob_entry(meta: dict, cli_container: Optional[str]) -> Tuple[str, str]:
    """
    Recognized manifest entry forms:
      - { "s3_key": "container/path/to/blob" }  (older S3 style)
      - { "s3_key": "path/to/blob" } with cli_container provided
      - { "blob_path": "/path/to/blob", "azure_container": "container" }
      - { "azure_container": "container", "blob": "path/to/blob" }
    Returns (container, blob_key)
    """
    if not meta:
        raise RuntimeError("empty manifest entry")
    if "s3_key" in meta and meta.get("s3_key"):
        val = meta["s3_key"]
        if "/" in val:
            parts = val.split("/", 1)
            return parts[0], parts[1]
        if cli_container:
            return cli_container, val
        raise RuntimeError("manifest s3_key lacks container prefix and --azure-container not provided")
    if "blob_path" in meta and meta.get("blob_path"):
        blob = meta["blob_path"]
        if "azure_container" in meta and meta.get("azure_container"):
            return meta["azure_container"], blob.lstrip("/")
        if cli_container:
            return cli_container, blob.lstrip("/")
        raise RuntimeError("manifest contains blob_path but no container; provide --azure-container")
    if "azure_container" in meta and "blob" in meta:
        return meta["azure_container"], meta["blob"]
    # Fallback: if manifest stores a direct path relative to the backup prefix, assume cli_container
    if "path" in meta and meta.get("path"):
        if cli_container:
            return cli_container, meta["path"].lstrip("/")
    raise RuntimeError("manifest entry lacks recognized blob key (s3_key|blob_path|azure_container+blob)")


def _attempt_restore_to_target_azure(container: str, blob_key: str, azure_prefix: str, base_url: str, collection: str, api_key: Optional[str]) -> Tuple[bool, str]:
    """
    Try recover via presigned URL first; if that fails, download locally and POST to upload endpoint.
    """
    presigned = None
    try:
        presigned = azure_generate_blob_sas_url(container, blob_key, expiry_seconds=3600)
    except Exception:
        presigned = None

    if presigned:
        ok, detail = qdrant_recover_from_url(base_url, collection, presigned, api_key=api_key, timeout=600)
        if ok:
            return True, detail

    tmpf = Path(tempfile.mkdtemp(prefix="qdrant-restore-")) / Path(blob_key).name
    try:
        azure_download_blob_to_file(container, blob_key, tmpf)
    except Exception as e:
        return False, f"download-azure-failed: {e}"
    ok2, detail2 = qdrant_upload_snapshot_file(base_url, collection, tmpf.name, tmpf, api_key=api_key, timeout=1800)
    try:
        tmpf.unlink()
    except Exception:
        pass
    return ok2, detail2


def restore_from_manifest_azure(
    container_arg: Optional[str],
    azure_prefix: str,
    backup_id: Optional[str],
    namespace: str,
    api_key: Optional[str],
    restart: bool,
    per_pod: bool,
    port_base: int = DEFAULT_PORT_BASE,
    wait_seconds: int = 20,
):
    azure_prefix = azure_prefix.rstrip("/")
    client = azure_client_from_env()  # validate credentials early

    # Determine backup_id if not provided
    if not backup_id:
        latest_blob = f"{azure_prefix}/latest.manifest.json"
        tmp_latest = Path(tempfile.mkdtemp(prefix="qdrant-restore-")) / "latest.manifest.json"
        try:
            if not container_arg:
                raise RuntimeError("azure container not provided and latest.manifest.json required")
            azure_download_blob_to_file(container_arg, latest_blob, tmp_latest)
        except Exception as e:
            raise RuntimeError(f"Failed to download latest.manifest.json from azure://{container_arg}/{latest_blob}: {e}")
        j = json.loads(tmp_latest.read_text())
        if "backup_id" not in j:
            raise RuntimeError("latest.manifest.json missing backup_id")
        backup_id = j["backup_id"]

    manifest_blob = f"{azure_prefix}/{backup_id}/manifest.json"
    tmpdir = Path(tempfile.mkdtemp(prefix=f"qdrant-restore-{backup_id}-"))
    manifest_local = tmpdir / "manifest.json"
    try:
        if not container_arg:
            raise RuntimeError("azure container (--azure-container or BACKUP_AZ_CONTAINER) is required")
        azure_download_blob_to_file(container_arg, manifest_blob, manifest_local)
    except Exception as e:
        raise RuntimeError(f"Failed to download manifest {manifest_blob}: {e}")

    manifest = json.loads(manifest_local.read_text())
    pods_in_manifest = list(manifest.get("pods", {}).keys())

    # Decide on which pods to target
    if pods_in_manifest == ["service"]:
        if per_pod:
            pods = discover_qdrant_pods(namespace)
            if not pods:
                raise RuntimeError("No qdrant pods discovered to target for per-pod restore")
        else:
            pods = ["service"]
    else:
        pods = pods_in_manifest
        if not per_pod:
            pods = ["service"]

    results = {"backup_id": backup_id, "created_at": manifest.get("created_at"), "pods": {}}
    port_forwards = []
    try:
        if per_pod:
            discovered_pods = discover_qdrant_pods(namespace)
            if not discovered_pods:
                raise RuntimeError("No qdrant pods discovered for per-pod restore")
            for idx, pod in enumerate(discovered_pods):
                local_port = port_base + idx
                pf = start_port_forward(pod, namespace, local_port, remote_port=6333, timeout=KUBECTL_PROBE_TIMEOUT)
                port_forwards.append(pf)
                base_url = f"http://127.0.0.1:{local_port}"
                # Choose manifest entry: prefer exact pod key, then "service"
                pod_entry = manifest.get("pods", {}).get(pod) or manifest.get("pods", {}).get("service") or {}
                collections = pod_entry.get("collections", {}) or {}
                results["pods"].setdefault(pod, {"collections": {}})
                if not collections:
                    continue
                for col, meta in collections.items():
                    if not meta:
                        results["pods"][pod]["collections"][col] = {"_restore_status": {"ok": False, "detail": "empty manifest entry"}}
                        continue
                    try:
                        container, blob_key = parse_manifest_blob_entry(meta, container_arg)
                    except Exception as e:
                        results["pods"][pod]["collections"][col] = {**(meta or {}), "_restore_status": {"ok": False, "detail": str(e)}}
                        continue
                    ok, detail = _attempt_restore_to_target_azure(container, blob_key, azure_prefix, base_url, col, api_key)
                    results["pods"][pod]["collections"][col] = {**meta, "_restore_status": {"ok": ok, "detail": detail}}
        else:
            base_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
            collections_map = {}
            for pod_key, podinfo in manifest.get("pods", {}).items():
                for col, meta in (podinfo.get("collections") or {}).items():
                    if col not in collections_map:
                        collections_map[col] = meta
            results["pods"].setdefault("service", {"collections": {}})
            for col, meta in collections_map.items():
                if not meta:
                    results["pods"]["service"]["collections"][col] = {"_restore_status": {"ok": False, "detail": "empty manifest entry"}}
                    continue
                try:
                    container, blob_key = parse_manifest_blob_entry(meta, container_arg)
                except Exception as e:
                    results["pods"]["service"]["collections"][col] = {**meta, "_restore_status": {"ok": False, "detail": str(e)}}
                    continue
                ok, detail = _attempt_restore_to_target_azure(container, blob_key, azure_prefix, base_url, col, api_key)
                results["pods"]["service"]["collections"][col] = {**meta, "_restore_status": {"ok": ok, "detail": detail}}
    finally:
        for pf in port_forwards:
            try:
                stop_port_forward(pf)
            except Exception:
                pass

    if restart:
        if kubectl_exists():
            rc, out, err = run_cmd(["kubectl", "rollout", "restart", "statefulset/qdrant", "-n", namespace], timeout=60)
            if rc != 0:
                raise RuntimeError(f"Failed to restart statefulset/qdrant: {err or out}")
            deadline = time.time() + 600
            while time.time() < deadline:
                rc2, out2, err2 = run_cmd(["kubectl", "rollout", "status", "statefulset/qdrant", "-n", namespace], timeout=30)
                if rc2 == 0:
                    break
                time.sleep(3)
            time.sleep(wait_seconds)

    print(json.dumps(results, indent=2))
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Restore Qdrant cluster from Azure Blob manifest. Supports --per-pod or service/cluster mode.")
    p.add_argument("--azure-container", required=False, default=os.environ.get("BACKUP_AZ_CONTAINER") or os.environ.get("BACKUP_AZ_CONTAINER".upper()), help="Azure container where backups are stored. (env: BACKUP_AZ_CONTAINER)")
    p.add_argument("--azure-prefix", default=os.environ.get("BACKUP_PREFIX", DEFAULT_AZURE_PREFIX), help=f"Azure prefix for backups (default {DEFAULT_AZURE_PREFIX}). (env: BACKUP_PREFIX)")
    p.add_argument("--backup-id", default=os.environ.get("BACKUP_ID", ""), help="Backup ID to restore (if omitted, uses latest.manifest.json). (env: BACKUP_ID)")
    p.add_argument("--namespace", default=os.environ.get("QDRANT_NAMESPACE", DEFAULT_NAMESPACE), help="Kubernetes namespace for qdrant. (env: QDRANT_NAMESPACE)")
    p.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT__SERVICE__API_KEY", ""), help="Optional Qdrant API key for auth (header 'api-key'). (env: QDRANT__SERVICE__API_KEY)")
    p.add_argument("--restart", action="store_true", help="Rollout restart statefulset/qdrant after restore and wait for readiness.")
    # per-pod flag: CLI wins, otherwise fallback to env var PER_POD
    p.add_argument("--per-pod", action="store_true", help="Use per-pod port-forwarded restore (target pods individually). Can be set via PER_POD env var.")
    p.add_argument("--port-base", type=int, default=int(os.environ.get("PORT_BASE", DEFAULT_PORT_BASE)), help=f"Base local port used for port-forwarding ({DEFAULT_PORT_BASE}). (env: PORT_BASE)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve per_pod from CLI or ENV
    per_pod_env = os.environ.get("PER_POD", "")
    per_pod = args.per_pod or (str(per_pod_env).lower() in ("1", "true", "yes"))

    azure_container = args.azure_container or os.environ.get("BACKUP_AZ_CONTAINER")
    azure_prefix = args.azure_prefix or os.environ.get("BACKUP_PREFIX", DEFAULT_AZURE_PREFIX)
    backup_id = args.backup_id or os.environ.get("BACKUP_ID", "")
    namespace = args.namespace or DEFAULT_NAMESPACE
    qdrant_api_key = args.qdrant_api_key or os.environ.get("QDRANT__SERVICE__API_KEY", "")

    try:
        if BlobServiceClient is None:
            print("ERROR: azure-storage-blob not installed. pip install azure-storage-blob azure-identity", file=sys.stderr)
            sys.exit(2)
        if per_pod and not kubectl_exists():
            print("ERROR: per-pod restore requires kubectl in PATH", file=sys.stderr)
            sys.exit(2)
        if not azure_container:
            print("ERROR: Azure container not specified. Set BACKUP_AZ_CONTAINER or pass --azure-container", file=sys.stderr)
            sys.exit(2)

        res = restore_from_manifest_azure(
            container_arg=azure_container,
            azure_prefix=azure_prefix,
            backup_id=backup_id or None,
            namespace=namespace,
            api_key=qdrant_api_key or None,
            restart=args.restart,
            per_pod=per_pod,
            port_base=args.port_base,
        )
        print("Restore finished successfully.")
        sys.exit(0)
    except Exception as e:
        print("Restore failed:", str(e), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
