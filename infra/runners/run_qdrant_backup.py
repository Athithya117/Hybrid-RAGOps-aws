#!/usr/bin/env python3
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
import requests
try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError, AzureError
except Exception:
    BlobServiceClient = None
def now_isoutc():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
SENSITIVE = {"AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT_KEY"}
def canonical_inputs_hash(obj: dict) -> str:
    serial = {}
    for k in sorted(obj.keys()):
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
def azure_client_from_env():
    if BlobServiceClient is None:
        raise RuntimeError("azure-storage-blob is not installed; pip install azure-storage-blob azure-identity")
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
    if conn:
        return BlobServiceClient.from_connection_string(conn)
    if account and key:
        url = f"https://{account}.blob.core.windows.net"
        return BlobServiceClient(account_url=url, credential=key)
    try:
        from azure.identity import DefaultAzureCredential
        cred = DefaultAzureCredential()
        account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
        if not account:
            raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME required when using DefaultAzureCredential")
        url = f"https://{account}.blob.core.windows.net"
        return BlobServiceClient(account_url=url, credential=cred)
    except Exception as e:
        raise RuntimeError("Azure storage credentials not found; set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME+AZURE_STORAGE_ACCOUNT_KEY or configure DefaultAzureCredential") from e
def azure_upload_file(container: str, blob_key: str, filename: str):
    client = azure_client_from_env()
    try:
        container_client = client.get_container_client(container=container)
        try:
            container_client.create_container()
        except Exception:
            pass
    except Exception:
        pass
    blob_client = client.get_blob_client(container=container, blob=blob_key)
    with open(filename, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
def azure_download_file(container: str, blob_key: str, filename: str):
    client = azure_client_from_env()
    blob_client = client.get_blob_client(container=container, blob=blob_key)
    try:
        downloader = blob_client.download_blob()
    except Exception as e:
        raise RuntimeError(f"azure download failed: {e}")
    ensure_dir(Path(filename).parent)
    with open(filename, "wb") as f:
        for chunk in downloader.chunks():
            if chunk:
                f.write(chunk)
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
    candidates = [
        f"{qdrant_url}/collections/{collection}/snapshots/{snapshot_name}/download",
        f"{qdrant_url}/collections/{collection}/snapshots/{snapshot_name}",
    ]
    last_err = None
    for url in candidates:
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 200:
                    ensure_dir(dest_path.parent)
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
def list_qdrant_pods(namespace: str) -> List[str]:
    if not kubectl_exists():
        return []
    selectors = ["app.kubernetes.io/name=qdrant","app=qdrant","app.kubernetes.io/instance=qdrant"]
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
def create_staging_secret_in_cluster(namespace: str) -> bool:
    if not kubectl_exists():
        print("kubectl not found; cannot create in-cluster secret. Proceeding without creating secret.", file=sys.stderr)
        return False
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    if not (conn or (account and key)):
        print("Azure credentials not present in environment; skipping staging secret creation.", file=sys.stderr)
        return False
    literals = []
    if conn:
        literals += ["--from-literal", f"AZURE_STORAGE_CONNECTION_STRING={conn}"]
    if account:
        literals += ["--from-literal", f"AZURE_STORAGE_ACCOUNT_NAME={account}"]
    if key:
        literals += ["--from-literal", f"AZURE_STORAGE_ACCOUNT_KEY={key}"]
    cmd = ["kubectl", "create", "secret", "generic", "qdrant-backup-azure", "-n", namespace, "--dry-run=client", "-o", "yaml"] + literals
    rc, out, err = run_cmd(cmd, timeout=20)
    if rc != 0:
        print("Failed to render secret YAML for in-cluster apply:", err or out, file=sys.stderr)
        return False
    rc2, out2, err2 = run_cmd(["kubectl", "apply", "-f", "-"], input_bytes=out.encode("utf-8"), timeout=20)
    if rc2 != 0:
        print("Failed to apply staging secret in-cluster:", err2 or out2, file=sys.stderr)
        return False
    return True
def backup_cluster_service_mode(args, cfg):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_id = f"{timestamp}-{cfg['UUID_SHORT']}"
    local_tmp = Path(args.local_dir or "tmp") / backup_id
    ensure_dir(local_tmp)
    prefix = args.azure_prefix.rstrip("/")
    azure_container = args.azure_container or None
    if not azure_container:
        raise RuntimeError("azure container (--azure-container) is required for backup")
    collections = args.collections or list_collections(cfg["QDRANT_URL"], timeout=args.timeout)
    if not collections:
        raise RuntimeError("No collections found to backup.")
    inputs = {"collections": collections, "qdrant_url": cfg["QDRANT_URL"], "azure_container": azure_container, "azure_prefix": prefix, "mode": "service", "namespace": args.namespace}
    inputs_hash = canonical_inputs_hash(inputs)
    inputs_hash_path = Path(args.local_dir or "tmp") / ".inputs_hash"
    existing = inputs_hash_path.read_text().strip() if inputs_hash_path.exists() else None
    if existing == inputs_hash:
        print("No non-secret changes detected (inputs hash unchanged); proceeding (still performs backup).")
    else:
        inputs_hash_path.write_text(inputs_hash)
    manifest = {"backup_id": backup_id, "created_at": datetime.datetime.utcnow().isoformat() + "Z", "namespace": args.namespace, "pods": {"service": {"collections": {}}}, "collections": collections, "mode": "service"}
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
        blob_path = f"{prefix}/{backup_id}/{target.name}"
        print(f"Uploading to Azure Blob: container={azure_container} blob={blob_path}")
        azure_upload_file(azure_container, blob_path, str(target))
        manifest["pods"]["service"]["collections"][col] = {"snapshot_name": snap_name, "azure_container": azure_container, "blob_path": blob_path, "sha256": sha, "size": size, "local_path": str(target)}
    manifest_json = json.dumps(manifest, indent=2)
    manifest_local = local_tmp / "manifest.json"
    manifest_local.write_text(manifest_json)
    latest_local = local_tmp / "latest.manifest.json"
    latest_local.write_text(manifest_json)
    manifest_blob_path = f"{prefix}/{backup_id}/manifest.json"
    latest_blob_path = f"{prefix}/latest.manifest.json"
    azure_upload_file(azure_container, manifest_blob_path, str(manifest_local))
    azure_upload_file(azure_container, latest_blob_path, str(latest_local))
    print("\n=== BACKUP MANIFEST ===")
    print(manifest_json)
    print("=== END MANIFEST ===\n")
    print("Backup finished. backup_id:", backup_id, "local:", str(local_tmp))
    return backup_id, str(local_tmp)
def backup_cluster_per_pod(args, cfg):
    if not kubectl_exists():
        raise RuntimeError("--per-pod requires kubectl in PATH.")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_id = f"{timestamp}-{cfg['UUID_SHORT']}"
    local_tmp = Path(args.local_dir or "tmp") / backup_id
    ensure_dir(local_tmp)
    prefix = args.azure_prefix.rstrip("/")
    azure_container = args.azure_container or None
    if not azure_container:
        raise RuntimeError("azure container (--azure-container) is required for backup")
    pods = list_qdrant_pods(args.namespace)
    if not pods:
        raise RuntimeError("No qdrant pods found in namespace: " + args.namespace)
    collections = args.collections or list_collections(cfg["QDRANT_URL"], timeout=args.timeout)
    if not collections:
        raise RuntimeError("No collections found to backup.")
    inputs = {"collections": collections, "qdrant_url": cfg["QDRANT_URL"], "azure_container": azure_container, "azure_prefix": prefix, "mode": "per-pod", "namespace": args.namespace, "pods": pods}
    inputs_hash = canonical_inputs_hash(inputs)
    inputs_hash_path = Path(args.local_dir or "tmp") / ".inputs_hash"
    existing = inputs_hash_path.read_text().strip() if inputs_hash_path.exists() else None
    if existing == inputs_hash:
        print("No non-secret changes detected (inputs hash unchanged); proceeding (still performs backup).")
    else:
        inputs_hash_path.write_text(inputs_hash)
    manifest = {"backup_id": backup_id, "created_at": datetime.datetime.utcnow().isoformat() + "Z", "namespace": args.namespace, "pods": {}, "collections": collections, "mode": "per-pod"}
    port_forwards = []
    try:
        for idx, pod in enumerate(pods):
            local_port = args.port_base + idx
            print(f"Starting port-forward for pod {pod} -> localhost:{local_port}")
            pf = start_port_forward(pod, args.namespace, local_port, remote_port=6333, timeout=10)
            port_forwards.append((pod, pf, local_port))
        for pod, pf, local_port in port_forwards:
            pod_dir = local_tmp / pod
            ensure_dir(pod_dir)
            qurl = f"http://127.0.0.1:{local_port}"
            manifest["pods"].setdefault(pod, {"collections": {}})
            for col in collections:
                pod_file_path = None
                print(f"[{pod}] requesting snapshot for collection '{col}' ...")
                snap_name = request_snapshot_and_get_name(qurl, col, wait=True, timeout=args.timeout)
                print(f"[{pod}] snapshot created: {snap_name}")
                fname = f"{pod}-{col}-{snap_name}.snapshot"
                target = pod_dir / fname
                try:
                    print(f"[{pod}] attempting HTTP download via {qurl} ...")
                    download_snapshot_via_service(qurl, col, snap_name, target, timeout=args.timeout)
                    print(f"[{pod}] downloaded snapshot via HTTP -> {target}")
                except Exception as e_http:
                    print(f"[{pod}] HTTP download failed: {e_http!s}; attempting kubectl cp from pod filesystem...")
                    pod_snap_dir = f"/qdrant/snapshots/{col}"
                    files = kubectl_list_files_in_pod(args.namespace, pod, pod_snap_dir)
                    candidate = None
                    for f in files:
                        if snap_name in f:
                            candidate = f
                            pod_file_path = f"{pod_snap_dir}/{f}"
                            break
                    if not candidate:
                        files2 = kubectl_list_files_in_pod(args.namespace, pod, "/qdrant/snapshots")
                        for f in files2:
                            if snap_name in f:
                                candidate = f
                                pod_file_path = f"/qdrant/snapshots/{f}"
                                break
                    if not candidate:
                        raise RuntimeError(f"[{pod}] Unable to locate snapshot file in pod filesystem for snapshot {snap_name}. ls {pod_snap_dir} -> {files}")
                    print(f"[{pod}] copying snapshot from pod path {pod_file_path} -> {target}")
                    kubectl_cp_from_pod(args.namespace, pod, pod_file_path, target)
                sha = sha256_of_file(target)
                size = target.stat().st_size
                blob_path = f"{prefix}/{backup_id}/{target.name}"
                print(f"[{pod}] uploading to Azure Blob: container={azure_container} blob={blob_path}")
                azure_upload_file(azure_container, blob_path, str(target))
                entry = {"snapshot_name": snap_name, "azure_container": azure_container, "blob_path": blob_path, "sha256": sha, "size": size, "local_path": str(target)}
                if pod_file_path:
                    entry["pod_path"] = pod_file_path
                manifest["pods"][pod]["collections"][col] = entry
    finally:
        for pod, pf, local_port in port_forwards:
            try:
                stop_port_forward(pf)
            except Exception:
                pass
    manifest_json = json.dumps(manifest, indent=2)
    manifest_local = local_tmp / "manifest.json"
    manifest_local.write_text(manifest_json)
    latest_local = local_tmp / "latest.manifest.json"
    latest_local.write_text(manifest_json)
    manifest_blob_path = f"{prefix}/{backup_id}/manifest.json"
    latest_blob_path = f"{prefix}/latest.manifest.json"
    azure_upload_file(azure_container, manifest_blob_path, str(manifest_local))
    azure_upload_file(azure_container, latest_blob_path, str(latest_local))
    print("\n=== BACKUP MANIFEST ===")
    print(manifest_json)
    print("=== END MANIFEST ===\n")
    print("Backup finished. backup_id:", backup_id, "local:", str(local_tmp))
    return backup_id, str(local_tmp)
def restore_cluster(args, cfg):
    tmpdir = Path(args.local_dir or "tmp") / f"restore-{args.restore}"
    ensure_dir(tmpdir)
    prefix = args.azure_prefix.rstrip("/")
    azure_container = args.azure_container or None
    if not azure_container:
        raise RuntimeError("azure container (--azure-container) is required for restore")
    manifest_blob_path = f"{prefix}/{args.restore}/manifest.json"
    local_manifest = tmpdir / "manifest.json"
    print(f"Downloading manifest azure://{azure_container}/{manifest_blob_path} -> {local_manifest}")
    azure_download_file(azure_container, manifest_blob_path, str(local_manifest))
    manifest = json.loads(local_manifest.read_text())
    files_to_download = []
    for pod_key, podinfo in manifest.get("pods", {}).items():
        for col, meta in (podinfo.get("collections") or {}).items():
            blob_path = meta.get("blob_path")
            if blob_path:
                fname = Path(blob_path).name
                local_target = tmpdir / fname
                files_to_download.append((blob_path, local_target))
    print(f"Downloading {len(files_to_download)} snapshot files referenced in manifest...")
    for key, target in files_to_download:
        print(" ", f"azure://{azure_container}/{key}", "->", str(target))
        ensure_dir(target.parent)
        azure_download_file(azure_container, key, str(target))
    print("Verifying snapshot files checksums...")
    for pod_key, podinfo in manifest.get("pods", {}).items():
        for col, meta in (podinfo.get("collections") or {}).items():
            blob_path = meta.get("blob_path")
            if not blob_path:
                continue
            fname = Path(blob_path).name
            localf = tmpdir / fname
            if not localf.exists():
                raise RuntimeError(f"Missing expected snapshot file: {localf}")
            actual = sha256_of_file(localf)
            expected = meta.get("sha256")
            if expected and actual != expected:
                raise RuntimeError(f"Checksum mismatch for {localf}: expected {expected} actual {actual}")
    print("Checksums OK.")
    pod_keys = list(manifest.get("pods", {}).keys())
    if pod_keys == ["service"]:
        pods = list_qdrant_pods(args.namespace)
    else:
        pods = pod_keys
    if not pods:
        raise RuntimeError("No qdrant pods found to restore into.")
    print("Pods to restore:", pods)
    for pod in pods:
        pod_collections = manifest["pods"].get(pod) or manifest["pods"].get("service") or {}
        if not pod_collections:
            continue
        for col, meta in pod_collections.get("collections", {}).items():
            blob_path = meta.get("blob_path")
            if not blob_path:
                continue
            fname = Path(blob_path).name
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
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            rc, out, err = run_cmd(["kubectl", "rollout", "status", f"statefulset/qdrant", "-n", args.namespace], timeout=30)
            if rc == 0:
                break
            time.sleep(2)
        time.sleep(5)
    print("Restore completed successfully.")
    return
def parse_args():
    p = argparse.ArgumentParser(description="Qdrant backup & restore utility (Azure-native).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--backup", action="store_true", help="Create snapshots and upload to Azure Blob.")
    grp.add_argument("--restore", metavar="BACKUP_ID", help="Restore backup ID from Azure Blob.")
    p.add_argument("--azure-container", required=False, help="Azure container to store backups.")
    p.add_argument("--azure-prefix", default="qdrant/backups", help="Azure prefix for backups.")
    p.add_argument("--local-dir", default=str("tmp"), help="Local directory to store temporary backup files.")
    p.add_argument("--collections", nargs="+", help="Collections to backup. If omitted, auto-detected.")
    p.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"), help="Qdrant service URL.")
    p.add_argument("--namespace", default=os.environ.get("QDRANT_NAMESPACE", "qdrant"), help="Kubernetes namespace where qdrant runs.")
    p.add_argument("--timeout", type=int, default=300, help="Timeout seconds for HTTP/storage operations.")
    p.add_argument("--env", default=os.environ.get("ENV", "STAGING"), help="ENV (STAGING/PROD).")
    p.add_argument("--restart", action="store_true", help="When restoring, rollout restart the statefulset/qdrant after copying files.")
    p.add_argument("--per-pod", action="store_true", help="Use per-pod snapshot & filesystem copy (required for node-local NVMe).")
    p.add_argument("--port-base", type=int, default=7000, help="Base port for per-pod port-forwarding (7000).")
    return p.parse_args()
def main():
    args = parse_args()
    cfg = {"ENV": args.env.upper(), "QDRANT_URL": args.qdrant_url, "UUID_SHORT": str(uuid.uuid4())[:8]}
    if cfg["ENV"] == "STAGING":
        print("ENV=STAGING: will create/update in-cluster secret qdrant-backup-azure if Azure creds present.")
        created = create_staging_secret_in_cluster(args.namespace)
        if created:
            print("Staging secret qdrant-backup-azure created/updated in-cluster.")
        else:
            print("Staging secret not created/updated (check messages).")
    else:
        print("ENV=PROD: no in-cluster secret created (expect managed identity).")
    try:
        try:
            client = azure_client_from_env()
            try:
                client.get_service_properties()
            except Exception:
                print("WARNING: cannot access Azure Blob account (will attempt anyway).")
        except Exception as e:
            print("WARNING: azure client not configured:", e)
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
            args.restore = args.restore
            restore_cluster(args, cfg)
            print("Restore completed.")
            sys.exit(0)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(3)
if __name__ == "__main__":
    main()
