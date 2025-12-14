#!/usr/bin/env python3
"""
force_sync_azure_and_local_fs.py

Mirror local <-> Azure Blob (key-auth first - connection string or account key).
Modes:
  --upload         mirror local -> Azure (delete remote orphans)
  --download       mirror Azure -> local (delete local orphans)
  --merge-upload   upload changed only, DO NOT delete remote orphans
  --merge-download download changed only, DO NOT delete local orphans

Deterministic behavior:
 - Sorted iteration for deterministic ordering
 - Pre-validate required envs (fail fast)
 - Dry-run supported
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- imports (fail-fast) ----
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception as e:
    print(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "level": "ERROR",
        "event": "import_failure",
        "msg": "missing dependency 'azure-storage-blob'. Install: pip install azure-storage-blob",
        "exception": str(e)
    }))
    raise SystemExit(2)

# ---- helpers ----
def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def log(level: str, event: str, msg: str, **kwargs) -> None:
    o = {"ts": ts(), "level": level, "event": event, "msg": msg}
    if kwargs:
        o.update(kwargs)
    print(json.dumps(o, default=str), flush=True)

def info(event: str, msg: str, **k): log("INFO", event, msg, **k)
def warn(event: str, msg: str, **k): log("WARN", event, msg, **k)
def error(event: str, msg: str, **k): log("ERROR", event, msg, **k)

# ---- hashing and checksum helpers ----
def compute_hashes(path: str, chunk_size: int = 8 * 1024 * 1024) -> Tuple[str, str]:
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()

def _hex_from_base64(b64: str) -> Optional[str]:
    try:
        raw = base64.b64decode(b64)
        return raw.hex()
    except Exception:
        return None

def _normalize_etag(etag: str) -> str:
    if not etag:
        return ""
    e = etag.strip()
    if e.startswith("W/"):
        e = e[2:]
    e = e.strip('"').strip("'")
    if e.startswith("0x") or e.startswith("0X"):
        e = e[2:]
    return e.lower()

# ---- shell runner for az fallback ----
def run(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"Command not found: {cmd[0]}. Install Azure CLI (az) or provide key/env.")
    out = (proc.stdout or "").strip()
    er = (proc.stderr or "").strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout: {out}\nstderr: {er}")
    return proc.returncode, out, er

# ---- config ----
DEFAULT_PREFIX = os.environ.get("DEFAULT_PREFIX", "data")
LOCAL_BASE = os.environ.get("LOCAL_BASE", "data")
DEFAULT_CONCURRENCY = int(os.environ.get("CONCURRENT_FILES", "4"))
VERIFY_META_RETRIES = int(os.environ.get("VERIFY_META_RETRIES", "3"))
VERIFY_META_SLEEP = float(os.environ.get("VERIFY_META_SLEEP", "0.7"))

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
AZURE_ENDPOINT_SUFFIX = os.environ.get("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP_NAME = os.environ.get("AZURE_RESOURCE_GROUP_NAME")
AZURE_CONTAINER = os.environ.get("AZURE_CONTAINER") or os.environ.get("AZ_CONTAINER")

# ---- validate auth preconditions ----
def validate_auth_preconditions():
    if AZURE_STORAGE_CONNECTION_STRING:
        info("auth", "Using AZURE_STORAGE_CONNECTION_STRING")
        return
    if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
        info("auth", "Using AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY")
        return
    # allow az fallback only when full context exists
    if AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP_NAME and AZURE_STORAGE_ACCOUNT_NAME:
        info("auth", "No key/connstring present — will attempt az CLI fetch of storage key")
        return
    error("auth_missing", "Missing storage auth: set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_KEY, or provide AZURE_SUBSCRIPTION_ID + AZURE_RESOURCE_GROUP_NAME + AZURE_STORAGE_ACCOUNT_NAME")
    raise SystemExit(2)

def fetch_key_via_az(account_name: str, resource_group: str) -> Optional[str]:
    try:
        info("az_fetch", "Attempting to fetch storage account key via az CLI", account=account_name, resource_group=resource_group)
        rc, out, err = run(["az", "storage", "account", "keys", "list", "--account-name", account_name, "--resource-group", resource_group, "-o", "json"], check=True)
        arr = json.loads(out)
        if arr and isinstance(arr, list) and len(arr) > 0:
            key = arr[0].get("value")
            if key:
                info("az_fetch_ok", "Fetched storage account key via az CLI (primary key)")
                return key
        warn("az_fetch_empty", "az returned empty keys array")
        return None
    except Exception as e:
        warn("az_fetch_failed", "Failed to fetch account key via az CLI", error=str(e))
        return None

def get_blob_service_client() -> BlobServiceClient:
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
        account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.{AZURE_ENDPOINT_SUFFIX}"
        return BlobServiceClient(account_url=account_url, credential=AZURE_STORAGE_ACCOUNT_KEY)
    if AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP_NAME and AZURE_STORAGE_ACCOUNT_NAME:
        key = fetch_key_via_az(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME)
        if key:
            account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.{AZURE_ENDPOINT_SUFFIX}"
            return BlobServiceClient(account_url=account_url, credential=key)
    error("auth_missing", "No storage key/connection string available and az fetch failed.")
    raise SystemExit(2)

# ---- adapter ----
class AzureBlobFs:
    def __init__(self, client: BlobServiceClient):
        self.client = client

    def _parse(self, full: str) -> Tuple[str, Optional[str]]:
        parts = full.split("/", 1)
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]

    def find(self, root: str) -> List[str]:
        container, prefix = self._parse(root)
        prefix = (prefix or "").lstrip("/")
        out: List[str] = []
        container_client = self.client.get_container_client(container)
        try:
            for b in container_client.list_blobs(name_starts_with=prefix):
                out.append(f"{container}/{b.name}")
        except Exception as e:
            warn("find_failed", "list_blobs failed", root=root, exception=str(e))
        return out

    def info(self, full: str) -> Dict:
        container, blob = self._parse(full)
        if not blob:
            return {}
        container_client = self.client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob)
        try:
            props = blob_client.get_blob_properties()
            meta = props.metadata or {}
            etag = props.etag
            size = getattr(props, "size", 0) or 0
            content_md5 = None
            cs = getattr(props, "content_settings", None)
            if cs and getattr(cs, "content_md5", None):
                v = cs.content_md5
                if isinstance(v, (bytes, bytearray)):
                    content_md5 = base64.b64encode(v).decode()
                else:
                    content_md5 = v
            info_obj = {"name": blob, "path": full, "size": int(size), "etag": etag, "metadata": meta}
            if content_md5:
                info_obj["content_md5"] = content_md5
            return info_obj
        except Exception as e:
            warn("info_failed", "get_blob_properties failed", blob=full, exception=str(e))
            return {}

    def put(self, local_path: str, full_remote_path: str, metadata: Optional[Dict[str,str]] = None, content_type: str = "application/octet-stream"):
        container, blob = self._parse(full_remote_path)
        if not blob:
            raise ValueError("remote path must include blob name")
        container_client = self.client.get_container_client(container)
        try:
            container_client.create_container()
        except Exception:
            pass
        blob_client = container_client.get_blob_client(blob)
        with open(local_path, "rb") as data:
            cs = ContentSettings(content_type=content_type)
            blob_client.upload_blob(data, overwrite=True, metadata=metadata or {}, content_settings=cs)

    def get(self, full_remote_path: str, local_target: str):
        container, blob = self._parse(full_remote_path)
        if not blob:
            raise ValueError("remote path must include blob name")
        container_client = self.client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob)
        downloader = blob_client.download_blob()
        target_path = Path(local_target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            for chunk in downloader.chunks():
                if chunk:
                    f.write(chunk)

    def rm(self, full_remote_path: str):
        container, blob = self._parse(full_remote_path)
        container_client = self.client.get_container_client(container)
        if not blob:
            try:
                container_client.delete_container()
            except Exception as e:
                warn("rm_failed", "delete_container failed", container=container, exception=str(e))
            return
        blob_client = container_client.get_blob_client(blob)
        try:
            blob_client.delete_blob()
        except Exception as e:
            warn("rm_failed", "delete_blob failed", blob=full_remote_path, exception=str(e))

    def setxattrs(self, full_remote_path: str, metadata: Dict[str,str]):
        container, blob = self._parse(full_remote_path)
        if not blob:
            return False
        container_client = self.client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob)
        try:
            blob_client.set_blob_metadata(metadata or {})
            return True
        except Exception as e:
            warn("setxattrs_failed", "set_blob_metadata failed", blob=full_remote_path, exception=str(e))
            return False

# ---- remote helpers ----
def get_fs(_protocol: Optional[str] = None):
    client = get_blob_service_client()
    fs = AzureBlobFs(client)
    return fs, "azureblob-sdk-key"

def safe_rel_normalize(p: str) -> str:
    # always posix and no leading slash
    return p.replace("\\", "/").lstrip("/")

def join_remote(container: str, prefix: str, rel: str) -> str:
    reln = safe_rel_normalize(rel)
    if prefix:
        p = prefix.strip("/").rstrip("/")
        blob = f"{p}/{reln}"
    else:
        blob = reln
    return f"{container}/{blob}"

def list_remote_objects(fs: AzureBlobFs, container: str, prefix: str) -> List[Tuple[str, str, int, Dict]]:
    prefix_clean = prefix.strip("/").rstrip("/")
    root = f"{container}/{prefix_clean}" if prefix_clean else f"{container}"
    out: List[Tuple[str,str,int,Dict]] = []
    found = fs.find(root)
    for full in sorted(found):
        info_obj = fs.info(full)
        if not info_obj:
            continue
        # compute rel relative to provided prefix
        # full format: container/blobname...
        lead = f"{container}/"
        rel = full[len(lead):] if full.startswith(lead) else full
        if prefix_clean:
            if rel.startswith(prefix_clean + "/"):
                rel = rel[len(prefix_clean)+1:]
            elif rel == prefix_clean:
                rel = ""
        rel = safe_rel_normalize(rel)
        size = int(info_obj.get("size", 0) or 0)
        out.append((full, rel, size, info_obj))
    return out

def extract_remote_values(info_obj: Dict) -> Dict[str, Optional[str]]:
    meta = (info_obj.get("metadata") or {}) if isinstance(info_obj, dict) else {}
    metadata_sha = None
    for k in ("sha256","SHA256","Sha256"):
        if meta.get(k):
            metadata_sha = meta.get(k)
            break
    content_md5 = info_obj.get("content_md5") or info_obj.get("Content-MD5")
    etag = info_obj.get("etag") or info_obj.get("ETag") or ""
    return {"metadata_sha256": metadata_sha, "content_md5": content_md5, "etag": etag, "raw_info": info_obj}

def upload_file_fs(fs: AzureBlobFs, local_path: str, full_remote_path: str, sha256: Optional[str], content_type: str = "application/octet-stream", dry_run: bool = False, verify_retries: int = VERIFY_META_RETRIES):
    if dry_run:
        return {"rel_path": full_remote_path, "action": "dry_run"}
    metadata = {"sha256": sha256} if sha256 else {}
    fs.put(local_path, full_remote_path, metadata=metadata, content_type=content_type)
    # best-effort set metadata (setxattrs also verifies)
    try:
        fs.setxattrs(full_remote_path, metadata)
    except Exception:
        pass
    for attempt in range(1, verify_retries+1):
        try:
            info_obj = fs.info(full_remote_path)
            meta = (info_obj.get("metadata") or {}) if isinstance(info_obj, dict) else {}
            remote_sha = meta.get("sha256") or meta.get("SHA256") or meta.get("Sha256")
            if sha256 and remote_sha == sha256:
                return {"rel_path": full_remote_path, "action": "uploaded", "verified": True}
        except Exception:
            pass
        time.sleep(VERIFY_META_SLEEP)
    return {"rel_path": full_remote_path, "action": "uploaded", "verified": False}

def download_file_fs(fs: AzureBlobFs, full_remote_path: str, local_target: str, dry_run: bool = False):
    if dry_run:
        return {"rel_path": full_remote_path, "action": "dry_run"}
    fs.get(full_remote_path, local_target)
    return {"rel_path": full_remote_path, "action": "downloaded"}

def delete_remote_file_fs(fs: AzureBlobFs, full_remote_path: str, dry_run: bool = False):
    if dry_run:
        return full_remote_path
    fs.rm(full_remote_path)
    return full_remote_path

# ---- local helpers ----
def list_local_files(base_dir: str) -> List[Tuple[str,str]]:
    base = Path(base_dir)
    if not base.exists():
        return []
    out: List[Tuple[str,str]] = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            try:
                rel = p.relative_to(base).as_posix()
            except Exception:
                rel = p.name
            out.append((str(p.resolve()), safe_rel_normalize(rel)))
    return out

def safe_remove_local(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except PermissionError:
        try:
            os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
            os.remove(path)
            return True
        except Exception as e:
            warn("delete_local_perm_failed", "chmod+delete failed", path=path, error=str(e))
            return False
    except FileNotFoundError:
        return True
    except Exception as e:
        warn("delete_local_failed", "delete local failed", path=path, error=str(e))
        return False

# ---- skip logic (unchanged but robust) ----
def should_skip_upload(local_path: str, remote_info: Optional[Dict], verbose: bool = False) -> Tuple[bool,str]:
    if not remote_info:
        return False, "remote_missing"
    try:
        local_size = Path(local_path).stat().st_size
    except Exception:
        local_size = None
    remote_meta_sha = remote_info.get("metadata_sha256")
    remote_etag = (remote_info.get("etag") or "") or ""
    remote_content_md5 = remote_info.get("content_md5")
    if remote_meta_sha:
        try:
            _, local_sha = compute_hashes(local_path)
            if local_sha == remote_meta_sha:
                return True, "match_metadata_sha256"
            return False, "metadata_sha256_mismatch"
        except Exception as e:
            return False, f"local_hash_failed:{e}"
    if remote_content_md5:
        try:
            local_md5, _ = compute_hashes(local_path)
            if local_md5 == remote_content_md5:
                return True, "match_content_md5_hex"
            hex_from_b64 = _hex_from_base64(remote_content_md5)
            if hex_from_b64 and local_md5 == hex_from_b64:
                return True, "match_content_md5_base64"
            return False, "content_md5_mismatch"
        except Exception as e:
            return False, f"local_hash_failed:{e}"
    if local_size is not None:
        remote_size = int(remote_info.get("size", 0) or 0)
        if local_size == remote_size and remote_etag:
            norm = _normalize_etag(remote_etag)
            if all(c in "0123456789abcdef" for c in norm) and len(norm) == 32:
                try:
                    local_md5, _ = compute_hashes(local_path)
                    if local_md5 == norm:
                        return True, "match_etag_md5"
                    return False, "etag_mismatch"
                except Exception as e:
                    return False, f"local_hash_failed:{e}"
    return False, "no_reliable_remote_checksum"

def should_skip_download(local_path: str, remote_info: Dict) -> bool:
    try:
        if not Path(local_path).exists():
            return False
        local_size = Path(local_path).stat().st_size
    except Exception:
        return False
    remote_meta_sha = remote_info.get("metadata_sha256")
    remote_etag = (remote_info.get("etag") or "") or ""
    remote_content_md5 = remote_info.get("content_md5")
    if remote_meta_sha:
        try:
            _, local_sha = compute_hashes(local_path)
            return local_sha == remote_meta_sha
        except Exception:
            return False
    if remote_content_md5:
        try:
            local_md5, _ = compute_hashes(local_path)
            if local_md5 == remote_content_md5:
                return True
            hex_from_b64 = _hex_from_base64(remote_content_md5)
            if hex_from_b64 and local_md5 == hex_from_b64:
                return True
        except Exception:
            return False
    remote_size = int(remote_info.get("size", 0) or 0)
    if local_size == remote_size and remote_etag:
        norm = _normalize_etag(remote_etag)
        if all(c in "0123456789abcdef" for c in norm) and len(norm) == 32:
            try:
                local_md5, _ = compute_hashes(local_path)
                return local_md5 == norm
            except Exception:
                return False
    return False

# ---- core operations ----
def upload_directory(base_dir: str, container: str, prefix: str, concurrency: int, dry_run: bool = False, verbose: bool = False, delete_orphans: bool = True):
    info("upload_start", "Upload mirror starting", local=base_dir, container=container, prefix=prefix, concurrency=concurrency, delete_orphans=delete_orphans)
    fs, proto = get_fs(None)
    local_entries = list_local_files(base_dir)
    local_rel_map = {rel: abs_path for abs_path, rel in local_entries}
    remote_entries = list_remote_objects(fs, container, prefix)
    remote_map: Dict[str, Dict] = {}
    for full, rel, size, info_obj in remote_entries:
        vals = extract_remote_values(info_obj)
        vals["full"] = full
        vals["size"] = size
        remote_map[safe_rel_normalize(rel)] = vals

    # compute orphans
    remote_rels = set(remote_map.keys())
    local_rels = set(local_rel_map.keys())
    stale_remote = sorted(remote_rels - local_rels)
    info("delete_orphans", "Deleting remote orphans (if enabled)", orphan_count=len(stale_remote), delete_orphans=delete_orphans)
    if delete_orphans and stale_remote:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(delete_remote_file_fs, fs, remote_map[rel]["full"], dry_run): rel for rel in stale_remote}
            for fut in as_completed(futures):
                rel = futures[fut]
                try:
                    key = fut.result()
                    info("deleted_remote_orphan", "Deleted remote orphan", rel=rel, remote=key)
                except Exception as e:
                    warn("delete_orphan_failed", "Failed deleting remote orphan", rel=rel, error=str(e))

    # refresh remote map deterministically
    remote_entries = list_remote_objects(fs, container, prefix)
    remote_map = {}
    for full, rel, size, info_obj in remote_entries:
        vals = extract_remote_values(info_obj)
        vals["full"] = full
        vals["size"] = size
        remote_map[safe_rel_normalize(rel)] = vals

    successes = skipped = failed = 0
    errors: List[str] = []
    tasks = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rel in sorted(local_rel_map.keys()):
            local_path = local_rel_map[rel]
            remote_info = remote_map.get(rel)
            try:
                skip, reason = should_skip_upload(local_path, remote_info, verbose=verbose)
            except Exception as e:
                warn("skip_check_failed", "Checksum decision failed; will upload", rel=rel, error=str(e))
                skip, reason = False, "skip_check_error"
            if skip:
                skipped += 1
                info("skipped_upload", "Skipped upload (unchanged)", rel=rel, local=local_path, reason=reason)
                continue
            try:
                _, local_sha = compute_hashes(local_path)
            except Exception as e:
                warn("hash_failed", "Failed computing hashes; will upload without sha metadata", rel=rel, error=str(e))
                local_sha = None
            full_remote = join_remote(container, prefix, rel)
            tasks[ex.submit(upload_file_fs, fs, local_path, full_remote, local_sha, "application/octet-stream", dry_run)] = (rel, local_path, full_remote, local_sha)
        for fut in as_completed(tasks):
            rel, local_path, full_remote, sha256 = tasks[fut]
            try:
                result = fut.result()
                action = result.get("action")
                verified = result.get("verified", False)
                if action == "dry_run":
                    info("upload_dryrun", "Dry-run would upload", rel=rel, remote=full_remote, sha256=sha256)
                else:
                    successes += 1
                    info("uploaded", "Uploaded file", rel=rel, remote=full_remote, verified=bool(verified), sha256=sha256)
            except Exception as e:
                failed += 1
                errors.append(f"{rel}: {e}")
                warn("upload_failed", "Upload failed", rel=rel, error=str(e))

    info("upload_finished", "Upload finished", succeeded=successes, skipped=skipped, failed=failed)
    for e in errors[:20]:
        warn("upload_error", "Upload error detail", detail=e)

def download_directory(container: str, base_dir: str, prefix: str, concurrency: int, dry_run: bool = False, verbose: bool = False, delete_orphans: bool = True):
    info("download_start", "Download mirror starting", container=container, local=base_dir, prefix=prefix, concurrency=concurrency, delete_orphans=delete_orphans)
    fs, proto = get_fs(None)
    remote_entries = list_remote_objects(fs, container, prefix)
    remote_map: Dict[str, Dict] = {}
    for full, rel, size, info_obj in remote_entries:
        vals = extract_remote_values(info_obj)
        vals["full"] = full
        vals["size"] = size
        remote_map[safe_rel_normalize(rel)] = vals

    local_entries = list_local_files(base_dir)
    local_rel_map = {rel: abs_path for abs_path, rel in local_entries}

    # compute local orphans
    remote_rels = set(remote_map.keys())
    local_rels = set(local_rel_map.keys())
    stale_local = sorted(local_rels - remote_rels)
    info("delete_local_orphans", "Deleting local orphans (if enabled)", orphan_count=len(stale_local), delete_orphans=delete_orphans)
    if delete_orphans and stale_local:
        for rel in stale_local:
            path = local_rel_map[rel]
            try:
                if dry_run:
                    info("delete_local_dryrun", "Would delete local orphan", rel=rel, path=path)
                    continue
                ok = safe_remove_local(path)
                if ok:
                    info("deleted_local_orphan", "Deleted local orphan", rel=rel, path=path)
                else:
                    warn("delete_local_failed", "Failed to delete local orphan", rel=rel, path=path)
            except Exception as e:
                warn("delete_local_failed", "Failed to delete local orphan", rel=rel, error=str(e))

    # refresh local map deterministically
    local_entries = list_local_files(base_dir)
    local_rel_map = {rel: abs_path for abs_path, rel in local_entries}

    successes = skipped = failed = 0
    errors: List[str] = []
    tasks = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rel in sorted(remote_map.keys()):
            rinfo = remote_map[rel]
            full = rinfo["full"]
            local_path = str(Path(base_dir) / rel)
            try:
                if should_skip_download(local_path, rinfo):
                    skipped += 1
                    info("skipped_download", "Skipped download (unchanged)", rel=rel, local=local_path)
                    continue
            except Exception as e:
                warn("skip_download_failed", "Checksum decision failed for download; will attempt download", rel=rel, error=str(e))
            tasks[ex.submit(download_file_fs, fs, full, local_path, dry_run)] = rel
        for fut in as_completed(tasks):
            rel = tasks[fut]
            try:
                result = fut.result()
                if result.get("action") == "dry_run":
                    info("download_dryrun", "Dry-run would download", rel=rel)
                else:
                    successes += 1
                    info("downloaded", "Downloaded file", rel=rel)
            except Exception as e:
                failed += 1
                errors.append(f"{rel}: {e}")
                warn("download_failed", "Download failed", rel=rel, error=str(e))
    info("download_finished", "Download finished", succeeded=successes, skipped=skipped, failed=failed)
    for e in errors[:20]:
        warn("download_error", "Download error detail", detail=e)

# ---- CLI ----
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart deterministic mirror sync local <-> Azure Blob (key-auth first).")
    gp = p.add_mutually_exclusive_group(required=True)
    gp.add_argument("--upload", action="store_true", help="Mirror local -> Azure (skip unchanged, delete remote orphans)")
    gp.add_argument("--download", action="store_true", help="Mirror Azure -> local (skip unchanged, delete local orphans)")
    gp.add_argument("--merge-upload", action="store_true", help="Merge upload: upload changed files only, DO NOT delete remote orphans")
    gp.add_argument("--merge-download", action="store_true", help="Merge download: download changed files only, DO NOT delete local orphans")
    p.add_argument("--max-concurrency", type=int, default=0, help="Override concurrency (0 = auto/env)")
    p.add_argument("--dry-run", action="store_true", help="Do not perform state-changing operations; print actions only")
    p.add_argument("--verbose", action="store_true", help="Emit additional debug logs")
    return p.parse_args()

def compute_concurrency(override: int = 0) -> int:
    if override and override > 0:
        return max(1, override)
    return max(1, DEFAULT_CONCURRENCY)

def main() -> None:
    args = parse_args()
    container = AZURE_CONTAINER
    if not container:
        error("missing_env", "AZURE_CONTAINER (or AZ_CONTAINER) env variable is not set")
        raise SystemExit(2)

    validate_auth_preconditions()

    concurrency = compute_concurrency(args.max_concurrency)
    prefix = os.environ.get("DEFAULT_PREFIX", DEFAULT_PREFIX).strip("/")
    dry_run = args.dry_run
    verbose = args.verbose

    try:
        fs, proto = get_fs(None)
    except Exception as e:
        error("fs_init_failed", "Filesystem initialization failed", exception=str(e))
        raise SystemExit(3)

    try:
        probe = fs.find(container)
        info("fs_ok", "Filesystem initialized and container probe OK", protocol=proto, container=container, sample_count=len(probe))
    except Exception as e:
        warn("container_access", "Container may not exist or probe failed", container=container, error=str(e))

    if args.upload:
        upload_directory(LOCAL_BASE, container, prefix, concurrency, dry_run=dry_run, verbose=verbose, delete_orphans=True)
    elif args.download:
        download_directory(container, LOCAL_BASE, prefix, concurrency, dry_run=dry_run, verbose=verbose, delete_orphans=True)
    elif args.merge_upload:
        upload_directory(LOCAL_BASE, container, prefix, concurrency, dry_run=dry_run, verbose=verbose, delete_orphans=False)
    elif args.merge_download:
        download_directory(container, LOCAL_BASE, prefix, concurrency, dry_run=dry_run, verbose=verbose, delete_orphans=False)
    else:
        error("cli_usage", "Please specify --upload/--download/--merge-upload/--merge-download")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
