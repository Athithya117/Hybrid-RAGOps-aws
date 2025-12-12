#!/usr/bin/env python3
"""
force_sync_azure_and_local_fs.py

Smart mirror sync local <-> Azure Blob using key-based auth (Option A).

Modes:
  --upload         mirror local -> Azure (deletes remote orphans)
  --download       mirror Azure -> local (deletes local orphans)
  --merge-upload   non-destructive upload: upload changed files only, DO NOT delete remote orphans
  --merge-download non-destructive download: download changed files only, DO NOT delete local orphans

Auth precedence:
  1) AZURE_STORAGE_CONNECTION_STRING
  2) AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
  3) (fallback) Attempt to fetch primary key via az CLI:
       az storage account keys list --resource-group <RG> --account-name <ACCOUNT>
     This requires the caller (az login) to have permission to list storage keys.

Environment:
  AZURE_CONTAINER (required)
  AZURE_STORAGE_CONNECTION_STRING OR
  AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
  (optional fallback) AZURE_SUBSCRIPTION_ID + AZURE_RESOURCE_GROUP_NAME + AZURE_STORAGE_ACCOUNT_NAME

Other envs:
  LOCAL_BASE (default "data")
  DEFAULT_PREFIX (default "data")
  CONCURRENT_FILES (default 4)
  MULTIPART_CHUNKSIZE_MB (default 100)
  VERIFY_META_RETRIES, VERIFY_META_SLEEP

Dependencies:
  pip install azure-storage-blob

Usage:
  export AZURE_CONTAINER="rag-data-prod"
  export AZURE_STORAGE_ACCOUNT_NAME="storeragprod42"
  export AZURE_STORAGE_ACCOUNT_KEY="..."
  python infra/base_infra/force_sync_azure_and_local_fs.py --merge-upload --dry-run
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------- imports (fail-fast) --------
try:
    from azure.storage.blob import (
        BlobServiceClient,
        ContentSettings,
    )
except Exception as e:
    print(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "level": "ERROR",
        "event": "import_failure",
        "msg": "missing dependency 'azure-storage-blob'. Install: pip install azure-storage-blob",
        "exception": str(e)
    }))
    raise

# -------- small helpers & logging --------
def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def log(level: str, event: str, msg: str, **kwargs) -> None:
    obj = {"ts": ts(), "level": level, "event": event, "msg": msg}
    if kwargs:
        obj.update(kwargs)
    print(json.dumps(obj, default=str), flush=True)

def info(event: str, msg: str, **kwargs) -> None:
    log("INFO", event, msg, **kwargs)

def warn(event: str, msg: str, **kwargs) -> None:
    log("WARN", event, msg, **kwargs)

def error(event: str, msg: str, **kwargs) -> None:
    log("ERROR", event, msg, **kwargs)

# -------- hashing helpers --------
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

# -------- runner (az fallback) --------
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

# -------- config (env-driven) --------
DEFAULT_PREFIX = os.environ.get("DEFAULT_PREFIX", "data")
LOCAL_BASE = os.environ.get("LOCAL_BASE", "data")
DEFAULT_CONCURRENCY = int(os.environ.get("CONCURRENT_FILES", "4"))
DEFAULT_CHUNKSIZE_MB = int(os.environ.get("MULTIPART_CHUNKSIZE_MB", "100"))
VERIFY_META_RETRIES = int(os.environ.get("VERIFY_META_RETRIES", "3"))
VERIFY_META_SLEEP = float(os.environ.get("VERIFY_META_SLEEP", "0.7"))

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
AZURE_ENDPOINT_SUFFIX = os.environ.get("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"

# optional for az-key fetch
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP_NAME = os.environ.get("AZURE_RESOURCE_GROUP_NAME")

# required target container
AZURE_CONTAINER = os.environ.get("AZURE_CONTAINER") or os.environ.get("AZ_CONTAINER")

# -------- validate minimal env for key-fetch path --------
def validate_auth_preconditions():
    if AZURE_STORAGE_CONNECTION_STRING:
        info("auth", "Using AZURE_STORAGE_CONNECTION_STRING")
        return
    if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
        info("auth", "Using AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY")
        return
    # allow az fallback only when full inputs exist
    if AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP_NAME and AZURE_STORAGE_ACCOUNT_NAME:
        info("auth", "No key/connstring present — will attempt az CLI fetch of storage key")
        return
    error("auth_missing", "Missing storage auth: set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_KEY, or provide AZURE_SUBSCRIPTION_ID + AZURE_RESOURCE_GROUP_NAME + AZURE_STORAGE_ACCOUNT_NAME to let az fetch keys")
    raise SystemExit(2)

# -------- helper: fetch key via az CLI if needed --------
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

# -------- BlobServiceClient factory (key-first) --------
def get_blob_service_client() -> BlobServiceClient:
    """
    Prefer connection string or account key. If missing and az CLI is available, try to fetch key.
    Fail fast with clear message if no usable auth found.
    """
    if AZURE_STORAGE_CONNECTION_STRING:
        info("auth", "Using AZURE_STORAGE_CONNECTION_STRING")
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
        account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.{AZURE_ENDPOINT_SUFFIX}"
        info("auth", "Using account_name + account_key from env", account=AZURE_STORAGE_ACCOUNT_NAME)
        return BlobServiceClient(account_url=account_url, credential=AZURE_STORAGE_ACCOUNT_KEY)

    # attempt to fetch key via az CLI
    if AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP_NAME and AZURE_STORAGE_ACCOUNT_NAME:
        key = fetch_key_via_az(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME)
        if key:
            account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.{AZURE_ENDPOINT_SUFFIX}"
            info("auth", "Using account_key fetched via az CLI", account=AZURE_STORAGE_ACCOUNT_NAME)
            return BlobServiceClient(account_url=account_url, credential=key)

    # no key possible
    error("auth_missing", "No storage key/connection string available and az fetch failed. Provide AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_KEY or run 'az login' and ensure you can fetch keys.")
    raise SystemExit(2)

# -------- AzureBlobFs adapter (SDK-backed, key/connstring auth) --------
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
        container_client = self.client.get_container_client(container)
        out: List[str] = []
        try:
            blobs = container_client.list_blobs(name_starts_with=prefix)
            for b in blobs:
                out.append(f"{container}/{b.name}")
        except Exception as e:
            warn("find_failed", "list_blobs failed", root=root, exception=str(e))
            return []
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
            size = props.size
            content_md5 = None
            cs = getattr(props, "content_settings", None)
            if cs and getattr(cs, "content_md5", None):
                v = cs.content_md5
                if isinstance(v, (bytes, bytearray)):
                    content_md5 = base64.b64encode(v).decode()
                else:
                    content_md5 = v
            info_obj = {
                "name": blob,
                "path": full,
                "size": size,
                "etag": etag,
                "metadata": meta,
            }
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
        blob_client = container_client.get_blob_client(blob)
        try:
            # ensure container exists (ignore if already exists)
            container_client.create_container()
        except Exception:
            pass
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
        if not blob:
            try:
                container_client = self.client.get_container_client(container)
                container_client.delete_container()
            except Exception as e:
                warn("rm_failed", "delete_container failed", container=container, exception=str(e))
            return
        container_client = self.client.get_container_client(container)
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

# -------- remote helpers (adapter-backed) --------
def get_fs(protocol: Optional[str] = None):
    client = get_blob_service_client()
    fs = AzureBlobFs(client)
    return fs, "azureblob-sdk-key"

def list_remote_objects(fs, container: str, prefix: str) -> List[Tuple[str, str, int, Dict]]:
    prefix_key = (prefix.rstrip("/") + "/") if prefix else ""
    root = f"{container}/{prefix_key}" if prefix_key else f"{container}"
    out: List[Tuple[str, str, int, Dict]] = []
    found = fs.find(root)
    for full in found:
        info_obj = fs.info(full)
        if not info_obj:
            continue
        rel = full
        lead = f"{container}/{prefix_key}" if prefix_key else f"{container}/"
        if rel.startswith(lead):
            rel = rel[len(lead):]
        elif rel.startswith(f"{container}/"):
            rel = rel[len(f"{container}/"):]
        size = int(info_obj.get("size", 0) or 0)
        out.append((full, rel, size, info_obj))
    return out

def extract_remote_values(info_obj: Dict) -> Dict[str, Optional[str]]:
    meta = (info_obj.get("metadata") or {}) if isinstance(info_obj, dict) else {}
    metadata_sha = None
    for k in ("sha256", "SHA256", "Sha256"):
        if meta.get(k):
            metadata_sha = meta.get(k)
            break
    content_md5 = None
    for k in ("content_md5", "Content-MD5", "content-md5", "ContentMD5", "content_md5"):
        if info_obj.get(k):
            content_md5 = info_obj.get(k)
            break
    etag = info_obj.get("etag") or info_obj.get("ETag") or ""
    return {"metadata_sha256": metadata_sha, "content_md5": content_md5, "etag": etag, "raw_info": info_obj}

def upload_file_fs(fs, local_path: str, full_remote_path: str, sha256: Optional[str], content_type: str = "application/octet-stream", dry_run: bool = False, verify_retries: int = VERIFY_META_RETRIES):
    if dry_run:
        return {"rel_path": full_remote_path, "action": "dry_run"}
    metadata = {"sha256": sha256} if sha256 else {}
    fs.put(local_path, full_remote_path, metadata=metadata, content_type=content_type)
    try:
        fs.setxattrs(full_remote_path, {"sha256": sha256} if sha256 else {})
    except Exception:
        pass
    for attempt in range(1, verify_retries + 1):
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

def download_file_fs(fs, full_remote_path: str, local_target: str, dry_run: bool = False):
    tgt = Path(local_target)
    tgt.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"rel_path": full_remote_path, "action": "dry_run"}
    fs.get(full_remote_path, str(tgt))
    return {"rel_path": full_remote_path, "action": "downloaded"}

def delete_remote_file_fs(fs, full_remote_path: str, dry_run: bool = False):
    if dry_run:
        return full_remote_path
    fs.rm(full_remote_path)
    return full_remote_path

# -------- local helpers --------
def list_local_files(base_dir: str) -> List[Tuple[str, str]]:
    base = Path(base_dir)
    if not base.exists():
        return []
    out: List[Tuple[str, str]] = []
    for p in base.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(base).as_posix()
            except Exception:
                rel = p.name
            out.append((str(p.resolve()), rel))
    return out

# -------- skip logic (unchanged) --------
def should_skip_upload(local_path: str, remote_info: Optional[Dict], verbose: bool = False) -> Tuple[bool, str]:
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
        except Exception as e:
            return False, f"local_hash_failed:{e}"
        if local_sha == remote_meta_sha:
            return True, "match_metadata_sha256"
        return False, "metadata_sha256_mismatch"
    if remote_content_md5:
        try:
            local_md5, _ = compute_hashes(local_path)
        except Exception as e:
            return False, f"local_hash_failed:{e}"
        if local_md5 == remote_content_md5:
            return True, "match_content_md5_hex"
        hex_from_b64 = _hex_from_base64(remote_content_md5)
        if hex_from_b64 and local_md5 == hex_from_b64:
            return True, "match_content_md5_base64"
        return False, "content_md5_mismatch"
    if local_size is not None:
        remote_size = int(remote_info.get("size", 0) or 0)
        if local_size == remote_size and remote_etag:
            norm = _normalize_etag(remote_etag)
            if all(c in "0123456789abcdef" for c in norm) and len(norm) == 32:
                try:
                    local_md5, _ = compute_hashes(local_path)
                except Exception as e:
                    return False, f"local_hash_failed:{e}"
                if local_md5 == norm:
                    return True, "match_etag_md5"
                return False, "etag_mismatch"
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

# -------- orchestration (parameterized delete_orphans) --------
def upload_directory(base_dir: str, container: str, prefix: str, concurrency: int, chunksize_mb: int, dry_run: bool = False, verbose: bool = False, delete_orphans: bool = True):
    info("upload_start", "Upload mirror starting", local=base_dir, container=container, prefix=prefix, concurrency=concurrency, delete_orphans=delete_orphans)
    fs, proto = get_fs(None)
    local = list_local_files(base_dir)
    local_rel_map = {rel: lp for lp, rel in local}
    remote_entries = list_remote_objects(fs, container, prefix)
    remote_map: Dict[str, Dict] = {}
    for full, rel, size, info_obj in remote_entries:
        vals = extract_remote_values(info_obj)
        vals["full"] = full
        vals["size"] = size
        remote_map[rel] = vals

    # delete remote orphans (only when delete_orphans True)
    remote_rels = set(remote_map.keys())
    local_rels = set(local_rel_map.keys())
    stale_remote = sorted(list(remote_rels - local_rels))
    info("delete_orphans", "Deleting remote orphans", orphan_count=len(stale_remote), delete_orphans=delete_orphans)
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

    # refresh remote map
    remote_entries = list_remote_objects(fs, container, prefix)
    remote_map = {}
    for full, rel, size, info_obj in remote_entries:
        vals = extract_remote_values(info_obj)
        vals["full"] = full
        vals["size"] = size
        remote_map[rel] = vals

    successes = skipped = failed = 0
    errors: List[str] = []
    tasks = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rel, local_path in local_rel_map.items():
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
                local_md5, local_sha = compute_hashes(local_path)
            except Exception as e:
                warn("hash_failed", "Failed computing hashes; will upload without sha metadata", rel=rel, error=str(e))
                local_md5, local_sha = None, None
            full_remote = f"{container}/{prefix.rstrip('/')}/{rel}" if prefix else f"{container}/{rel}"
            tasks[ex.submit(upload_file_fs, fs, local_path, full_remote, local_sha, "application/octet-stream", dry_run) ] = (rel, local_path, full_remote, local_sha)
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
    info("download_start", "Download mirror starting", container=container, prefix=prefix, local=base_dir, concurrency=concurrency, delete_orphans=delete_orphans)
    fs, proto = get_fs(None)
    remote_entries = list_remote_objects(fs, container, prefix)
    remote_map: Dict[str, Dict] = {}
    for full, rel, size, info_obj in remote_entries:
        vals = extract_remote_values(info_obj)
        vals["full"] = full
        vals["size"] = size
        remote_map[rel] = vals
    local_entries = list_local_files(base_dir)
    local_rel_map = {rel: lp for lp, lp in local_entries}

    remote_rels = set(remote_map.keys())
    local_rels = set(local_rel_map.keys())
    stale_local = sorted(list(local_rels - remote_rels))
    info("delete_local_orphans", "Deleting local orphans", orphan_count=len(stale_local), delete_orphans=delete_orphans)
    if delete_orphans and stale_local:
        for rel in stale_local:
            path = local_rel_map[rel]
            try:
                if dry_run:
                    info("delete_local_dryrun", "Would delete local orphan", rel=rel, path=path)
                    continue
                os.remove(path)
                info("deleted_local_orphan", "Deleted local orphan", rel=rel, path=path)
            except FileNotFoundError:
                pass
            except Exception as e:
                warn("delete_local_failed", "Failed to delete local orphan", rel=rel, error=str(e))

    # refresh local map
    local_entries = list_local_files(base_dir)
    local_rel_map = {rel: lp for lp, lp in local_entries}

    successes = skipped = failed = 0
    errors: List[str] = []
    tasks = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rel, rinfo in remote_map.items():
            full = rinfo["full"]
            local_path = Path(base_dir) / rel
            try:
                if should_skip_download(str(local_path), rinfo):
                    skipped += 1
                    info("skipped_download", "Skipped download (unchanged)", rel=rel, local=str(local_path))
                    continue
            except Exception as e:
                warn("skip_download_failed", "Checksum decision failed for download; will attempt download", rel=rel, error=str(e))
            tasks[ex.submit(download_file_fs, fs, full, str(local_path), dry_run)] = rel
        for fut in as_completed(tasks):
            rel = tasks[fut]
            try:
                fut.result()
                successes += 1
                info("downloaded", "Downloaded file", rel=rel)
            except Exception as e:
                failed += 1
                errors.append(f"{rel}: {e}")
                warn("download_failed", "Download failed", rel=rel, error=str(e))
    info("download_finished", "Download finished", succeeded=successes, skipped=skipped, failed=failed)
    for e in errors[:20]:
        warn("download_error", "Download error detail", detail=e)

# -------- CLI --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart mirror sync local <-> Azure Blob (key-auth first).")
    gp = p.add_mutually_exclusive_group(required=True)
    gp.add_argument("--upload", action="store_true", help="Mirror local -> Azure (skip unchanged, delete remote orphans)")
    gp.add_argument("--download", action="store_true", help="Mirror Azure -> local (skip unchanged, delete local orphans)")
    gp.add_argument("--merge-upload", action="store_true", help="Merge upload: upload changed files only, DO NOT delete remote orphans")
    gp.add_argument("--merge-download", action="store_true", help="Merge download: download changed files only, DO NOT delete local orphans")
    p.add_argument("--max-concurrency", type=int, default=0, help="Override concurrency (0 = auto/env)")
    p.add_argument("--multipart-chunksize-mb", type=int, default=DEFAULT_CHUNKSIZE_MB, help="Multipart chunk size in MiB (not used by SDK but kept for compatibility)")
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

    # ensure auth availability (fail fast)
    validate_auth_preconditions()

    concurrency = compute_concurrency(args.max_concurrency)
    prefix = os.environ.get("DEFAULT_PREFIX", DEFAULT_PREFIX)
    dry_run = args.dry_run
    verbose = args.verbose
    chunksize_mb = args.multipart_chunksize_mb

    # build fs
    try:
        fs, proto_used = get_fs(None)
    except SystemExit:
        raise
    except Exception as e:
        error("fs_init_failed", "Filesystem initialization failed", exception=str(e))
        raise SystemExit(3)

    try:
        probe = fs.find(container)
        info("fs_ok", "Filesystem initialized and container probe OK", protocol=proto_used, container=container, sample_count=len(probe))
    except Exception as e:
        warn("container_access", "Container may not exist or probe failed", container=container, error=str(e))

    # dispatch modes
    if args.upload:
        upload_directory(LOCAL_BASE, container, prefix, concurrency, chunksize_mb, dry_run=dry_run, verbose=verbose, delete_orphans=True)
    elif args.download:
        download_directory(container, LOCAL_BASE, prefix, concurrency, dry_run=dry_run, verbose=verbose, delete_orphans=True)
    elif args.merge_upload:
        upload_directory(LOCAL_BASE, container, prefix, concurrency, chunksize_mb, dry_run=dry_run, verbose=verbose, delete_orphans=False)
    elif args.merge_download:
        download_directory(container, LOCAL_BASE, prefix, concurrency, dry_run=dry_run, verbose=verbose, delete_orphans=False)
    else:
        error("cli_usage", "Please specify --upload/--download/--merge-upload/--merge-download")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
