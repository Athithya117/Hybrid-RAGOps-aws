#!/usr/bin/env python3
"""
Finalized app-layer storage glue (deterministic).

Auth decision:
 - AZURE_USE_MANAGED_IDENTITY=true  => Managed Identity (User Assigned)
     Required: AZURE_STORAGE_ACCOUNT_NAME, UAI_RAG_RW_CLIENT_ID
 - AZURE_USE_MANAGED_IDENTITY!=true => Connection string mode (default for local/CI)
     Required: AZURE_STORAGE_CONNECTION_STRING

On startup the code validates credentials by calling get_container_properties()
for the configured container. This is intentional: fail fast and deterministic.
"""
from __future__ import annotations
import os
import sys
import time
import json
import uuid
import hashlib
import importlib
import importlib.util
import mimetypes
import urllib.parse
import logging
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# --- Minimal dependencies / logging setup ---
_root = logging.getLogger()
_root.setLevel(logging.WARNING)
for n in ("urllib3", "requests", "httpx", "azure", "adlfs"):
    lg = logging.getLogger(n)
    lg.setLevel(logging.WARNING)
    lg.propagate = False

def ts_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

# -------------------- Config & validation --------------------
CONTAINER = (
    os.getenv("AZURE_CONTAINER")
    or os.getenv("STORAGE_CONTAINER")
    or os.getenv("AZ_CONTAINER")
)
if not CONTAINER:
    print(json.dumps({"ts": ts_now(), "level": "error", "event": "startup", "msg": "AZURE_CONTAINER (or STORAGE_CONTAINER or AZ_CONTAINER) must be set"}), file=sys.stderr)
    sys.exit(1)

# deterministic auth switch: only this variable decides MI mode
USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower() in ("1", "true", "yes")

# In non-MI mode the app *requires* a single env: AZURE_STORAGE_CONNECTION_STRING
# In MI mode it *requires* AZURE_STORAGE_ACCOUNT_NAME and UAI_RAG_RW_CLIENT_ID
AZ_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
AZ_ACCOUNT = (os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or os.getenv("AZURE_ACCOUNT_NAME") or "").strip()
AZ_ENDPOINT_SUFFIX = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net").strip()
UAI_RAG_RW_CLIENT_ID = os.getenv("UAI_RAG_RW_CLIENT_ID", "").strip()

if USE_MANAGED_IDENTITY:
    missing = []
    if not AZ_ACCOUNT:
        missing.append("AZURE_STORAGE_ACCOUNT_NAME")
    if not UAI_RAG_RW_CLIENT_ID:
        missing.append("UAI_RAG_RW_CLIENT_ID")
    if missing:
        print(json.dumps({"ts": ts_now(), "level": "error", "event": "config", "msg": f"Missing required envs for managed identity mode: {', '.join(missing)}"}), file=sys.stderr)
        sys.exit(2)
else:
    if not AZ_CONN:
        # fail fast and actionable message
        print(json.dumps({"ts": ts_now(), "level": "error", "event": "config", "msg": "Non-managed-identity mode requires AZURE_STORAGE_CONNECTION_STRING (mount it as a secret)"}), file=sys.stderr)
        sys.exit(2)

RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("S3_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("S3_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"

def log(level: str, event: str, msg: str, **extra) -> None:
    o = {"ts": ts_now(), "level": level, "event": event, "msg": msg, "container": CONTAINER}
    if extra:
        o.update(extra)
    print(json.dumps(o, ensure_ascii=False), flush=True)

# -------------------- Azure client factory (deterministic) --------------------
def build_blob_service_client():
    """
    Deterministic:
      - If USE_MANAGED_IDENTITY: use DefaultAzureCredential(managed_identity_client_id=...) + account_url
      - Else: use connection string (required)
    Fail fast with clear messages and container validation.
    """
    if USE_MANAGED_IDENTITY:
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except Exception as e:
            log("error", "azure_import", "azure.identity and azure.storage.blob are required for managed identity mode", error=str(e))
            raise SystemExit(2)

        account_url = f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
        try:
            # provide the managed identity client id explicitly (user-assigned MI)
            cred = DefaultAzureCredential(managed_identity_client_id=UAI_RAG_RW_CLIENT_ID)
            client = BlobServiceClient(account_url=account_url, credential=cred)
            # HARD VALIDATION: ensure container exists and credentials valid
            try:
                _ = client.get_container_client(CONTAINER).get_container_properties()
            except Exception as e_check:
                log("error", "mi_validation_failed", "Managed Identity client created but container validation failed; verify Workload Identity, role assignment, and network/DNS", error=str(e_check))
                raise SystemExit(2)
            log("info", "client_init", "Initialized BlobServiceClient (managed identity)", account=AZ_ACCOUNT)
            return client
        except SystemExit:
            raise
        except Exception as e:
            log("error", "mi_client_failed", "Failed to initialize BlobServiceClient with managed identity; check cluster workload identity configuration", error=str(e))
            raise SystemExit(2)
    else:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except Exception as e:
            log("error", "azure_import", "azure.storage.blob package required for connection-string mode (pip install azure-storage-blob)", error=str(e))
            raise SystemExit(2)
        try:
            client = BlobServiceClient.from_connection_string(AZ_CONN)
            # HARD VALIDATION
            try:
                _ = client.get_container_client(CONTAINER).get_container_properties()
            except Exception as e_check:
                log("error", "connstr_validation_failed", "Connection string provided but container validation failed; verify connection string and container name", error=str(e_check))
                raise SystemExit(2)
            log("info", "client_init", "Initialized BlobServiceClient (connection string)")
            return client
        except SystemExit:
            raise
        except Exception as e:
            log("error", "connstr_failed", "Failed to initialize BlobServiceClient.from_connection_string; verify the connection string", error=str(e))
            raise SystemExit(2)

# -------------------- StorageBackend using azure sdk only (simple surface) --------------------
class StorageBackend:
    def __init__(self, container: str):
        self.container = container
        self.storage_url = f"az://{container.rstrip('/')}/"
        self.blob_service = build_blob_service_client()
        try:
            self.container_client = self.blob_service.get_container_client(container)
        except Exception as e:
            log("error", "container_client_failed", f"Unable to get container client for {container}", error=str(e))
            raise SystemExit(2)

    def _strip_az_prefix(self, full: str) -> str:
        if full.startswith("az://"):
            rest = full[len("az://") :]
            if rest.startswith(self.container + "/"):
                return rest[len(self.container) + 1 :]
            if rest == self.container:
                return ""
            return rest
        if full.startswith(self.container + "/"):
            return full[len(self.container) + 1 :]
        return full

    def find(self, root_path: str) -> List[str]:
        prefix = self._strip_az_prefix(root_path).lstrip("/")
        out: List[str] = []
        try:
            for b in self.container_client.list_blobs(name_starts_with=prefix):
                out.append(f"az://{self.container}/{b.name}")
        except Exception as e:
            log("warn", "list_blobs_failed", "list_blobs error", error=str(e))
        return out

    def glob(self, pattern: str) -> List[str]:
        prefix = self._strip_az_prefix(pattern).lstrip("/")
        out: List[str] = []
        try:
            for b in self.container_client.list_blobs(name_starts_with=prefix):
                out.append(f"az://{self.container}/{b.name}")
        except Exception as e:
            log("warn", "glob_failed", "list_blobs error", error=str(e))
        return out

    def info(self, full_path: str) -> Dict[str, Any]:
        name = self._strip_az_prefix(full_path).lstrip("/")
        try:
            blob_client = self.container_client.get_blob_client(name)
            props = blob_client.get_blob_properties()
            meta = getattr(props, "metadata", {}) or {}
            content_type = ""
            try:
                if getattr(props, "content_settings", None) is not None:
                    content_type = getattr(props.content_settings, "content_type", "") or ""
            except Exception:
                content_type = getattr(props, "content_type", "") or ""
            info_obj: Dict[str, Any] = {
                "size": int(getattr(props, "size", 0) or 0),
                "etag": getattr(props, "etag", "") or "",
                "ETag": getattr(props, "etag", "") or "",
                "eTag": getattr(props, "etag", "") or "",
                "Content-Type": content_type,
                "content-type": content_type,
                "content_type": content_type,
                "last_modified": getattr(props, "last_modified", ""),
                "Last-Modified": getattr(props, "last_modified", ""),
                "metadata": meta,
                "meta": meta,
                "type": "file",
            }
            return info_obj
        except Exception as e:
            raise

    def exists(self, full_path: str) -> bool:
        name = self._strip_az_prefix(full_path).lstrip("/")
        try:
            blob_client = self.container_client.get_blob_client(name)
            return blob_client.exists()
        except Exception:
            return False

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        # noop for blob storage
        return

    def open(self, full_path: str, mode: str = "rb"):
        name = self._strip_az_prefix(full_path).lstrip("/")
        blob_client = self.container_client.get_blob_client(name)
        if "r" in mode:
            stream = blob_client.download_blob()
            data = stream.readall()
            return io.BytesIO(data)
        # write: context manager that uploads on close
        class _BlobWriter(io.BytesIO):
            def __init__(self, bc):
                super().__init__()
                self._bc = bc
            def close(self):
                try:
                    self.seek(0)
                    data = self.read()
                    # Azure BlobClient.upload_blob works with bytes
                    self._bc.upload_blob(data, overwrite=True)
                except Exception as e:
                    log("warn", "upload_failed", "blob upload failed in writer.close", error=str(e))
                    raise
                finally:
                    super().close()
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False
        return _BlobWriter(blob_client)

    def rm(self, full_path: str) -> None:
        name = self._strip_az_prefix(full_path).lstrip("/")
        blob_client = self.container_client.get_blob_client(name)
        try:
            blob_client.delete_blob()
        except Exception:
            pass

    def delete(self, full_path: str) -> None:
        self.rm(full_path)

# instantiate backend
storage = StorageBackend(CONTAINER)
STORAGE_URL = f"az://{CONTAINER.rstrip('/')}/"

# ------------------- Adapted helper functions (same behaviour) -------------------
def full_path_from_key(key: str) -> str:
    return STORAGE_URL + key.lstrip("/")

def strip_root_from_path(full: str) -> str:
    if full.startswith(STORAGE_URL):
        return full[len(STORAGE_URL):]
    proto_prefix = "az://"
    if full.startswith(proto_prefix):
        rest = full[len(proto_prefix):]
        if rest.startswith(CONTAINER + "/"):
            return rest[len(CONTAINER) + 1 :]
        if rest == CONTAINER:
            return ""
    if full.startswith(CONTAINER + "/"):
        return full[len(CONTAINER) + 1 :]
    return full

def retry(func, retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log("warn", "retry", f"attempt={attempt+1} error={str(e)}")
            time.sleep(delay)
            delay *= backoff

def list_raw_files() -> List[str]:
    base = RAW_PREFIX
    root_path = STORAGE_URL + base
    out: List[str] = []
    try:
        found = storage.find(root_path)
    except Exception:
        try:
            found = storage.glob(root_path + "**")
        except Exception:
            found = []
    for full in found:
        try:
            info_obj = storage.info(full)
        except Exception:
            continue
        if info_obj.get("type") == "directory":
            continue
        rel = strip_root_from_path(full)
        if rel.endswith("/"):
            continue
        if rel.lower().endswith(".manifest.json"):
            continue
        out.append(rel)
    return out

def head_remote_metadata(full_remote_path: str) -> Dict[str, str]:
    try:
        info_obj = storage.info(full_remote_path)
        meta: Dict = {}
        for k in ("metadata", "meta", "Metadata"):
            if k in info_obj and isinstance(info_obj[k], dict):
                meta = info_obj[k]
                break
        if not meta:
            for k, v in info_obj.items():
                if isinstance(v, dict) and any(x in k.lower() for x in ("meta", "metadata", "content")):
                    meta = v
                    break
        return {k.lower(): v for k, v in (meta or {}).items()}
    except Exception:
        return {}

def list_remote_objects(container: str, prefix: str) -> List[Tuple[str, str, int, str]]:
    prefix_key = prefix.rstrip("/") + "/"
    root = STORAGE_URL + prefix_key
    out: List[Tuple[str, str, int, str]] = []
    try:
        found = storage.find(root)
    except Exception:
        try:
            found = storage.glob(root + "**")
        except Exception:
            found = []
    for full in found:
        try:
            info_obj = storage.info(full)
        except Exception:
            continue
        if info_obj.get("type") == "directory":
            continue
        rel = strip_root_from_path(full)
        size = int(info_obj.get("size", 0) or 0)
        etag = ""
        for k in ("etag", "ETag", "eTag"):
            if k in info_obj:
                etag = str(info_obj.get(k) or "")
                break
        out.append((full, rel, size, etag))
    return out

def upload_file_fs(local_path: str, full_remote_path: str, sha256: Optional[str], content_type: str = "application/octet-stream"):
    with open(local_path, "rb") as lf:
        data = lf.read()
    parent = str(Path(full_remote_path).parent)
    try:
        storage.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    try:
        with storage.open(full_remote_path, "wb") as f:
            if hasattr(f, "write"):
                f.write(data)
            else:
                try:
                    f.write(data)
                except Exception:
                    pass
    except Exception as e:
        raise

def download_file_fs(full_remote_path: str, local_target: str):
    target = Path(local_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with storage.open(full_remote_path, "rb") as f:
        data = f.read()
    target.write_bytes(data)
    return {"rel_path": full_remote_path}

def delete_remote_file_fs(full_remote_path: str):
    try:
        storage.rm(full_remote_path)
    except Exception:
        try:
            storage.delete(full_remote_path)
        except Exception:
            pass
    return full_remote_path

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

def compute_md5(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def compute_sha256(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def file_sha256(s3_key: str) -> str:
    full = full_path_from_key(s3_key)
    def _read():
        h = hashlib.sha256()
        with storage.open(full, "rb") as stream:
            for chunk in iter(lambda: stream.read(8192), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    return retry(_read)

def manifest_path(s3_key: str, file_hash: Optional[str] = None) -> str:
    return f"{s3_key}.manifest.json"

def is_already_processed(file_hash: str) -> bool:
    if os.getenv("FORCE_PROCESS", "false").lower() == "true":
        return False
    base_prefix = CHUNKED_PREFIX
    search_prefix = f"{base_prefix}{file_hash}_"
    glob_pattern = STORAGE_URL + search_prefix + "*"
    try:
        matches = storage.glob(glob_pattern)
    except Exception:
        matches = []
    if matches:
        return True
    for ext in ("json", "jsonl"):
        test_key = f"{base_prefix}{file_hash}_1.{ext}"
        full = full_path_from_key(test_key)
        try:
            if storage.exists(full):
                return True
        except Exception:
            pass
    return False

def save_manifest(s3_key: str, manifest: dict) -> bool:
    key = manifest_path(s3_key, manifest.get("file_hash"))
    full = full_path_from_key(key)
    try:
        parent = str(Path(full).parent)
        try:
            storage.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        with storage.open(full, "wb") as f:
            if hasattr(f, "write"):
                f.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
            else:
                try:
                    f.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
                except Exception:
                    pass
        log("info", "saved_manifest", "manifest_written", key=full)
        return True
    except Exception as e:
        log("error", "save_manifest_failed", str(e), key=full)
        return False

def get_format_module(ext: str) -> Optional[str]:
    mapping = {
        "pdf": "pdf", "pptx": "_pptx", "ppt": "_pptx", "html": "_html", "htm": "_html",
        "md": "md", "markdown": "md", "mdown": "md", "txt": "txt",
        "wav": "wav", "mp3": "wav",
        "jpg": "images", "jpeg": "images", "png": "images", "webp": "images",
        "tiff": "images", "tif": "images", "gif": "images", "bmp": "images",
        "csv": "_csv", "jsonl": "jsonl", "ndjson": "jsonl",
    }
    return mapping.get(ext.lower())

def detect_mime(key: str) -> str:
    mime, _ = mimetypes.guess_type(key)
    return mime or "application/octet-stream"

def detect_ext_from_key(some_fs, bucket: str, key: str) -> str:
    k = urllib.parse.unquote(key.split("?", 1)[0].split("#", 1)[0])
    base, ext = os.path.splitext(k)
    ext = ext.lstrip(".").lower()
    if ext in ("markdown", "mdown"):
        ext = "md"
    if ext:
        return ext
    try:
        full = full_path_from_key(key)
        head = storage.info(full)
        ctype = ((head.get("Content-Type") or head.get("content-type") or head.get("content_type") or "")).lower()
        metadata = head.get("metadata") or head.get("meta") or head.get("Metadata") or {}
        meta_fn = metadata.get("filename") or metadata.get("originalname") or ""
        if meta_fn:
            _, mext = os.path.splitext(meta_fn)
            mext = mext.lstrip(".").lower()
            if mext in ("markdown", "mdown"): return "md"
            if mext: return mext
        if "markdown" in ctype or "text/markdown" in ctype: return "md"
        if "text/html" in ctype: return "html"
        if ctype.startswith("text/"): return "txt"
        if "application/pdf" in ctype: return "pdf"
        if "presentation" in ctype or "powerpoint" in ctype or "officedocument.presentationml" in ctype: return "pptx"
        if "wordprocessingml" in ctype or "officedocument.wordprocessingml" in ctype: return "docx"
        if "officedocument.spreadsheetml" in ctype or "excel" in ctype: return "xlsx"
        if ctype.startswith("image/"): return "jpg"
        if ctype.startswith("audio/"): return "wav"
    except Exception:
        pass
    return ""

def load_module_from_path(module_name: str, path: Path):
    loader_name = f"local_formats_{module_name}"
    spec = importlib.util.spec_from_file_location(loader_name, str(path))
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(f"Cannot load module {module_name} from {path}")

def _import_format_module(module_name: str):
    tried: List[str] = []
    for pkg in ("indexing_pipeline.parse_chunk.formats", "parse_chunk.formats"):
        fq = f"{pkg}.{module_name}"
        try:
            return importlib.import_module(fq)
        except Exception:
            tried.append(fq)
    workdir = Path(__file__).resolve().parent.parent
    candidates = [
        workdir / "parse_chunk" / "formats" / f"{module_name}.py",
        workdir / "indexing_pipeline" / "parse_chunk" / "formats" / f"{module_name}.py",
        Path(__file__).resolve().parent / "formats" / f"{module_name}.py",
    ]
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if p.exists():
            try:
                return load_module_from_path(module_name, p)
            except Exception:
                tried.append(str(p))
    raise ImportError(f"Failed to import module for format '{module_name}', tried: {', '.join(tried)}")

def main() -> None:
    run_id = os.getenv("RUN_ID") or str(uuid.uuid4())
    parser_version = os.getenv("PARSER_VERSION", "2.42.1")

    keys = list_raw_files()
    log("info", "scan", "found_files", count=len(keys))

    for key in keys:
        try:
            if key.lower().endswith(".manifest.json"):
                log("debug", "skip", "manifest", key=key)
                continue

            ext = detect_ext_from_key(None, CONTAINER, key)
            module_name = get_format_module(ext)
            if not module_name:
                log("warn", "skip_unsupported", "unsupported_ext", key=key, ext=ext)
                continue

            try:
                mod = _import_format_module(module_name)
            except Exception as e:
                log("error", "import_failed", str(e), module=module_name, key=key)
                continue

            if not hasattr(mod, "parse_file"):
                log("warn", "skip_no_parse", "no_parse_file", module=module_name, key=key)
                continue

            try:
                file_hash = file_sha256(key)
            except Exception as e:
                log("error", "hash_failed", str(e), key=key)
                continue

            if is_already_processed(file_hash):
                log("info", "already_processed", "skipping", file_hash=file_hash, key=key)
                continue

            sd = os.getenv("SOURCE_DATE_EPOCH")
            if sd:
                try:
                    ts = datetime.utcfromtimestamp(int(sd)).isoformat() + "Z"
                except Exception:
                    ts = datetime.utcnow().isoformat() + "Z"
            else:
                ts = datetime.utcnow().isoformat() + "Z"

            manifest = {
                "file_hash": file_hash,
                "s3_key": key,
                "pipeline_run_id": run_id,
                "mime_type": detect_mime(key),
                "timestamp": ts,
                "parser_version": parser_version,
            }

            try:
                result = mod.parse_file(key, manifest)
                if not isinstance(result, dict) or "saved_chunks" not in result:
                    raise ValueError("Invalid parse_file() return. Expected dict with 'saved_chunks'.")
            except Exception as e:
                log("error", "parse_failed", str(e), key=key)
                try:
                    manifest.setdefault("error", str(e))
                    save_manifest(key, manifest)
                except Exception:
                    pass
                continue

            count = int(result.get("saved_chunks", 0) or 0)
            log("info", "parsed", "parsed_and_stored", key=key, saved_chunks=count)
            save_manifest(key, manifest)
        except Exception as exc_outer:
            log("error", "loop_failure", str(exc_outer), key=key if "key" in locals() else None)
            continue

if __name__ == "__main__":
    main()
