#!/usr/bin/env python3
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

# filesystem (may be optional depending on mode)
try:
    import fsspec
    from fsspec.spec import AbstractFileSystem  # type: ignore
except Exception:
    fsspec = None
    AbstractFileSystem = object  # type: ignore

# --- logging setup: silence noisy third-party loggers ---
_root = logging.getLogger()
_root.setLevel(logging.WARNING)
_noisy = (
    "adlfs",
    "azure",
    "azure.storage",
    "azure.core",
    "azure.identity",
    "urllib3",
    "botocore",
    "requests",
    "httpx",
)
for n in _noisy:
    lg = logging.getLogger(n)
    lg.setLevel(logging.WARNING)
    lg.propagate = False


def ts_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


CONTAINER = (
    os.getenv("AZURE_CONTAINER")
    or os.getenv("STORAGE_CONTAINER")
    or os.getenv("AZ_CONTAINER")
)
if not CONTAINER:
    print(
        json.dumps(
            {
                "ts": ts_now(),
                "level": "error",
                "event": "startup",
                "msg": "AZURE_CONTAINER (or STORAGE_CONTAINER or AZ_CONTAINER) must be set",
            }
        ),
        file=sys.stderr,
    )
    sys.exit(1)

RAW_PREFIX = (
    (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("S3_RAW_PREFIX") or "data/raw/")
    .rstrip("/")
    + "/"
)
CHUNKED_PREFIX = (
    (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("S3_CHUNKED_PREFIX") or "data/chunked/")
    .rstrip("/")
    + "/"
)


def log(level: str, event: str, msg: str, **extra) -> None:
    o = {
        "ts": ts_now(),
        "level": level,
        "event": event,
        "msg": msg,
        "container": CONTAINER,
    }
    if extra:
        o.update(extra)
    print(json.dumps(o, ensure_ascii=False), flush=True)


# ------------------- Deterministic auth switch -------------------
# Use AZURE_USE_MANAGED_IDENTITY exclusively (true/false)
USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower() in ("1", "true", "yes")

# ------------------- Azure SDK detection helpers -------------------
def azure_storage_sdk_available() -> bool:
    try:
        import azure.storage.blob  # noqa: F401
        return True
    except Exception:
        return False


# Helper to build fsspec options (same as before)
def build_storage_options() -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        opts["connection_string"] = conn
        return opts
    acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
    key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
    sas = os.environ.get("AZURE_SAS_TOKEN")
    eps = os.environ.get("AZURE_ENDPOINT_SUFFIX") or "core.windows.net"
    if acct and key:
        opts["account_name"] = acct
        opts["account_key"] = key
        opts["endpoint_suffix"] = eps
        return opts
    if acct and sas:
        opts["account_name"] = acct
        opts["sas_token"] = sas
        opts["endpoint_suffix"] = eps
        return opts
    if os.environ.get("AZURE_ANON"):
        if acct:
            opts["account_name"] = acct
        opts["anon"] = True
        return opts
    return opts


STORAGE_URL = f"az://{CONTAINER.rstrip('/')}/"


# Storage backend abstraction implementing the small surface used across this file
class StorageBackend:
    def __init__(self, container: str):
        self.container = container
        self.storage_url = f"az://{container.rstrip('/')}/"
        self.fs: Optional[AbstractFileSystem] = None
        self.blob_service = None
        self.container_client = None

        # Endpoint suffix
        endpoint_suffix = os.environ.get("AZURE_ENDPOINT_SUFFIX", "core.windows.net")

        # MANAGED IDENTITY PATH
        if USE_MANAGED_IDENTITY:
            # require azure.identity + azure.storage.blob
            try:
                from azure.identity import DefaultAzureCredential  # type: ignore
                from azure.storage.blob import BlobServiceClient  # type: ignore
            except Exception as e:
                log(
                    "error",
                    "azure_sdk_missing",
                    "azure.identity + azure.storage.blob required for managed identity mode",
                    error=str(e),
                )
                # Fail fast: MI selected -> cannot silently fallback
                sys.exit(2)

            account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
            if not account_name:
                log(
                    "error",
                    "config",
                    "AZURE_STORAGE_ACCOUNT_NAME required when AZURE_USE_MANAGED_IDENTITY=true",
                )
                sys.exit(2)
            account_url = f"https://{account_name}.{endpoint_suffix}"
            try:
                cred = DefaultAzureCredential()
                self.blob_service = BlobServiceClient(account_url=account_url, credential=cred)
                self.container_client = self.blob_service.get_container_client(self.container)
                log("info", "storage.init", "managed_identity_initialized", account=account_name)
            except Exception as e:
                # MI chosen but couldn't create client -> fail fast, with helpful guidance.
                log(
                    "error",
                    "blobclient.managed.init.failed",
                    "failed creating BlobServiceClient with managed identity. Are you running in an environment with Workload Identity / MSI available? If not, set AZURE_USE_MANAGED_IDENTITY=0 and provide AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_CONNECTION_STRING or AZURE_SAS_TOKEN.",
                    error=str(e),
                )
                sys.exit(2)
            return

        # NON-MANAGED IDENTITY PATH: prefer azure.storage.blob if available
        sdk_ok = azure_storage_sdk_available()
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        acct = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") or os.environ.get("AZURE_ACCOUNT_NAME")
        acct_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY") or os.environ.get("AZURE_ACCOUNT_KEY")
        sas = os.environ.get("AZURE_SAS_TOKEN")

        if sdk_ok:
            try:
                from azure.storage.blob import BlobServiceClient  # type: ignore
                # 1) connection string
                if conn_str:
                    try:
                        self.blob_service = BlobServiceClient.from_connection_string(conn_str)  # type: ignore
                        self.container_client = self.blob_service.get_container_client(self.container)
                        log("info", "storage.init", "azure_sdk_connstr", method="connection_string")
                        return
                    except Exception as e:
                        log("warn", "connstr.failed", "from_connection_string failed", error=str(e))
                # 2) account key
                if acct and acct_key:
                    try:
                        account_url = f"https://{acct}.{endpoint_suffix}"
                        self.blob_service = BlobServiceClient(account_url=account_url, credential=acct_key)  # type: ignore
                        self.container_client = self.blob_service.get_container_client(self.container)
                        log("info", "storage.init", "azure_sdk_account_key", account=acct)
                        return
                    except Exception as e:
                        log("warn", "account_key.failed", "BlobServiceClient(account_key) failed", error=str(e))
                # 3) SAS token
                if acct and sas:
                    try:
                        token = sas if sas.startswith("?") else ("?" + sas)
                        account_url = f"https://{acct}.{endpoint_suffix}{token}"
                        self.blob_service = BlobServiceClient(account_url=account_url)  # type: ignore
                        self.container_client = self.blob_service.get_container_client(self.container)
                        log("info", "storage.init", "azure_sdk_sas", account=acct)
                        return
                    except Exception as e:
                        log("warn", "sas.failed", "BlobServiceClient(sas) failed", error=str(e))
                # If azure SDK present but none of above succeeded -> fall through to fsspec fallback (if available)
                log("info", "storage.init", "azure_sdk_present_but_no_valid_credentials", account=acct is not None)
            except Exception as e:
                log("warn", "azure_sdk_import.failed", "import azure.storage.blob failed after availability check", error=str(e))

        # fsspec/adlfs fallback – requires fsspec installed (and adlfs plugin)
        opts = build_storage_options()
        if not opts:
            # no mechanism to authenticate in non-MI mode
            log(
                "error",
                "no_credentials",
                "non-managed-identity mode requires AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY) or (AZURE_STORAGE_ACCOUNT_NAME + AZURE_SAS_TOKEN) or AZURE_ANON",
            )
            sys.exit(2)

        if fsspec is None:
            log(
                "error",
                "fsspec_missing",
                "fsspec (and adlfs) required for key/SAS mode fallback (pip install fsspec adlfs)",
            )
            sys.exit(2)

        try:
            self.fs = fsspec.filesystem("az", **opts)  # type: ignore
            log("info", "storage.init", "fsspec_initialized", opts_summary=list(opts.keys()))
        except Exception as e:
            log("error", "fsspec_init_failed", "failed to init fsspec az", error=str(e))
            sys.exit(2)

    # helpers to convert paths
    def _strip_az_prefix(self, full: str) -> str:
        # given az://container/... or container/... or ... -> return blob name
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
        # root_path is like STORAGE_URL + base
        if self.fs is not None:
            try:
                return self.fs.find(root_path)  # type: ignore
            except Exception:
                try:
                    return self.fs.glob(root_path + "**", recursive=True)  # type: ignore
                except Exception:
                    return []
        else:
            # blob listing
            prefix = self._strip_az_prefix(root_path)
            prefix = prefix.lstrip("/")
            out: List[str] = []
            try:
                for b in self.container_client.list_blobs(name_starts_with=prefix):
                    out.append(f"az://{self.container}/{b.name}")
            except Exception as e:
                log("warn", "list_blobs_failed", "list_blobs error", error=str(e))
                return []
            return out

    def glob(self, pattern: str) -> List[str]:
        if self.fs is not None:
            try:
                return self.fs.glob(pattern, recursive=True)  # type: ignore
            except Exception:
                return []
        else:
            prefix = self._strip_az_prefix(pattern)
            prefix = prefix.lstrip("/")
            out: List[str] = []
            try:
                for b in self.container_client.list_blobs(name_starts_with=prefix):
                    out.append(f"az://{self.container}/{b.name}")
            except Exception as e:
                log("warn", "glob_failed", "list_blobs error", error=str(e))
                return []
            return out

    def info(self, full_path: str) -> Dict[str, Any]:
        if self.fs is not None:
            try:
                return self.fs.info(full_path)  # type: ignore
            except Exception:
                raise
        else:
            name = self._strip_az_prefix(full_path).lstrip("/")
            blob_client = self.container_client.get_blob_client(name)
            props = blob_client.get_blob_properties()
            meta = getattr(props, "metadata", {}) or {}
            # return keys similar to fsspec info
            info_obj: Dict[str, Any] = {
                "size": int(getattr(props, "size", 0) or 0),
                "etag": getattr(props, "etag", "") or "",
                "ETag": getattr(props, "etag", "") or "",
                "eTag": getattr(props, "etag", "") or "",
                "Content-Type": (getattr(props, "content_settings", None).content_type if getattr(props, "content_settings", None) else getattr(props, "content_type", "")) or "",
                "content-type": (getattr(props, "content_settings", None).content_type if getattr(props, "content_settings", None) else getattr(props, "content_type", "")) or "",
                "content_type": (getattr(props, "content_settings", None).content_type if getattr(props, "content_settings", None) else getattr(props, "content_type", "")) or "",
                "last_modified": getattr(props, "last_modified", ""),
                "Last-Modified": getattr(props, "last_modified", ""),
                "metadata": meta,
                "meta": meta,
                "type": "file",
            }
            return info_obj

    def exists(self, full_path: str) -> bool:
        if self.fs is not None:
            try:
                return self.fs.exists(full_path)  # type: ignore
            except Exception:
                return False
        else:
            name = self._strip_az_prefix(full_path).lstrip("/")
            blob_client = self.container_client.get_blob_client(name)
            try:
                return blob_client.exists()
            except Exception:
                return False

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        # blob storage doesn't need directories - noop
        if self.fs is not None:
            try:
                if hasattr(self.fs, "makedirs"):
                    self.fs.makedirs(path, exist_ok=exist_ok)  # type: ignore
            except Exception:
                pass

    def open(self, full_path: str, mode: str = "rb"):
        if self.fs is not None:
            return self.fs.open(full_path, mode)  # type: ignore
        else:
            name = self._strip_az_prefix(full_path).lstrip("/")
            blob_client = self.container_client.get_blob_client(name)
            if "r" in mode:
                stream = blob_client.download_blob()
                data = stream.readall()
                return io.BytesIO(data)
            # write modes: provide a context manager that uploads on close
            class _BlobWriter(io.BytesIO):
                def __init__(self, bc):
                    super().__init__()
                    self._bc = bc

                def close(self):
                    try:
                        self.seek(0)
                        data = self.read()
                        # upload_blob expects bytes
                        self._bc.upload_blob(data, overwrite=True)
                    except Exception as e:
                        log("warn", "upload_failed", "blob upload failed in writer.close", error=str(e))
                    super().close()

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    self.close()
                    return False

            return _BlobWriter(blob_client)

    def rm(self, full_path: str) -> None:
        if self.fs is not None:
            try:
                self.fs.rm(full_path)  # type: ignore
            except Exception:
                try:
                    self.fs.delete(full_path)  # type: ignore
                except Exception:
                    pass
        else:
            name = self._strip_az_prefix(full_path).lstrip("/")
            blob_client = self.container_client.get_blob_client(name)
            try:
                blob_client.delete_blob()
            except Exception:
                pass

    def delete(self, full_path: str) -> None:
        # alias
        self.rm(full_path)


# instantiate backend
storage = StorageBackend(CONTAINER)


# ------------------- original functions (adapted to storage backend) -------------------

def full_path_from_key(key: str) -> str:
    return STORAGE_URL + key.lstrip("/")


def strip_root_from_path(full: str) -> str:
    if full.startswith(STORAGE_URL):
        return full[len(STORAGE_URL) :]
    proto_prefix = "az://"
    if full.startswith(proto_prefix):
        rest = full[len(proto_prefix) :]
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
    # ensure directory/parent exists when using implementations that require it
    parent = str(Path(full_remote_path).parent)
    try:
        storage.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    # use storage.open for write
    try:
        with storage.open(full_remote_path, "wb") as f:
            if hasattr(f, "write"):
                f.write(data)
            else:
                # fallback — write via blob client possibly returned a BytesIO-like
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
                # attempt to write via fallback
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
        "pdf": "pdf",
        "pptx": "_pptx",
        "ppt": "_pptx",
        "html": "_html",
        "htm": "_html",
        "md": "md",
        "markdown": "md",
        "mdown": "md",
        "txt": "txt",
        "wav": "wav",
        "mp3": "wav",
        "jpg": "images",
        "jpeg": "images",
        "png": "images",
        "webp": "images",
        "tiff": "images",
        "tif": "images",
        "gif": "images",
        "bmp": "images",
        "csv": "_csv",
        "jsonl": "jsonl",
        "ndjson": "jsonl",
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

    # No extension on filename; inspect metadata / content-type
    try:
        full = full_path_from_key(key)
        head = storage.info(full)
        ctype = (
            (head.get("Content-Type") or head.get("content-type") or head.get("content_type") or "")
            .lower()
        )
        metadata = head.get("metadata") or head.get("meta") or head.get("Metadata") or {}
        meta_fn = metadata.get("filename") or metadata.get("originalname") or ""
        if meta_fn:
            _, mext = os.path.splitext(meta_fn)
            mext = mext.lstrip(".").lower()
            if mext in ("markdown", "mdown"):
                return "md"
            if mext:
                return mext
        if "markdown" in ctype or "text/markdown" in ctype:
            return "md"
        if "text/html" in ctype:
            return "html"
        if ctype.startswith("text/"):
            return "txt"
        if "application/pdf" in ctype:
            return "pdf"
        if "presentation" in ctype or "powerpoint" in ctype or "officedocument.presentationml" in ctype:
            return "pptx"
        if "wordprocessingml" in ctype or "officedocument.wordprocessingml" in ctype:
            return "docx"
        if "officedocument.spreadsheetml" in ctype or "excel" in ctype:
            return "xlsx"
        if ctype.startswith("image/"):
            return "jpg"
        if ctype.startswith("audio/"):
            return "wav"
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
