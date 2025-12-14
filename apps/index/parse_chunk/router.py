#!/usr/bin/env python3
"""
Robust router for parse_chunk formats.

Goals:
 - Deterministic startup and strict auth validation.
 - Try to import user format modules; if import fails, record full traceback and
   attach a fallback parser so files are not skipped. Every file gets a manifest:
   either successful parse (saved_chunks > 0) or an error manifest (saved_chunks=0).
 - Log full tracebacks for import failures and parse exceptions.
 - Be defensive when reading numeric envs and when calling external libs.
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
import importlib.machinery
import mimetypes
import urllib.parse
import logging
import io
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# ---------------- logging ----------------
_root = logging.getLogger()
_root.setLevel(logging.WARNING)
for n in ("urllib3", "requests", "httpx", "azure", "adlfs"):
    lg = logging.getLogger(n)
    lg.setLevel(logging.WARNING)
    lg.propagate = False

def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def log(level: str, event: str, msg: str, **extra):
    o = {"ts": now_ts(), "level": level, "event": event, "msg": msg}
    if extra:
        o.update(extra)
    # prefer JSON lines to stderr for structured logging
    out = json.dumps(o, ensure_ascii=False)
    if level in ("error", "warn", "warning"):
        print(out, file=sys.stderr, flush=True)
    else:
        print(out, flush=True)

# ---------------- Config & validation ----------------
CONTAINER = (
    os.getenv("AZURE_CONTAINER")
    or os.getenv("STORAGE_CONTAINER")
    or os.getenv("AZ_CONTAINER")
)
if not CONTAINER:
    log("error", "startup", "AZURE_CONTAINER (or STORAGE_CONTAINER or AZ_CONTAINER) must be set")
    sys.exit(1)

# deterministic auth switch
USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower() in ("1", "true", "yes")

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
        log("error", "config", f"Missing required envs for managed identity mode: {', '.join(missing)}")
        sys.exit(2)
else:
    if not AZ_CONN:
        log("error", "config", "Non-managed-identity mode requires AZURE_STORAGE_CONNECTION_STRING (mount it as a secret)")
        sys.exit(2)

# prefixes
RAW_PREFIX = (os.getenv("STORAGE_RAW_PREFIX") or os.getenv("S3_RAW_PREFIX") or "data/raw/").rstrip("/") + "/"
CHUNKED_PREFIX = (os.getenv("STORAGE_CHUNKED_PREFIX") or os.getenv("S3_CHUNKED_PREFIX") or "data/chunked/").rstrip("/") + "/"

# ---------------- Azure client factory (deterministic) ----------------
def build_blob_service_client():
    if USE_MANAGED_IDENTITY:
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except Exception as e:
            log("error", "azure_import", "azure.identity and azure.storage.blob are required for managed identity mode", error=str(e))
            raise SystemExit(2)
        account_url = f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
        try:
            cred = DefaultAzureCredential(managed_identity_client_id=UAI_RAG_RW_CLIENT_ID)
            client = BlobServiceClient(account_url=account_url, credential=cred)
            # validate container
            try:
                _ = client.get_container_client(CONTAINER).get_container_properties()
            except Exception as e_check:
                log("error", "mi_validation_failed", "Managed Identity client created but container validation failed; verify WI, role assignment, and network/DNS", error=str(e_check))
                raise SystemExit(2)
            log("info", "client_init", "Initialized BlobServiceClient (managed identity)", account=AZ_ACCOUNT)
            return client
        except SystemExit:
            raise
        except Exception as e:
            log("error", "mi_client_failed", "Failed to initialize BlobServiceClient with managed identity", error=str(e))
            raise SystemExit(2)
    else:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except Exception as e:
            log("error", "azure_import", "azure.storage.blob required for connection-string mode", error=str(e))
            raise SystemExit(2)
        try:
            client = BlobServiceClient.from_connection_string(AZ_CONN)
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
            log("error", "connstr_failed", "Failed to initialize BlobServiceClient.from_connection_string", error=str(e))
            raise SystemExit(2)

# ---------------- StorageBackend ----------------
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
        return self.find(pattern)

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
            return {
                "size": int(getattr(props, "size", 0) or 0),
                "etag": getattr(props, "etag", "") or "",
                "Content-Type": content_type,
                "content-type": content_type,
                "content_type": content_type,
                "last_modified": getattr(props, "last_modified", ""),
                "metadata": meta,
                "type": "file",
            }
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
        return

    def open(self, full_path: str, mode: str = "rb"):
        name = self._strip_az_prefix(full_path).lstrip("/")
        blob_client = self.container_client.get_blob_client(name)
        if "r" in mode:
            stream = blob_client.download_blob()
            data = stream.readall()
            return io.BytesIO(data)
        class _BlobWriter(io.BytesIO):
            def __init__(self, bc):
                super().__init__()
                self._bc = bc
            def close(self):
                try:
                    self.seek(0)
                    data = self.read()
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

# instantiate
storage = StorageBackend(CONTAINER)
STORAGE_URL = f"az://{CONTAINER.rstrip('/')}/"

# ---------------- helpers ----------------
def ts_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

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

def save_manifest(s3_key: str, manifest: dict) -> bool:
    key = f"{s3_key}.manifest.json"
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

def detect_mime(key: str) -> str:
    mime, _ = mimetypes.guess_type(key)
    return mime or "application/octet-stream"

def detect_ext_from_key(key: str) -> str:
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

# mapping ext -> module name
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

# ---------------- robust import machinery ----------------
MODULE_CACHE: Dict[str, Any] = {}

def load_module_by_name(pkg_candidates: List[str]) -> Any:
    """
    Try to import by package-qualified name(s). Raise last exception on failure.
    """
    last_exc = None
    for name in pkg_candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_exc = e
    raise last_exc

def load_module_from_path(module_name: str, path: Path):
    """Load module from a specific file path with full traceback on error"""
    loader_name = f"local_formats_{module_name}"
    spec = importlib.util.spec_from_file_location(loader_name, str(path))
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        except Exception:
            raise
    raise ImportError(f"Cannot load module {module_name} from {path}")

def make_fallback_parser(module_name: str, ext: str):
    """
    Create a fallback module object with parse_file(key, manifest) that:
      - Writes an error manifest (saved_chunks = 0) with import traceback
      - Returns {'saved_chunks': 0}
    This ensures the router never silently skips files.
    """
    import types
    mod = types.SimpleNamespace()
    def parse_file(key: str, manifest: dict) -> dict:
        err_msg = manifest.get("error", "")
        fallback_error = f"Fallback parser used for module '{module_name}' (ext='{ext}'). Original error: {err_msg}"
        # include diagnostic timestamp and node
        manifest.setdefault("error", fallback_error)
        # attempt to save manifest (router will write it)
        return {"saved_chunks": 0}
    mod.parse_file = parse_file
    return mod

def _import_format_module(module_name: str, ext_hint: str):
    """
    Robust loader: try package imports, then file-system loads; on any import failure
    produce a fallback parser instead of raising. Full tracebacks are logged.
    """
    if module_name in MODULE_CACHE:
        return MODULE_CACHE[module_name]
    tried = []
    # try package imports first (two common pkg roots)
    pkg_roots = ("indexing_pipeline.parse_chunk.formats", "parse_chunk.formats")
    pkg_candidates = [f"{root}.{module_name}" for root in pkg_roots]
    try:
        try:
            m = load_module_by_name(pkg_candidates)
            MODULE_CACHE[module_name] = m
            return m
        except Exception as e_pkg:
            tried.extend(pkg_candidates)
            # try filesystem candidates (workdir relative)
            workdir = Path(__file__).resolve().parent.parent
            candidates = [
                workdir / "parse_chunk" / "formats" / f"{module_name}.py",
                workdir / "indexing_pipeline" / "parse_chunk" / "formats" / f"{module_name}.py",
                Path(__file__).resolve().parent / "formats" / f"{module_name}.py",
            ]
            for p in candidates:
                try:
                    p_res = p.resolve()
                except Exception:
                    continue
                if p_res.exists():
                    try:
                        m = load_module_from_path(module_name, p_res)
                        MODULE_CACHE[module_name] = m
                        return m
                    except Exception:
                        tb = traceback.format_exc()
                        tried.append(str(p_res))
                        log("error", "import_failed_traceback", f"Failed importing module file {p_res}", module=module_name, traceback=tb)
            # if we reach here, all import attempts failed
            tb_pkg = "".join(traceback.format_exception_only(type(e_pkg), e_pkg)).strip()
            log("error", "import_failed", f"Cannot import module '{module_name}' (ext hint '{ext_hint}'). Will use fallback parser. Package error: {tb_pkg}", tried=";".join(tried))
            # produce fallback with saved traceback in manifest later
            fallback = make_fallback_parser(module_name, ext_hint)
            MODULE_CACHE[module_name] = fallback
            return fallback
    except Exception as e:
        # catch-all: create fallback and log complete traceback
        tb = traceback.format_exc()
        log("error", "import_unexpected", f"Unexpected import error for module '{module_name}'; using fallback", module=module_name, traceback=tb)
        fallback = make_fallback_parser(module_name, ext_hint)
        MODULE_CACHE[module_name] = fallback
        return fallback

# ---------------- main processing ----------------
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

            ext = detect_ext_from_key(key)
            module_name = get_format_module(ext)
            if not module_name:
                # If ext is empty or unknown, still attempt to run a generic fallback parser that
                # writes an "unsupported_ext" manifest (not skipped).
                log("warn", "skip_unsupported", "unsupported_ext", key=key, ext=ext)
                fake_module = make_fallback_parser("unknown", ext)
                mod = fake_module
            else:
                mod = _import_format_module(module_name, ext)

            # ensure parse_file exists
            if not hasattr(mod, "parse_file"):
                log("warn", "skip_no_parse", "no_parse_file", module=getattr(mod, "__name__", str(mod)), key=key)
                # create fallback parse_file that notes missing function and continues
                mod = make_fallback_parser(getattr(mod, "__name__", "anon"), ext)

            # compute file hash before parsing
            try:
                file_hash = file_sha256(key)
            except Exception as e:
                log("error", "hash_failed", f"file hash failed: {str(e)}", key=key)
                # still create an error manifest
                manifest = {
                    "file_hash": None,
                    "s3_key": key,
                    "pipeline_run_id": run_id,
                    "mime_type": detect_mime(key),
                    "timestamp": now_ts(),
                    "parser_version": parser_version,
                    "error": f"hash_failed: {str(e)}",
                }
                try:
                    save_manifest(key, manifest)
                except Exception:
                    pass
                continue

            if is_already_processed(file_hash):
                log("info", "already_processed", "skipping", file_hash=file_hash, key=key)
                continue

            ts = datetime.utcnow().isoformat() + "Z"
            manifest = {
                "file_hash": file_hash,
                "s3_key": key,
                "pipeline_run_id": run_id,
                "mime_type": detect_mime(key),
                "timestamp": ts,
                "parser_version": parser_version,
            }

            # Now actually call parse_file with strong exception capture and full tracebacks
            try:
                result = mod.parse_file(key, manifest)
                if not isinstance(result, dict) or "saved_chunks" not in result:
                    raise ValueError("Invalid parse_file() return. Expected dict with 'saved_chunks'.")
            except Exception as e:
                tb = traceback.format_exc()
                # augment manifest with error and traceback
                manifest.setdefault("error", str(e))
                manifest.setdefault("traceback", tb)
                try:
                    save_manifest(key, manifest)
                except Exception:
                    pass
                log("error", "parse_failed", f"parse_file raised: {str(e)}", key=key, module=getattr(mod, "__name__", str(mod)), traceback=tb)
                continue

            count = int(result.get("saved_chunks", 0) or 0)
            log("info", "parsed", "parsed_and_stored", key=key, saved_chunks=count)
            try:
                save_manifest(key, manifest)
            except Exception as e:
                log("warn", "manifest_save_failed", "failed to save manifest after parse", key=key, error=str(e))
        except Exception as exc_outer:
            tb = traceback.format_exc()
            log("error", "loop_failure", str(exc_outer), key=key if 'key' in locals() else None, traceback=tb)
            try:
                # best-effort manifest if possible
                if 'key' in locals():
                    save_manifest(key, {"file_hash": None, "s3_key": key, "pipeline_run_id": run_id, "error": str(exc_outer), "traceback": tb})
            except Exception:
                pass
            continue

if __name__ == "__main__":
    main()
