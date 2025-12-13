from __future__ import annotations
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

TS = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ")

def log(level: str, event: str, msg: str, **k: Any) -> None:
    o = {"ts": TS(), "level": level, "event": event, "msg": msg}
    if k:
        o.update(k)
    print(json.dumps(o), flush=True)

USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", os.getenv("USE_MANAGED_IDENTITY", "")).strip().lower() in ("1", "true", "yes")
UAI_RAG_RW_CLIENT_ID = os.getenv("UAI_RAG_RW_CLIENT_ID", "").strip()

AZ_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
AZ_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
AZ_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
AZ_SAS = os.getenv("AZURE_SAS_TOKEN", "").strip()
AZ_ENDPOINT_SUFFIX = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net").strip()

AZURE_STRICT_VALIDATE = os.getenv("AZURE_STRICT_VALIDATE", "true").strip().lower() in ("1", "true", "yes")

CONTAINER = os.getenv("AZURE_CONTAINER") or os.getenv("STORAGE_CONTAINER") or os.getenv("AZ_CONTAINER")
if not CONTAINER:
    log("ERROR", "missing_env", "AZURE_CONTAINER (or STORAGE_CONTAINER/AZ_CONTAINER) must be set")
    raise SystemExit(2)

S3_PREFIX = (os.getenv("AZURE_RAW_PREFIX", os.getenv("S3_RAW_PREFIX", "data/raw/")).lstrip("/").rstrip("/") + "/")
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/preconv"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

OVERWRITE_ALL_AUDIO_FILES = os.getenv("OVERWRITE_ALL_AUDIO_FILES", "false").lower() == "true"
OVERWRITE_OTHER_TO_PDF = os.getenv("OVERWRITE_OTHER_TO_PDF", "true").lower() == "true"
OVERWRITE_SPREADSHEETS_WITH_CSV = os.getenv("OVERWRITE_SPREADSHEETS_WITH_CSV", "false").lower() == "true"

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings  # type: ignore
except Exception as e:
    log("ERROR", "import_failure", "missing azure-storage-blob; pip install azure-storage-blob", error=str(e))
    raise SystemExit(2)

if USE_MANAGED_IDENTITY:
    try:
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential  # type: ignore
    except Exception as e:
        log("ERROR", "import_failure", "azure-identity required for managed identity mode (pip install azure-identity)", error=str(e))
        raise SystemExit(2)

def build_blob_service_client() -> "BlobServiceClient":
    if USE_MANAGED_IDENTITY:
        if not AZ_ACCOUNT:
            log("ERROR", "env_missing", "AZURE_STORAGE_ACCOUNT_NAME required for managed identity mode")
            raise SystemExit(2)
        account_url = f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
        try:
            if UAI_RAG_RW_CLIENT_ID:
                try:
                    cred = ManagedIdentityCredential(client_id=UAI_RAG_RW_CLIENT_ID)
                    log("INFO", "mi_cred", "Using ManagedIdentityCredential with client_id", client_id=UAI_RAG_RW_CLIENT_ID)
                except Exception as e_mi:
                    log("WARN", "mi_cred_failed", "ManagedIdentityCredential(client_id=...) failed, trying DefaultAzureCredential", error=str(e_mi))
                    cred = DefaultAzureCredential()
            else:
                cred = DefaultAzureCredential()
                log("INFO", "mi_cred", "Using DefaultAzureCredential for managed identity")
            client = BlobServiceClient(account_url=account_url, credential=cred)
            log("INFO", "client_init", "Initialized BlobServiceClient (managed identity)", account=AZ_ACCOUNT)
            return client
        except Exception as e:
            log("ERROR", "mi_client_failed", "Failed to init BlobServiceClient with managed identity; ensure Workload Identity / MSI available and envs are correct", error=str(e))
            raise SystemExit(2)
    if AZ_CONN:
        try:
            client = BlobServiceClient.from_connection_string(AZ_CONN)
            log("INFO", "client_init", "Initialized BlobServiceClient (connection string)")
            return client
        except Exception as e:
            log("ERROR", "connstr_init_failed", "Failed to init client from connection string", error=str(e))
            raise SystemExit(2)
    if AZ_ACCOUNT and AZ_KEY:
        try:
            account_url = f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
            client = BlobServiceClient(account_url=account_url, credential=AZ_KEY)
            log("INFO", "client_init", "Initialized BlobServiceClient (account + key)", account=AZ_ACCOUNT)
            return client
        except Exception as e:
            log("ERROR", "acctkey_init_failed", "Failed to init client from account+key", error=str(e))
            raise SystemExit(2)
    if AZ_ACCOUNT and AZ_SAS:
        try:
            account_url = f"https://{AZ_ACCOUNT}.{AZ_ENDPOINT_SUFFIX}"
            token = AZ_SAS if AZ_SAS.startswith("?") else ("?" + AZ_SAS)
            client = BlobServiceClient(account_url=account_url + token)
            log("INFO", "client_init", "Initialized BlobServiceClient (account + SAS)", account=AZ_ACCOUNT)
            return client
        except Exception as e:
            log("ERROR", "sas_init_failed", "Failed to init client from SAS token", error=str(e))
            raise SystemExit(2)
    msg = "non-managed identity mode requires AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME+AZURE_STORAGE_ACCOUNT_KEY or AZURE_SAS_TOKEN"
    log("ERROR", "auth_missing", msg)
    raise SystemExit(2)

BLOB_SERVICE = build_blob_service_client()
try:
    CONTAINER_CLIENT = BLOB_SERVICE.get_container_client(CONTAINER)
except Exception as e:
    log("ERROR", "container_client_failed", f"Unable to get container client for {CONTAINER}", error=str(e))
    raise SystemExit(2)

if AZURE_STRICT_VALIDATE:
    try:
        CONTAINER_CLIENT.get_container_properties()
        log("INFO", "container_validation", f"container '{CONTAINER}' exists and is accessible")
    except Exception as e:
        log("ERROR", "container_validation_failed", f"Failed to validate container '{CONTAINER}'. Check auth and network.", error=str(e))
        raise SystemExit(2)
else:
    try:
        CONTAINER_CLIENT.get_container_properties()
        log("INFO", "container_validation", f"container '{CONTAINER}' exists and is accessible")
    except Exception as e:
        log("WARN", "container_validation_skipped", f"Could not validate container '{CONTAINER}', continuing (AZURE_STRICT_VALIDATE=false)", error=str(e))

def compute_hashes(path: str, chunk_size: int = 8*1024*1024) -> Tuple[str, str]:
    md5 = hashlib.md5(); sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            md5.update(chunk); sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()

def sanitize_key(k: str) -> str:
    k = k.lower()
    safe = []
    for c in k:
        if ('a' <= c <= 'z') or ('0' <= c <= '9') or c == '_':
            safe.append(c)
        else:
            safe.append('_')
    out = ''.join(safe).strip('_')
    if not out:
        out = 'm'
    if out[0].isdigit():
        out = "_" + out
    return out[:128]

def encode_value(v: str) -> str:
    if v is None: return ""
    b = v.encode('utf-8')
    return base64.b64encode(b).decode('ascii')

def decode_value(v: str) -> str:
    try:
        return base64.b64decode(v.encode('ascii')).decode('utf-8')
    except Exception:
        return v

def prepare_metadata(d: Dict[str, Optional[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (d or {}).items():
        kk = sanitize_key(str(k))
        vv = encode_value(str(v)) if v is not None else ""
        out[kk] = vv
    return out

def get_blob_props(name: str) -> Dict[str, Any]:
    try:
        bc = CONTAINER_CLIENT.get_blob_client(name)
        props = bc.get_blob_properties()
        md = props.metadata or {}
        content_md5 = None
        try:
            if getattr(props, "content_settings", None) is not None:
                content_md5 = getattr(props.content_settings, "content_md5", None)
        except Exception:
            content_md5 = None
        return {"exists": True, "etag": props.etag, "size": int(getattr(props, "size", 0) or 0), "content_md5": content_md5, "metadata": md}
    except Exception:
        return {"exists": False}

def download_blob(name: str, dst: str) -> bool:
    try:
        bc = CONTAINER_CLIENT.get_blob_client(name)
        stream = bc.download_blob()
        data = stream.readall()
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        log("WARN", "download_failed", f"download failed for {name}", error=str(e))
        return False

def upload_blob(name: str, src_path: str, metadata: Dict[str, str], content_type: str = "application/octet-stream", overwrite: bool = True) -> Dict[str, Any]:
    bc = CONTAINER_CLIENT.get_blob_client(name)
    cs = ContentSettings(content_type=content_type)
    try:
        with open(src_path, "rb") as data:
            bc.upload_blob(data, overwrite=overwrite, content_settings=cs, metadata=metadata)
        props = bc.get_blob_properties()
        remote_meta = props.metadata or {}
        verified = all((k in remote_meta and remote_meta[k] == metadata[k]) for k in metadata)
        return {"action": "uploaded", "verified": bool(verified), "etag": props.etag}
    except Exception as e:
        log("ERROR", "az_upload_failed", f"upload failed {name}", error=str(e))
        return {"action": "failed", "error": str(e)}

def run_soffice_convert(src: str, outdir: str, convert_to: str) -> Tuple[bool, str]:
    env = os.environ.copy()
    env["SAL_USE_VCLPLUGIN"] = env.get("SAL_USE_VCLPLUGIN", "gen")
    env["HOME"] = env.get("HOME", "/tmp")
    cmd = [
        "soffice", "--headless", "--invisible", "--nologo", "--nodefault",
        "--nofirststartwizard", "--nolockcheck", "--convert-to", convert_to,
        "--outdir", outdir, src
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=120)
        stderr = res.stderr.decode('utf-8', errors='ignore').strip()
        if res.returncode == 0:
            return True, ""
        return False, (stderr.splitlines()[:6] and "\n".join(stderr.splitlines()[:6])) or stderr
    except Exception as e:
        return False, str(e)

def run_ffmpeg_convert(src: str, dst: str, extra_args: Optional[list] = None) -> Tuple[bool, str]:
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst]
    if extra_args:
        args = args[:-1] + extra_args + [args[-1]]
    try:
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if res.returncode == 0:
            return True, ""
        out = (res.stderr.decode('utf-8', errors='ignore').splitlines()[:6])
        return False, ("\n".join(out) if out else "ffmpeg error")
    except Exception as e:
        return False, str(e)

def process_audio(key: str):
    name = Path(key).name
    ext = name.split('.')[-1].lower()
    if ext not in ("mp3","m4a","aac","wav","flac","ogg","opus","webm","amr","wma","aiff","aif"):
        return
    src_local = str(TMP_DIR / "src" / name)
    out_local = str(TMP_DIR / "out" / (Path(name).stem + ".wav"))
    os.makedirs(os.path.dirname(src_local), exist_ok=True)
    os.makedirs(os.path.dirname(out_local), exist_ok=True)
    if not download_blob(key, src_local):
        log("WARN", "audio_download_failed", f"download failed {key}")
        return
    src_md5, src_sha = compute_hashes(src_local)
    src_props = get_blob_props(key)
    src_etag = (src_props.get("etag") or "").strip('"') if src_props.get("exists") else ""
    ok, err = run_ffmpeg_convert(src_local, out_local)
    if not ok:
        log("ERROR", "audio_convert_failed", f"ffmpeg failed for {key}", error=(err or "unknown"))
        try: os.remove(src_local)
        except Exception: pass
        return
    if ext == "wav":
        s3_target_key = f"{S3_PREFIX}audio/{name}"
    else:
        s3_target_key = f"{S3_PREFIX}audio/{Path(name).stem}.wav"
    metadata = prepare_metadata({
        "sha256": src_sha,
        "converted-from": key,
        "converted-etag": src_etag,
        "converted-sr": "16000",
        "converted-ch": "1",
    })
    up = upload_blob(s3_target_key, out_local, metadata, content_type="audio/wav", overwrite=True)
    if up.get("action") == "uploaded":
        log("INFO", "audio_uploaded", "Uploaded converted audio", target=s3_target_key, result=up)
        if OVERWRITE_ALL_AUDIO_FILES and s3_target_key != key:
            try:
                CONTAINER_CLIENT.delete_blob(key)
                log("INFO", "audio_deleted_old", "Deleted original audio", name=key)
            except Exception as e:
                log("WARN", "delete_old_failed", "failed deleting original audio", name=key, error=str(e))
    else:
        log("ERROR", "audio_upload_failed", "Upload failed", target=s3_target_key, result=up)
    try:
        os.remove(src_local)
        os.remove(out_local)
    except Exception:
        pass

def process_doc(key: str):
    name = Path(key).name
    ext = name.split('.')[-1].lower()
    if ext not in ("doc","docx"):
        return
    src_local = str(TMP_DIR / "src" / name)
    outdir = str(TMP_DIR / "out")
    os.makedirs(os.path.dirname(src_local), exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    if not download_blob(key, src_local):
        log("WARN", "doc_download_failed", f"download failed {key}")
        return
    ok, err = run_soffice_convert(src_local, outdir, "pdf:writer_pdf_Export")
    if not ok:
        qname = f"{S3_PREFIX}quarantine/{name}.corrupt"
        metadata = prepare_metadata({"quarantined_from": key, "error": err})
        up = upload_blob(qname, src_local, metadata, content_type="application/octet-stream", overwrite=True)
        log("WARN", "doc_convert_failed", "soffice failed or could not open file", key=key, err=(err or "unknown"), quarantine=qname, result=up)
        try: os.remove(src_local)
        except Exception: pass
        return
    out_pdf = None
    for f in Path(outdir).glob(f"{Path(name).stem}*.pdf"):
        out_pdf = str(f); break
    if not out_pdf:
        log("WARN", "doc_no_pdf", "conversion produced no pdf", source=key)
        try: os.remove(src_local)
        except Exception: pass
        return
    tgt_key = f"{S3_PREFIX}pdfs/{name}.pdf"
    src_md5, src_sha = compute_hashes(src_local)
    src_props = get_blob_props(key)
    src_etag = (src_props.get("etag") or "").strip('"') if src_props.get("exists") else ""
    metadata = prepare_metadata({"sha256": src_sha, "converted-from": key, "converted-etag": src_etag})
    up = upload_blob(tgt_key, out_pdf, metadata, content_type="application/pdf", overwrite=True)
    if up.get("action") == "uploaded":
        log("INFO", "doc_uploaded", "Uploaded pdf", target=tgt_key, result=up)
        if OVERWRITE_OTHER_TO_PDF:
            try:
                CONTAINER_CLIENT.delete_blob(key)
                log("INFO", "doc_deleted_old", "Deleted original doc", name=key)
            except Exception as e:
                log("WARN", "delete_old_failed", "failed deleting original doc", name=key, error=str(e))
    else:
        log("ERROR", "doc_upload_failed", "Upload failed", target=tgt_key, result=up)
    try:
        os.remove(src_local)
        os.remove(out_pdf)
    except Exception:
        pass

def process_sheet(key: str):
    name = Path(key).name
    ext = name.split('.')[-1].lower()
    if ext not in ("xls","xlsx","ods","xlsm","xlsb"):
        return
    src_local = str(TMP_DIR / "src" / name)
    outdir = str(TMP_DIR / "out")
    os.makedirs(os.path.dirname(src_local), exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    if not download_blob(key, src_local):
        log("WARN", "sheet_download_failed", f"download failed {key}")
        return
    ok, err = run_soffice_convert(src_local, outdir, "csv")
    if not ok:
        log("WARN", "sheet_convert_failed", "soffice csv conversion produced no output", key=key, err=(err or "unknown"))
        try: os.remove(src_local)
        except Exception: pass
        return
    created = list(Path(outdir).glob("*.csv"))
    if not created:
        log("WARN", "sheet_no_csv", "no csvs produced", key=key)
        try: os.remove(src_local)
        except Exception: pass
        return
    for f in created:
        tgt_key = f"{S3_PREFIX}csvs/{name}.{f.name}"
        src_md5, src_sha = compute_hashes(src_local)
        src_props = get_blob_props(key)
        src_etag = (src_props.get("etag") or "").strip('"') if src_props.get("exists") else ""
        metadata = prepare_metadata({"sha256": src_sha, "converted-from": key, "converted-etag": src_etag})
        up = upload_blob(tgt_key, str(f), metadata, content_type="text/csv", overwrite=True)
        if up.get("action") == "uploaded":
            log("INFO", "sheet_uploaded", "Uploaded csv", target=tgt_key, result=up)
    if OVERWRITE_SPREADSHEETS_WITH_CSV:
        try:
            CONTAINER_CLIENT.delete_blob(key)
            log("INFO", "sheet_deleted_old", "Deleted original sheet", name=key)
        except Exception as e:
            log("WARN", "delete_old_failed", "failed deleting original sheet", name=key, error=str(e))
    try:
        os.remove(src_local)
        for f in created:
            try: os.remove(str(f))
            except Exception: pass
    except Exception:
        pass

def group_remaining():
    prefix = S3_PREFIX
    try:
        generator = CONTAINER_CLIENT.list_blobs(name_starts_with=prefix)
    except Exception as e:
        log("ERROR", "list_blobs_failed", "failed listing blobs for grouping", error=str(e))
        return
    for b in generator:
        try:
            if b.name.endswith("/") or b.name.endswith(".manifest.json"):
                continue
            rel = b.name[len(prefix):] if b.name.startswith(prefix) else b.name
            ext = rel.split('.')[-1].lower()
            if ext in ("mp3","m4a","wav","flac","ogg","opus","webm","amr","wma","aiff","aif"):
                sub="audio/"
            elif ext in ("jpg","jpeg","png","webp","tif","tiff","bmp","gif"):
                sub="images/"
            elif ext == "pdf":
                sub="pdfs/"
            elif ext in ("doc","docx"):
                sub="docs/"
            elif ext in ("ppt","pptx"):
                sub="ppts/"
            elif ext in ("xls","xlsx","ods","xlsm","xlsb"):
                sub="others/"
            elif ext in ("txt",):
                sub="txts/"
            elif ext in ("csv",):
                sub="csvs/"
            elif ext in ("md",):
                sub="mds/"
            else:
                sub="others/"
            dst = prefix + sub + Path(rel).name
            if dst == b.name:
                continue
            tgt_props = get_blob_props(dst)
            if tgt_props.get("exists"):
                try:
                    CONTAINER_CLIENT.delete_blob(b.name)
                    log("INFO", "group_deleted_old", "deleted original", name=b.name)
                except Exception as e:
                    log("WARN", "group_delete_failed", "failed deleting", name=b.name, error=str(e))
                continue
            tmp_local = str(TMP_DIR / "src" / Path(rel).name)
            os.makedirs(os.path.dirname(tmp_local), exist_ok=True)
            if download_blob(b.name, tmp_local):
                md5, sha = compute_hashes(tmp_local)
                metadata = prepare_metadata({"sha256": sha})
                up = upload_blob(dst, tmp_local, metadata, overwrite=True)
                if up.get("action") == "uploaded":
                    try:
                        CONTAINER_CLIENT.delete_blob(b.name)
                    except Exception:
                        pass
                    log("INFO", "group_uploaded", "group uploaded", from_name=b.name, to=dst, result=up)
                else:
                    log("WARN", "group_move_failed_upload", "upload failed", to=dst)
                try: os.remove(tmp_local)
                except Exception: pass
        except Exception as e:
            log("WARN", "group_iteration_failed", "skipping blob", blob=getattr(b, "name", None), error=str(e))

def main():
    log("INFO", "startup", "pre_conversions python start", container=CONTAINER, prefix=S3_PREFIX, use_managed_identity=str(USE_MANAGED_IDENTITY))
    os.makedirs(TMP_DIR / "src", exist_ok=True)
    os.makedirs(TMP_DIR / "out", exist_ok=True)
    try:
        blobs = CONTAINER_CLIENT.list_blobs(name_starts_with=S3_PREFIX)
    except Exception as e:
        log("ERROR", "list_failed", "failed listing blobs", error=str(e))
        raise SystemExit(2)
    keys = [b.name for b in blobs]
    for key in keys:
        rel = key[len(S3_PREFIX):] if key.startswith(S3_PREFIX) else key
        ext = rel.split('.')[-1].lower()
        try:
            if ext in ("mp3","m4a","aac","wav","flac","ogg","opus","webm","amr","wma","aiff","aif"):
                process_audio(key)
            elif ext in ("doc","docx"):
                process_doc(key)
            elif ext in ("xls","xlsx","ods","xlsm","xlsb"):
                process_sheet(key)
        except Exception as e:
            log("ERROR", "processing_failed", f"processing {key} failed", error=str(e))
    group_remaining()
    log("INFO", "finished", "pre_conversions python completed")

if __name__ == "__main__":
    main()
