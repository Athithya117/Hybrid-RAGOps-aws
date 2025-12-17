#!/usr/bin/env python3
"""
Query helpers (Azure-only) — UI-field selection, Azure presign, and small utilities.

Notes:
 - This file is Azure-only; S3/boto3 removed.
 - The UI field builder does NOT produce a 'snippet' key. Full text can be obtained
   via _full_text_from_payload(payload) and will be added to UI meta_items as ("content", <full-text>)
   by the caller (query.py) when required.
 - presign_azure_blob_blocking(path, expires=3600, inline=True) returns an Azure SAS URL.
"""
from __future__ import annotations
import os
import re
import mimetypes
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus, urlparse

# Azure imports used only by presign function (lazy usage allowed)
try:
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient
except Exception:
    generate_blob_sas = None  # type: ignore
    BlobSasPermissions = None  # type: ignore
    BlobServiceClient = None  # type: ignore

try:
    from azure.identity import DefaultAzureCredential
except Exception:
    DefaultAzureCredential = None  # type: ignore

# env knobs (Azure-focused)
ENV = os.getenv("ENV", "STAGING").upper()
AZURE_USE_MANAGED_IDENTITY = os.getenv("AZURE_USE_MANAGED_IDENTITY", "").strip().lower() in ("1", "true", "yes")
if ENV == "PROD":
    AZURE_USE_MANAGED_IDENTITY = True

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
AZURE_ENDPOINT_SUFFIX = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")


def iso_ts():
    return datetime.now(timezone.utc).isoformat()


def _ext_from_url_or_name(val: Optional[str]) -> str:
    if not val:
        return ""
    try:
        base = val.split("?", 1)[0].split("#", 1)[0]
        _, ext = base.rsplit(".", 1) if "." in base else ("", "")
        return ext.strip().lower()
    except Exception:
        return ""


def _guess_content_type_from_key(key: str) -> Optional[str]:
    try:
        ctype, _ = mimetypes.guess_type(key)
        return ctype
    except Exception:
        return None


def _detect_type(file_type: Optional[str], source_url: Optional[str], file_name: Optional[str], chunk_type: Optional[str]) -> str:
    ft = (file_type or "").lower()
    if ft:
        if "pdf" in ft: return "pdf"
        if "presentation" in ft or "powerpoint" in ft or "ppt" in ft or "pptx" in ft: return "pptx"
        if ft.startswith("audio/"): return "audio"
        if ft.startswith("image/"): return "image"
        if "csv" in ft: return "csv"
        if "json" in ft and "jsonl" in ft: return "jsonl"
        if "markdown" in ft: return "md"
        if "html" in ft or "xml" in ft: return "html"
        if "text" in ft: return "txt"
    ext = _ext_from_url_or_name(source_url) or _ext_from_url_or_name(file_name)
    if ext in ("pdf",): return "pdf"
    if ext in ("ppt", "pptx", "pptm", "odp"): return "pptx"
    if ext in ("mp3", "wav", "m4a", "flac", "aac", "ogg"): return "audio"
    if ext in ("jpg", "jpeg", "png", "webp", "tiff", "tif", "gif", "bmp"): return "image"
    if ext in ("csv", "tsv"): return "csv"
    if ext in ("json", "jsonl", "ndjson"): return "jsonl"
    if ext in ("md", "markdown"): return "md"
    if ext in ("html", "htm", "xhtml"): return "html"
    if ext in ("txt", "text"): return "txt"
    ct = (chunk_type or "").lower()
    if "audio" in ct: return "audio"
    if "slide" in ct or "slides" in ct: return "pptx"
    if "row" in ct or "csv" in ct: return "csv"
    if "image" in ct or "frame" in ct: return "image"
    return "unknown"


def _strip_html(content: str) -> str:
    # remove scripts/styles and tags conservatively
    try:
        t = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', content)
        t = re.sub(r'(?is)<[^>]+>', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    except Exception:
        return re.sub(r'\s+', ' ', content or "").strip()


def _full_text_from_payload(payload: Dict[str, Any]) -> str:
    """
    Return the full textual content for a chunk, with minimal processing (preserve content).
    This intentionally does NOT truncate.
    Order of preference: content -> text -> html (stripped) -> headings/title.
    """
    if not isinstance(payload, dict):
        return ""
    if payload.get("content"):
        return str(payload.get("content") or "")
    if payload.get("text"):
        return str(payload.get("text") or "")
    if payload.get("html"):
        return _strip_html(str(payload.get("html") or ""))
    # fallback to headings/title concatenation
    h = payload.get("headings") or payload.get("heading_path") or payload.get("title") or ""
    if isinstance(h, (list, tuple)):
        return " - ".join([str(x) for x in h])
    return str(h or "")


def ui_fields_from_payload(payload: Dict[str, Any], prefer_snippet_len: Optional[int] = None) -> List[Tuple[str, Any]]:
    """
    Build ordered UI fields from a payload.
    IMPORTANT: This function DOES NOT add a 'snippet' field. Callers that want the full text
    should call _full_text_from_payload(payload) and insert ("content", <full_text>) if needed.
    """
    p = payload or {}
    file_name = p.get("file_name") or (p.get("source_url") or "").split("/")[-1] or None
    source_url = p.get("source_url") or p.get("s3_path") or p.get("raw_key") or None
    file_type = p.get("file_type") or None
    chunk_type = p.get("chunk_type") or None
    detected = _detect_type(file_type, source_url, file_name, chunk_type)

    ordered: List[Tuple[str, Any]] = []

    if source_url:
        ordered.append(("source_url", source_url))
    if file_name:
        ordered.append(("file_name", file_name))
    if p.get("chunk_id"):
        ordered.append(("chunk_id", p.get("chunk_id")))
    if p.get("chunk_index") is not None:
        ordered.append(("chunk_index", p.get("chunk_index")))
    if p.get("token_count") is not None:
        try:
            ordered.append(("token_count", int(p.get("token_count"))))
        except Exception:
            ordered.append(("token_count", p.get("token_count")))

    # additional fields by detected type
    if detected == "pdf":
        if p.get("page_number") is not None:
            ordered.append(("page_number", int(p.get("page_number"))))
        if p.get("line_start") is not None or p.get("line_end") is not None:
            ls = int(p.get("line_start") or 0); le = int(p.get("line_end") or 0)
            ordered.append(("line_range", [ls, le]))
        if p.get("layout_tags"):
            ordered.append(("layout_tags", p.get("layout_tags")))
        if p.get("figures"):
            ordered.append(("figures", p.get("figures")))
        if p.get("semantic_region"):
            ordered.append(("semantic_region", p.get("semantic_region")))
    elif detected == "pptx":
        if p.get("slide_range"):
            ordered.append(("slide_range", p.get("slide_range")))
        if p.get("layout"):
            ordered.append(("layout", p.get("layout")))
    elif detected == "audio":
        if p.get("audio_range"):
            ordered.append(("audio_range", p.get("audio_range")))
        if p.get("duration"):
            ordered.append(("duration", p.get("duration")))
    elif detected == "csv":
        if p.get("row_range"):
            ordered.append(("row_range", p.get("row_range")))
        if p.get("headings"):
            ordered.append(("headings", p.get("headings")))
        if p.get("semantic_region"):
            ordered.append(("semantic_region", p.get("semantic_region")))
    elif detected == "image":
        if p.get("layout_bbox"):
            ordered.append(("layout_bbox", p.get("layout_bbox")))
        if p.get("used_ocr") is not None:
            ordered.append(("used_ocr", bool(p.get("used_ocr"))))
    elif detected == "html":
        if p.get("headings"):
            ordered.append(("headings", p.get("headings")))
        if p.get("line_range"):
            ordered.append(("line_range", p.get("line_range")))
        if p.get("semantic_region"):
            ordered.append(("semantic_region", p.get("semantic_region")))
    elif detected in ("md", "txt"):
        if p.get("headings"):
            ordered.append(("headings", p.get("headings")))
        if p.get("line_range"):
            ordered.append(("line_range", p.get("line_range")))
        if p.get("semantic_region"):
            ordered.append(("semantic_region", p.get("semantic_region")))
    elif detected == "jsonl":
        if p.get("line_range"):
            ordered.append(("line_range", p.get("line_range")))
        if p.get("semantic_region"):
            ordered.append(("semantic_region", p.get("semantic_region")))
    else:
        if p.get("headings"):
            ordered.append(("headings", p.get("headings")))
        if p.get("line_range"):
            ordered.append(("line_range", p.get("line_range")))
        if p.get("semantic_region"):
            ordered.append(("semantic_region", p.get("semantic_region")))

    if p.get("tags"):
        ordered.append(("tags", p.get("tags")))

    # optional verbose fields
    if os.getenv("UI_VERBOSE", "false").lower() in ("1", "true", "yes"):
        if p.get("parser_version"):
            ordered.append(("parser_version", p.get("parser_version")))
        if p.get("timestamp"):
            ordered.append(("timestamp", p.get("timestamp")))
        if p.get("file_type"):
            ordered.append(("file_type", p.get("file_type")))
        if p.get("document_id"):
            ordered.append(("document_id", p.get("document_id")))

    out = [(k, v) for k, v in ordered if v is not None and v != ""]
    return out


# -----------------------
# Azure path parsing & presign (blocking)
# -----------------------
def _parse_az_path(path: str) -> Tuple[str, str, str]:
    if not path:
        raise ValueError("empty path")
    path = path.strip()
    if path.startswith("az://"):
        stripped = path[5:]
        parts = stripped.split("/", 1)
        if len(parts) != 2:
            raise ValueError("az:// path must be az://container/blob")
        container, blob = parts[0], parts[1]
        account = AZURE_STORAGE_ACCOUNT_NAME or ""
        if not account:
            raise ValueError("AZURE_STORAGE_ACCOUNT_NAME env required for az:// paths")
        return account, container, blob
    if path.startswith("http://") or path.startswith("https://"):
        u = urlparse(path)
        hostparts = u.netloc.split(".")
        if not hostparts:
            raise ValueError("invalid blob URL")
        account = hostparts[0]
        p = u.path.lstrip("/")
        parts = p.split("/", 1)
        if len(parts) != 2:
            raise ValueError("blob URL must include container and blob path")
        container, blob = parts[0], parts[1]
        return account, container, blob
    if "/" in path:
        if not AZURE_STORAGE_ACCOUNT_NAME:
            raise ValueError("AZURE_STORAGE_ACCOUNT_NAME env required when passing container/blob style path")
        container, blob = path.split("/", 1)
        return AZURE_STORAGE_ACCOUNT_NAME, container, blob
    raise ValueError("unrecognized path format; expected az://, https://... or container/blob")


def presign_azure_blob_blocking(path: str, expires: int = 3600, inline: bool = True) -> str:
    """
    Generate a read-only SAS URL for the blob at `path` (Azure-only).
    Modes:
      - If AZURE_SAS_TOKEN configured -> append token to resource URL.
      - If AZURE_STORAGE_ACCOUNT_KEY present and not using managed identity -> account-key SAS via generate_blob_sas.
      - Else use DefaultAzureCredential to obtain a user-delegation SAS (requires RBAC).
    This tries both keyword and positional get_user_delegation_key signatures for SDK compatibility.
    """
    if generate_blob_sas is None:
        raise RuntimeError("azure.storage.blob not installed; install azure-storage-blob")

    account, container, blob = _parse_az_path(path)
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=5)
    expiry = now + timedelta(seconds=int(expires))
    endpoint = f"https://{account}.blob.{AZURE_ENDPOINT_SUFFIX}"
    content_type = _guess_content_type_from_key(blob) or "application/octet-stream"
    content_disp = None
    if inline:
        filename = blob.split("/")[-1] or "file"
        content_disp = f'inline; filename="{filename}"'

    # If explicit SAS token provided, apply it (simplest)
    if AZURE_SAS_TOKEN:
        token = AZURE_SAS_TOKEN if AZURE_SAS_TOKEN.startswith("?") else ("?" + AZURE_SAS_TOKEN)
        return f"{endpoint}/{container}/{quote_plus(blob)}{token}"

    # Account key -> account-level SAS
    if AZURE_STORAGE_ACCOUNT_KEY and not AZURE_USE_MANAGED_IDENTITY:
        sas = generate_blob_sas(
            account_name=account,
            container_name=container,
            blob_name=blob,
            account_key=AZURE_STORAGE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            start=start,
            content_disposition=content_disp,
            content_type=content_type,
        )
        return f"{endpoint}/{container}/{quote_plus(blob)}?{sas}"

    # Connection string -> try client (if present)
    if AZURE_STORAGE_CONNECTION_STRING and not AZURE_USE_MANAGED_IDENTITY:
        try:
            bsc = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BlobServiceClient from AZURE_STORAGE_CONNECTION_STRING: {e}")

    # User-delegation SAS via DefaultAzureCredential
    if DefaultAzureCredential is None or BlobServiceClient is None:
        raise RuntimeError("azure.identity or azure.storage.blob missing for user-delegation SAS")

    cred = DefaultAzureCredential()
    try:
        bsc = BlobServiceClient(account_url=endpoint, credential=cred)
    except Exception as e:
        raise RuntimeError(f"Failed to create BlobServiceClient with DefaultAzureCredential: {e}")

    # SDK signature compatibility: try keyword args first, fall back to positional
    try:
        try:
            udk = bsc.get_user_delegation_key(key_start_time=start, key_expiry_time=expiry)
        except TypeError:
            udk = bsc.get_user_delegation_key(start, expiry)
    except Exception as e:
        raise RuntimeError(f"Failed to obtain user delegation key (ensure RBAC and time propagation): {e}")

    sas = generate_blob_sas(
        account_name=account,
        container_name=container,
        blob_name=blob,
        user_delegation_key=udk,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
        start=start,
        content_disposition=content_disp,
        content_type=content_type,
    )
    return f"{endpoint}/{container}/{quote_plus(blob)}?{sas}"
