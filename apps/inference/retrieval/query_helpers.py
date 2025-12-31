#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import mimetypes
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus, urlparse

try:
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
except Exception:
    generate_blob_sas = None
    BlobSasPermissions = None

ENV = os.getenv("ENV", "STAGING").upper()
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
AZURE_ENDPOINT_SUFFIX = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")

RERANKER_MODE = os.getenv("RERANKER_MODE", "AUTO").upper()
RERANK_TOPK = int(os.getenv("RERANK_TOPK", os.getenv("RERANKER_TOP_K", "20")))
RERANKER_TOP_K = RERANK_TOPK
RERANK_AUTO_THRESHOLD = float(os.getenv("RERANK_AUTO_THRESHOLD", "0.75"))
RERANK_THRESHOLD = int(os.getenv("RERANK_THRESHOLD", "30"))
RERANK_MARGIN = float(os.getenv("RERANK_MARGIN", "0.08"))
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.6"))
MAX_CHUNKS_TO_LLM = int(os.getenv("MAX_CHUNKS_TO_LLM", "6"))
QUERY_TOPK_DENSE = int(os.getenv("QUERY_TOPK_DENSE", os.getenv("QUERY_TOPK", "200")))
QUERY_TOPK_SPARSE = int(os.getenv("QUERY_TOPK_SPARSE", "200"))
RRF_TOP_N = int(os.getenv("RRF_TOP_N", "10"))

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
    try:
        t = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', content)
        t = re.sub(r'(?is)<[^>]+>', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    except Exception:
        return re.sub(r'\s+', ' ', content or "").strip()

def _full_text_from_payload(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    if payload.get("content"):
        return str(payload.get("content") or "")
    if payload.get("text"):
        return str(payload.get("text") or "")
    if payload.get("html"):
        return _strip_html(str(payload.get("html") or ""))
    h = payload.get("headings") or payload.get("heading_path") or payload.get("title") or ""
    if isinstance(h, (list, tuple)):
        return " - ".join([str(x) for x in h])
    return str(h or "")

def ui_fields_from_payload(payload: Dict[str, Any], prefer_snippet_len: Optional[int] = None) -> List[Tuple[str, Any]]:
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
        account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "") or ""
        if account:
            return account, container, blob
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "") or ""
        if conn:
            acct_from_conn, _ = _extract_account_key_from_connection_string(conn)
            if acct_from_conn:
                return acct_from_conn, container, blob
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME env required for az:// paths (or supply AZURE_STORAGE_CONNECTION_STRING containing AccountName)")

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
        acct = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "") or ""
        if acct:
            container, blob = path.split("/", 1)
            return acct, container, blob
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "") or ""
        if conn:
            acct_from_conn, _ = _extract_account_key_from_connection_string(conn)
            if acct_from_conn:
                container, blob = path.split("/", 1)
                return acct_from_conn, container, blob
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME env required when passing container/blob style path (or supply AZURE_STORAGE_CONNECTION_STRING containing AccountName)")

    raise ValueError("unrecognized path format; expected az://, https://... or container/blob")

def _extract_account_key_from_connection_string(conn: str) -> Tuple[Optional[str], Optional[str]]:
    if not conn:
        return None, None
    parts = dict()
    for part in conn.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            parts[k.strip().lower()] = v.strip()
    account = parts.get("accountname")
    key = parts.get("accountkey")
    return account, key

def presign_azure_blob_blocking(path: str, expires: int = 3600, inline: bool = True) -> str:
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

    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "") or ""
    if conn:
        acct_from_conn, acct_key = _extract_account_key_from_connection_string(conn)
        if acct_key:
            sas = generate_blob_sas(
                account_name=account or acct_from_conn,
                container_name=container,
                blob_name=blob,
                account_key=acct_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry,
                start=start,
                content_disposition=content_disp,
                content_type=content_type,
            )
            return f"{endpoint}/{container}/{quote_plus(blob)}?{sas}"

    acct_key_env = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "") or ""
    acct_name_env = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "") or ""
    if acct_key_env and (acct_name_env or account):
        acct_name = acct_name_env or account
        sas = generate_blob_sas(
            account_name=acct_name,
            container_name=container,
            blob_name=blob,
            account_key=acct_key_env,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            start=start,
            content_disposition=content_disp,
            content_type=content_type,
        )
        return f"{endpoint}/{container}/{quote_plus(blob)}?{sas}"

    sas_token = os.getenv("AZURE_SAS_TOKEN", "") or ""
    if sas_token:
        token = sas_token if sas_token.startswith("?") else ("?" + sas_token)
        return f"{endpoint}/{container}/{quote_plus(blob)}{token}"

    raise RuntimeError("No AZURE_STORAGE_CONNECTION_STRING with AccountKey, nor AZURE_STORAGE_ACCOUNT_KEY, nor AZURE_SAS_TOKEN configured for presign")
