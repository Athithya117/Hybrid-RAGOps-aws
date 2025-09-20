#!/usr/bin/env python3
# indexing_pipeline/ray_indexing_pipeline.py
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import numpy as np
import ray
from boto3.s3.transfer import TransferConfig
from ray import serve
from ray.util.placement_group import placement_group, remove_placement_group
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests

# try to import local parse streaming function if present
try:
    from parse_chunk.router import stream_chunks as local_stream_chunks  # type: ignore
except Exception:
    local_stream_chunks = None

# Optional metrics module (silently ignored if absent)
try:
    import metrics  # type: ignore
except Exception:
    metrics = None

# ---- Environment & defaults ----
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DEFAULT_WORKDIR = os.environ.get("WORKDIR", "/workspace/indexing_pipeline")
LIBREOFFICE_SCRIPT = "parse_chunk/pre_conversions/libreoffice_server.sh"
PRE_CONVERSION_SCRIPTS = [
    "parse_chunk/pre_conversions/group_similar_raw_files.py",
    "parse_chunk/pre_conversions/all_audio_to_wav.py",
    "parse_chunk/pre_conversions/doc_docx_to_pdf.py",
    "parse_chunk/pre_conversions/spreadsheets_to_csv.py",
    "parse_chunk/pre_conversions/ppt_to_pptx.py",
]
ROUTER = "parse_chunk/router.py"

# Embedding & batching
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "512"))
QDRANT_UPSERT_BATCH_SIZE = int(
    os.environ.get(
        "QDRANT_UPSERT_BATCH_SIZE",
        os.environ.get("QDRANT_UPSERT_BATCH_SIZES", str(EMBED_BATCH_SIZE)),
    )
)
MAX_EMBED_CALL_RETRIES = int(os.environ.get("EMBED_CALL_MAX_RETRIES", "3"))
UPsert_MAX_RETRIES = int(os.environ.get("INDEX_UPSERT_MAX_RETRIES", "3"))
UPsert_BASE_BACKOFF = float(os.environ.get("INDEX_UPSERT_BASE_BACKOFF", "1.0"))

# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_PRIVATE_IP", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION_NAME", "my_documents")
QDRANT_SHARDS = int(os.environ.get("QDRANT_SHARDS", "1"))
QDRANT_REPLICATION_FACTOR = int(os.environ.get("QDRANT_REPLICATION_FACTOR", "1"))
QDRANT_ON_DISK_PAYLOAD = (os.environ.get("QDRANT__STORAGE__PAYLOAD_ON_DISK", "false").lower() == "true")

# S3 / chunks
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_CHUNKED_PREFIX = (os.environ.get("S3_CHUNKED_PREFIX", "") or "").lstrip("/").rstrip("/") + "/"
CHUNK_FORMAT = (os.environ.get("CHUNK_FORMAT", "json") or "json").lower()
STORE_ONE_FILE_PER_CHUNK = (os.environ.get("STORE_ONE_FILE_PER_CHUNK", "true").lower() == "true")

# Embedding model constraints
TEXT_SNIPPET_LEN = int(os.environ.get("TEXT_SNIPPET_LEN", "300"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1200"))

# Embedder discovery
EMBEDDER_ACTOR_NAME = os.environ.get("EMBEDDER_ACTOR_NAME", "embedder-actor")
EMBEDDER_SERVE_DEPLOYMENT = os.environ.get("EMBEDDER_SERVE_DEPLOYMENT", "Embedder")

# Ray settings (optional overrides)
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")
RAY_NAMESPACE = os.environ.get("RAY_NAMESPACE")

# Snapshot/backup settings
ENABLE_QDRANT_SNAPSHOT = (os.environ.get("ENABLE_QDRANT_SNAPSHOT", "false").lower() == "true")
SNAPSHOT_S3_BUCKET = os.environ.get("SNAPSHOT_S3_BUCKET")
SNAPSHOT_S3_PREFIX = (os.environ.get("SNAPSHOT_S3_PREFIX", "") or "").lstrip("/").rstrip("/") + "/"
# Local snapshot folder (where Qdrant writes snapshots on host/container). Pipeline must be able to read this path.
BACKUP_FROM_PATH = os.environ.get("BACKUP_FROM_PATH", "/workspace/qdrant/backups/snapshots/")
# Where snapshot contents map into node storage path (info only)
BACKUP_AND_RESTORE_PATH = os.environ.get("BACKUP_AND_RESTORE_PATH", "/workspace/qdrant/data/")
BACKUP_TO_PREFIX = os.environ.get("BACKUP_TO_PREFIX", "qdrant/backups")
MULTIPART_CHUNKSIZE_MB = int(os.environ.get("MULTIPART_CHUNKSIZE_MB", "100"))
BACKUP_MAX_CONCURRENCY = os.environ.get("BACKUP_MAX_CONCURRENCY")  # optional numeric
OUTPUT_MANIFEST_KEY = os.environ.get("OUTPUT_MANIFEST_KEY", "latest_qdrant_backup.manifest.json")
SNAPSHOT_POLL_SECONDS = int(os.environ.get("SNAPSHOT_POLL_SECONDS", "3"))
SNAPSHOT_WAIT_TIMEOUT = int(os.environ.get("SNAPSHOT_WAIT_TIMEOUT", "900"))  # seconds to wait for snapshot file to appear

# Metrics
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9000"))

# Logging
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
root.addHandler(handler)
root.setLevel(LOG_LEVEL)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger("ray_indexing_pipeline")

# boto3 client (region/creds via env/profile/instance role)
s3 = boto3.client("s3")

# Qdrant singleton
_qdrant_client_singleton: Optional[QdrantClient] = None


def qdrant_client() -> QdrantClient:
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        _qdrant_client_singleton = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    return _qdrant_client_singleton


# ---- ID / payload helpers ----
def point_id_for(chunk: Dict[str, Any]) -> str:
    doc = chunk.get("document_id") or chunk.get("doc_id") or chunk.get("source") or "unknown-doc"
    chk = chunk.get("chunk_id") or chunk.get("id") or str(abs(hash(chunk.get("text", "")[:64])))
    return f"{doc}::{chk}"


def payload_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for k, v in chunk.items():
        if k in ("embedding", "text", "vector"):
            continue
        payload[k] = v
    text = chunk.get("text")
    if text:
        payload["text_snippet"] = text[:TEXT_SNIPPET_LEN]
    if "parser_version" in chunk:
        payload["parser_version"] = chunk["parser_version"]
    if "source_url" in chunk:
        payload["source_url"] = chunk["source_url"]
    if "document_id" in chunk:
        payload["document_id"] = chunk["document_id"]
    return payload


def make_qdrant_points(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[dict]:
    pts: List[dict] = []
    for chunk, emb in zip(chunks, embeddings):
        pid = point_id_for(chunk)
        payload = payload_from_chunk(chunk)
        pts.append({"id": pid, "vector": list(map(float, emb)), "payload": payload})
    return pts


# ---- Qdrant collection/index creation ----
def ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    mappings = [
        ("document_id", models.PayloadSchemaType.UUID),
        ("chunk_id", models.PayloadSchemaType.UUID),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ("file_type", models.PayloadSchemaType.KEYWORD),
        ("text_checksum", models.PayloadSchemaType.KEYWORD),
        ("tags", models.PayloadSchemaType.KEYWORD),
        ("layout_tags", models.PayloadSchemaType.KEYWORD),
        ("timestamp", models.PayloadSchemaType.DATETIME),
        ("page_number", models.PayloadSchemaType.INTEGER),
        ("token_count", models.PayloadSchemaType.INTEGER),
        ("used_ocr", models.PayloadSchemaType.BOOL),
        ("chunk_duration_ms", models.PayloadSchemaType.INTEGER),
        ("headings", models.PayloadSchemaType.KEYWORD),
        ("heading_path", models.PayloadSchemaType.KEYWORD),
    ]
    for field, schema in mappings:
        try:
            client.create_payload_index(collection_name=collection, field_name=field, field_schema=schema)
        except Exception:
            pass


def ensure_collection(client: QdrantClient, collection: str, vector_size: Optional[int] = None) -> None:
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection in existing:
            ensure_payload_indexes(client, collection)
            return
        vs = int(vector_size) if vector_size is not None else 768
        client.create_collection(
            collection_name=collection,
            vector_size=vs,
            distance="Cosine",
            shards=QDRANT_SHARDS,
            replication_factor=QDRANT_REPLICATION_FACTOR,
            on_disk_payload=QDRANT_ON_DISK_PAYLOAD,
        )
        logger.info("Created Qdrant collection %s vector_size=%s", collection, vs)
        ensure_payload_indexes(client, collection)
    except Exception as e:
        logger.warning("ensure_collection warning: %s", e)


# ---- chunk iteration (S3 or local stream) ----
def parse_json_objects_from_text(blob_text: str) -> List[dict]:
    import re

    out: List[dict] = []
    blob_text = blob_text.strip()
    if not blob_text:
        return out
    try:
        parsed = json.loads(blob_text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)]
    except Exception:
        pass
    lines = [ln for ln in blob_text.splitlines() if ln.strip()]
    for ln in lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    if out:
        return out
    parts = [p.strip() for p in re.split(r"\n\s*\n", blob_text) if p.strip()]
    for part in parts:
        try:
            obj = json.loads(part)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    if out:
        return out
    json_obj_re = re.compile(r"\{(?:[^{}]|(?R))*\}", flags=re.DOTALL) if hasattr(re, "compile") else None
    if json_obj_re:
        for m in json_obj_re.finditer(blob_text):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    if out:
        return out
    logger.warning("Failed to robustly parse combined file into JSON objects (len=%d)", len(blob_text))
    return out


def iter_chunks_from_s3_prefix() -> Iterable[Dict[str, Any]]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set and no local parser present")
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": S3_BUCKET, "Prefix": S3_CHUNKED_PREFIX}
    logger.info("Listing S3 chunk files under s3://%s/%s", S3_BUCKET, S3_CHUNKED_PREFIX)
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []) if page.get("Contents") else []:
            key = obj.get("Key")
            if not key or key.endswith("/"):
                continue
            logger.debug("Found chunk file s3://%s/%s", S3_BUCKET, key)
            try:
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                text = body.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("Failed to read s3://%s/%s: %s", S3_BUCKET, key, e)
                continue
            if STORE_ONE_FILE_PER_CHUNK:
                if CHUNK_FORMAT == "jsonl":
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                            yield obj
                        except Exception:
                            logger.warning("Skipping invalid jsonl line in %s", key)
                            continue
                else:
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            yield parsed
                        elif isinstance(parsed, list):
                            for p in parsed:
                                if isinstance(p, dict):
                                    yield p
                    except Exception:
                        objs = parse_json_objects_from_text(text)
                        for o in objs:
                            yield o
            else:
                if CHUNK_FORMAT == "jsonl":
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                            yield obj
                        except Exception:
                            logger.warning("Skipping invalid jsonl line in combined file %s", key)
                            continue
                else:
                    objs = []
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            objs = [parsed]
                        elif isinstance(parsed, list):
                            objs = [p for p in parsed if isinstance(p, dict)]
                    except Exception:
                        objs = parse_json_objects_from_text(text)
                    for o in objs:
                        yield o


# ---- embedder discovery & calls (actor or serve) ----
def get_embedder_actor(name: str = EMBEDDER_ACTOR_NAME):
    try:
        actor = ray.get_actor(name)
        logger.info("Found embedder actor: %s", name)
        return actor
    except Exception:
        return None


def get_serve_handle(deployment: str = EMBEDDER_SERVE_DEPLOYMENT):
    try:
        handle = serve.get_deployment(deployment).get_handle(sync=False)
        logger.info("Found serve handle: %s", deployment)
        return handle
    except Exception:
        return None


def call_embedder_via_actor(actor, texts: List[str], max_length: Optional[int] = None) -> List[List[float]]:
    last_exc = None
    for attempt in range(1, MAX_EMBED_CALL_RETRIES + 1):
        try:
            start = time.time()
            ref = actor.embed.remote(texts, max_length=max_length)
            embeddings = ray.get(ref)
            elapsed = time.time() - start
            logger.debug("Actor embed call: %d texts in %.2fs", len(texts), elapsed)
            if metrics:
                try:
                    metrics.observe_embedder_latency(elapsed)
                except Exception:
                    pass
            return [np.asarray(e, dtype=float).tolist() for e in embeddings]
        except Exception as e:
            last_exc = e
            backoff = 2 ** attempt
            logger.warning("Actor embed attempt %s failed: %s; backoff %s", attempt, e, backoff)
            time.sleep(backoff)
    logger.exception("Embedder actor failed after retries: %s", last_exc)
    raise RuntimeError("Embedder actor call failed") from last_exc


def call_embedder_via_serve(handle, texts: List[str], max_length: Optional[int] = None) -> List[List[float]]:
    body = {"texts": texts}
    if max_length is not None:
        body["max_length"] = int(max_length)
    try:
        start = time.time()
        fut = handle.remote(body)
        res = ray.get(fut)
        elapsed = time.time() - start
        logger.debug("Serve embed call: %d texts in %.2fs", len(texts), elapsed)
        if metrics:
            try:
                metrics.observe_embedder_latency(elapsed)
            except Exception:
                pass
        if isinstance(res, dict) and "embeddings" in res:
            embeddings = res["embeddings"]
        else:
            embeddings = res
        return [np.asarray(e, dtype=float).tolist() for e in embeddings]
    except Exception as e:
        logger.exception("Serve embed failed: %s", e)
        raise


# ---- Qdrant upsert with retries ----
def upsert_batch_with_retries(client: QdrantClient, collection: str, points: List[dict]) -> None:
    n = len(points)
    for i in range(0, n, QDRANT_UPSERT_BATCH_SIZE):
        sub = points[i : i + QDRANT_UPSERT_BATCH_SIZE]
        for attempt in range(1, UPsert_MAX_RETRIES + 1):
            try:
                start = time.time()
                client.upsert(collection_name=collection, points=sub)
                elapsed = time.time() - start
                logger.debug("Upserted %d points in %.2fs", len(sub), elapsed)
                if metrics:
                    try:
                        metrics.observe_upsert_duration(elapsed)
                    except Exception:
                        pass
                break
            except Exception as e:
                logger.warning("Qdrant upsert attempt %s failed: %s", attempt, e)
                if metrics:
                    try:
                        metrics.record_upsert_failure()
                    except Exception:
                        pass
                if attempt == UPsert_MAX_RETRIES:
                    logger.exception("Max upsert retries reached")
                    raise
                time.sleep(UPsert_BASE_BACKOFF * (2 ** (attempt - 1)))


# ---- batch processing pipeline ----
def process_batch(chunks: List[Dict[str, Any]], embedder_actor, serve_handle, qclient: QdrantClient) -> None:
    texts = [c.get("text", "") for c in chunks]
    if embedder_actor:
        embeddings = call_embedder_via_actor(embedder_actor, texts, max_length=MAX_LENGTH)
    elif serve_handle:
        embeddings = call_embedder_via_serve(serve_handle, texts, max_length=MAX_LENGTH)
    else:
        raise RuntimeError("No embedder available (actor or Serve)")
    if not embeddings:
        raise RuntimeError("Embedder returned no embeddings")
    if len(embeddings) < len(chunks):
        vec_len = len(embeddings[0]) if embeddings else 768
        for _ in range(len(chunks) - len(embeddings)):
            embeddings.append([0.0] * vec_len)
    elif len(embeddings) > len(chunks):
        embeddings = embeddings[: len(chunks)]
    vec_dim = len(embeddings[0])
    ensure_collection(qclient, QDRANT_COLLECTION, vector_size=vec_dim)
    points = make_qdrant_points(chunks, embeddings)
    upsert_batch_with_retries(qclient, QDRANT_COLLECTION, points)
    if metrics:
        try:
            metrics.record_indexed(len(points))
        except Exception:
            pass
    logger.info("Indexed %d points", len(points))


# ---- Snapshot wait / local packaging / upload / manifest ----
def find_snapshot_file(snapshot_name: Optional[str], search_dir: str) -> Optional[str]:
    p = Path(search_dir)
    if not p.exists():
        logger.debug("Snapshot search dir does not exist: %s", search_dir)
        return None
    # If snapshot_name is defined, look for any file that contains that name
    if snapshot_name:
        # try exact match first
        cand = p / snapshot_name
        if cand.exists():
            return str(cand)
        # try files containing the name
        for f in sorted(p.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if snapshot_name in f.name:
                return str(f)
    # else pick newest .snapshot* or .tar.zst file
    candidates = list(p.glob("*.snapshot*")) + list(p.glob("*.tar.zst")) + list(p.glob("*.zst"))
    if not candidates:
        # fallback: any file
        all_files = [f for f in p.iterdir() if f.is_file()]
        if not all_files:
            return None
        candidates = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def wait_for_snapshot_file(snapshot_name: Optional[str], search_dir: str, timeout: int = SNAPSHOT_WAIT_TIMEOUT, poll: int = SNAPSHOT_POLL_SECONDS) -> Optional[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = find_snapshot_file(snapshot_name, search_dir)
        if found:
            logger.info("Found snapshot file: %s", found)
            return found
        logger.debug("Waiting for snapshot file in %s; sleeping %ss", search_dir, poll)
        time.sleep(poll)
    logger.warning("Timed out waiting for snapshot file in %s (timeout %s sec).", search_dir, timeout)
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_local_snapshot_to_s3(
    snapshot_path: str,
    s3_bucket: str,
    s3_prefix: str,
    backup_and_restore_path: str = BACKUP_AND_RESTORE_PATH,
    cand_concurrency: Optional[int] = None,
    multipart_chunksize_mb: int = MULTIPART_CHUNKSIZE_MB,
    output_manifest_key: str = OUTPUT_MANIFEST_KEY,
) -> Dict[str, Any]:
    """
    Upload a local snapshot file (or directory) to S3 using multipart transfer + concurrency.
    Returns a dict with metadata including s3_key and sha256.
    """
    src = Path(snapshot_path)
    if not src.exists():
        raise FileNotFoundError(snapshot_path)

    # If snapshot path is a directory, tar+zstd it
    if src.is_dir():
        tmp_tar = Path(tempfile.mktemp(suffix=".tar.zst"))
        logger.info("Packaging snapshot directory %s -> %s", src, tmp_tar)
        # create tar.zst via system tar + zstd if available
        # fallback to python tarfile + external zstd if needed (we prefer system tar)
        cmd = ["tar", "-C", str(src), "-I", "zstd -T0 -19", "-cf", str(tmp_tar), "."]
        subprocess.check_call(cmd)
        upload_path = str(tmp_tar)
        remove_after = True
    else:
        # if file already endswith .tar.zst or .zst, upload as-is
        upload_path = str(src)
        remove_after = False

    size_bytes = Path(upload_path).stat().st_size
    size_mb = (size_bytes + 1048575) // 1048576
    logger.info("Preparing to upload %s (size: %d MB)", upload_path, size_mb)

    # concurrency heuristic
    vcpus = os.cpu_count() or 1
    if cand_concurrency is None:
        try:
            if BACKUP_MAX_CONCURRENCY:
                cand = int(BACKUP_MAX_CONCURRENCY)
            else:
                cand = max(1, vcpus // 2)
        except Exception:
            cand = max(1, vcpus // 2)
    else:
        cand = cand_concurrency

    chunk_bytes = multipart_chunksize_mb * 1024 * 1024
    transfer_config = TransferConfig(multipart_chunksize=chunk_bytes, max_concurrency=cand, multipart_threshold=chunk_bytes)

    s3_key = f"{s3_prefix.rstrip('/')}/{Path(upload_path).name}"
    dest_s3_uri = f"s3://{s3_bucket}/{s3_key}"
    logger.info("Uploading to %s (chunksize=%d MB, concurrency=%d)", dest_s3_uri, multipart_chunksize_mb, cand)

    t0 = time.time()
    # upload
    s3.upload_file(upload_path, s3_bucket, s3_key, Config=transfer_config)
    t1 = time.time()
    upload_duration = int(t1 - t0)

    # compute sha256
    sha256 = sha256_file(upload_path)
    logger.info("Uploaded to %s in %d sec; sha256=%s", dest_s3_uri, upload_duration, sha256)

    # write manifest locally then upload atomically
    manifest = {
        "backup_and_restore_path": backup_and_restore_path,
        "latest_snapshot": Path(upload_path).name,
        "latest_snapshot_from_path": str(Path(snapshot_path).resolve()),
        "latest_snapshot_to_path": dest_s3_uri,
        "latest_snapshot_size_in_mb": int(size_mb),
        "multipart_upload_concurrency": int(cand),
        "latest_snapshot_upload_duration_in_seconds": int(upload_duration),
        "sha256": sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    tmp_manifest = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
    try:
        tmp_manifest.write(json.dumps(manifest, indent=2))
        tmp_manifest.flush()
        tmp_manifest.close()
        tmp_key = f"{output_manifest_key}.tmp.{int(time.time())}"
        final_key = output_manifest_key
        logger.info("Uploading manifest to s3://%s/%s (temp key: %s)", s3_bucket, final_key, tmp_key)
        s3.upload_file(tmp_manifest.name, s3_bucket, tmp_key)
        # copy tmp to final (copy_object)
        copy_source = {"Bucket": s3_bucket, "Key": tmp_key}
        s3.copy_object(CopySource=copy_source, Bucket=s3_bucket, Key=final_key)
        # delete temp manifest
        s3.delete_object(Bucket=s3_bucket, Key=tmp_key)
        logger.info("Manifest written atomically to s3://%s/%s", s3_bucket, final_key)
    finally:
        try:
            os.unlink(tmp_manifest.name)
        except Exception:
            pass
        if remove_after:
            try:
                os.unlink(upload_path)
                logger.debug("Removed packaged upload file %s", upload_path)
            except Exception:
                pass

    return {"s3_key": s3_key, "s3_uri": dest_s3_uri, "sha256": sha256, "size_mb": size_mb, "upload_duration_sec": upload_duration}


# ---- Snapshot trigger (optional) ----
def trigger_qdrant_snapshot(collection: str) -> Optional[str]:
    if not ENABLE_QDRANT_SNAPSHOT:
        return None
    url_base = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    headers = {}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    create_paths = [
        f"{url_base}/collections/{collection}/snapshots/create",
        f"{url_base}/collections/{collection}/snapshots",
    ]
    for method_url in create_paths:
        try:
            resp = requests.post(method_url, headers=headers, timeout=30)
            if resp.ok:
                try:
                    data = resp.json()
                    snapshot = data.get("snapshot") or data.get("name") or data.get("snapshot_name")
                    logger.info("Triggered snapshot for collection %s: %s", collection, snapshot)
                    return str(snapshot) if snapshot else None
                except Exception:
                    logger.info("Snapshot trigger OK via %s", method_url)
                    return None
            else:
                logger.debug("Snapshot trigger returned status %s for %s", resp.status_code, method_url)
        except Exception as e:
            logger.debug("Snapshot trigger attempt failed for %s: %s", method_url, e)
    logger.warning("Could not trigger qdrant snapshot via HTTP API; please run snapshots externally")
    return None


# ---- run local script & stream logs ----
def run_local_and_stream(script_path: Path, workdir: str) -> int:
    cmd = [sys.executable, str(script_path)]
    logger.info("Starting local script: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    except Exception as e:
        logger.exception("Failed to start %s: %s", script_path, e)
        return 1
    out_lines = []
    err_lines = []

    def reader(stream, collect, tag):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                collect.append(line)
                logger.info("[%s] %s", tag, line.rstrip())
        except Exception:
            pass

    t_out = threading.Thread(target=reader, args=(proc.stdout, out_lines, script_path.name), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, err_lines, script_path.name), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)
    return proc.returncode


# ---- LibreOffice actor ----
LIBREOFFICE_LOG = "/tmp/libreoffice.log"
SOFFICE_LOG = "/tmp/libreoffice_soffice.log"


@ray.remote
class LibreOfficeActor:
    def __init__(self, workdir: str, script_relpath: str, port: int, log_path: str = LIBREOFFICE_LOG, force_direct: bool = False):
        self.workdir = Path(workdir)
        self.script = self.workdir / script_relpath
        self.port = int(port)
        self.log_path = log_path
        self.soffice_log = SOFFICE_LOG
        self.proc = None
        self.mode = None
        self.force_direct = bool(force_direct)
        self._start()

    def _which_soffice(self) -> Optional[str]:
        for name in ("soffice", "libreoffice"):
            p = shutil.which(name)
            if p:
                return p
        return None

    def _start_soffice(self, explicit_bin: Optional[str] = None):
        soffice_bin = explicit_bin or self._which_soffice()
        if not soffice_bin:
            raise FileNotFoundError("soffice/libreoffice not found in PATH")
        try:
            open(self.log_path, "a").close()
        except Exception:
            pass
        logf = open(self.soffice_log, "a")
        cmd = [
            soffice_bin,
            "--headless",
            "--nologo",
            "--invisible",
            "--nodefault",
            "--norestore",
            f"--accept=socket,host=127.0.0.1,port={self.port};urp;"
        ]
        self.proc = subprocess.Popen(cmd, cwd=str(self.workdir), stdout=logf, stderr=logf, preexec_fn=os.setsid)
        self.mode = "soffice"
        self.pid = getattr(self.proc, "pid", None)

    def _start_wrapper(self):
        try:
            os.chmod(self.script, os.stat(self.script).st_mode | 0o111)
        except Exception:
            pass
        try:
            open(self.log_path, "a").close()
        except Exception:
            pass
        logf = open(self.log_path, "a")
        self.proc = subprocess.Popen(["bash", str(self.script), str(self.port)], cwd=str(self.workdir), stdout=logf, stderr=logf, preexec_fn=os.setsid)
        self.mode = "script"
        self.pid = getattr(self.proc, "pid", None)

    def _start(self):
        try:
            if self.force_direct:
                self._start_soffice()
                return
            if self.script.exists():
                try:
                    self._start_wrapper()
                except Exception:
                    self._start_soffice()
                    return
            else:
                self._start_soffice()
        except Exception:
            raise

    def info(self):
        return {"pid": getattr(self, "pid", None), "port": self.port, "log": self.log_path, "mode": self.mode}

    def wait_until_ready(self, timeout: int = 30, interval: float = 0.5) -> bool:
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if getattr(self, "proc", None) and self.proc.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return True
            except Exception:
                time.sleep(interval)
        return False

    def stop(self):
        if getattr(self, "proc", None):
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        return True


# ---- pipeline run orchestration ----
def run_pipeline(workdir: str, libreoffice_port: int, libreoffice_ready_timeout: int):
    workdir = str(Path(workdir).resolve())

    # Connect/start Ray
    try:
        if RAY_ADDRESS and RAY_ADDRESS != "auto":
            ray.init(address=RAY_ADDRESS, namespace=RAY_NAMESPACE)
        else:
            ray.init(address="auto", namespace=RAY_NAMESPACE)
        logger.info("Connected / started Ray (address=%s)", RAY_ADDRESS)
    except Exception:
        logger.exception("Ray init failed; trying local init()")
        ray.init(namespace=RAY_NAMESPACE)

    # optional metrics server
    if metrics:
        try:
            metrics.start_metrics_server(port=METRICS_PORT)
            logger.info("Metrics server started on %s", METRICS_PORT)
        except Exception:
            logger.exception("Failed to start metrics server")

    # Check chunk source availability
    if (local_stream_chunks is None) and (not S3_BUCKET):
        logger.error("No chunk source available: neither local parse_chunk.router.stream_chunks() nor S3_BUCKET configured.")
        raise SystemExit("Missing chunk source: set S3_BUCKET or provide local parser")

    # placement group + LibreOffice actor
    try:
        pg = placement_group([{"CPU": 2}], strategy="STRICT_PACK")
        ray.get(pg.ready())
    except Exception:
        logger.exception("placement_group creation failed")
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        raise

    try:
        libre_actor = LibreOfficeActor.options(placement_group=pg, placement_group_bundle_index=0, num_cpus=1).remote(
            workdir, LIBREOFFICE_SCRIPT, libreoffice_port, LIBREOFFICE_LOG, False
        )
        info = ray.get(libre_actor.info.remote(), timeout=30)
        logger.info("LibreOffice actor info: %s", info)
    except Exception:
        logger.exception("Failed to create LibreOfficeActor")
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        raise

    # wait for libreoffice readiness (best-effort)
    try:
        ready = ray.get(libre_actor.wait_until_ready.remote(timeout=libreoffice_ready_timeout), timeout=libreoffice_ready_timeout + 5)
    except Exception:
        ready = False
    if not ready:
        logger.warning("LibreOffice not ready after %s seconds; continuing but conversions may fail.", libreoffice_ready_timeout)

    # run pre-conversion scripts (if present)
    for script_rel in PRE_CONVERSION_SCRIPTS:
        script_path = Path(workdir) / script_rel
        if not script_path.exists():
            logger.info("Skipping missing pre-conversion script: %s", script_path)
            continue
        rc = run_local_and_stream(script_path, workdir)
        if rc != 0:
            logger.error("Pre-conversion failed: %s (rc=%s)", script_path, rc)
            try:
                ray.get(libre_actor.stop.remote(), timeout=10)
            except Exception:
                pass
            try:
                remove_placement_group(pg)
            except Exception:
                pass
            raise SystemExit(f"Pre-conversion failed: {script_path} rc={rc}")
        logger.info("Pre-conversion succeeded: %s", script_path)

    # run router (if present)
    router_path = Path(workdir) / ROUTER
    if router_path.exists():
        rc = run_local_and_stream(router_path, workdir)
        if rc != 0:
            logger.error("Router failed (rc=%s).", rc)
            try:
                ray.get(libre_actor.stop.remote(), timeout=10)
            except Exception:
                pass
            try:
                remove_placement_group(pg)
            except Exception:
                pass
            raise SystemExit("Router failed")
        logger.info("Router completed successfully.")
    else:
        logger.info("Router missing; expecting chunks via S3 or local stream")

    # discover embedder (actor or serve)
    embedder_actor = None
    serve_handle = None
    try:
        embedder_actor = get_embedder_actor()
    except Exception:
        logger.exception("Error checking embedder actor")
    if embedder_actor is None:
        try:
            serve_handle = get_serve_handle()
        except Exception:
            logger.debug("No Serve deployment found")

    if (embedder_actor is None) and (serve_handle is None):
        logger.error("No embedder actor or Serve deployment found. Aborting.")
        try:
            ray.get(libre_actor.stop.remote(), timeout=10)
        except Exception:
            pass
        try:
            remove_placement_group(pg)
        except Exception:
            pass
        raise SystemExit("No embedder available in Ray cluster")

    # Qdrant client
    qclient = qdrant_client()
    if metrics:
        try:
            metrics.set_qdrant_connected(True)
        except Exception:
            pass

    # choose chunk iterator
    if local_stream_chunks is not None:
        chunk_iter = local_stream_chunks()
        logger.info("Using local parser stream_chunks() as chunk source")
    else:
        chunk_iter = iter_chunks_from_s3_prefix()
        logger.info("Using S3 chunk files under s3://%s/%s as chunk source", S3_BUCKET, S3_CHUNKED_PREFIX)

    # main processing loop: batch -> embed -> upsert
    buffer: List[Dict[str, Any]] = []
    total = 0
    try:
        for chunk in chunk_iter:
            if not isinstance(chunk, dict):
                logger.warning("Skipping non-dict chunk")
                continue
            buffer.append(chunk)
            if len(buffer) >= EMBED_BATCH_SIZE:
                try:
                    process_batch(buffer, embedder_actor, serve_handle, qclient)
                    total += len(buffer)
                except Exception:
                    logger.exception("Batch processing failed; continuing")
                buffer = []
        if buffer:
            try:
                process_batch(buffer, embedder_actor, serve_handle, qclient)
                total += len(buffer)
            except Exception:
                logger.exception("Final batch failed")
    finally:
        logger.info("Indexing finished. total indexed chunks=%s", total)
        # optional snapshot trigger + upload
        if ENABLE_QDRANT_SNAPSHOT:
            snap = trigger_qdrant_snapshot(QDRANT_COLLECTION)
            if snap is None:
                logger.info("Snapshot trigger returned no name; attempting to locate most recent snapshot file")
            # Wait for snapshot file in BACKUP_FROM_PATH
            snapshot_path = wait_for_snapshot_file(snap, BACKUP_FROM_PATH)
            if snapshot_path:
                if not SNAPSHOT_S3_BUCKET:
                    logger.warning("SNAPSHOT_S3_BUCKET not configured; snapshot available at %s but will not be uploaded", snapshot_path)
                else:
                    try:
                        result = upload_local_snapshot_to_s3(
                            snapshot_path,
                            s3_bucket=SNAPSHOT_S3_BUCKET,
                            s3_prefix=SNAPSHOT_S3_PREFIX or BACKUP_TO_PREFIX,
                            backup_and_restore_path=BACKUP_AND_RESTORE_PATH,
                        )
                        logger.info("Snapshot uploaded: %s", result)
                    except Exception:
                        logger.exception("Snapshot upload failed")
            else:
                logger.warning("No snapshot file detected to upload (snapshot trigger=%s).", snap)
        # teardown
        try:
            ray.get(libre_actor.stop.remote(), timeout=30)
        except Exception:
            logger.exception("Exception while stopping libre actor")
        try:
            remove_placement_group(pg)
        except Exception:
            logger.exception("Exception while removing placement_group")
        try:
            ray.shutdown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--libreoffice-port", type=int, default=7003)
    parser.add_argument("--libreoffice-ready-timeout", type=int, default=10)
    args = parser.parse_args()

    def _handler(sig, frame):
        logger.info("Signal %s received, exiting.", sig)
        try:
            ray.shutdown()
        except Exception:
            pass
        sys.exit(1)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    run_pipeline(args.workdir, args.libreoffice_port, args.libreoffice_ready_timeout)


if __name__ == "__main__":
    main()
