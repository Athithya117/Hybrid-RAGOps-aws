from __future__ import annotations
import os
import sys
import time
import json
import logging
import re
from typing import Iterable, Dict, List, Any, Optional

import boto3
import ray
from ray import serve
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import requests

try:
    from parse_chunk.router import stream_chunks
except Exception:
    stream_chunks = None

try:
    import metrics
except Exception:
    metrics = None

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("indexer")

RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")
RAY_NAMESPACE = os.environ.get("RAY_NAMESPACE", None)
EMBEDDER_ACTOR_NAME = os.environ.get("EMBEDDER_ACTOR_NAME", "embedder-actor")
EMBEDDER_SERVE_DEPLOYMENT = os.environ.get("EMBEDDER_SERVE_DEPLOYMENT", "Embedder")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "512"))
QDRANT_UPSERT_BATCH_SIZE = int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", str(EMBED_BATCH_SIZE)))
MAX_EMBED_CALL_RETRIES = int(os.environ.get("EMBED_CALL_MAX_RETRIES", "3"))
QDRANT_HOST = os.environ.get("QDRANT_PRIVATE_IP", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION_NAME", "my_documents")
QDRANT_SHARDS = int(os.environ.get("QDRANT_SHARDS", "1"))
QDRANT_REPLICATION_FACTOR = int(os.environ.get("QDRANT_REPLICATION_FACTOR", "1"))
QDRANT_WRITE_CONSISTENCY = int(os.environ.get("QDRANT_WRITE_CONSISTENCY", "1"))
QDRANT_ON_DISK_PAYLOAD = (os.environ.get("QDRANT_ON_DISK_PAYLOAD", "false").lower() == "true")
UPsert_MAX_RETRIES = int(os.environ.get("INDEX_UPSERT_MAX_RETRIES", "3"))
UPsert_BASE_BACKOFF = float(os.environ.get("INDEX_UPSERT_BASE_BACKOFF", "1.0"))
TEXT_SNIPPET_LEN = int(os.environ.get("TEXT_SNIPPET_LEN", "300"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1200"))
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_CHUNKED_PREFIX = (os.environ.get("S3_CHUNKED_PREFIX", "") or "").rstrip("/") + "/"
CHUNK_FORMAT = (os.environ.get("CHUNK_FORMAT", "json") or "json").lower()
STORE_ONE_FILE_PER_CHUNK = (os.environ.get("STORE_ONE_FILE_PER_CHUNK", "true").lower() == "true")
ENABLE_QDRANT_SNAPSHOT = (os.environ.get("ENABLE_QDRANT_SNAPSHOT", "false").lower() == "true")
SNAPSHOT_S3_BUCKET = os.environ.get("SNAPSHOT_S3_BUCKET")
SNAPSHOT_S3_PREFIX = (os.environ.get("SNAPSHOT_S3_PREFIX", "") or "").rstrip("/") + "/"
SNAPSHOT_POLL_SECONDS = int(os.environ.get("SNAPSHOT_POLL_SECONDS", "3"))

if CHUNK_FORMAT not in ("json", "jsonl"):
    raise SystemExit("CHUNK_FORMAT must be 'json' or 'jsonl'")
if not S3_BUCKET and stream_chunks is None:
    raise SystemExit("S3_BUCKET must be set in environment to read chunk files when stream_chunks is not present")

s3 = boto3.client("s3")
_qdrant_client_singleton: Optional[QdrantClient] = None

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

def qdrant_client() -> QdrantClient:
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        _qdrant_client_singleton = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    return _qdrant_client_singleton

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
            logger.info("Created payload index %s:%s", field, schema)
        except Exception as exc:
            logger.debug("Create payload index skipped or failed for %s: %s", field, exc)

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
        logger.info(
            "Created Qdrant collection %s vector_size=%s shards=%s replicas=%s on_disk_payload=%s",
            collection,
            vs,
            QDRANT_SHARDS,
            QDRANT_REPLICATION_FACTOR,
            QDRANT_ON_DISK_PAYLOAD,
        )
        ensure_payload_indexes(client, collection)
    except Exception as e:
        logger.warning("ensure_collection warning: %s", e)

def parse_json_objects_from_text(blob_text: str) -> List[dict]:
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

def get_embedder_actor():
    try:
        actor = ray.get_actor(EMBEDDER_ACTOR_NAME)
        logger.info("Found embedder actor: %s", EMBEDDER_ACTOR_NAME)
        return actor
    except Exception:
        return None

def get_serve_handle():
    try:
        handle = serve.get_deployment(EMBEDDER_SERVE_DEPLOYMENT).get_handle(sync=False)
        logger.info("Found serve handle: %s", EMBEDDER_SERVE_DEPLOYMENT)
        return handle
    except Exception:
        return None

def call_embedder_via_actor(actor, text_refs: List[ray.ObjectRef], max_length: Optional[int] = None) -> List[List[float]]:
    last_exc = None
    for attempt in range(1, MAX_EMBED_CALL_RETRIES + 1):
        try:
            start = time.time()
            embeddings_ref = actor.embed.remote(text_refs, max_length=max_length)
            embeddings = ray.get(embeddings_ref)
            elapsed = time.time() - start
            if metrics:
                metrics.observe_embedder_latency(elapsed)
            return [np.asarray(e, dtype=float).tolist() for e in embeddings]
        except Exception as e:
            last_exc = e
            backoff = 2 ** attempt
            logger.warning("Actor embed attempt %s failed: %s; backoff %s", attempt, e, backoff)
            time.sleep(backoff)
    logger.exception("Embedder actor failed after retries: %s", last_exc)
    raise RuntimeError("Embedder actor call failed") from last_exc

def call_embedder_via_serve(handle, text_refs: List[ray.ObjectRef], max_length: Optional[int] = None) -> List[List[float]]:
    try:
        start = time.time()
        future = handle.remote(text_refs, max_length)
        res = ray.get(future)
        if isinstance(res, ray.ObjectRef):
            embeddings = ray.get(res)
        else:
            embeddings = res
        elapsed = time.time() - start
        if metrics:
            metrics.observe_embedder_latency(elapsed)
        return [np.asarray(e, dtype=float).tolist() for e in embeddings]
    except Exception:
        logger.debug("Serve passthrough with ObjectRefs failed; trying inline JSON fallback")
    texts = ray.get(text_refs)
    body = {"texts": texts}
    if max_length is not None:
        body["max_length"] = int(max_length)
    try:
        start = time.time()
        res_future = handle.remote(body)
        res = ray.get(res_future)
        elapsed = time.time() - start
        if isinstance(res, dict) and "embeddings" in res:
            embeddings = res["embeddings"]
        else:
            embeddings = res
        if metrics:
            metrics.observe_embedder_latency(elapsed)
        return [np.asarray(e, dtype=float).tolist() for e in embeddings]
    except Exception as e:
        logger.exception("Serve inline embed failed: %s", e)
        raise

def make_qdrant_points(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[dict]:
    pts: List[dict] = []
    for chunk, emb in zip(chunks, embeddings):
        pid = point_id_for(chunk)
        payload = payload_from_chunk(chunk)
        pts.append({"id": pid, "vector": list(map(float, emb)), "payload": payload})
    return pts

def upsert_batch_with_retries(client: QdrantClient, collection: str, points: List[dict]) -> None:
    n = len(points)
    for i in range(0, n, QDRANT_UPSERT_BATCH_SIZE):
        sub = points[i: i + QDRANT_UPSERT_BATCH_SIZE]
        for attempt in range(1, UPsert_MAX_RETRIES + 1):
            try:
                start = time.time()
                client.upsert(collection_name=collection, points=sub)
                elapsed = time.time() - start
                if metrics:
                    metrics.observe_upsert_duration(elapsed)
                break
            except Exception as e:
                logger.warning("Qdrant upsert attempt %s failed: %s", attempt, e)
                if metrics:
                    metrics.record_upsert_failure()
                if attempt == UPsert_MAX_RETRIES:
                    logger.exception("Max upsert retries reached")
                    raise
                time.sleep(UPsert_BASE_BACKOFF * (2 ** (attempt - 1)))

def process_batch(chunks: List[Dict[str, Any]], embedder_actor, serve_handle, qclient: QdrantClient) -> None:
    texts = [c.get("text", "") for c in chunks]
    text_refs = [ray.put(t) for t in texts]
    logger.debug("Placed %d text ObjectRefs", len(text_refs))
    if embedder_actor:
        embeddings = call_embedder_via_actor(embedder_actor, text_refs, max_length=MAX_LENGTH)
    elif serve_handle:
        embeddings = call_embedder_via_serve(serve_handle, text_refs, max_length=MAX_LENGTH)
    else:
        raise RuntimeError("No embedder available")
    if len(embeddings) < len(chunks):
        logger.warning("Embeddings fewer than chunks; padding with zeros")
    if not embeddings:
        raise RuntimeError("Embedder returned no embeddings")
    if len(embeddings) < len(chunks):
        vec_len = len(embeddings[0])
        for _ in range(len(chunks) - len(embeddings)):
            embeddings.append([0.0] * vec_len)
    elif len(embeddings) > len(chunks):
        embeddings = embeddings[:len(chunks)]
    vec_dim = len(embeddings[0])
    ensure_collection(qclient, QDRANT_COLLECTION, vector_size=vec_dim)
    points = make_qdrant_points(chunks, embeddings)
    upsert_batch_with_retries(qclient, QDRANT_COLLECTION, points)
    try:
        ray.internal.free(text_refs)
    except Exception:
        pass
    if metrics:
        metrics.record_indexed(len(points))
    logger.info("Indexed %d points", len(points))

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
            resp = requests.post(method_url, headers=headers, timeout=20)
            if resp.ok:
                try:
                    data = resp.json()
                    snapshot = data.get("snapshot")
                    if not snapshot:
                        snapshot = data.get("name") or data.get("snapshot_name")
                    logger.info("Triggered snapshot for collection %s: %s", collection, snapshot)
                    return str(snapshot) if snapshot else None
                except Exception:
                    logger.info("Snapshot trigger response ok for %s", method_url)
                    return None
            else:
                logger.debug("Snapshot trigger returned status %s for %s", resp.status_code, method_url)
        except Exception as e:
            logger.debug("Snapshot trigger attempt failed for %s: %s", method_url, e)
    logger.warning("Could not trigger qdrant snapshot via HTTP API; please run snapshots externally")
    return None

def upload_snapshot_to_s3(snapshot_name: str) -> None:
    if not snapshot_name or not SNAPSHOT_S3_BUCKET:
        return
    logger.info("Operator action required: copy snapshot %s from Qdrant server filesystem to S3 bucket %s under prefix %s", snapshot_name, SNAPSHOT_S3_BUCKET, SNAPSHOT_S3_PREFIX)

def main() -> None:
    if metrics:
        try:
            metrics.start_metrics_server(port=int(os.environ.get("METRICS_PORT", "9000")))
            logger.info("Started metrics server on port %s", os.environ.get("METRICS_PORT", "9000"))
        except Exception as e:
            logger.warning("Failed to start metrics server: %s", e)
    logger.info("Starting indexer; connecting to Ray (address=%s)", RAY_ADDRESS)
    ray.init(address=RAY_ADDRESS, namespace=RAY_NAMESPACE, ignore_reinit_error=True)
    embedder_actor = None
    serve_handle = None
    try:
        embedder_actor = get_embedder_actor()
        if embedder_actor is None:
            serve_handle = get_serve_handle()
    except Exception:
        logger.exception("Embedder discovery error; proceeding to attempt Serve handle")
    qclient = qdrant_client()
    try:
        if metrics:
            metrics.set_qdrant_connected(True)
    except Exception:
        pass
    logger.info("Connected to Qdrant at %s:%s (collection=%s)", QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION)
    if stream_chunks is not None:
        chunk_iter = stream_chunks()
        logger.info("Using local parser stream_chunks() as chunk source")
    else:
        chunk_iter = iter_chunks_from_s3_prefix()
        logger.info("Using S3 chunk files under s3://%s/%s as chunk source", S3_BUCKET, S3_CHUNKED_PREFIX)
    buffer: List[Dict[str, Any]] = []
    total = 0
    for chunk in chunk_iter:
        if not isinstance(chunk, dict):
            logger.warning("Skipping non-dict chunk object")
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
    logger.info("Indexing finished. total indexed chunks=%s", total)
    if ENABLE_QDRANT_SNAPSHOT:
        snap = trigger_qdrant_snapshot(QDRANT_COLLECTION)
        if snap and SNAPSHOT_S3_BUCKET:
            upload_snapshot_to_s3(snap)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Indexer terminated: %s", exc)
        try:
            if metrics:
                metrics.set_qdrant_connected(False)
        except Exception:
            pass
        sys.exit(2)
