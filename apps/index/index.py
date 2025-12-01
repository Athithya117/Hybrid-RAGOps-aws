import os, json, hashlib, logging, time, signal, sys, re, traceback
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_third_party_names = (
    "httpx", "httpcore", "urllib3", "qdrant_client",
    "uvicorn.access", "uvicorn.error", "asyncio",
    "botocore", "botocore.client", "botocore.hooks", "botocore.parsers",
    "boto3", "s3transfer", "aiobotocore", "boto3.resources"
)
for _n in _third_party_names:
    logging.getLogger(_n).handlers.clear()
    logging.getLogger(_n).setLevel(logging.CRITICAL)
    logging.getLogger(_n).propagate = False
_root = logging.getLogger()
_root.handlers.clear()
_root.setLevel(logging.CRITICAL)
_app_level = logging.DEBUG if LOG_LEVEL == "DEBUG" else (logging.INFO if LOG_LEVEL == "INFO" else logging.DEBUG)
logger = logging.getLogger("index")
logger.handlers.clear()
logger.setLevel(_app_level)
_h = logging.StreamHandler(stream=sys.stdout)
_h.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_h)
logger.propagate = False
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_rag_collection1")
S3_BUCKET = os.getenv("S3_BUCKET", "e2e-rag-system-42")
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")
DENSE_URL = os.getenv("DENSE_URL", "http://localhost:8205")
SPARSE_URL = os.getenv("SPARSE_URL", "http://localhost:8201")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
UPSERT_CHUNK = int(os.getenv("UPSERT_CHUNK", "500"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10.0"))
SPARSE_BATCH_FALLBACK = int(os.getenv("SPARSE_BATCH_FALLBACK", "8"))
QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "128"))
QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "32"))
QDRANT_HNSW_FULL_SCAN_THRESHOLD = int(os.getenv("QDRANT_HNSW_FULL_SCAN_THRESHOLD", "10000"))
QDRANT_ONDISK = os.getenv("QDRANT_ONDISK", "TRUE").upper() in ("1", "TRUE", "YES")
NORMALIZE_DENSE = True
SHUTDOWN = False
INFO_EVENTS = {"index.start", "batch.embedded", "index.prepared", "index.completed", "load.chunks", "collection.created", "collection.exists"}
def iso_ts():
    return datetime.now(timezone.utc).isoformat()
def _escape_stack(exc: BaseException) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__) if exc else [])
    return tb.replace("\n", "\\n")
def slog(level: str, evt: str, exc: Optional[BaseException] = None, **kw):
    entry = {"ts": iso_ts(), "lvl": level, "evt": evt}
    entry.update(kw)
    if level == "error" and exc is not None:
        entry["error"] = str(exc)
        entry["stack"] = _escape_stack(exc)
    elif level == "warning" and exc is not None:
        entry["error"] = str(exc)
    msg = json.dumps(entry, ensure_ascii=False)
    if level == "info":
        if LOG_LEVEL == "INFO":
            if evt in INFO_EVENTS:
                logger.info(msg)
        else:
            logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)
def handle_sigterm(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    slog("warning", "shutdown.requested", signal=signum)
signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)
def id_from_string(s: str) -> int:
    h = hashlib.md5(s.encode("utf8")).hexdigest()
    return int(h[:16], 16)
def l2_normalize(v: List[float]) -> List[float]:
    a = np.asarray(v, dtype=np.float32); n = np.linalg.norm(a)
    if n > 0: a = a / n
    return a.astype(float).tolist()
class DenseClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT):
        self.url = url.rstrip("/"); self.client = httpx.Client(timeout=timeout)
    def health(self) -> bool:
        try:
            r = self.client.get(f"{self.url}/health", timeout=HTTP_TIMEOUT); ok = r.status_code == 200
            slog("debug", "dense.health.check", url=self.url, status=r.status_code)
            return ok
        except Exception as e:
            slog("warning", "dense.health.error", exc=e)
            return False
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        start = time.time()
        r = self.client.post(f"{self.url}/embed", json={"texts": texts}, timeout=HTTP_TIMEOUT)
        elapsed = round(time.time() - start, 3)
        slog("debug", "dense.call", url=self.url, count=len(texts), status=r.status_code, elapsed=elapsed)
        if r.status_code == 200:
            j = r.json(); vecs = j.get("vectors")
            if not isinstance(vecs, list) or len(vecs) != len(texts): raise RuntimeError("dense embed invalid response")
            out = []
            for v in vecs:
                if not isinstance(v, list): raise RuntimeError("dense embed vector invalid")
                vv = [float(x) for x in v]
                if NORMALIZE_DENSE: vv = l2_normalize(vv)
                if len(vv) != DENSE_DIM: raise RuntimeError("dense embed dim mismatch")
                out.append(vv)
            slog("debug", "dense.embedded", count=len(out))
            return out
        raise RuntimeError(f"dense embed failed status={r.status_code} body={r.text}")
class SparseClient:
    def __init__(self, url: str, timeout: float = HTTP_TIMEOUT):
        self.url = url.rstrip("/"); self.client = httpx.Client(timeout=timeout)
    def health(self) -> bool:
        try:
            r = self.client.get(f"{self.url}/health", timeout=HTTP_TIMEOUT); ok = r.status_code == 200
            slog("debug", "sparse.health.check", url=self.url, status=r.status_code)
            return ok
        except Exception as e:
            slog("warning", "sparse.health.error", exc=e)
            return False
    def embed_chunked(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts: return []
        r = self.client.post(f"{self.url}/embed", json={"texts": texts}, timeout=HTTP_TIMEOUT)
        slog("debug", "sparse.call.attempt", url=self.url, count=len(texts), status=r.status_code)
        if r.status_code == 200:
            j = r.json(); vecs = j.get("vectors")
            if not isinstance(vecs, list) or len(vecs) != len(texts): raise RuntimeError("sparse embed invalid response")
            out = []
            for s in vecs:
                if not isinstance(s, dict) or "indices" not in s or "values" not in s: raise RuntimeError("sparse embed element invalid")
                inds = [int(x) for x in s.get("indices", [])]; vals = [float(x) for x in s.get("values", [])]; out.append({"indices": inds, "values": vals})
            slog("debug", "sparse.embedded", count=len(out))
            return out
        if r.status_code == 400:
            try:
                j = r.json(); detail = j.get("detail", ""); m = re.search(r"max=(\d+)", str(detail))
                if m:
                    maxb = int(m.group(1))
                    if maxb <= 0: raise RuntimeError("invalid server max")
                    out = []; i = 0
                    while i < len(texts):
                        chunk = texts[i:i+maxb]
                        out.extend(self.embed_chunked(chunk))
                        i += maxb
                    slog("debug", "sparse.batch.split", original=len(texts), split_to=maxb)
                    return out
            except Exception as e:
                slog("warning", "sparse.batch.split.failed", exc=e)
        if r.status_code == 422:
            maxb = SPARSE_BATCH_FALLBACK; out = []; i = 0
            while i < len(texts):
                out.extend(self.embed_chunked(texts[i:i+maxb])); i += maxb
            slog("debug", "sparse.batch.fallback", original=len(texts), fallback=maxb)
            return out
        raise RuntimeError(f"sparse embed failed status={r.status_code} body={r.text}")
def sparse_to_qdrant_sparsevector(sparse_obj: Any) -> SparseVector:
    if sparse_obj is None: return SparseVector(indices=[], values=[])
    if isinstance(sparse_obj, dict):
        inds = list(map(int, sparse_obj.get("indices", []))); vals = list(map(float, sparse_obj.get("values", [])))
        return SparseVector(indices=inds, values=vals)
    raise RuntimeError("unsupported sparse object")
def _parse_list_like(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            inner = s[1:-1].strip()
            if not inner: return []
            parts = [p.strip() for p in inner.split(",")]
            out = []
            for p in parts:
                if p.isdigit():
                    out.append(int(p))
                else:
                    out.append(p.strip('"').strip("'"))
            return out
    if "-" in s and all(part.strip().isdigit() for part in s.split("-", 1)):
        a, b = s.split("-", 1)
        return [int(a), int(b)]
    return [s]
def _to_optional_int(v):
    if v is None: return None
    if isinstance(v, int): return v
    s = str(v).strip()
    if s == "": return None
    try:
        return int(s)
    except Exception:
        return None
def _to_optional_list_of_ints(v):
    parsed = _parse_list_like(v)
    out = []
    for e in parsed:
        try:
            out.append(int(e))
        except Exception:
            pass
    return out if out else None
def _to_optional_list_of_str(v):
    parsed = _parse_list_like(v)
    out = [str(x) for x in parsed]
    return out if out else None
def _to_optional_audio_range(v):
    parsed = _parse_list_like(v)
    out = [str(x) for x in parsed]
    return out if out else None
def normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(chunk)
    c["headings"] = _to_optional_list_of_str(c.get("headings")) or []
    c["heading_path"] = _to_optional_list_of_str(c.get("heading_path")) or []
    c["tags"] = _to_optional_list_of_str(c.get("tags")) or []
    c["layout_tags"] = _to_optional_list_of_str(c.get("layout_tags")) or []
    c["figures"] = _parse_list_like(c.get("figures")) or []
    c["row_range"] = _to_optional_list_of_ints(c.get("row_range"))
    c["line_range"] = _to_optional_list_of_ints(c.get("line_range"))
    c["token_range"] = _to_optional_list_of_ints(c.get("token_range"))
    c["slide_range"] = _parse_list_like(c.get("slide_range")) or None
    c["audio_range"] = _to_optional_audio_range(c.get("audio_range"))
    c["page_number"] = _to_optional_int(c.get("page_number"))
    c["token_count"] = _to_optional_int(c.get("token_count"))
    if "used_ocr" in c:
        if isinstance(c["used_ocr"], bool):
            pass
        else:
            s = str(c["used_ocr"]).lower()
            c["used_ocr"] = True if s in ("1", "true", "yes") else False
    if "timestamp" in c and c.get("timestamp") is not None:
        c["timestamp"] = str(c.get("timestamp"))
    for t in ("text", "file_name", "file_type", "source_url", "parser_version", "chunk_type", "chunk_id", "document_id"):
        if t in c and c.get(t) is not None:
            c[t] = str(c.get(t))
    return c
def create_collection_hybrid(client, name, dense_dim):
    try:
        if client.collection_exists(name): slog("info", "collection.exists", name=name); return
    except Exception as e:
        slog("warning", "collection.exists.check.failed", exc=e)
    hnsw = {"m": QDRANT_HNSW_M, "ef_construct": QDRANT_HNSW_EF_CONSTRUCT, "full_scan_threshold": QDRANT_HNSW_FULL_SCAN_THRESHOLD, "on_disk": QDRANT_ONDISK}
    try:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": {"size": dense_dim, "distance": "Cosine", "hnsw_config": hnsw}},
            sparse_vectors_config={"sparse": {}}
        )
        slog("info", "collection.created", name=name, dense_dim=dense_dim)
    except Exception as e:
        slog("error", "collection.create.failed", exc=e); raise
def create_collection_sparse_only(client, name):
    try:
        if client.collection_exists(name): slog("info", "collection.exists", name=name); return
    except Exception as e:
        slog("warning", "collection.exists.check.failed", exc=e)
    try:
        client.create_collection(collection_name=name, sparse_vectors_config={"sparse": {}})
        slog("info", "collection.created.sparse", name=name)
    except Exception as e:
        slog("error", "collection.create.sparse.failed", exc=e); raise
FULL_PAYLOAD_KEYS = [
    "document_id", "file_name", "chunk_id", "chunk_type", "text", "token_count", "source_url", "timestamp",
    "parser_version", "page_number", "row_range", "line_range", "token_range", "audio_range", "slide_range",
    "headings", "heading_path", "tags", "layout_tags", "figures", "file_type", "token_count", "used_ocr",
    "parser_version", "layout", "layout_tags"
]
def make_pointstruct(pid: int, vectors_payload: Dict[str, Any], payload: Dict[str, Any]):
    try:
        return {"id": pid, "vector": vectors_payload, "payload": payload}
    except Exception as e:
        slog("error", "pointstruct.failed", exc=e); raise
def chunk_and_vectors_to_pointstructs(items: List[Dict[str, Any]], hybrid: bool) -> List[Dict[str, Any]]:
    pts = []
    for chunk, dvec, sparse_obj in items:
        cid = chunk.get("chunk_id") or (str(chunk.get("document_id")) + "_0")
        pid = id_from_string(cid)
        q_sv = sparse_to_qdrant_sparsevector(sparse_obj) if sparse_obj is not None else None
        vectors_payload = {}
        if hybrid:
            if dvec is not None: vectors_payload["dense"] = dvec
            if q_sv is not None and q_sv.indices: vectors_payload["sparse"] = q_sv
            if not vectors_payload:
                slog("warning", "skip.no_vectors", chunk_id=cid); continue
        else:
            if q_sv is None or (not q_sv.indices):
                slog("warning", "skip.no_sparse", chunk_id=cid); continue
            vectors_payload["sparse"] = q_sv
        payload = {}
        for k in FULL_PAYLOAD_KEYS:
            payload[k] = chunk.get(k) if k in chunk else None
        pts.append(make_pointstruct(pid, vectors_payload, payload))
    return pts
def existing_point_ids(client, collection_name, ids: List[int]) -> set:
    if not ids: return set()
    try:
        res = client.retrieve(collection_name=collection_name, ids=ids)
    except Exception as e:
        slog("warning", "retrieve.failed", exc=e); return set()
    out = set()
    if isinstance(res, list):
        for r in res:
            try:
                m = r.model_dump() if hasattr(r, "model_dump") else (r.dict() if hasattr(r, "dict") else r)
                pid = m.get("id")
                if pid is not None: out.add(pid)
            except Exception:
                continue
    elif isinstance(res, dict):
        for k in ("result", "points", "data"):
            if k in res and isinstance(res[k], list):
                for p in res[k]:
                    pid = p.get("id")
                    if pid is not None: out.add(pid)
    return out
def safe_embed_and_points(chunks: List[Dict[str, Any]], sparse_client: Optional[SparseClient], dense_client: Optional[DenseClient], hybrid: bool):
    texts = [c.get("text", "") or "" for c in chunks]
    dense_vecs = [None] * len(texts)
    sparse_objs = [None] * len(texts)
    if dense_client is not None:
        try:
            dense_vecs = dense_client.embed(texts)
            slog("debug", "dense.emb.ok", count=len(dense_vecs))
        except Exception as e:
            slog("warning", "dense.embed.failed", exc=e); dense_vecs = [None] * len(texts)
    if sparse_client is not None:
        try:
            sparse_objs = sparse_client.embed_chunked(texts)
            slog("debug", "sparse.emb.ok", count=len(sparse_objs))
        except Exception as e:
            slog("warning", "sparse.embed.failed", exc=e); sparse_objs = [None] * len(texts)
    items = []
    for i in range(len(chunks)):
        items.append((chunks[i], dense_vecs[i] if i < len(dense_vecs) else None, sparse_objs[i] if i < len(sparse_objs) else None))
    return chunk_and_vectors_to_pointstructs(items, hybrid)
def embed_and_upsert(client, collection_name, chunks, sparse_client, dense_client, hybrid):
    total = len(chunks); slog("info", "index.start", total_input_chunks=total, batch=BATCH_SIZE, hybrid=hybrid)
    to_upsert = []; processed = 0
    for i in range(0, total, BATCH_SIZE):
        if SHUTDOWN: slog("warning", "shutdown.during_index"); break
        batch = chunks[i:i+BATCH_SIZE]; ids = [id_from_string(c.get("chunk_id") or (str(c.get("document_id")) + "_0")) for c in batch]
        start = time.time(); existing = existing_point_ids(client, collection_name, ids); elapsed = round(time.time() - start, 3)
        to_process = [c for c, pid in zip(batch, ids) if pid not in existing]; skipped = len(batch) - len(to_process)
        slog("debug", "batch.check", batch_range=f"{i}..{i+len(batch)-1}", retrieve_time=elapsed, skipped=skipped)
        if to_process:
            pts = safe_embed_and_points(to_process, sparse_client, dense_client, hybrid)
            to_upsert.extend(pts); processed += len(pts)
            slog("info", "batch.embedded", embedded=len(pts), processed=processed)
    total_prepared = len(to_upsert); slog("info", "index.prepared", total_points_to_upsert=total_prepared)
    if total_prepared == 0: slog("warning", "index.no_new_points"); return
    for j in range(0, total_prepared, UPSERT_CHUNK):
        if SHUTDOWN: slog("warning", "shutdown.before_upsert"); break
        slice_pts = to_upsert[j:j+UPSERT_CHUNK]
        try:
            client.upsert(collection_name=collection_name, points=slice_pts)
            slog("debug", "upsert.chunk", start=j, end=j+len(slice_pts)-1)
        except Exception as e:
            slog("error", "upsert.failed", exc=e); raise
    slog("info", "index.completed")
def load_chunks_from_s3(bucket: str, prefix: str) -> List[Dict[str, Any]]:
    try:
        import boto3
    except Exception as e:
        slog("error", "boto3.missing", exc=e); raise SystemExit("boto3 required")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = []
    for p in pages:
        for c in p.get("Contents", []):
            k = c.get("Key")
            if k and k.endswith(".json"):
                keys.append(k)
    if not keys:
        slog("error", "no_s3_chunks", bucket=bucket, prefix=prefix); raise SystemExit(f"No .json chunk files in s3://{bucket}/{prefix}")
    chunks = []
    for k in sorted(keys):
        try:
            obj = s3.get_object(Bucket=bucket, Key=k)
            body = obj["Body"].read().decode("utf8")
            data = json.loads(body)
            if isinstance(data, list):
                for raw in data:
                    try:
                        norm = normalize_chunk(raw)
                        chunks.append(norm)
                    except Exception as e:
                        slog("warning", "chunk.normalize.failed", key=k, exc=e)
            else:
                slog("error", "s3.chunk.format.invalid", key=k, type=str(type(data))); raise SystemExit(f"Expected list in s3://{bucket}/{k}")
        except Exception as e:
            slog("error", "s3.get_object.failed", key=k, exc=e); raise
    slog("info", "load.chunks", original_chunks=len(chunks))
    return chunks
def validate_and_build_clients():
    dc = DenseClient(DENSE_URL); sc = SparseClient(SPARSE_URL)
    dense = None; sparse = None
    if dc.health(): dense = dc; slog("debug", "dense.ready", url=DENSE_URL)
    else: slog("warning", "dense.unhealthy", url=DENSE_URL)
    if sc.health(): sparse = sc; slog("debug", "sparse.ready", url=SPARSE_URL)
    else: slog("warning", "sparse.unhealthy", url=SPARSE_URL)
    if dense is None and sparse is None: slog("error", "no_embed_services"); raise SystemExit("Neither dense nor sparse service healthy")
    return dense, sparse
def retrieve_and_index():
    chunks = load_chunks_from_s3(S3_BUCKET, S3_CHUNKED_PREFIX)
    dense_client, sparse_client = validate_and_build_clients()
    hybrid_mode = (dense_client is not None and sparse_client is not None)
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)
    except Exception as e:
        slog("error", "qdrant.client.init.failed", exc=e); raise SystemExit(f"Unable to contact Qdrant: {e}")
    try:
        _ = client.get_collections()
    except Exception as e:
        slog("error", "qdrant.unreachable", exc=e); raise SystemExit(f"Unable to contact Qdrant: {e}")
    if hybrid_mode:
        create_collection_hybrid(client, COLLECTION_NAME, dense_dim=DENSE_DIM)
    else:
        create_collection_sparse_only(client, COLLECTION_NAME)
    embed_and_upsert(client, COLLECTION_NAME, chunks, sparse_client, dense_client, hybrid_mode)
if __name__ == "__main__":
    retrieve_and_index()
