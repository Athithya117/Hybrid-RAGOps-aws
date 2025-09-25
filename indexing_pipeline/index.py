#!/usr/bin/env python3
import os, sys, json, time, uuid, hashlib, asyncio
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
import boto3, httpx, weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth
from weaviate.exceptions import WeaviateInsertManyAllFailedError
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_RAW_PREFIX = os.environ.get("S3_RAW_PREFIX", "data/raw/").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.environ.get("S3_CHUNKED_PREFIX", "data/chunked/").rstrip("/") + "/"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", "./data")
AWS_REGION = os.environ.get("AWS_REGION")
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8001/embed")
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "32"))
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "8"))
EMBED_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", "60"))
INDEX_BATCH = int(os.environ.get("INDEX_BATCH", "128"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "")
COLLECTION_NAME = os.environ.get("WEAVIATE_COLLECTION", "Chunk")
SELECTIVE_UPSERT = os.environ.get("SELECTIVE_UPSERT", "1") in ("1", "true", "True")
FORCE_REHASH = os.environ.get("FORCE_REHASH", "0") in ("1", "true", "True")
MAX_HASH_FETCH = int(os.environ.get("MAX_HASH_FETCH", "5000"))
INT_ARRAY_FIELDS = {"row_range", "token_range"}
NUMBER_ARRAY_FIELDS = {"audio_range"}
TEXT_ARRAY_FIELDS = {"tags", "layout_tags", "heading_path", "headings"}
session = boto3.Session()
creds = session.get_credentials()
HAS_AWS_CREDS = creds is not None and getattr(creds, "access_key", None) and getattr(creds, "secret_key", None)
s3 = None
if HAS_AWS_CREDS and S3_BUCKET:
    s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else boto3.client("s3")
def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
def make_uuid_v5(document_id: str, chunk_id: str) -> str:
    ns_s = os.environ.get("NAMESPACE_UUID", "7f3d2a6e-1c5f-4f7a-9d3c-2d5f7a6b8e9c")
    try:
        ns = uuid.UUID(ns_s)
    except Exception:
        ns = uuid.UUID("7f3d2a6e-1c5f-4f7a-9d3c-2d5f7a6b8e9c")
    return str(uuid.uuid5(ns, f"{document_id}:{chunk_id}"))
def make_vector_id(document_id: str, chunk_id: str) -> int:
    h = hashlib.sha256(f"{document_id}:{chunk_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)
def parse_json_objects_from_text(s: str) -> List[Dict[str, Any]]:
    if not s or not s.strip():
        return []
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return j
        except Exception:
            pass
    parts = []
    lines = s.splitlines()
    maybe_jsonl = True
    for ln in lines:
        t = ln.strip()
        if not t:
            continue
        try:
            parts.append(json.loads(t))
        except Exception:
            maybe_jsonl = False
            break
    if maybe_jsonl and parts:
        return parts
    objs = []
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == '"' and not esc:
            in_str = not in_str
        if in_str and ch == "\\" and not esc:
            esc = True
            continue
        if not in_str:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    raw = s[start:i + 1]
                    try:
                        objs.append(json.loads(raw))
                    except Exception:
                        pass
                    start = None
        if esc:
            esc = False
    if objs:
        return objs
    try:
        j = json.loads(s)
        if isinstance(j, dict):
            return [j]
    except Exception:
        pass
    raise RuntimeError("Unable to parse JSON/JSONL content")
def try_connect_with_params(host, port, secure, grpc_host, grpc_port, grpc_secure, auth, headers):
    try:
        client = weaviate.connect_to_custom(http_host=host, http_port=port, http_secure=secure, grpc_host=grpc_host, grpc_port=grpc_port, grpc_secure=grpc_secure, auth_credentials=auth, headers=headers)
        return client
    except TypeError:
        client = weaviate.connect_to_custom(http_host=host, http_port=port, http_secure=secure, grpc_host=grpc_host, grpc_port=grpc_port, grpc_secure=grpc_secure, auth_client_secret=auth, headers=headers)
        return client
def connect_weaviate():
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = (parsed.scheme == "https")
    grpc_host = host
    grpc_port = 50051
    grpc_secure = False
    headers = {}
    auth = None
    if WEAVIATE_API_KEY:
        try:
            auth = Auth.api_key(WEAVIATE_API_KEY)
        except Exception:
            auth = None
    if auth:
        try:
            client = try_connect_with_params(host, port, secure, grpc_host, grpc_port, grpc_secure, auth, headers)
            return client
        except Exception as e:
            msg = str(e).lower()
            if "oidc" in msg or "401" in msg or "unauthor" in msg:
                client = try_connect_with_params(host, port, secure, grpc_host, grpc_port, grpc_secure, None, headers)
                return client
            raise
    else:
        client = try_connect_with_params(host, port, secure, grpc_host, grpc_port, grpc_secure, None, headers)
        return client
def get_server_schema_map(client) -> Dict[str, Tuple[str, bool]]:
    out = {}
    try:
        schema = client.schema.get()
    except Exception:
        return out
    classes = schema.get("classes", [])
    for cls in classes:
        if cls.get("class") == COLLECTION_NAME:
            for prop in cls.get("properties", []):
                name = prop.get("name")
                dtypes = prop.get("dataType", [])
                if not dtypes:
                    continue
                dtype = dtypes[0]
                if isinstance(dtype, str):
                    is_array = dtype.endswith("[]")
                    base = dtype.replace("[]", "")
                    if base in ("text", "string"):
                        base = "text"
                    elif base in ("int", "integer", "long"):
                        base = "int"
                    elif base in ("number", "float", "double"):
                        base = "number"
                    elif base in ("date", "dateTime"):
                        base = "date"
                    elif base in ("boolean", "bool"):
                        base = "bool"
                    else:
                        base = base.lower()
                    out[name] = (base, is_array)
    return out
def ensure_collection(client):
    try:
        client.collections.get(COLLECTION_NAME)
        return
    except Exception:
        pass
    try:
        int_array_type = DataType.INT_ARRAY
    except Exception:
        int_array_type = getattr(DataType, "NUMBER_ARRAY", DataType.NUMBER)
    try:
        number_array_type = DataType.NUMBER_ARRAY
    except Exception:
        number_array_type = getattr(DataType, "NUMBER_ARRAY", DataType.NUMBER)
    props = [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="document_id", data_type=DataType.TEXT),
        Property(name="chunk_id", data_type=DataType.TEXT),
        Property(name="chunk_type", data_type=DataType.TEXT),
        Property(name="token_count", data_type=DataType.INT),
        Property(name="file_type", data_type=DataType.TEXT),
        Property(name="file_name", data_type=DataType.TEXT),
        Property(name="source_url", data_type=DataType.TEXT),
        Property(name="page_number", data_type=DataType.INT),
        Property(name="timestamp", data_type=DataType.DATE),
        Property(name="tags", data_type=DataType.TEXT_ARRAY),
        Property(name="layout_tags", data_type=DataType.TEXT_ARRAY),
        Property(name="heading_path", data_type=DataType.TEXT_ARRAY),
        Property(name="headings", data_type=DataType.TEXT_ARRAY),
        Property(name="parser_version", data_type=DataType.TEXT),
        Property(name="audio_range_human", data_type=DataType.TEXT),
        Property(name="parse_chunk_duration_ms", data_type=DataType.INT),
        Property(name="content_hash", data_type=DataType.TEXT),
        Property(name="source_file_hash", data_type=DataType.TEXT),
        Property(name="vector_id", data_type=DataType.NUMBER),
        Property(name="deleted", data_type=DataType.BOOL),
        Property(name="row_range", data_type=int_array_type),
        Property(name="token_range", data_type=int_array_type),
        Property(name="audio_range", data_type=number_array_type),
    ]
    try:
        client.collections.create(name=COLLECTION_NAME, vector_config=Configure.Vectors.self_provided(), properties=props)
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "class already exists" in msg:
            pass
        else:
            raise
    for _ in range(20):
        try:
            client.collections.get(COLLECTION_NAME)
            return
        except Exception:
            time.sleep(0.2)
    return
def coerce_value_to_type(base: str, is_array: bool, value: Any):
    if is_array:
        arr = []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, tuple)):
                    value = parsed
                else:
                    value = [parsed]
            except Exception:
                if "," in value:
                    parts = [p.strip() for p in value.split(",") if p.strip()]
                    value = parts
                elif "-" in value:
                    parts = [p.strip() for p in value.split("-") if p.strip()]
                    value = parts
                else:
                    value = [value]
        if isinstance(value, (list, tuple)):
            for x in value:
                if x is None:
                    continue
                try:
                    if base == "int":
                        if isinstance(x, str):
                            arr.append(int(float(x)))
                        else:
                            arr.append(int(x))
                    elif base == "number":
                        arr.append(float(x))
                    elif base == "bool":
                        if isinstance(x, str):
                            arr.append(x.strip().lower() in ("1", "true", "yes"))
                        else:
                            arr.append(bool(x))
                    else:
                        arr.append(str(x))
                except Exception:
                    continue
        else:
            try:
                if base == "int":
                    arr = [int(value)]
                elif base == "number":
                    arr = [float(value)]
                elif base == "bool":
                    arr = [bool(value)]
                else:
                    arr = [str(value)]
            except Exception:
                arr = [str(value)]
        return arr
    else:
        if isinstance(value, (list, tuple, dict)):
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            else:
                return json.dumps(value)
        if base == "int":
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    return 0
        elif base == "number":
            try:
                return float(value)
            except Exception:
                try:
                    return float(str(value))
                except Exception:
                    return 0.0
        elif base == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes")
            return bool(value)
        elif base == "date":
            return str(value)
        else:
            return str(value)
def sanitize_props_for_schema(props: Dict[str, Any], schema_map: Dict[str, Tuple[str, bool]]) -> Dict[str, Any]:
    out = {}
    schema_known = bool(schema_map)
    for k, v in props.items():
        if k in schema_map:
            base, is_array = schema_map[k]
            try:
                out[k] = coerce_value_to_type(base, is_array, v)
            except Exception:
                try:
                    out[k] = json.dumps(v)
                except Exception:
                    out[k] = str(v)
            continue
        # unknown property: only send real arrays if server schema explicitly expects arrays (schema_known and property recorded as array)
        if k in INT_ARRAY_FIELDS:
            if schema_known and (k in schema_map and schema_map[k][1]):
                out[k] = coerce_value_to_type("int", True, v)
            else:
                try:
                    out[k] = json.dumps(v)
                except Exception:
                    out[k] = str(v)
            continue
        if k in NUMBER_ARRAY_FIELDS:
            if schema_known and (k in schema_map and schema_map[k][1]):
                out[k] = coerce_value_to_type("number", True, v)
            else:
                try:
                    out[k] = json.dumps(v)
                except Exception:
                    out[k] = str(v)
            continue
        if k in TEXT_ARRAY_FIELDS:
            if schema_known and (k in schema_map and schema_map[k][1]):
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, (list, tuple)):
                            out[k] = [str(x) for x in parsed if x is not None]
                            continue
                    except Exception:
                        pass
                if isinstance(v, (list, tuple)):
                    out[k] = [str(x) for x in v if x is not None]
                    continue
                out[k] = [str(v)]
            else:
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, (list, tuple)):
                            out[k] = json.dumps(parsed)
                            continue
                    except Exception:
                        pass
                    out[k] = v
                elif isinstance(v, (list, tuple, dict)):
                    try:
                        out[k] = json.dumps(v)
                    except Exception:
                        out[k] = str(v)
                else:
                    out[k] = str(v)
            continue
        if isinstance(v, (list, dict, tuple)):
            try:
                out[k] = json.dumps(v)
            except Exception:
                out[k] = str(v)
        elif v is None:
            continue
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out
def rest_upsert_object(weaviate_url: str, api_key: str, class_name: str, uuid_str: str, props: Dict[str, Any], vector: Optional[List[float]] = None, timeout: float = 15.0):
    url = weaviate_url.rstrip("/") + "/v1/objects"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {"class": class_name, "id": uuid_str, "properties": props}
    if vector is not None:
        payload["vector"] = vector
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return True
    except Exception as e:
        try:
            up_url = weaviate_url.rstrip("/") + f"/v1/objects/{uuid_str}"
            r2 = httpx.put(up_url, json=payload, headers=headers, timeout=timeout)
            r2.raise_for_status()
            return True
        except Exception:
            print("REST upsert failed for uuid", uuid_str, "err:", str(e), file=sys.stderr)
            return False
def weaviate_batch_upsert(client, items: List[Dict[str, Any]], schema_map: Dict[str, Tuple[str, bool]]):
    coll = client.collections.use(COLLECTION_NAME)
    try:
        with coll.batch.fixed_size(batch_size=INDEX_BATCH, concurrent_requests=1) as batch:
            for it in items:
                sanitized_props = sanitize_props_for_schema(it["properties"], schema_map)
                batch.add_object(properties=sanitized_props, vector=it.get("vector"), uuid=it.get("uuid"))
        return True
    except WeaviateInsertManyAllFailedError as e:
        print("Batch insert failed, falling back to per-object upserts:", str(e), file=sys.stderr)
        for it in items:
            try:
                sanitized = sanitize_props_for_schema(it["properties"], schema_map)
                ok = rest_upsert_object(WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION_NAME, it["uuid"], sanitized, it.get("vector"))
                if not ok:
                    print("Failed per-object upsert uuid", it["uuid"], file=sys.stderr)
            except Exception as ex:
                print("Per-object upsert exception:", ex, file=sys.stderr)
        return False
    except Exception as e:
        print("Batch insert generic failure, trying per-object upserts:", str(e), file=sys.stderr)
        for it in items:
            try:
                sanitized = sanitize_props_for_schema(it["properties"], schema_map)
                ok = rest_upsert_object(WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION_NAME, it["uuid"], sanitized, it.get("vector"))
                if not ok:
                    print("Failed per-object upsert uuid", it["uuid"], file=sys.stderr)
            except Exception as ex:
                print("Per-object upsert exception:", ex, file=sys.stderr)
        return False
class EmbedClient:
    def __init__(self, concurrency: int, batch_size: int):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.batch_size = batch_size
        limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
        self._client = httpx.AsyncClient(limits=limits)
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        async def do_batch(batch_texts: List[str]):
            async with self.semaphore:
                resp = await self._client.post(EMBEDDING_URL, json={"texts": batch_texts}, timeout=EMBED_TIMEOUT)
                resp.raise_for_status()
                j = resp.json()
                if isinstance(j, dict):
                    if "embeddings" in j and isinstance(j["embeddings"], list):
                        return j["embeddings"]
                    if "data" in j and isinstance(j["data"], list):
                        out = []
                        for it in j["data"]:
                            if isinstance(it, dict) and "embedding" in it:
                                out.append(it["embedding"])
                            elif isinstance(it, list):
                                out.append(it)
                        if out:
                            return out
                if isinstance(j, list) and all(isinstance(x, list) for x in j):
                    return j
                raise RuntimeError("Unsupported embedder response shape")
        tasks = [asyncio.create_task(do_batch(texts[i:i + self.batch_size])) for i in range(0, len(texts), self.batch_size)]
        results: List[List[float]] = []
        for t in asyncio.as_completed(tasks):
            res = await t
            results.extend(res)
        return results
    async def close(self):
        await self._client.aclose()
def list_chunked_files_local(base_dir: str) -> List[str]:
    out = []
    chunk_dir = os.path.join(os.path.abspath(base_dir), "chunked")
    if not os.path.isdir(chunk_dir):
        return out
    for fn in os.listdir(chunk_dir):
        if fn.endswith(".json") or fn.endswith(".jsonl"):
            out.append(os.path.join(chunk_dir, fn))
    return out
def list_chunked_files_s3(bucket: str, prefix: str) -> List[str]:
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    prefix_key = prefix.rstrip("/") + "/"
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_key):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or key.lower().endswith(".manifest.json"):
                continue
            if key.lower().endswith(".json") or key.lower().endswith(".jsonl"):
                out.append(key)
    return out
def read_manifest_local(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
def write_manifest_local(path: str, manifest: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)
def s3_get_manifest(bucket: str, key: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key + ".manifest.json")
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None
def s3_write_manifest(bucket: str, key: str, manifest: dict):
    tmp_key = key + ".manifest.json.tmp"
    final_key = key + ".manifest.json"
    s3.put_object(Bucket=bucket, Key=tmp_key, Body=json.dumps(manifest, indent=2).encode("utf-8"), ContentType="application/json")
    s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': tmp_key}, Key=final_key)
    s3.delete_object(Bucket=bucket, Key=tmp_key)
def compute_local_file_hash(path: str, chunk_size=8*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk_size), b""):
            h.update(blk)
    return h.hexdigest()
def compute_s3_file_hash(bucket: str, key: str, chunk_size=8*1024):
    obj = s3.get_object(Bucket=bucket, Key=key)
    stream = obj["Body"]
    h = hashlib.sha256()
    for blk in iter(lambda: stream.read(chunk_size), b""):
        h.update(blk)
    return h.hexdigest()
def weaviate_aggregate_count_http(weaviate_url, file_hash, api_key=None, timeout=15.0):
    q = '{ Aggregate { ' + COLLECTION_NAME + '(where: { path: ["source_file_hash"], operator: Equal, valueString: "%s" }) { meta { count } } } }' % file_hash
    url = weaviate_url.rstrip("/") + "/v1/graphql"
    headers = {"Content-Type": "application/json"}
    if api_key: headers["X-API-Key"] = api_key
    r = httpx.post(url, json={"query": q}, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    try:
        return int(j["data"]["Aggregate"][COLLECTION_NAME][0]["meta"]["count"])
    except Exception:
        return 0
def weaviate_get_hashes_http(weaviate_url, file_hash, api_key=None, limit=1000, timeout=30.0):
    url = weaviate_url.rstrip("/") + "/v1/graphql"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    hashes = []
    offset = 0
    while True:
        q = '{ Get { ' + COLLECTION_NAME + '(where: { path:["source_file_hash"], operator: Equal, valueString: "%s" }, limit: %d, offset: %d) { content_hash } } }' % (file_hash, limit, offset)
        r = httpx.post(url, json={"query": q}, headers=headers, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        items = j.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
        if not items:
            break
        for it in items:
            h = it.get("content_hash")
            if h:
                hashes.append(h)
        if len(items) < limit:
            break
        offset += limit
    return hashes
def compute_chunks_fingerprint(list_hashes: List[str]) -> str:
    s = "".join(sorted(list_hashes))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
async def index_chunk_files_for_file(file_hash: str, embed_client: EmbedClient, client_weaviate, mode: str) -> Tuple[int, List[str]]:
    all_chunks = []
    if mode == "s3":
        paginator = s3.get_paginator("list_objects_v2")
        prefix = S3_CHUNKED_PREFIX + file_hash
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.lower().endswith(".json") or k.lower().endswith(".jsonl"):
                    body = s3.get_object(Bucket=S3_BUCKET, Key=k)["Body"].read().decode("utf-8")
                    parts = parse_json_objects_from_text(body)
                    for p in parts:
                        p["_chunk_source_key"] = k
                        all_chunks.append(p)
    else:
        base = os.path.abspath(LOCAL_DATA_DIR)
        chunk_dir = os.path.join(base, "chunked")
        if not os.path.isdir(chunk_dir):
            return 0, []
        candidates = [f for f in os.listdir(chunk_dir) if f.startswith(file_hash) and (f.endswith(".json") or f.endswith(".jsonl"))]
        for fn in candidates:
            path = os.path.join(chunk_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                parts = parse_json_objects_from_text(f.read())
                for p in parts:
                    p["_chunk_source_key"] = fn
                    all_chunks.append(p)
    if not all_chunks:
        return 0, []
    for c in all_chunks:
        c["content_hash"] = c.get("content_hash") or sha256_text(c.get("text", ""))
    existing_hashes = set()
    if SELECTIVE_UPSERT:
        try:
            hashes = weaviate_get_hashes_http(WEAVIATE_URL, file_hash, WEAVIATE_API_KEY, limit=MAX_HASH_FETCH)
            existing_hashes = set(hashes)
        except Exception:
            existing_hashes = set()
    indices_to_index = []
    for i, c in enumerate(all_chunks):
        if FORCE_REHASH or (not SELECTIVE_UPSERT) or (c["content_hash"] not in existing_hashes):
            indices_to_index.append(i)
    if not indices_to_index:
        combined = list({h for h in existing_hashes}.union({c["content_hash"] for c in all_chunks}))
        return 0, combined
    texts = [all_chunks[i].get("text", "") for i in indices_to_index]
    embeddings = await embed_client.embed_texts(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError("embedding count mismatch")
    schema_map = get_server_schema_map(client_weaviate)
    idx_map = {pos: idx for pos, idx in enumerate(indices_to_index)}
    total_indexed = 0
    collected_hashes = set(c["content_hash"] for c in all_chunks)
    batch_items = []
    for i, c in enumerate(all_chunks):
        if i not in idx_map:
            continue
        vec = embeddings[idx_map[i]]
        doc_id = c.get("document_id") or c.get("_chunk_source_key") or file_hash
        chunk_id = c.get("chunk_id") or hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        uuid_str = make_uuid_v5(doc_id, chunk_id)
        v_id = make_vector_id(doc_id, chunk_id)
        props = {k: v for k, v in c.items() if k != "embedding" and not k.startswith("_")}
        props["text"] = c.get("text", "")
        props["document_id"] = doc_id
        props["chunk_id"] = chunk_id
        props["source_file_hash"] = file_hash
        props["content_hash"] = c["content_hash"]
        props["vector_id"] = v_id
        props["deleted"] = bool(props.get("deleted", False))
        for m in ["file_name", "chunk_type", "file_type", "tags", "timestamp"]:
            if m in c and m not in props:
                props[m] = c[m]
        sanitized = sanitize_props_for_schema(props, schema_map)
        batch_items.append({"properties": sanitized, "vector": [float(x) for x in vec], "uuid": uuid_str})
        if len(batch_items) >= INDEX_BATCH:
            weaviate_batch_upsert(client_weaviate, batch_items, schema_map)
            total_indexed += len(batch_items)
            batch_items = []
    if batch_items:
        weaviate_batch_upsert(client_weaviate, batch_items, schema_map)
        total_indexed += len(batch_items)
    return total_indexed, list(collected_hashes)
async def process_raw_file(item: str, embed_client: EmbedClient, client_weaviate, mode: str):
    if mode == "s3":
        bucket = S3_BUCKET
        key = item
        manifest = s3_get_manifest(bucket, key) or {}
        file_hash = manifest.get("file_hash")
        if not file_hash or FORCE_REHASH:
            file_hash = compute_s3_file_hash(bucket, key)
            manifest["file_hash"] = file_hash
        expected_chunks = manifest.get("expected_chunks")
        if manifest.get("status") == "completed" and manifest.get("file_hash") == file_hash and expected_chunks:
            try:
                cnt = weaviate_aggregate_count_http(WEAVIATE_URL, file_hash, WEAVIATE_API_KEY)
                if cnt == int(expected_chunks):
                    return
            except Exception:
                pass
        manifest["status"] = "in-progress"
        manifest["s3_key"] = key
        manifest["pipeline_run_id"] = manifest.get("pipeline_run_id") or str(uuid.uuid4())
        manifest["mime_type"] = manifest.get("mime_type") or ""
        s3_write_manifest(bucket, key, manifest)
        indexed_count, collected_hashes = await index_chunk_files_for_file(file_hash, embed_client, client_weaviate, "s3")
        manifest["indexed_chunks"] = manifest.get("indexed_chunks", 0) + indexed_count
        manifest["weaviate_count"] = weaviate_aggregate_count_http(WEAVIATE_URL, file_hash, WEAVIATE_API_KEY)
        if manifest.get("expected_chunks"):
            try:
                weav_fhash = None
                if collected_hashes:
                    weav_fhash = compute_chunks_fingerprint(collected_hashes)
                if manifest.get("expected_chunks") == manifest["weaviate_count"] and (not manifest.get("chunks_fingerprint") or (weav_fhash and weav_fhash == manifest.get("chunks_fingerprint"))):
                    manifest["status"] = "completed"
                    manifest["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                else:
                    manifest["status"] = "partial"
            except Exception:
                manifest["status"] = "partial"
        else:
            manifest["status"] = "completed"
            manifest["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        s3_write_manifest(bucket, key, manifest)
    else:
        path = item
        manifest_path = path + ".manifest.json"
        manifest = read_manifest_local(manifest_path) or {}
        file_hash = manifest.get("file_hash")
        if not file_hash or FORCE_REHASH:
            file_hash = compute_local_file_hash(path)
            manifest["file_hash"] = file_hash
        expected_chunks = manifest.get("expected_chunks")
        if manifest.get("status") == "completed" and manifest.get("file_hash") == file_hash and expected_chunks:
            try:
                cnt = weaviate_aggregate_count_http(WEAVIATE_URL, file_hash, WEAVIATE_API_KEY)
                if cnt == int(expected_chunks):
                    return
            except Exception:
                pass
        manifest["status"] = "in-progress"
        manifest["s3_key"] = manifest.get("s3_key") or path
        manifest["pipeline_run_id"] = manifest.get("pipeline_run_id") or str(uuid.uuid4())
        manifest["mime_type"] = manifest.get("mime_type") or ""
        write_manifest_local(manifest_path, manifest)
        indexed_count, collected_hashes = await index_chunk_files_for_file(file_hash, embed_client, client_weaviate, "local")
        manifest["indexed_chunks"] = manifest.get("indexed_chunks", 0) + indexed_count
        manifest["weaviate_count"] = weaviate_aggregate_count_http(WEAVIATE_URL, file_hash, WEAVIATE_API_KEY)
        if manifest.get("expected_chunks"):
            try:
                weav_fhash = None
                if collected_hashes:
                    weav_fhash = compute_chunks_fingerprint(collected_hashes)
                if manifest.get("expected_chunks") == manifest["weaviate_count"] and (not manifest.get("chunks_fingerprint") or (weav_fhash and weav_fhash == manifest.get("chunks_fingerprint"))):
                    manifest["status"] = "completed"
                    manifest["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                else:
                    manifest["status"] = "partial"
            except Exception:
                manifest["status"] = "partial"
        else:
            manifest["status"] = "completed"
            manifest["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_manifest_local(manifest_path, manifest)
async def discover_raw_inputs() -> List[Dict[str, str]]:
    items = []
    if HAS_AWS_CREDS and S3_BUCKET:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith("/") or k.lower().endswith(".manifest.json"):
                    continue
                items.append({"mode": "s3", "key": k})
        return items
    base = os.path.abspath(LOCAL_DATA_DIR)
    raw_dir = os.path.join(base, "raw")
    if not os.path.isdir(raw_dir):
        return []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".manifest.json"):
                continue
            path = os.path.join(root, f)
            items.append({"mode": "local", "key": path})
    return items
async def main():
    client_weaviate = None
    embed_client = None
    try:
        client_weaviate = connect_weaviate()
        ensure_collection(client_weaviate)
        embed_client = EmbedClient(concurrency=EMBED_CONCURRENCY, batch_size=EMBED_BATCH)
        items = await discover_raw_inputs()
        if not items:
            print("No raw inputs discovered.", file=sys.stderr)
            await embed_client.close()
            try:
                client_weaviate.close()
            except Exception:
                pass
            return
        sem = asyncio.Semaphore(CONCURRENCY)
        async def worker(it):
            async with sem:
                try:
                    await process_raw_file(it["key"], embed_client, client_weaviate, it["mode"])
                except Exception as e:
                    print("error processing", it["key"], str(e), file=sys.stderr)
        tasks = [asyncio.create_task(worker(it)) for it in items]
        for t in asyncio.as_completed(tasks):
            try:
                await t
            except Exception as e:
                print("task error", e, file=sys.stderr)
        await embed_client.close()
        try:
            client_weaviate.close()
        except Exception:
            pass
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        if embed_client:
            try:
                await embed_client.close()
            except Exception:
                pass
        if client_weaviate:
            try:
                client_weaviate.close()
            except Exception:
                pass
    except Exception as e:
        print("Fatal error in main:", str(e), file=sys.stderr)
        if embed_client:
            try:
                await embed_client.close()
            except Exception:
                pass
        if client_weaviate:
            try:
                client_weaviate.close()
            except Exception:
                pass
        raise
if __name__ == "__main__":
    asyncio.run(main())
