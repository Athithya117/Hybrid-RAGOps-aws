# indexing_pipeline/index/arangodb_indexer.py
import json
import logging
import requests
from typing import Dict, List
from arango import ArangoClient
from config import settings

logger = logging.getLogger(__name__)

# HTTP API helper for index creation (because python-arango may not support 'vector' index type)
def create_vector_index_http(db_name: str, collection: str, index_def: Dict):
    """
    Create index via ArangoDB HTTP API:
    POST /_api/index?collection=<collection>
    index_def example:
    {
      "type": "vector",
      "name": "vec_cosine_idx",
      "fields": ["embedding"],
      "params": {
        "metric": "cosine",
        "dimension": 768,
        "nLists": 256,
        "factory": "IVF256,PQ64"
      }
    }
    """
    url = settings.ARANGO_URL.rstrip("/")
    api_path = f"{url}/_db/{db_name}/_api/index?collection={collection}"
    auth = (settings.ARANGO_USER, settings.ARANGO_PASSWORD)
    resp = requests.post(api_path, json=index_def, auth=auth)
    if resp.status_code not in (200, 201):
        # if index already exists, Arango might return 409 or 400; log and raise for safety
        logger.error(f"Vector index creation failed: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    logger.info(f"Created/verified vector index for {collection}: {resp.json()}")

def init_arango_collections_and_vector_index():
    # plain Arango client flow for DB and collections
    client = ArangoClient(hosts=settings.ARANGO_URL)
    sys_db = client.db("_system", username=settings.ARANGO_USER, password=settings.ARANGO_PASSWORD)

    if not sys_db.has_database(settings.ARANGO_DB):
        sys_db.create_database(settings.ARANGO_DB)
        logger.info(f"Created ArangoDB database {settings.ARANGO_DB}")

    db = client.db(settings.ARANGO_DB, username=settings.ARANGO_USER, password=settings.ARANGO_PASSWORD)

    # Create document collection (chunks)
    if not db.has_collection(settings.ARANGO_COLLECTION):
        db.create_collection(settings.ARANGO_COLLECTION)
        logger.info(f"Created document collection {settings.ARANGO_COLLECTION}")

    # Create edge collection
    if not db.has_collection(settings.ARANGO_EDGE_COLLECTION):
        db.create_collection(settings.ARANGO_EDGE_COLLECTION, edge=True)
        logger.info(f"Created edge collection {settings.ARANGO_EDGE_COLLECTION}")

    # Create vector index via HTTP API (idempotent-ish; servers return error if exact index exists)
    index_def = {
        "type": "vector",
        "name": settings.ARANGO_VECTOR_INDEX_NAME,
        "fields": [settings.ARANGO_VECTOR_FIELD],
        "params": {
            "metric": settings.ARANGO_VECTOR_METRIC,
            "dimension": settings.VECTOR_DIM,
            # Optional tuning parameters:
            # nLists (IVF size), defaultNProbe (search effort), factory (FAISS factory string)
            **({} if not settings.ARANGO_VECTOR_PARAMS else settings.ARANGO_VECTOR_PARAMS)
        }
    }

    try:
        create_vector_index_http(settings.ARANGO_DB, settings.ARANGO_COLLECTION, index_def)
    except Exception as e:
        # If index exists or there is a transient error, log a warning but continue.
        logger.warning(f"Vector index creation returned error (might already exist): {e}")

    return db

def build_edges(chunk: Dict) -> List[Dict]:
    edges = []
    src = f"{settings.ARANGO_COLLECTION}/{chunk['chunk_id']}"
    graph = chunk.get("graph", {})
    for rel, target in [
        ("parent_id", graph.get("parent_id")),
        ("previous_id", graph.get("previous_id")),
        ("next_id", graph.get("next_id")),
    ]:
        if target:
            edges.append({
                "_from": src,
                "_to": f"{settings.ARANGO_COLLECTION}/{target}",
                "relation": rel
            })
    return edges

def index_chunks_arango(chunks: List[Dict]):
    db = init_arango_collections_and_vector_index()
    doc_coll = db.collection(settings.ARANGO_COLLECTION)
    edge_coll = db.collection(settings.ARANGO_EDGE_COLLECTION)

    doc_batch, edge_batch = [], []
    for i, chunk in enumerate(chunks, 1):
        # ensure embedding attribute is present and valid
        emb = chunk.get("embedding")
        if not isinstance(emb, list) or len(emb) != settings.VECTOR_DIM:
            logger.warning(f"Skipping invalid embedding for chunk {chunk.get('chunk_id')}")
            continue

        # build the document: save embedding in `embedding` field (vector index expects this)
        # don't store huge redundant text in payload if you want smaller DB — adjust as needed
        doc = {"_key": chunk["chunk_id"], **chunk}
        # move embedding into explicit field (already present in chunk schema), ensure name correct
        doc[settings.ARANGO_VECTOR_FIELD] = emb

        doc_batch.append(doc)
        edge_batch.extend(build_edges(chunk))

        if i % settings.ARANGO_BATCH_SIZE == 0:
            doc_coll.insert_many(doc_batch, overwrite=True)
            if edge_batch:
                edge_coll.insert_many(edge_batch, overwrite=True)
            logger.info(f"Arango inserted batch ending at {i}")
            doc_batch.clear()
            edge_batch.clear()

    if doc_batch:
        doc_coll.insert_many(doc_batch, overwrite=True)
        if edge_batch:
            edge_coll.insert_many(edge_batch, overwrite=True)
        logger.info("Arango inserted final batch")
