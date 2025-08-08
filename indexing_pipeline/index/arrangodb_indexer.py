import logging
from typing import Dict, List
from arango import ArangoClient
from config import settings

logger = logging.getLogger(__name__)

def init_arango_collections():
    client = ArangoClient(hosts=settings.ARANGO_URL)
    sys_db = client.db("_system", username=settings.ARANGO_USER, password=settings.ARANGO_PASSWORD)

    if not sys_db.has_database(settings.ARANGO_DB):
        sys_db.create_database(settings.ARANGO_DB)
        logger.info(f"Created ArangoDB database {settings.ARANGO_DB}")

    db = client.db(settings.ARANGO_DB, username=settings.ARANGO_USER, password=settings.ARANGO_PASSWORD)

    if not db.has_collection(settings.ARANGO_COLLECTION):
        db.create_collection(settings.ARANGO_COLLECTION)
        logger.info(f"Created document collection {settings.ARANGO_COLLECTION}")

    if not db.has_collection(settings.ARANGO_EDGE_COLLECTION):
        db.create_collection(settings.ARANGO_EDGE_COLLECTION, edge=True)
        logger.info(f"Created edge collection {settings.ARANGO_EDGE_COLLECTION}")

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
    db = init_arango_collections()
    doc_coll = db.collection(settings.ARANGO_COLLECTION)
    edge_coll = db.collection(settings.ARANGO_EDGE_COLLECTION)

    doc_batch, edge_batch = [], []
    for i, chunk in enumerate(chunks, 1):
        doc_batch.append({"_key": chunk["chunk_id"], **chunk})
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
