import logging
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from config import settings
from utils import normalize_payload

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

def init_collection(client: QdrantClient):
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        metric = {
            settings.DistanceMetric.COSINE: Distance.COSINE,
            settings.DistanceMetric.EUCLID: Distance.EUCLID,
            settings.DistanceMetric.DOT: Distance.DOT,
        }[settings.DISTANCE_METRIC]
        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.VECTOR_DIM,
                distance=metric
            )
        )
        logger.info(f"Created Qdrant collection {settings.QDRANT_COLLECTION}")

def build_points(chunks: List[Dict]) -> List[PointStruct]:
    points = []
    for c in chunks:
        emb = c.get("embedding")
        if not isinstance(emb, list) or len(emb) != settings.VECTOR_DIM:
            logger.warning(f"Skipping invalid embedding for chunk {c.get('chunk_id')}")
            continue
        points.append(PointStruct(
            id=c["chunk_id"],
            vector=emb,
            payload=normalize_payload(c)
        ))
    return points

def index_chunks_qdrant(chunks: List[Dict]):
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    init_collection(client)
    points = build_points(chunks)
    for i in range(0, len(points), settings.BATCH_SIZE):
        batch = points[i : i + settings.BATCH_SIZE]
        try:
            client.upsert(collection_name=settings.QDRANT_COLLECTION, points=batch)
            logger.info(f"Qdrant indexed batch {i // settings.BATCH_SIZE + 1}")
        except UnexpectedResponse as e:
            logger.error(f"Qdrant batch {i} failed: {e}")
