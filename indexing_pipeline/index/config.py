import os
from enum import Enum

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLID = "euclid"
    DOT = "dot"

class Settings:
    # Qdrant
    QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "rag8s_chunks")
    VECTOR_DIM       = int(os.getenv("VECTOR_DIM", 768))
    DISTANCE_METRIC  = DistanceMetric(os.getenv("DISTANCE_METRIC", "cosine"))
    BATCH_SIZE       = int(os.getenv("BATCH_SIZE", 256))

    # ArangoDB
    ARANGO_URL           = os.getenv("ARANGO_URL", "http://localhost:8529")
    ARANGO_USER          = os.getenv("ARANGO_USER", "root")
    ARANGO_PASSWORD      = os.getenv("ARANGO_PASSWORD", "")
    ARANGO_DB            = os.getenv("ARANGO_DB", "rag_database")
    ARANGO_COLLECTION    = os.getenv("ARANGO_COLLECTION", "rag_chunks")
    ARANGO_EDGE_COLLECTION = os.getenv("ARANGO_EDGE_COLLECTION", "rag_edges")
    ARANGO_BATCH_SIZE    = int(os.getenv("ARANGO_BATCH_SIZE", 256))

    # Logging
    LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
