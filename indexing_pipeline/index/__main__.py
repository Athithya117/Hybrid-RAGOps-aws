# indexing_pipeline/index/__main__.py
import json
from pathlib import Path
import logging
from config import settings
from arangodb_indexer import index_chunks_arango

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parents[1] / "data" / "parsed_chunks.jsonl"

def load_chunks(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    chunks = list(load_chunks(DATA_FILE))
    logger.info(f"Loaded {len(chunks)} chunks from {DATA_FILE}")

    # Push to ArangoDB (vector + docs + edges)
    index_chunks_arango(chunks)
