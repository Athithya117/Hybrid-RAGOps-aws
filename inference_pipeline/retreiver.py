# inference_pipeline/retriever.py
"""
RAG8s retriever with GeAR-style expansion (using precomputed triples),
FAISS sidecar for vectors, ArangoSearch BM25 for keywords, and Arango graph traversal.

Designed to be pragmatic and production-ready:
 - precomputed triples (indexing) -> no ReLiK at inference time
 - parallel candidate gen: FAISS + BM25 + direct graph
 - GeAR beam expansion over precomputed triples stored in ArangoDB graph
 - per-query normalization and weighted fusion
 - optional cross-encoder reranker hook
"""

import time
import math
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# External deps (install in your inference image):
# pip install faiss-cpu python-arango numpy more-itertools
import numpy as np
import faiss
from arango import ArangoClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------
# CONFIG / TUNABLES
# -----------------------
ARANGO_URL = "http://arangodb:8529"           # change as needed
ARANGO_DB = "rag8s"
ARANGO_USER = "root"
ARANGO_PASS = "openSesame"                    # use secret store in prod
CHUNKS_COLLECTION = "chunks"                  # document collection for chunks
CHUNKS_VIEW = "chunksView"                    # ArangoSearch view name over chunks
ENTITY_GRAPH_NAME = "EntityGraph"             # ArangoDB graph with entity nodes/edges
FAISS_INDEX_PATH = "/mnt/faiss/index.ivf"     # or use in-memory index / networked index
FAISS_INDEX_DIM = 768                         # embedding dimensionality (adjust)
FAISS_INDEX = None                             # will be loaded on init

# Candidate sizes
N_VEC = 50   # vector candidates
N_BM25 = 30  # BM25 candidates
N_GRAPH = 20 # direct graph neighbors
MAX_GEARS_HOPS = 2  # GeAR expansion max hops
GEAR_BEAM_WIDTH = 20

# Fusion weights (start here, tune with eval.py)
W_VEC = 0.40
W_BM25 = 0.25
W_GRAPH = 0.20
W_GEAR = 0.15

# Final context size
TOP_K = 10

# Optional reranker settings
USE_RERANKER = False
RERANKER_BATCH_SIZE = 16

# -----------------------
# Helpers: embeddings & FAISS
# -----------------------
def load_faiss_index(path: str):
    """Load FAISS index from disk (or adapt to your FAISS service)."""
    global FAISS_INDEX
    try:
        FAISS_INDEX = faiss.read_index(path)
        logger.info("FAISS index loaded from %s", path)
    except Exception as e:
        logger.error("Failed to load FAISS index: %s", e)
        raise

def faiss_search(vec: np.ndarray, topk: int = N_VEC) -> List[Tuple[int, float]]:
    """Query FAISS index. vec: 1D numpy vector (float32). Returns list of (id, score)."""
    if FAISS_INDEX is None:
        raise RuntimeError("FAISS index not loaded")
    # ensure correct shape
    q = np.asarray(vec).astype('float32').reshape(1, -1)
    distances, ids = FAISS_INDEX.search(q, topk)
    # depending on index metric, distances may be L2 (lower better) or inner product (higher better).
    # Here we return id, raw_score where raw_score is cosine-like (you may need to transform)
    results = []
    for idx, dist in zip(ids[0], distances[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(dist)))
    return results

# Placeholder: replace with your embedding function (ONNX/transformer)
def embed_text(text: str) -> np.ndarray:
    """
    Compute embedding for `text`. Must return 1D numpy float32 vector.
    Replace with your ONNX / GPU model call.
    """
    # Dummy random vector for template; replace with actual model
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.normal(size=(FAISS_INDEX_DIM,)).astype('float32')
    # normalize for cosine
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

# -----------------------
# ArangoDB client
# -----------------------
def init_arango_client():
    client = ArangoClient(hosts=ARANGO_URL)
    db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
    return db

# -----------------------
# BM25 via ArangoSearch
# -----------------------
def bm25_search(db, query_text: str, limit: int = N_BM25) -> List[Dict[str, Any]]:
    """
    Run ArangoSearch BM25 over the chunks view.
    Returns list of dicts: { _id, _key, bm25_score }
    """
    aql = f"""
    LET q = TOKENS(@q, "text_en")
    FOR doc IN {CHUNKS_VIEW}
      SEARCH ANALYZER(doc.text IN q, "text_en")
      SORT BM25(doc) DESC
      LIMIT @limit
      RETURN {{ _id: doc._id, _key: doc._key, bm25: BM25(doc) }}
    """
    cursor = db.aql.execute(aql, bind_vars={'q': query_text, 'limit': limit})
    return [r for r in cursor]

# -----------------------
# Direct graph neighbors (1..N hops) - used as simple graph retrieval
# -----------------------
def graph_neighbors(db, seed_entity_keys: List[str], hops: int = 1, limit: int = N_GRAPH):
    """
    Traverse EntityGraph outwards from the seed entities for up to `hops`.
    Expects precomputed entity nodes to be keyed by some id you store at indexing time.
    Returns list of { chunk_id, hop_distance, path_vertices }
    """
    if not seed_entity_keys:
        return []
    aql = f"""
    LET seeds = @seeds
    FOR v, e, p IN 1..{hops} OUTBOUND seeds GRAPH '{ENTITY_GRAPH_NAME}'
      OPTIONS {{ bfs: true, uniqueVertices: "global" }}
      FILTER HAS(v, 'chunk_id')  /* only entity nodes with linked chunk_id */
      LIMIT @limit
      RETURN DISTINCT {{ chunk_id: v.chunk_id, hop: LENGTH(p.edges), path: p.vertices[*]._key }}
    """
    cursor = db.aql.execute(aql, bind_vars={'seeds': seed_entity_keys, 'limit': limit})
    return [r for r in cursor]

# -----------------------
# GeAR expansion (beam over precomputed triples / graph)
# -----------------------
def gear_expand(db, seed_entity_keys: List[str], max_hops: int = MAX_GEARS_HOPS,
                beam_width: int = GEAR_BEAM_WIDTH) -> List[Dict[str, Any]]:
    """
    GeAR-like expansion using precomputed triples / graph:
      - starts from seed entities
      - does a beam-limited multi-hop traversal and returns candidate chunks with path scores
    Returns list of { chunk_id, gear_score, path }
    """
    # Strategy:
    # 1) get 1-hop neighbors (entities) and their linked chunk ids
    # 2) compute a simple path score = 1/(1 + hop_distance) (or use edge weights if available)
    # 3) maintain beam by selecting top beam_width entity paths at each hop (we simulate beam by LIMIT in AQL)
    # You can extend to a more complex beam with custom scoring & local expand.
    candidates = []
    for hop in range(1, max_hops + 1):
        aql = f"""
        LET seeds = @seeds
        FOR v, e, p IN 1..{hop} OUTBOUND seeds GRAPH '{ENTITY_GRAPH_NAME}'
          OPTIONS {{ bfs: true, uniqueVertices: "global" }}
          FILTER HAS(v, 'chunk_id')
          SORT LENGTH(p.edges) ASC
          LIMIT @beam
          RETURN DISTINCT {{ chunk_id: v.chunk_id, hop: LENGTH(p.edges), path: p.vertices[*]._key }}
        """
        cursor = db.aql.execute(aql, bind_vars={'seeds': seed_entity_keys, 'beam': beam_width})
        for r in cursor:
            # simple score
            score = 1.0 / (1.0 + r.get('hop', 1))
            candidates.append({'chunk_id': r['chunk_id'], 'gear_score': float(score), 'path': r['path']})
    # dedupe by chunk_id keeping max score
    out = {}
    for c in candidates:
        cid = c['chunk_id']
        if cid not in out or c['gear_score'] > out[cid]['gear_score']:
            out[cid] = c
    return list(out.values())

# -----------------------
# Fetch chunk docs and metadata (from Arango)
# -----------------------
def fetch_chunks(db, chunk_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict keyed by chunk_key with chunk document fields and any stored metadata.
    Expects chunk _key values.
    """
    if not chunk_keys:
        return {}
    aql = f"""
    LET keys = @keys
    FOR doc IN {CHUNKS_COLLECTION}
      FILTER doc._key IN keys
      RETURN {{
        _key: doc._key,
        _id: doc._id,
        text: doc.text,
        meta: doc.meta,
        arango_bm25_precomputed: doc.bm25_score  /* optional precomputed */
      }}
    """
    cursor = db.aql.execute(aql, bind_vars={'keys': chunk_keys})
    out = {}
    for r in cursor:
        out[r['_key']] = r
    return out

# -----------------------
# Utility: normalization & fusion
# -----------------------
def minmax_normalize(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mx <= mn + 1e-12:
        return [1.0 for _ in vals]
    return [ (v - mn) / (mx - mn) for v in vals ]

def normalize_and_fuse(candidates: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    candidates: dict keyed by chunk_key with fields:
      { 'vec': float or None, 'bm25': float or None, 'graph': float or None, 'gear': float or None, ... }
    weights: dict with keys 'vec','bm25','graph','gear'
    Returns list of candidates with 'final_score' sorted desc.
    """
    keys = list(candidates.keys())
    vecs = [ candidates[k].get('vec', 0.0) for k in keys ]
    bm25s = [ candidates[k].get('bm25', 0.0) for k in keys ]
    graphs = [ candidates[k].get('graph', 0.0) for k in keys ]
    gears = [ candidates[k].get('gear', 0.0) for k in keys ]

    vec_n = minmax_normalize(vecs)
    bm25_n = minmax_normalize(bm25s)
    graph_n = minmax_normalize(graphs)
    gear_n = minmax_normalize(gears)

    fused = []
    for i,k in enumerate(keys):
        s = (weights['vec'] * vec_n[i]
             + weights['bm25'] * bm25_n[i]
             + weights['graph'] * graph_n[i]
             + weights['gear'] * gear_n[i])
        rec = candidates[k].copy()
        rec['final_score'] = float(s)
        fused.append(rec)
    fused.sort(key=lambda x: x['final_score'], reverse=True)
    return fused

# -----------------------
# Optional rerank placeholder
# -----------------------
def rerank_cross_encoder(query: str, top_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Placeholder: call your cross-encoder reranker here.
    Input: query, list of candidates with fields 'text' and 'final_score'
    Output: reordered list (same elements) with updated 'final_score' and ranking.
    """
    # Implement reranker inference call (batch inputs to gte-modernbert-reranker-base).
    # For now: return unchanged list.
    return top_candidates

# -----------------------
# Main retrieval API
# -----------------------
def retrieve(query: str, db=None, top_k: int = TOP_K,
             use_reranker: bool = USE_RERANKER) -> List[Dict[str, Any]]:
    """
    End-to-end retrieval:
      1) embed query
      2) parallel: vector(FAISS), bm25(ArangoSearch), direct graph traversal
      3) GeAR expansion over precomputed triples
      4) fetch chunk docs & build candidate dict
      5) normalize & fuse scores
      6) optional rerank
      7) return top_k chunks with provenance
    """
    start = time.time()
    if db is None:
        db = init_arango_client()

    # 1) embedding
    q_vec = embed_text(query)

    # 2) vector search -> returns (faiss_id, score)
    vec_hits = faiss_search(q_vec, topk=N_VEC)
    # Map FAISS ids to arango chunk keys:
    # It's recommended you maintain a small Arango collection mapping faiss_id -> chunk_key.
    # Here we assume FAISS id == numeric mapping to stored chunk keys (replace with your mapping)
    faiss_to_chunk_keys = []
    for fid, score in vec_hits:
        # placeholder mapping; replace with actual lookup
        faiss_to_chunk_keys.append((str(fid), float(score)))  # chunk_key, raw_vec_score

    # 3) BM25 search
    bm_hits = bm25_search(db, query_text=query, limit=N_BM25)  # returns docs with bm25
    bm25_keys = [(r['_key'], float(r['bm25'])) for r in bm_hits]

    # 4) Entity detection for graph seeds (cheap lookup)
    # We assume you have an entity inverted index collection 'entityIndex' mapping token -> entity_key(s).
    # For template, we extract tokens simply by splitting; replace with production entity linker.
    entity_tokens = [t.lower() for t in query.split() if len(t) > 2]
    # Lookup entity keys in Arango (pseudo)
    entity_keys = []
    if entity_tokens:
        aql = """
        LET toks = @toks
        FOR e IN entityIndex
          FILTER e.name IN toks
          RETURN e._key
        """
        cursor = db.aql.execute(aql, bind_vars={'toks': entity_tokens})
        entity_keys = [r for r in cursor]

    # 5) direct graph neighbors (1..N_GRAPH hops, small)
    graph_hits = graph_neighbors(db, seed_entity_keys=entity_keys, hops=1, limit=N_GRAPH)
    graph_keys = [(r['chunk_id'], 1.0 / (1.0 + r.get('hop', 1))) for r in graph_hits]

    # 6) GeAR expansion (multi-hop beam over precomputed triples)
    gear_candidates = gear_expand(db, seed_entity_keys=entity_keys, max_hops=MAX_GEARS_HOPS, beam_width=GEAR_BEAM_WIDTH)
    gear_keys = [(c['chunk_id'], float(c['gear_score']), c.get('path')) for c in gear_candidates]

    # 7) Build unified candidate set
    candidates = {}  # keyed by chunk_key
    # add vectors
    for key, raw_vec_score in faiss_to_chunk_keys:
        candidates.setdefault(key, {})['vec'] = float(raw_vec_score)
    # add bm25
    for key, bm25 in bm25_keys:
        candidates.setdefault(key, {})['bm25'] = float(bm25)
    # add graph
    for key, gscore in graph_keys:
        candidates.setdefault(key, {})['graph'] = float(gscore)
    # add gear
    for key, gscore, path in gear_keys:
        candidates.setdefault(key, {})['gear'] = float(gscore)
        candidates[key]['gear_path'] = path

    # 8) fetch chunk texts and metadata for all candidate keys
    all_keys = list(candidates.keys())
    chunk_docs = fetch_chunks(db, all_keys)
    for k, doc in chunk_docs.items():
        candidates.setdefault(k, {})['text'] = doc.get('text', '')
        candidates[k]['meta'] = doc.get('meta', {})

    # 9) normalization & fusion
    weights = {'vec': W_VEC, 'bm25': W_BM25, 'graph': W_GRAPH, 'gear': W_GEAR}
    fused = normalize_and_fuse(candidates, weights)

    # 10) optional rerank on top_K
    top = fused[:top_k]
    if use_reranker:
        top = rerank_cross_encoder(query, top)

    end = time.time()
    logger.info("retrieve: query=%s, candidates=%d, returned=%d, time=%.3fs",
                query, len(fused), len(top), end - start)
    # attach provenance info, return
    for r in top:
        # ensure chunk text present
        r['text'] = r.get('text', '')
        r['provenance'] = {
            'vec': r.get('vec'),
            'bm25': r.get('bm25'),
            'graph': r.get('graph'),
            'gear': r.get('gear'),
            'gear_path': r.get('gear_path', None)
        }
    return top

# -----------------------
# Example usage (if module run directly)
# -----------------------
if __name__ == "__main__":
    # init clients
    load_faiss_index(FAISS_INDEX_PATH)  # ensure index exists
    db = init_arango_client()
    q = "What is the efficiency of carbon capture methods in industrial plants?"
    results = retrieve(q, db=db, top_k=TOP_K, use_reranker=False)
    for i, r in enumerate(results, 1):
        print(f"RANK {i} | score={r['final_score']:.4f} | key={r.get('_key','?')}")
        print(" PROV:", r['provenance'])
        print(" TEXT SNIPPET:", r['text'][:300].replace('\n', ' '))
        print("-" * 80)
