#!/usr/bin/env python3
import os, time, urllib.parse
from typing import List, Dict, Tuple, Any, Optional
import requests
import weaviate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class SearchRequest(BaseModel):
    q: str
    candidate_k: int = 64
    rrf_k: int = 60
    top_k_sparse: int = 256
    threshold: float = 1e-6

class SearchResultItem(BaseModel):
    document_id: Optional[str]
    chunk_id: Optional[str]
    text: Optional[str]
    dense_sim: float
    sparse_score: float
    rrf_score: float

app = FastAPI(title="Weaviate Hybrid Search API")

def parse_host_port(url: str) -> Tuple[str,int]:
    p = urllib.parse.urlparse(url)
    host = p.hostname or "localhost"
    port = p.port or (443 if p.scheme == "https" else 80)
    return host, int(port)

def connect_weaviate(url: str, api_key: Optional[str] = None):
    host, port = parse_host_port(url)
    if api_key:
        auth = weaviate.AuthApiKey(api_key)
        client = weaviate.Client(url=f"http://{host}:{port}", auth_client_secret=auth, timeout_config=(30,120))
    else:
        client = weaviate.connect_to_local(host=host, port=port)
    return client

def call_dense(embedder_url: str, text: str, timeout: int = 60) -> List[float]:
    r = requests.post(f"{embedder_url.rstrip('/')}/embed", json={"text": text, "max_length": 512}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    embs = j.get("embeddings")
    if not embs or not embs[0]:
        raise RuntimeError("bad embed response")
    return [float(x) for x in embs[0]]

def call_sparse(embedder_url: str, text: str, top_k: int = 256, threshold: float = 1e-6, timeout: int = 60) -> Tuple[List[int], List[float]]:
    r = requests.post(f"{embedder_url.rstrip('/')}/sparse_embed", json={"text": text, "top_k": top_k, "threshold": threshold}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    sp = j.get("sparse")
    if not sp:
        return [], []
    first = sp[0]
    return [int(x) for x in first.get("indices", [])], [float(x) for x in first.get("values", [])]

def sparse_dot_score(qi: List[int], qv: List[float], di: List[int], dv: List[float]) -> float:
    if not qi or not di:
        return 0.0
    qmap = {int(i): float(v) for i, v in zip(qi, qv)}
    s = 0.0
    for i, v in zip(di, dv):
        ii = int(i)
        if ii in qmap:
            s += qmap[ii] * float(v)
    return float(s)

def rrf_fuse(items: List[Dict[str,Any]], rrf_k: int) -> List[Dict[str,Any]]:
    if not items:
        return []
    dense_order = sorted(items, key=lambda x: -x.get("dense_sim", 0.0))
    sparse_order = sorted(items, key=lambda x: -x.get("sparse_score", 0.0))
    dense_rank = {item["chunk_id"]: rank+1 for rank, item in enumerate(dense_order)}
    sparse_rank = {item["chunk_id"]: rank+1 for rank, item in enumerate(sparse_order)}
    fused = []
    for it in items:
        cid = it["chunk_id"]
        dr = dense_rank.get(cid, len(items)+1)
        sr = sparse_rank.get(cid, len(items)+1)
        rrf_score = 1.0/(rrf_k + dr) + 1.0/(rrf_k + sr)
        fused.append({**it, "rrf_score": rrf_score})
    fused_sorted = sorted(fused, key=lambda x: -x["rrf_score"])
    return fused_sorted

def hybrid_search(client, collection_name: str, query_text: str, embedder_url: str, candidate_k: int = 64, rrf_k: int = 60, top_k_sparse: int = 256, threshold: float = 1e-6) -> List[Dict[str,Any]]:
    dense_q = call_dense(embedder_url, query_text)
    q_idx, q_vals = call_sparse(embedder_url, query_text, top_k=top_k_sparse, threshold=threshold)
    props = ["document_id", "chunk_id", "text", "sparse_indices", "sparse_values"]
    q = client.query.get(collection_name, props).with_near_vector({"vector": dense_q}).with_limit(candidate_k)
    res = q.do()
    hits = res.get("data", {}).get("Get", {}).get(collection_name, [])
    items = []
    for h in hits:
        add = h.get("_additional", {})
        sim = 0.0
        if "certainty" in add:
            sim = float(add.get("certainty") or 0.0)
        elif "distance" in add:
            try:
                dist = float(add.get("distance"))
                sim = 1.0 / (1.0 + dist)
            except Exception:
                sim = 0.0
        doc_idx = h.get("sparse_indices") or []
        doc_vals = h.get("sparse_values") or []
        sscore = sparse_dot_score(q_idx, q_vals, doc_idx, doc_vals)
        items.append({
            "document_id": h.get("document_id"),
            "chunk_id": h.get("chunk_id"),
            "text": h.get("text"),
            "dense_sim": sim,
            "sparse_score": sscore
        })
    fused = rrf_fuse(items, rrf_k)
    return fused

# configure client and embedder from env
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_ROOT_KEY", "")
EMBEDDER_URL = os.environ.get("APP_EMBEDDER_URL", "http://127.0.0.1:8001")
COLLECTION = os.environ.get("WEAVIATE_CLASS", "Chunk")

client = connect_weaviate(WEAVIATE_URL, api_key=(WEAVIATE_API_KEY or None))

@app.post("/search", response_model=List[SearchResultItem])
def search(req: SearchRequest):
    if not req.q:
        raise HTTPException(status_code=400, detail="empty query")
    try:
        results = hybrid_search(client, COLLECTION, req.q, EMBEDDER_URL, candidate_k=req.candidate_k, rrf_k=req.rrf_k, top_k_sparse=req.top_k_sparse, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out: List[SearchResultItem] = []
    for r in results:
        out.append(SearchResultItem(
            document_id=r.get("document_id"),
            chunk_id=r.get("chunk_id"),
            text=(r.get("text") or "")[:2000],
            dense_sim=float(r.get("dense_sim") or 0.0),
            sparse_score=float(r.get("sparse_score") or 0.0),
            rrf_score=float(r.get("rrf_score") or 0.0),
        ))
    return out

if __name__ == "__main__":
    uvicorn.run("query:app", host="0.0.0.0", port=int(os.environ.get("QUERY_PORT", "8002")), log_level="info")
