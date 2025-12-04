#!/usr/bin/env python3
# server.py -- deterministic mock embedder (self-contained executable)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import struct
import math
import os

# NOTE: If you run via "python server.py" this will start uvicorn.
# If you prefer to run via "uvicorn server:app" that's also fine.

app = FastAPI(title="mock-embedder", version="0.1")

DEFAULT_DIM = int(os.environ.get("EMBED_DIM", "256"))  # default dimension, override with EMBED_DIM env

def text_to_vector(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    """
    Deterministic hash-based pseudo-embedding:
    - Use repeated SHA256 of (text + counter) to fill dim floats.
    - Convert bytes -> uint32 -> map to [-1,1] and normalize.
    """
    if text is None:
        text = ""
    out = []
    counter = 0
    need_bytes = dim * 4
    digest_bytes = b""
    while len(digest_bytes) < need_bytes:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(counter.to_bytes(4, "little"))
        digest_bytes += h.digest()
        counter += 1
    for i in range(dim):
        chunk = digest_bytes[i*4:(i+1)*4]
        u = struct.unpack("<I", chunk)[0]
        f = (u / 0xFFFFFFFF) * 2.0 - 1.0
        out.append(f)
    norm = math.sqrt(sum(x*x for x in out))
    if norm > 0:
        out = [x / norm for x in out]
    return out

class EmbedRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    inputs: Optional[List[str]] = None
    model: Optional[str] = None
    dimension: Optional[int] = None

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    count: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    dim = req.dimension or DEFAULT_DIM
    candidates: List[str] = []
    if req.text is not None:
        candidates = [req.text]
    elif req.texts:
        candidates = list(req.texts)
    elif req.inputs:
        candidates = list(req.inputs)
    else:
        raise HTTPException(status_code=400, detail="No 'text', 'texts' or 'inputs' provided")
    embeddings = [text_to_vector(t or "", dim) for t in candidates]
    return EmbedResponse(
        embeddings=embeddings,
        model=(req.model or "mock-embedder"),
        dimension=dim,
        count=len(embeddings),
    )

if __name__ == "__main__":
    # lazy import of uvicorn so running "python server.py" works iff uvicorn is installed.
    import uvicorn
    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")
    # single-worker run is fine for testing
    uvicorn.run("mock_embedder:app", host=host, port=port, log_level="info", workers=1)
