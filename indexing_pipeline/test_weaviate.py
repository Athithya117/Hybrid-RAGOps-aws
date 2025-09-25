#!/usr/bin/env python3
import sys, os, json, requests, argparse, re, time

def robust_extract_vector(resp):
    """Extract vector from various possible embedder responses."""
    if isinstance(resp, list):
        if resp and isinstance(resp[0], list):
            return resp[0]
        if all(isinstance(x, (int, float)) for x in resp):
            return resp
    if isinstance(resp, dict):
        for k in ("embeddings", "data", "vectors", "vectors_list", "embedding", "vectors_array"):
            if k in resp:
                cand = resp[k]
                if isinstance(cand, list) and cand:
                    if isinstance(cand[0], list):
                        return cand[0]
                    if all(isinstance(x, (int, float)) for x in cand):
                        return cand
        for v in resp.values():
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return v
    return None

def post_graphql(weaviate_url, payload, timeout=30):
    url = weaviate_url.rstrip("/") + "/v1/graphql"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        return r.json()
    except Exception:
        return {"errors": [{"message": "non-json response", "raw": r.text}]}

def embed_text(embed_url, text, timeout=30, tries=3, backoff=0.3):
    payload = {"texts": [text]}
    last = None
    for i in range(tries):
        try:
            r = requests.post(embed_url, json=payload, timeout=timeout)
            j = r.json()
            v = robust_extract_vector(j)
            if v is None:
                last = j
                time.sleep(backoff * (2 ** i))
                continue
            return v
        except Exception as e:
            last = str(e)
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"Failed to extract vector from embedder response: {last}")

def search_near_vector(weaviate_url, vec, limit=10, certainty=0.25, timeout=30):
    q = {
        "query": f'{{Get{{Chunk(nearVector:{{vector:{json.dumps(vec)}, certainty:{certainty}}}, limit:{int(limit)}){{chunk_id,file_name,text,_additional{{distance}}}}}}}}'
    }
    return post_graphql(weaviate_url, q, timeout=timeout)

def search_lexical_contains(weaviate_url, text, limit=100, timeout=30):
    safe = json.dumps(text)
    q = {
        "query": f'{{Get{{Chunk(where:{{path:["text"], operator:Contains, valueString:{safe}}}, limit:{int(limit)}){{chunk_id,file_name,text}}}}}}'
    }
    return post_graphql(weaviate_url, q, timeout=timeout)

def vec_score_from_distance(d):
    try:
        d = float(d)
        return max(0.0, 1.0 - d)
    except Exception:
        return 0.0

def simple_lex_score(query, text):
    if not text:
        return 0.0
    qwords = set(re.findall(r"\w+", query.lower()))
    tw = set(re.findall(r"\w+", text.lower()))
    if not qwords:
        return 0.0
    return len(qwords & tw) / len(qwords)

def merge_results(vec_hits, lex_hits, vec_weight=0.7, lex_weight=0.3, topk=20):
    mapd = {}
    if vec_hits:
        for it in vec_hits:
            cid = it.get("chunk_id") or it.get("properties", {}).get("chunk_id")
            dist = it.get("_additional", {}).get("distance")
            mapd[cid] = {
                "vec_score": vec_score_from_distance(dist),
                "lex_score": 0.0,
                "file": it.get("file_name"),
                "text": it.get("text")
            }
    if lex_hits:
        for it in lex_hits:
            cid = it.get("chunk_id") or it.get("properties", {}).get("chunk_id")
            txt = it.get("text")
            if cid in mapd:
                mapd[cid]["lex_score"] = simple_lex_score(args.query, txt)
            else:
                mapd[cid] = {
                    "vec_score": 0.0,
                    "lex_score": simple_lex_score(args.query, txt),
                    "file": it.get("file_name"),
                    "text": txt
                }
    combined = []
    for cid, vals in mapd.items():
        score = vec_weight * vals.get("vec_score", 0.0) + lex_weight * vals.get("lex_score", 0.0)
        if score > 0:
            combined.append((score, cid, vals))
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:topk]

def rerank_chunks(reranker_url, query, contexts):
    """Call reranker API and log usage."""
    if not reranker_url:
        print("INFO: No reranker URL provided, skipping rerank.")
        return contexts
    passages = [c["text"] for c in contexts]
    try:
        print(f"INFO: Calling reranker at {reranker_url} with {len(passages)} passages...")
        r = requests.post(reranker_url.rstrip("/") + "/rerank",
                          json={"query": query, "passages": passages},
                          timeout=30)
        j = r.json()
        scores = j.get("scores", [])
        if len(scores) != len(contexts):
            print("WARNING: reranker returned unexpected number of scores, skipping rerank.")
            return contexts
        for c, s in zip(contexts, scores):
            c["score"] = 0.5 * c.get("score", 0.0) + 0.5 * s
        contexts.sort(key=lambda x: x["score"], reverse=True)
        print("INFO: Reranker applied successfully. Top score:", contexts[0]["score"])
        return contexts
    except Exception as e:
        print("WARNING: reranker call failed:", e)
        return contexts

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", "-q", default="what is AI")
    p.add_argument("--embed-url", default=os.environ.get("EMBEDDING_URL", "http://localhost:8001/embed"))
    p.add_argument("--weaviate-url", default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"))
    p.add_argument("--reranker-url", default=os.environ.get("RERANKER_URL", "http://localhost:8002"))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--certainty", type=float, default=0.25)
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--vec-weight", type=float, default=0.7)
    p.add_argument("--lex-weight", type=float, default=0.3)
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()

    # get query embedding
    try:
        vec = embed_text(args.embed_url, args.query)
    except Exception as e:
        print(json.dumps({"error": "embed_failed", "detail": str(e)}))
        sys.exit(2)

    # vector search
    near_resp = search_near_vector(args.weaviate_url, vec, limit=args.limit, certainty=args.certainty)
    hits_vec = near_resp.get("data", {}).get("Get", {}).get("Chunk") or []

    if not args.hybrid:
        out = [{"chunk_id": it.get("chunk_id"),
                "file_name": it.get("file_name"),
                "distance": it.get("_additional", {}).get("distance"),
                "snippet": (it.get("text") or "")[:800]} for it in hits_vec]
        print(json.dumps({"query": args.query, "vector_len": len(vec), "results": out}, indent=2))
        sys.exit(0)

    # lexical search
    lex_resp = search_lexical_contains(args.weaviate_url, args.query, limit=100)
    hits_lex = lex_resp.get("data", {}).get("Get", {}).get("Chunk") or []

    # merge vec + lexical
    merged = merge_results(hits_vec, hits_lex, vec_weight=args.vec_weight, lex_weight=args.lex_weight, topk=args.topk)

    # prepare for rerank
    contexts = [{"chunk_id": cid, "text": vals["text"], "score": score, "file": vals.get("file", "")}
                for score, cid, vals in merged]

    # rerank using external API
    contexts = rerank_chunks(args.reranker_url, args.query, contexts)

    # final output
    out = [{"score": c["score"],
            "chunk_id": c["chunk_id"],
            "file_name": c.get("file", ""),
            "snippet": (c["text"] or "")[:800]} for c in contexts]

    print(json.dumps({"query": args.query, "vector_len": len(vec), "merged_results": out}, indent=2))
