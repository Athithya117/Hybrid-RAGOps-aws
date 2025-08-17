# docs/inference_pipeline/llmless_retreival.md

When an LLM is unavailable, produce **precise, auditable, and human-usable answers** by combining semantic vectors, keyword search, and graph triplets — **without** relying on generation. Provide results as **structured JSON + concise human-readable text** (deterministic templates), with full provenance (file name + page) and graph paths.

---

# Principles (why & how)

1. **Deterministic**: produce outputs derived only from retrieved chunks/triplets (no generative invention).
2. **Traceable**: every fact references provenance `(filename, p.<page>)` and maps to `s3_path` in metadata.
3. **Compact**: token / bandwidth efficient; trim citations to `(filename, p.N)`.
4. **Human-friendly**: present both JSON and concise text templates for UI.
5. **Graceful fallback**: degrade quality predictably (e.g., fewer results, less scoring) when compute limits hit.

---

# High-level LLM-less workflow (summary)

```
User query (text)
  ├─→ Embed query → FAISS search [faiss_index_v1]
  ├─→ ArangoSearch (AQL) → keyword candidates [chunks_search_view]
  └─→ Graph/triplet lookup (ArangoDB) → triplets/paths [triplets collection / entity_graph]
        ↓
  Merge & deduplicate → normalize scores → cluster near-duplicates
        ↓
  Produce structured output:
      - JSON: list of chunk objects + graph paths/triplets
      - Text summary: deterministic templates (no generation)
        ↓
  Return to client (API), UI shows options to drill to full PDF via presigned URLs [s3://my-bucket/...]
```

---

# Detailed step-by-step (concrete)

## 0. Config / constants (example)

* `[FAISS index: faiss_index_v1]` (D = 768)
* ArangoDB collections: `[chunks]`, `[triplets]`, search view `[chunks_search_view]`
* Graph name: `[entity_graph]`
* Filestore base: `s3://my-bucket/docs/`
* Final outputs: top-M = 80 (prefilter), top-K = 12 (returned to client)
* Filename stored in chunk metadata as `filename` (e.g., `tesla_investments.pdf`) and `page` (int). Full s3 path stored in `s3_path`.

---

## 1. Query intake & normalization

* Lowercase / trim / strip punctuation as needed for ArangoSearch tokens.
* Parse optional user filters: `company=[Tesla|Rivian]`, `date`, `domain`. (Use regex / simple parser — **not** an LLM.)

**Example parsed object:**

```json
{
  "text": "Which founders of Tesla or Rivian have invested in solar energy startups, and what patents related to EV batteries do they hold?",
  "filters": {"companies": ["Tesla","Rivian"], "types": ["investment","patent"]}
}
```

---

## 2. Vector search (FAISS)

**Embed query** using local embedder: `[onnx-clip-768]` or `[sentence-transformers/multi-768]`.

```python
query_vector = embed_model.encode(query_text)  # shape (768,)
hits = faiss_index.search(query_vector, topk=100)
# returns [(chunk_id, distance/score), ...]
```

**Returned fields per hit**: `chunk_id`, `vec_score` (cosine), `chunk_text`, `filename`, `page`, `s3_path`.

---

## 3. ArangoSearch (keyword / BM25)

**AQL example**:

```aql
LET q = @query
FOR c IN chunks_search_view
  SEARCH ANALYZER(c.text IN TOKENS(q, 'text_en'), 'text_en')
  FILTER (c.filename IN @filter_filenames)   // optional
  SORT BM25(c) DESC
  LIMIT 60
  RETURN { chunk_id: c._key, bm25_score: BM25(c), text: c.text, filename: c.filename, page: c.page, s3_path: c.s3_path }
```

Collect top-N bm25 hits.

---

## 4. Graph / Triplet lookup (no runtime triplets from query)

Use **pre-extracted triplets** (`[triplets]`) and graph traversal (`[entity_graph]`). Two kinds of graph queries:

### 4.1 Direct triplet matching

```aql
FOR t IN triplets
  FILTER t.subject IN @entities OR t.object IN @entities
  RETURN { triplet_id: t._key, subject: t.subject, predicate: t.predicate, object: t.object, source_chunk_id: t.source_chunk_id, filename: t.filename, page: t.page }
```

### 4.2 Multi-hop traversal (precomputed paths)

```aql
LET seeds = @seed_entities  // e.g. ["Tesla","Rivian"]
FOR v, e, p IN 1..3 OUTBOUND seeds GRAPH "entity_graph"
  FILTER e.label IN ["has_founder","invested_in","holds_patent"]
  RETURN DISTINCT { path: p.vertices[*]._key, edges: p.edges[*].label, end_vertex: v, provenance: p }
```

**Return graph paths** with `path_nodes`, `edge_labels`, and `source_chunk_ids` (map to chunks).

---

## 5. Merge & deduplicate

* Combine unique chunk\_ids from FAISS, ArangoSearch, graph-derived chunks.
* Deduplicate exact duplicates by `chunk_id`.
* **Semantic dedupe**: cluster chunk embeddings (cosine similarity > 0.92) and keep canonical chunk with highest aggregated score.

**Canonical merge structure:**

```json
{
  "chunk_id": "chunk_101",
  "text": "...",
  "filenames": ["tesla_investments.pdf"],
  "pages": [12],
  "s3_paths": ["s3://my-bucket/docs/tesla_investments.pdf"],
  "vec_score": 0.923,
  "bm25_score": 8.1,
  "graph_score": 0.88,
  "sources_count": 3
}
```

---

## 6. Scoring & ranking (LLM-less)

Normalize each score into `[0..1]`. Example normalization:

```python
vec_norm = (vec_score + 1) / 2  # if cosine in [-1,1]
bm25_norm = bm25_score / (bm25_score + 10)  # example transform
graph_norm = graph_score  # already 0..1
gear_norm = gear_score
```

Example **fallback weighting** (favor deterministic evidence when no LLM):

```python
final_score = 0.40*vec_norm + 0.30*graph_norm + 0.20*bm25_norm + 0.10*(sources_count_normalized)
```

Sort by `final_score`. Keep top-K (12) for result list.

---

## 7. Construct LLM-less outputs (two forms)

### A. Structured JSON result (machine readable) — **primary**

Schema: `SearchResult` (return as JSON)

```json
{
  "query": "<user_query>",
  "results": [
    {
      "id": "chunk_101",
      "type": "chunk",
      "text": "Elon Musk, co-founder of Tesla, invested in SolarCity.",
      "filename": "tesla_investments.pdf",
      "page": 12,
      "s3_path": "s3://my-bucket/docs/tesla_investments.pdf",
      "scores": { "vec": 0.923, "bm25": 0.81, "graph": 0.88, "final": 0.91 },
      "retrieved_by": ["vector","graph"],
      "supporting_triplets": [
         {"subject":"Elon Musk","predicate":"invested_in","object":"SolarCity","source_chunk_id":"chunk_101"}
      ]
    },
    {
      "id": "triplet_path_1",
      "type": "triplet_path",
      "path": ["Tesla","Elon Musk","SolarCity"],
      "edges": ["has_founder","invested_in"],
      "supporting_chunks": [
          {"chunk_id":"chunk_101","filename":"tesla_investments.pdf","page":12}
      ],
      "score": 0.88
    }
  ],
  "metadata": {"topM": 80, "topK": 12, "timestamp": "2025-08-16T..."}
}
```

### B. Human-readable deterministic text summary — **secondary**

Constructed by templates and rules (no LLM):

* For each unique `person` found as founder of `Tesla` or `Rivian`:

  * If triplet path exists: produce statement:
    `"<Founder>, founder of <Company>, invested in <Startup> (filename.pdf, p.N) and holds patents <patent-list> (filename.pdf, p.N)."`
  * If investment missing: `"<Founder> — No investment information found (lookup files: filename.pdf, p.N)."`

**Example:**

```
Elon Musk, co-founder of Tesla, invested in SolarCity (tesla_investments.pdf, p.12) and holds patents for Tesla EV battery technology (tesla_patents.pdf, p.5).
RJ Scaringe, founder of Rivian, holds patents related to Rivian EV batteries (rivian_patents.pdf, p.5). No information found on RJ Scaringe investing in solar energy startups (rivian_investments.pdf, p.3).
```

**Rules for deterministic assembly**:

1. Prefer triplet-backed facts over single-chunk mentions.
2. If multiple chunks corroborate the same fact, show one citation + `(also: filename2.pdf, p.N)` optionally.
3. If patents are multiple, list patent IDs exactly as in triplet or chunk; otherwise show short phrase and citation.

---

## 8. Output examples (complete, concrete)

### 8.1 JSON (shortened)

```json
{
  "query": "Which founders of Tesla or Rivian have invested in solar energy startups, and what patents related to EV batteries do they hold?",
  "results": [
    {
      "id": "chunk_101",
      "type": "chunk",
      "text": "Elon Musk, co-founder of Tesla, invested in SolarCity.",
      "filename": "tesla_investments.pdf",
      "page": 12,
      "s3_path": "s3://my-bucket/docs/tesla_investments.pdf",
      "scores": {"vec":0.923,"bm25":0.81,"graph":0.88,"final":0.91},
      "retrieved_by": ["vector","graph"]
    },
    {
      "id": "chunk_202",
      "type": "chunk",
      "text": "Elon Musk holds patents for Tesla EV battery technology, including battery management systems.",
      "filename": "tesla_patents.pdf",
      "page": 5,
      "s3_path": "s3://my-bucket/docs/tesla_patents.pdf",
      "scores":{"vec":0.885,"bm25":0.75,"final":0.87}
    },
    {
      "id": "triplet_path_1",
      "type": "triplet_path",
      "path":["Tesla","Elon Musk","SolarCity"],
      "edges":["has_founder","invested_in"],
      "supporting_chunks":[{"chunk_id":"chunk_101","filename":"tesla_investments.pdf","page":12}],
      "score":0.88
    }
  ],
  "meta":{"returned":3}
}
```

### 8.2 Human-readable text (deterministic)

```
Elon Musk, co-founder of Tesla, invested in SolarCity (tesla_investments.pdf, p.12) and holds patents for Tesla EV battery technology (tesla_patents.pdf, p.5).
RJ Scaringe, founder of Rivian, holds patents related to Rivian EV batteries (rivian_patents.pdf, p.5). No information found about RJ Scaringe investing in solar energy startups (rivian_investments.pdf, p.3).
```

---

## 9. UI / UX behavior (recommended)

* **Search response panel**: show deterministic text summary at top (one-liners).
* **Evidence table**: list JSON results with columns: `type | snippet | filename | p | methods | final_score`.
* **Expand**: clicking a result shows full chunk text and `Open in PDF` button (server generates presigned URL from `s3_path` — \[trace\_file.py]).
* **Graph viewer**: show triplet paths visually; allow clicking nodes to reveal chunk(s).
* **“Show more evidence”**: paginated results (page size 12).
* **“LLM unavailable” banner**: explain that responses are deterministic and not generated.

---

## 10. Caching & performance

* Cache **recent query results** (normalized query string) for `ttl = 5–60m` depending on freshness; store mapping of query -> result JSON.
* Cache presigned URLs separately for UI click-through.
* Use in-memory FAISS shards for fast search; autoscale FAISS workers via Ray/Redis \[rayserve].
* Circuit-breaker: if reranker is overloaded, skip reranker and return prefiltered top-K.

---

## 11. Optional local reranker (non-generative)

If you can run a local cross-encoder (non-generative) like `[gte-modernbert-reranker-base]` on-prem, you can still **improve ranking** without calling an LLM. This is allowed in “LLM-less” mode (since reranker doesn’t generate text). Use small batches and GPU if available. If not possible, rely on the weighted score approach above.

---

## 12. Auditing & provenance

* Store for every query: `used_chunk_ids`, `final_output`, `timestamp`, `trace_id`, `full s3_paths`.
* UI provides `Open` that maps `(filename,page)` → full `s3_path` → presigned link and records click event for auditing.

---

## 13. Security & access control

* Do not expose raw S3 presigned URLs unless user authorized. Use backend that validates user token before generating URL.
* Redact sensitive fields from chunk text if policy says so (PII redaction) — perform redaction in preprocessing indexing pipeline where possible.

---

## 14. Metrics & monitoring

* Track: `retrieval_latency_ms`, `faiss_hits`, `arangosearch_hits`, `graph_hits`, `merge_time_ms`, `reranker_time_ms` (if used), `cache_hit_rate`.
* Evaluate: precision\@K (manual labeling), recall\@K (compare to gold), user click-through and fallback to LLM rates.

---

## 15. Edge cases & fallbacks

* **No results found**: return `"No information found."` plus suggestion of alternate queries (e.g., broaden terms).
* **Conflicting evidence**: return both statements with provenance, e.g., “x says A (file A), y says not A (file B).” No LLM synthesis — present facts only.
* **Duplicate filenames across docs**: include `filename + chunk_id` in UI to disambiguate.

---

## 16. Example implementation snippets

### 16.1 FAISS search (Python pseudo)

```python
def faiss_search(query_vector, topk=100):
    D, ids, dists = faiss_index.search(np.array([query_vector]), topk)
    results = []
    for cid,dist in zip(ids[0], dists[0]):
        chunk = get_chunk_by_id(cid)  # from ArangoDB or cache
        results.append({
            "chunk_id": cid,
            "text": chunk["text"],
            "filename": chunk["filename"],
            "page": chunk["page"],
            "s3_path": chunk["s3_path"],
            "vec_score": 1 - dist  # if dist normalized
        })
    return results
```

### 16.2 Score merge & normalize (Python)

```python
def normalize_bm25(x): return x/(x+10)
def normalize_vec(cosine): return (cosine+1)/2

for c in merged_chunks:
    c['vec_norm'] = normalize_vec(c.get('vec_score', 0))
    c['bm25_norm'] = normalize_bm25(c.get('bm25_score', 0))
    c['graph_norm'] = c.get('graph_score', 0)
    c['final'] = 0.4*c['vec_norm'] + 0.3*c['graph_norm'] + 0.2*c['bm25_norm'] + 0.1*(min(1,c.get('sources_count',1)/3))
```

### 16.3 Deterministic summary builder

```python
def build_summary(merged_results):
    # group by person entity if triplets exist, else by chunk
    persons = extract_person_entities(merged_results)  # simple heuristics
    lines = []
    for p in persons:
        invest = lookup_investment_triplet(p)
        patents = lookup_patent_triplets(p)
        if invest:
            lines.append(f"{p}, {invest['role']} of {invest['company']}, invested in {invest['target']} ({invest['filename']}, p.{invest['page']})")
        if patents:
            lines.append(f"{p} holds patents: {', '.join([pat['label'] for pat in patents])} ({patents[0]['filename']}, p.{patents[0]['page']})")
        if not invest and not patents:
            lines.append(f"{p} — No information found.")
    return "\n".join(lines)
```

---

## 17. Testing & QA

* Unit test: for a fixed dataset, assert deterministic output for a set of queries.
* Integration test: simulate FAISS + ArangoSearch + graph returns and verify JSON schema.
* E2E: simulate LLM outage and measure user satisfaction vs LLM-enabled baseline.

---

## 18. Deployment & scaling notes

* Put FAISS on GPU nodes (if embedding dims large) and shard by namespace.
* ArangoDB clusters should be sized for `chunks` collection and graph traversals; tune AQL timeouts.
* Reranker (if used) runs on dedicated CPU/GPU; set queue limits.
* Enable autoscaling via `[eks manifests / 12_rayservice.yaml]` for vector workers and `[13_karpenter-nodepools.yaml]` for autoscaling.

---

## 19. Checklist to enable LLM-less mode quickly

* [ ] Ensure each chunk stores `filename`, `page`, `s3_path`, `chunk_id`, `embedding`.
* [ ] Ensure `[triplets]` and `[entity_graph]` are populated and indexed.
* [ ] Implement FAISS search + ArangoSearch AQL endpoints.
* [ ] Implement merge/dedupe + scoring pipeline.
* [ ] Implement deterministic summary builder and JSON schema.
* [ ] Add UI hooks for presigned URL generation and graph viewer.
* [ ] Add monitoring (Prometheus) and alerts.

---

## 20. Concrete sample outputs (Tesla/Rivian query)

### JSON response (truncated)

```json
{
  "query": "Which founders of Tesla or Rivian have invested in solar energy startups, and what patents related to EV batteries do they hold?",
  "results": [
    {
      "id":"chunk_101",
      "type":"chunk",
      "text":"Elon Musk, co-founder of Tesla, invested in SolarCity.",
      "filename":"tesla_investments.pdf",
      "page":12,
      "s3_path":"s3://my-bucket/docs/tesla_investments.pdf",
      "scores":{"vec":0.923,"bm25":0.81,"graph":0.88,"final":0.91},
      "retrieved_by":["vector","graph"]
    },
    {
      "id":"chunk_202",
      "type":"chunk",
      "text":"Elon Musk holds patents for Tesla EV battery technology including battery management.",
      "filename":"tesla_patents.pdf",
      "page":5,
      "s3_path":"s3://my-bucket/docs/tesla_patents.pdf",
      "scores":{"vec":0.885,"bm25":0.75,"final":0.87},
      "retrieved_by":["vector"]
    },
    {
      "id":"triplet_path_1",
      "type":"triplet_path",
      "path":["Tesla","Elon Musk","SolarCity"],
      "edges":["has_founder","invested_in"],
      "supporting_chunks":[{"chunk_id":"chunk_101","filename":"tesla_investments.pdf","page":12}],
      "score":0.88
    }
  ],
  "summary": "Elon Musk, co-founder of Tesla, invested in SolarCity (tesla_investments.pdf, p.12) and holds patents for Tesla EV battery technology (tesla_patents.pdf, p.5). No information found on RJ Scaringe investing in solar energy startups (rivian_investments.pdf, p.3).",
  "meta":{"returned":3}
}
```

---

# Final notes (tradeoffs)

* **No LLM** → you get **deterministic, auditable** answers, but **no abstractive synthesis** or natural multi-sentence reasoning beyond deterministic templates.
* **Quality** depends on how rich your pre-extracted triplets and chunking are. Invest in high-quality chunking and triplet extraction at index time for best LLM-less performance.
* **Usability** is improved with a good UI (graph viewer, click-to-open PDF) and a friendly summary template.

---
