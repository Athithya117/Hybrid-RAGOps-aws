# docs/inference_pipeline/llm_retreival.md

**User query:**
`"Which founders of Tesla or Rivian have invested in solar energy startups, and what patents related to EV batteries do they hold?"`

---

## 0. Assumptions / config (prod-ready)

* Vector model: \[embedder: `onnx-clip-768` or `sentence-transformers/multi-768`]
* FAISS index: `[faiss_index_v1]`, dimension `D = 768`
* ArangoDB collections: `[chunks]` (documents), `[triplets]` (pre-extracted SRO triples)
* Reranker: `[cross-encoder: gte-modernbert-reranker-base]` (call via gRPC/HTTP)
* LLM: `[vLLM/generative-llm]` endpoint (returns text)
* Bucket base: `s3://my-bucket/docs/` (constant) — we trim citations for tokens.
* Top params: `M = 80` (prefilter), `K = 12` (final LLM chunks)
* Observability: OpenTelemetry traces + Prometheus metrics + Loki logs.
* All filenames stored in chunk metadata as `filename` (e.g. `tesla_investments.pdf`) and `page`. Full s3 path retained in `s3_path` metadata.

---

## 1. Ingestion / Indexing (already done)

* Documents processed by indexing pipeline. For each PDF/text:

  * Chunking → produce `chunk.text`, `chunk_id`, `filename`, `page`, `s3_path`.
  * Embed each chunk → store `chunk.embedding` in FAISS and optionally in ArangoDB.
  * Extract triplets via ReLiK → store as `(subject,predicate,object)` in `[triplets]` and create graph edges (vertices & edges) in ArangoDB.
* Example stored chunk (DB record in `[chunks]`):

```json
{
  "chunk_id": "chunk_123",
  "text": "Elon Musk, co-founder of Tesla, invested in SolarCity.",
  "filename": "tesla_investments.pdf",
  "page": 12,
  "s3_path": "s3://my-bucket/docs/tesla_investments.pdf",
  "embedding": [0.021, -0.134, 0.876, -0.432, 0.102, 0.511, ...],  // 768 floats
  "source": "pdf",
  "indexed_at": "2025-06-01T12:00:00Z"
}
```

* Example triplets inserted in `[triplets]` (ArangoDB vertices/edges):

```
("Tesla", "has_founder", "Elon Musk")
("Elon Musk", "invested_in", "SolarCity")
("Elon Musk", "holds_patent", "Tesla battery patent XYZ123")
("Rivian", "has_founder", "RJ Scaringe")
("RJ Scaringe", "holds_patent", "Rivian battery patent ABC456")
```

---

## 2. Inference: Step-by-step

### Step 2.0 — Receive query

* `user_query = "Which founders of Tesla or Rivian have invested in solar energy startups, and what patents related to EV batteries do they hold?"`

**Operations triggered in parallel**:

* 1. Embed query → \[FAISS search]
* 2. ArangoSearch keyword query → \[ArangoSearch]
* 3. Graph precomputed multi-hop traversal via GeAR → \[gear\_search] (uses pre-extracted triplets only)

---

### Step 2.1 — Query embedding & FAISS search

**Embed query** (example 8-dim truncation shown; real `D=768`):

```python
query_vector = [0.12, -0.03, 0.45, -0.21, 0.33, -0.02, 0.11, 0.07, ...]  # 768 floats
```

**FAISS nearest-neighbor search** (pseudo-call):

```python
# [FAISS index: faiss_index_v1]
faiss_index.search(query_vector, topk=100) -> returns [(chunk_id, vec_score, chunk_vector), ...]
```

**Example FAISS hits (top 5):**

| rank | chunk\_id  | vec\_score (cosine) | snippet text (truncated)                              | filename                | page |
| ---- | ---------- | ------------------: | ----------------------------------------------------- | ----------------------- | ---- |
| 1    | chunk\_101 |               0.923 | "Elon Musk invested in SolarCity."                    | tesla\_investments.pdf  | 12   |
| 2    | chunk\_202 |               0.885 | "Tesla holds patents for battery management."         | tesla\_patents.pdf      | 5    |
| 3    | chunk\_303 |               0.854 | "RJ Scaringe filed patents on modular battery packs." | rivian\_patents.pdf     | 5    |
| 4    | chunk\_404 |               0.740 | "Rivian investors include green energy funds."        | rivian\_investments.pdf | 8    |
| 5    | chunk\_505 |               0.712 | "SolarCity early fundraising and acquisition."        | solarcity\_history.pdf  | 3    |

*(vec\_score scaled 0..1 here for clarity)*

---

### Step 2.2 — ArangoSearch (keyword / BM25) via AQL

**AQL example** (ArangoSearch index `analyzer: text_en`):

```aql
LET q = @query
FOR c IN chunks_search_view
  SEARCH ANALYZER(c.text IN TOKENS(q, 'text_en'), 'text_en')
  SORT BM25(c) DESC
  LIMIT 60
  RETURN { chunk_id: c._key, bm25_score: BM25(c), filename: c.filename, page: c.page, text: c.text }
```

**ArangoSearch hits (example):**

* chunk\_101 (same as FAISS hit), bm25\_score 8.1
* chunk\_606 (Nikola Tesla historical doc) bm25\_score 6.5 (noise)

---

### Step 2.3 — Graph search + GeAR (multi-hop) using pre-extracted triplets

**Goal:** find founders of Tesla or Rivian who have `invested_in` nodes or `holds_patent` nodes.

**AQL graph traversal example (multi-hop):**

```aql
LET seed = ["Tesla", "Rivian"]
FOR v, e, p IN 1..3 OUTBOUND seed GRAPH "entity_graph"
  FILTER v.type == "person" OR v.type == "patent" OR e.label IN ["invested_in","holds_patent","has_founder"]
  RETURN DISTINCT { vertex: v, path: p }
```

**GeAR module (`gear_search`)** may prune/score traversal paths by heuristics (edge weight, confidence). Example graph candidates:

* chunk\_101 (Elon Musk → invested\_in → SolarCity) \[graph\_score: 0.88]
* chunk\_303 (RJ Scaringe → holds\_patent → Rivian battery patent ABC456) \[graph\_score: 0.79]

---

## 3. Merge & Deduplicate chunks

* Merge unique chunk\_ids from FAISS, ArangoSearch, and GeAR results.
* Deduplicate by `chunk_id`.
* For each chunk produce aggregated scores:

```python
# aggregated structure (example)
merged_chunk = {
  "chunk_id": "chunk_101",
  "text": "...",
  "filename": "tesla_investments.pdf",
  "page": 12,
  "s3_path": "s3://my-bucket/docs/tesla_investments.pdf",
  "vec_score": 0.923,
  "bm25_score": 8.1,   # normalized later
  "graph_score": 0.88,
  "gear_score": 0.87  # if GeAR assigns separate relevance
}
```

**Score normalization**: normalize `bm25_score` into `0..1` scale before combining.

---

## 4. Prefilter / Weighted scoring (first-pass)

**Weighted score formula (example production-tuned):**

```python
final_score = 0.45 * vec_score_norm + 0.20 * bm25_score_norm + 0.20 * graph_score + 0.15 * gear_score
```

* Rationale: prioritize vector semantic recall, keep graph+GeAR substantial weight for multi-hop reasoning.

**Compute final\_score for all merged chunks, sort, keep top-M (e.g., M=80).**

**Example top-6 after weighting:**

| rank | chunk\_id  | final\_score |
| ---- | ---------- | -----------: |
| 1    | chunk\_101 |         0.91 |
| 2    | chunk\_202 |         0.87 |
| 3    | chunk\_303 |         0.83 |
| 4    | chunk\_404 |         0.72 |
| 5    | chunk\_505 |         0.69 |
| 6    | chunk\_606 |         0.55 |

---

## 5. Cross-encoder reranking (fine-grained)

**Goal:** apply `[gte-modernbert-reranker-base]` to the prefiltered top-M (M=80) to get the most precise ordering.

**Batching**: split top-M into batches of `B=8` (to control latency) and call cross-encoder.

**Reranker input per item:**

```
{ "query": user_query, "candidate_text": chunk.text, "metadata": { "filename": chunk.filename, "page": chunk.page } }
```

**Reranker returns:** `rerank_score` (0..1). After reranking, sort and take top-K (K=12).

**Example top-K after reranking:**

| rank | chunk\_id  | rerank\_score |
| ---- | ---------- | ------------: |
| 1    | chunk\_101 |          0.97 |
| 2    | chunk\_202 |          0.94 |
| 3    | chunk\_303 |          0.90 |
| 4    | chunk\_505 |          0.82 |
| ...  | ...        |           ... |

**Notes:** Cross-encoder is comparatively slow — keep K small and batch.

---

## 6. Final chunk selection → prepare LLM input (K chunks)

* For each selected chunk, prepare the JSON chunk object **but** we will pass trimmed citation tokens to the model.

**We store both forms**:

* `s3_path` (full) and `filename` (for LLM display).
* LLM receives `filename` + `page` instead of full `s3://...` to save tokens.

**Example final top-K chunk objects (to LLM):**

```json
[
  {
    "chunk_id": "chunk_101",
    "text": "Elon Musk, co-founder of Tesla, invested in SolarCity.",
    "filename": "tesla_investments.pdf",
    "page": 12,
    "s3_path": "s3://my-bucket/docs/tesla_investments.pdf",
    "score": 0.97,
    "retrieval_method": "vector+graph"
  },
  {
    "chunk_id": "chunk_202",
    "text": "Elon Musk holds patents for Tesla EV battery technology including battery management and thermal regulation systems.",
    "filename": "tesla_patents.pdf",
    "page": 5,
    "s3_path": "s3://my-bucket/docs/tesla_patents.pdf",
    "score": 0.94,
    "retrieval_method": "vector"
  },
  {
    "chunk_id": "chunk_303",
    "text": "RJ Scaringe, founder of Rivian, holds patents related to Rivian EV batteries including modular battery pack designs.",
    "filename": "rivian_patents.pdf",
    "page": 5,
    "s3_path": "s3://my-bucket/docs/rivian_patents.pdf",
    "score": 0.90,
    "retrieval_method": "graph"
  }
]
```

---

## 7. LLM prompt (universal + trimmed citations) — production-ready

**Universal Prompt Template** (use this as the LLM prompt; we inject `{{user_query}}` and `{{chunks}}`):

```
You are a highly knowledgeable assistant. Answer the user's query using only the provided chunks. Follow these rules:

1. Answer in clear, human-readable text, not JSON.
2. Use only the information in the provided chunks.
3. Include inline sources for every fact, formatted as: (filename, p.<page>).
   Example: (tesla_investments.pdf, p.12)
4. Consolidate repeated information into a single statement.
5. If a fact requires multi-hop explanation, show the reasoning path concisely.
6. Be concise, precise, and complete. Do not invent facts.
7. If information is missing, answer: "No information found."

User Query:
Which founders of Tesla or Rivian have invested in solar energy startups, and what patents related to EV batteries do they hold?

Reference Chunks (top-K):
- Text: Elon Musk, co-founder of Tesla, invested in SolarCity.
  Source: (tesla_investments.pdf, p.12)
- Text: Elon Musk holds patents for Tesla EV battery technology including battery management and thermal regulation systems.
  Source: (tesla_patents.pdf, p.5)
- Text: RJ Scaringe, founder of Rivian, holds patents related to Rivian EV batteries including modular battery pack designs.
  Source: (rivian_patents.pdf, p.5)
- Text: No information found on RJ Scaringe investing in solar energy startups.
  Source: (rivian_investments.pdf, p.3)

Provide the answer in natural text with inline citations as specified.
```

**(Implementation note)**: The LLM sees the trimmed `(filename, p.N)` references and not the full s3 path — you keep `s3_path` in metadata for audit/use in UI clicks.

---

## 8. Expected LLM textual output (precise, token-efficient)

```
Elon Musk, co-founder of Tesla, invested in SolarCity (tesla_investments.pdf, p.12) and holds patents for Tesla EV battery technology, including battery management and thermal regulation systems (tesla_patents.pdf, p.5).
RJ Scaringe, founder of Rivian, holds patents related to Rivian EV batteries, including modular battery pack designs (rivian_patents.pdf, p.5). No information is available regarding RJ Scaringe investing in solar energy startups (rivian_investments.pdf, p.3).
```

* This is exactly the human-readable, inline-cited, trimmed format you requested.
* Token cost is minimized by using `filename` instead of `s3://...` on every citation.

---

## 9. Post-processing & UI behavior

* **Mapping citations to full S3 links**: when user clicks a citation in UI, resolve `filename + page` → `s3_path` from stored metadata, then generate presigned URL (via `trace_file.py` / backend).
* **Audit logs**: store `answer_id`, used chunk\_ids, full s3\_paths, reranker scores, LLM response, and trace id.
* **Confidence / provenance**: show each statement with `(filename, p.N)` and optionally a confidence score (from reranker or aggregated score).

---

## 10. Observability & metrics (practical)

* Metric points: `retrieval_time_ms`, `faiss_hits`, `arangosearch_time_ms`, `gear_time_ms`, `reranker_time_ms`, `llm_time_ms`, `tokens_used`. Export to Prometheus via instrumentation in `retriever.py` and `main.py`.
* Tracing: OpenTelemetry spans across FAISS query → AQL → GeAR → reranker → LLM.
* Alerts: high latency on reranker or LLM calls triggers circuit breaks; fallback: reduce M or serve non-reranked top-K.

---

## 11. Edge cases & fallback policy

* **If cross-encoder service down**: fallback = use first-pass `final_score` ordering and return top-K (less precise).
* **If LLM returns hallucination**: run `eval.py` (RAGAI-Catalyst) to detect inconsistencies; flag low confidence and show `No information found` for unclear facts.
* **If duplicate filenames with different paths**: use `chunk_id` mapping to disambiguate.

---

## 12. Example logs (what you will see)

```
[INFO] trace_id=abc123 GET /qa query="Which founders..." faiss_time=35ms arangosearch_time=20ms gear_time=40ms merged=74 prefilter_topM=80 reranker_time=420ms llm_time=210ms tokens=842
[DEBUG] used_chunks: ["chunk_101","chunk_202","chunk_303"]
```

---

## 13. Implementation snippets (Python-ish)

**Merging & prefilter (pseudocode):**

```python
# merged_chunks: list of dicts from FAISS, ArangoSearch, GeAR
for c in merged_chunks:
    c['bm25_norm'] = normalize(c.get('bm25_score', 0))
    c['vec_norm'] = normalize(c.get('vec_score', 0))
    # compute final_score
    c['final_score'] = 0.45*c['vec_norm'] + 0.20*c['bm25_norm'] + 0.20*c.get('graph_score',0) + 0.15*c.get('gear_score',0)

topM = sorted(merged_chunks, key=lambda x: x['final_score'], reverse=True)[:M]
```

**Reranker call (pseudo):**

```python
batch = topM[i:i+B]
payload = [{"query": user_query, "doc": c['text']} for c in batch]
scores = call_reranker(payload)  # returns list of floats
for c,score in zip(batch,scores):
    c['rerank_score'] = score

topK = sorted(topM, key=lambda x: x['rerank_score'], reverse=True)[:K]
```

**Prompt assembly (trim citations):**

```python
chunks_for_prompt = []
for c in topK:
    chunks_for_prompt.append({
        "text": c['text'],
        "filename": c['filename'],
        "page": c['page']
    })
prompt = render_template(UNIVERSAL_PROMPT, user_query=user_query, chunks=chunks_for_prompt)
response_text = call_llm(prompt)
```

---

## 14. Short checklist to implement immediately

* [ ] Ensure `filename` and `page` stored per chunk + keep `s3_path` in metadata.
* [ ] Implement score normalization and tuned weights.
* [ ] Prefilter top-M (80) then rerank with cross-encoder in small batches.
* [ ] Use trimmed citation format `(filename, p.N)` in prompt.
* [ ] Keep full S3 mapping for UI presigned links.
* [ ] Add metrics/traces for each stage.

---

