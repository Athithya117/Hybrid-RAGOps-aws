# Retrieval Service — Runtime Documentation (Updated)

## Overview

The retrieval service is a stateless FastAPI application implementing the online retrieval and answer-generation layer of the RAG pipeline. It accepts user queries, performs sparse and optional dense hybrid retrieval against Qdrant, applies Reciprocal Rank Fusion (RRF), conditionally reranks results, builds numbered prompts, invokes an LLM when available, and returns an answer with optional traceable source chunks.

This document reflects the current behavior of `apps/inference/retriever/query.py` and supersedes older documentation.

---

## High-Level Architecture

The service coordinates five external dependencies:

1. Dense embedding service (HTTP)
2. Sparse embedding service (HTTP)
3. Optional reranker service (HTTP)
4. Qdrant vector database
5. LLM provider (OpenAI- / Groq-compatible HTTP API)

All external calls are bounded by timeouts and instrumented with Prometheus metrics. Qdrant is accessed via the synchronous Python client but executed in `asyncio.to_thread` to avoid blocking the event loop.

---

## Service Lifecycle

### Startup

On startup, the service:

- Instantiates async HTTP clients:
  - `AsyncDenseClient`
  - `AsyncSparseClient`
  - `AsyncRerankerClient`
- Creates a `QdrantClient` in a background thread.
- Performs repeated `/health` checks against dense, sparse, and reranker services.
- Marks readiness (`service_ready` gauge) as soon as the Qdrant client is successfully created (model services are advisory, not gating).
- Launches a background health checker that continuously refreshes readiness state.

### Shutdown

On shutdown (`SIGINT` / `SIGTERM`):

- Background health task is cancelled.
- All HTTPX clients are closed.
- Readiness gauge is set to 0.
- Shutdown is logged deterministically.

---

## API Endpoints

### `/generate` (POST)

Primary query endpoint.

Request:
```json
{
  "query": "string",
  "enable_tracing": false,
  "top_k": 5,
  "return_chunks": true,
  "max_tokens": 512
}
````

Response:

```json
{
  "answer": "string",
  "chunks": [
    {
      "index": 1,
      "meta_items": [{"k": "source_url", "v": "..."}],
      "source_url": "..."
    }
  ]
}
```

Behavior summary:

* Validates input and availability of Qdrant.
* Executes hybrid or sparse-only retrieval.
* Optionally reranks results.
* Builds numbered prompt blocks.
* Calls LLM if configured, otherwise falls back to deterministic summarization.
* Filters invalid citations.
* Returns answer and (optionally) traceable chunks.

---

### `/presign` (POST)

Returns a presigned Azure Blob URL.

Request:

```json
{ "path": "az://container/blob", "expires": 3600, "inline": true }
```

Response:

```json
{ "url": "https://<account>.blob.core.windows.net/container/blob?<sas>" }
```

Presign logic is executed in a thread using `helpers.presign_azure_blob_blocking`.

---

### Health & Observability

* `/healthz` — liveness probe (`{"status":"ok"}`)
* `/readyz` — readiness probe (Qdrant + dependency health snapshot)
* `/metrics` — Prometheus exposition

---

## Retrieval Pipeline

### Step 1: Embedding

Depending on availability:

* Hybrid mode (default when dense + sparse clients exist):

  * Dense embedding via `dense_client.embed([query])`
  * Sparse embedding via `sparse_client.embed_chunked([query])`
* Sparse-only fallback:

  * Sparse embedding only

Dense vectors are L2-normalized and validated against `DENSE_DIM`.

Embedding failures are logged and degrade gracefully (signals set to `None`).

---

### Step 2: Qdrant Query

Hybrid (Dense + Sparse):

* Dense vector is used in a `Prefetch` query.
* Query executed with `FusionQuery(fusion=Fusion.RRF)`.
* Result size is capped by `RRF_TOP_N`.

Sparse-only:

* Sparse vector queried directly using `using="sparse"`.
* Result size capped by `QUERY_TOPK_SPARSE`.

All Qdrant calls are executed inside `asyncio.to_thread`.

Key distinction:

* `RRF_TOP_N` limits how many candidates survive hybrid fusion.

---

### Step 3: Normalization & Deduplication

* Qdrant responses are normalized via `query_response_to_items`.
* Each item is reduced to `{id, score, payload}`.
* Deduplication is performed by:

  * `payload.chunk_id` if present
  * otherwise by point `id`
* Results are truncated to request-level `top_k`.

---

## Reranking

### Reranker Modes

Controlled by `RERANKER_MODE`:

* `DISABLE` — never rerank
* `ALWAYS` — always rerank
* `AUTO` — rerank based on heuristics:

  * top fused score < `RERANK_AUTO_THRESHOLD`
  * or top–second score gap < `RERANK_MARGIN`

### Reranking Scope

* Only the top `RERANK_TOPK` retrieved results are reranked.
* Reranker returns per-document scores.
* Scores are combined with original fused scores using:

  * `combined = RERANK_ALPHA * rerank_score + (1 - RERANK_ALPHA) * fused_score`
* Only the reranked prefix is reordered; the tail remains unchanged.

Important distinction:

* `RRF_TOP_N` controls how many candidates exist at all.
* `RERANK_TOPK` controls how many of those candidates are reranked.

---

## LLM Invocation

### Prompt Construction

* Retrieved chunks are converted into numbered blocks:

  ```
  [1]
  Heading: ...
  Content: ...
  ```
* These blocks are embedded into the user prompt template.

### Execution

* If an API key is present:

  * `_call_llm_via_http()` is used (OpenAI/Groq-compatible).
* If missing or on failure:

  * `deterministic_summarize()` is used as fallback.

### Tracing Mode

* When `enable_tracing=true`, chunks and citations are preserved and returned.
* If tracing is requested but no API key exists, the service returns a diagnostic answer with source chunks only.

---

## Citation Filtering

Before returning the answer:

* All non-numeric citations (URLs, metadata fields) are stripped.
* Numeric citations `[n]` are retained only if `n` corresponds to a returned chunk index.
* Extraneous whitespace and URLs are removed.

---

## Configuration (Environment Variables)

Key variables used by the service:

* Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`
* Embeddings: `DENSE_URL`, `DENSE_DIM`, `SPARSE_URL`, `SPARSE_BATCH_FALLBACK`
* Retrieval: `QUERY_TOPK_DENSE`, `QUERY_TOPK_SPARSE`, `RRF_TOP_N`
* Reranking: `RERANKER_URL`, `RERANKER_MODE`, `RERANK_TOPK`, `RERANK_AUTO_THRESHOLD`, `RERANK_MARGIN`, `RERANK_ALPHA`
* LLM: `API_KEY`, `LLM_MODEL`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`
* Limits & runtime: `MAX_CHUNKS_TO_LLM`, `MAX_PROMPT_TOKENS`, `HTTP_TIMEOUT`, `SERVICE_NAME`, `ENV`, `LOG_LEVEL`

---

## Metrics

Exposed via `/metrics`:

* Request, error, and latency metrics
* Dense, sparse, reranker, Qdrant, and LLM call metrics
* Retrieved document count histogram
* Service readiness gauge

---

## Error Handling & Fallbacks

* Embedding failure: degrade to remaining signals
* Qdrant failure: empty results and safe error answer
* Reranker failure: skip rerank
* LLM failure or missing key: deterministic summarization
* Validation errors: structured 422 responses

---

## Implementation Notes

* Blocking operations (Qdrant, Azure presign) run in threads.
* HTTP calls obey `HTTP_TIMEOUT`.
* Logging is structured JSON with source-side log-level gating.
* The service is stateless and horizontally scalable.
