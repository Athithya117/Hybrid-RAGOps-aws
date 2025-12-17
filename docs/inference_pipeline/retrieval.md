# Retrieval Service — Runtime Documentation

## Overview

The retrieval service is a stateless FastAPI application that implements the online retrieval layer for the RAG pipeline. It accepts user queries, produces dense and sparse query embeddings, performs hybrid search against a Qdrant collection, applies fusion and optional reranking, assembles LLM prompts, and returns UI-ready results and optional presigned blob URLs.

---

## Components & Contracts

* **HTTP API (this service)**

  * `/generate` — main generate endpoint. Accepts `query`, returns `answer` and optional `chunks`.
  * `/presign` — returns an Azure Blob SAS URL for a stored object.
  * `/healthz`, `/readyz`, `/metrics` — health/readiness/Prometheus metrics.

* **Embedding services (external HTTP)**

  * Dense embedder: POST `{DENSE_URL}/embed` → `{"vectors":[[float,...],...]}`; health at `{DENSE_URL}/health`.
  * Sparse embedder: POST `{SPARSE_URL}/embed` → `{"vectors":[{"indices":[int...],"values":[float...]},...]}`; health at `{SPARSE_URL}/health`.
  * Reranker: POST `{RERANKER_URL}/rerank` → `{"scores":[float,...]}`; health at `{RERANKER_URL}/health`.

* **Vector DB**

  * Qdrant accessed via `qdrant_client` (sync client used inside `asyncio.to_thread`). Query functions return payload and scores. Collection name is `COLLECTION_NAME`.

* **LLM provider**

  * HTTP-based chat completion endpoint. `_call_llm_via_http()` posts to the configured LLM API (OpenAI/Groq-compatible).

---

## Runtime Control Flow (step-by-step)

### 1. Lifespan (startup)

1. Instantiate async thin clients:

   * `AsyncDenseClient(DENSE_URL)`, `AsyncSparseClient(SPARSE_URL)`, `AsyncRerankerClient(RERANKER_URL)`.
2. Create `QdrantClient` synchronously inside `asyncio.to_thread`.
3. Perform best-effort `/health` checks for dense, sparse, reranker and record readiness metrics.
4. Set `SERVICE_READY` gauge to reflect Qdrant availability.
5. On shutdown, close HTTPX clients and log lifecycle events.

### 2. Incoming request: `/generate`

1. Validate request (non-empty `query`, `top_k` bounds).
2. Verify Qdrant client exists; return a service-unavailable response if missing.
3. Call `hybrid_query(...)` to retrieve candidate chunks (top-K). Metrics and latency are recorded.
4. Trim results to `MAX_CHUNKS_TO_LLM`. If no documents, return `GenerateResponse(answer="no documents retrieved")`.
5. Build numbered LLM prompt blocks and UI metadata using `build_numbered_prompt_and_ui_chunks()`. This uses `ui_fields_from_payload()` to produce `snippet`, `source_url`, and other UI items.
6. If `enable_tracing` and an LLM API key is configured, call `_call_llm_via_http()` to fetch a traced answer; otherwise, call the configured LLM prompt or fallback `deterministic_summarize()` when LLM is not available or fails.
7. Validate and filter citations with `_validate_and_filter_citations()`.
8. Return `GenerateResponse(answer=<text>, chunks=<optional chunk list>)`.

### 3. Hybrid retrieval: `hybrid_query(...)`

1. **Embedding**

   * When hybrid mode is enabled, call:

     * `dense_client.embed([query_text])` → single dense vector (length = `DENSE_DIM`), normalized.
     * `sparse_client.embed_chunked([query_text])` → sparse `{indices, values}` object.
   * When hybrid disabled, only sparse embedding is used.
   * Embed calls are guarded by timeouts and instrumented; failures downgrade gracefully (embed result set to `None`).
2. **Qdrant search**

   * If dense is available and hybrid enabled:

     * Build `Prefetch(query=q_dense, using="dense", limit=prefetch_k)` and execute `client.query_points(..., query=FusionQuery(fusion=Fusion.RRF), limit=top_k, with_payload=True, with_vectors=False)` inside `asyncio.to_thread`.
   * Otherwise:

     * Build `SparseVector(indices=..., values=...)` and call `client.query_points(..., query=q_sparse_obj, using="sparse", limit=top_k, with_payload=True, with_vectors=False)` inside `asyncio.to_thread`.
3. **Response normalization & dedup**

   * Convert client response into a canonical list of items via `query_response_to_items()` and `extract_point_fields()`.
   * Deduplicate by `chunk_id` or `id` and return up to `top_k` candidate results.

### 4. Reranking decision and execution

1. Rerank is controlled by `RERANKER_MODE`:

   * `DISABLE`: skip reranking.
   * `ALWAYS`: rerank the top `RERANK_TOPK`.
   * `AUTO`: evaluate heuristics (fused top score vs `RERANK_AUTO_THRESHOLD`, rank disagreement > `RERANK_THRESHOLD`, top-second gap < `RERANK_MARGIN`) to decide rerank.
2. If rerank triggered:

   * Extract candidate documents into plain text list.
   * Call `reranker_client.rerank(query, documents)` → scores.
   * Merge reranker scores with original ordering and re-sort candidates.

### 5. LLM invocation & fallback

1. Build prompt body and LLM lines (numbered blocks) from selected chunks.
2. Call `_call_llm_via_http(system, user_prompt, model, max_tokens, temperature)` if API key present.
3. On LLM HTTP errors or missing API key, use deterministic summarizer `deterministic_summarize()` as fallback.
4. Post-process and filter citations.

### 6. Presign flow (`/presign`)

1. Accept `s3_path` (supports `az://container/blob`, `container/blob`, or full `https://...`).
2. Call `helpers.presign_azure_blob_blocking(path, expires, inline)` inside `asyncio.to_thread`.
3. Return a full SAS URL or error.

---

## Endpoints & JSON Contracts

### `/generate` (POST)

**Request**

```json
{
  "query": "string",
  "enable_tracing": false,
  "top_k": 5,
  "return_chunks": true,
  "max_tokens": 512
}
```

**Response**

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

### `/presign` (POST)

**Request**

```json
{ "s3_path": "az://container/blob", "expires": 3600, "inline": true }
```

**Response**

```json
{ "url": "https://<account>.blob.core.windows.net/container/blob?<sas>" }
```

### `/healthz` (GET)

Return: `{"status": "ok"}`

### `/readyz` (GET)

Return example:

```json
{
  "status":"ready",
  "service_ready": true,
  "qdrant": true,
  "dense": true,
  "sparse": true,
  "reranker": true
}
```

### `/metrics` (GET)

Prometheus exposition format.

---

## Configuration (Environment Variables)

Core variables used at runtime:

* `QDRANT_URL`, `QDRANT_API_KEY` — Qdrant connection.
* `COLLECTION_NAME` — Qdrant collection name.
* `DENSE_URL`, `DENSE_DIM`, `DENSE_BATCH_SIZE` — Dense embedder endpoint and expected vector dimension.
* `SPARSE_URL`, `SPARSE_BATCH_FALLBACK` — Sparse embedder endpoint and fallback batch size.
* `RERANKER_URL`, `RERANKER_MODE`, `RERANK_TOPK`, `RERANK_AUTO_THRESHOLD`, `RERANK_THRESHOLD`, `RERANK_MARGIN` — Reranker configuration and thresholds.
* `API_KEY`, `LLM_MODEL`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE` — LLM provider credentials and call parameters.
* `MAX_CHUNKS_TO_LLM`, `MAX_PROMPT_TOKENS`, `HTTP_TIMEOUT` — request safety limits and timeouts.
* Azure presign variables: `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_SAS_TOKEN`, `AZURE_ENDPOINT_SUFFIX`, `AZURE_USE_MANAGED_IDENTITY`, `ENV`.

---

## Metrics (exposed)

* `retrieval_requests_total`, `retrieval_request_duration_seconds`, `retrieval_errors_total`
* `dense_embed_requests_total`, `dense_embed_duration_seconds`
* `sparse_embed_requests_total`, `sparse_embed_duration_seconds`
* `qdrant_query_total`, `qdrant_query_duration_seconds`
* `llm_calls_total`, `llm_call_duration_seconds`
* `presign_requests_total`, `presign_duration_seconds`
* `retrieved_docs_count` (histogram)
* `service_ready` (gauge)

---

## Error Handling & Fallbacks

* **Embedding failure**: If dense or sparse embed fails, the service logs a warning and proceeds using the remaining signals. The corresponding client returns `None` or an empty list.
* **Qdrant failure**: If Qdrant query fails, `hybrid_query` returns an empty result list; `/generate` returns a safe failure message and records error metrics.
* **Reranker failure**: If reranker crashes or times out, skip rerank and continue with original fused results.
* **LLM failure**: If LLM HTTP call fails or no API key, fall back to `deterministic_summarize()` for a short answer.
* **Schema/Dimension mismatch**: Dense client verifies vector length; if mismatch is detected upstream, it raises an error to prevent silent corruption.

---

## Health, Readiness, and Probes

* **Liveness**: `/healthz` — lightweight check to confirm process is alive.
* **Readiness**: `/readyz` — composite check: Qdrant client availability and model service `/health` responses. Use readiness probe to prevent routing traffic until Qdrant client is created.
* **Docker-level**: HEALTHCHECK configured to call `/healthz` in the runtime image.

---

## Shutdown Behavior

* `SIGINT` / `SIGTERM` set a shutdown flag. Lifespan teardown closes HTTPX clients and logs shutdown events. External load balancer should be drained before pod termination to allow in-flight requests to finish.

---

## Implementation Notes (concise)

* Blocking Qdrant client calls are executed inside `asyncio.to_thread` to avoid blocking the event loop.
* Presign operations that require Azure SDKs are executed in a thread to avoid blocking.
* All external calls (embed, rerank, LLM, Qdrant) obey `HTTP_TIMEOUT` and are instrumented for latency and error telemetry.

