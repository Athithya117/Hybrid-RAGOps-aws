# Platform Model Architecture — default models overview

## At-a-glance component table (exact names, ports, default envs)

| Role             |                                              Model (default) |                              Runtime & port | Default env knobs (concrete)                                                  | Response shape / contract                                                                      |
| ---------------- | -----------------------------------------------------------: | ------------------------------------------: | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Dense embedder   |      `BAAI/bge-small-en-v1.5` (384-dim). ([Hugging Face][4]) |  Self-hosted FastAPI `/embed` port **8200** | `DENSE_MODEL_NAME`, `DENSE_DIM=384`, `DENSE_BATCH_SIZE=16`, `DENSE_NORMALIZE` | Req: `{"texts":["..."]}` → Res: `{"vectors":[[float,...],...]}` (vector length = `DENSE_DIM`)  |
| Sparse embedder  |    `Qdrant/minicoil-v1` (sparse lexical) ([Hugging Face][5]) |  Self-hosted FastAPI `/embed` port **8201** | `SPARSE_MODEL_NAME`, `SPARSE_BATCH_SIZE=8`, `SPARSE_CUDA`                     | Req: `{"texts":["..."]}` → Res: `{"vectors":[{"indices":[int,...],"values":[float,...]},...]}` |
| Reranker         | `ms-marco-MiniLM-L-6-v2` (cross-encoder) ([Hugging Face][6]) | Self-hosted FastAPI `/rerank` port **8202** | `RERANKER_MODEL_NAME`, `RERANKER_MAX_DOCS=50`                                 | Req: `{"query":"...","documents":["d1","d2",...]}` → Res: `{"scores":[float,...]}`             |
| LLM (generation) |       `llama-3.1-8b-instant` (Groq-managed) ([GroqCloud][3]) |                     Managed provider (HTTP) | API key / endpoint (provider-managed)                                         | Standard chat/completion API; deterministic prompt contract required by platform               |

---

## Concrete deployment & runtime contracts

1. **Image build pattern**

   * Builder stage MUST prewarm the model (download weights) and copy `/models_cache` into final image layer to reduce runtime cold-start. (Your Dockerfiles already follow this pattern.)
   * Provide a writable mount for `/models_cache` (PV) in production to *avoid* baking extremely large images and enable fast redeploys.

2. **HTTP contract details (exact endpoints + validation)**

   * Dense `/embed` (POST): accept `{"texts": ["s1",... ]}`; reject if batch size > `DENSE_BATCH_SIZE` with 400; respond `{"vectors":[... ]}`; vectors must be `float` and have length `DENSE_DIM`. Validate and fail-fast on dimension mismatch (server must `exit(1)` on init mismatch). (Matches your host_dense behavior.)
   * Sparse `/embed` (POST): accept `{"texts": [...]}`; return sparse objects `{indices, values}` normalized to ints/floats. Reject oversize batches > `SPARSE_BATCH_SIZE`.
   * Reranker `/rerank` (POST): accept `{"query": "...", "documents": ["..."]}`; reject if `len(documents) > RERANKER_MAX_DOCS` or model not `READY` (503).
   * Health: `/health` returns `{"status":"ok"|"not_ready","model":<name>,"ready":bool,"model_error":<string?>}` — use this for K8s readiness/liveness.

3. **Qdrant collection (hybrid) — minimal concrete template**

   * Create a collection that exposes **named vectors**: one dense vector (name `dense`) and one sparse vector (name `sparse_bm25` or `sparse_minicoil`). Example (pseudocode / qdrant-client):

```python
client.create_collection(
  collection_name="docs",
  vectors_config={
    "dense": {"size": 384, "distance": "Cosine"},
    "sparse_minicoil": {"type": "sparse", "config": {"bm25": True}}
  }
)
```

* Use `dense.size = DENSE_DIM`. Configure sparse vector per Qdrant sparse config (BM25 or SPLADE-style) — Qdrant supports named sparse vectors and IDF at collection level. ([Qdrant][7])

4. **Indexing contract**

   * Embedding pipeline must:

     * Normalize chunk schema: `document_id`, `chunk_id`, `token_count`, `token_range`, `source_url`, `parser_version`.
     * Compute deterministic point id from `chunk_id` (e.g., md5 hex → integer slice).
     * For each chunk: compute dense vector (if available) and sparse representation; upsert into Qdrant point with `vectors={"dense": [..], "sparse_minicoil": {"indices":[..],"values":[..]}}` plus `payload` including full chunk metadata.
   * Dedup: prior to embedding use `client.retrieve` (IDs) to skip already-present points (saves compute).

---

## Autoscaling, resource sizing & SLO guidance (concrete recommendations)

> Assumptions: CPU-only production for cost-efficiency (you can enable GPU later per-service).

* **Dense embedder (BGE-small, CPU)**

  * Pod resources (start): `requests: cpu=1000m, mem=4Gi`; `limits: cpu=2000m, mem=8Gi`.
  * HPA: scale on custom metric `requests_per_pod` or `cpu` with target 60% CPU; set min=2, max=10. Batch concurrency: prefer `DENSE_BATCH_SIZE=16`. Expect p95 latency per 16-text batch on x86 CPU ≈ 50–300ms (measure & tune per instance). ([Hugging Face][4])

* **Sparse embedder (miniCOIL)**

  * Pod resources: `requests: cpu=500m, mem=2Gi`; `limits: cpu=1500m, mem=4Gi`.
  * HPA: min=1, max=6; scale on CPU or `request_latency` custom metric; `SPARSE_BATCH_SIZE=8`.

* **Reranker (cross-encoder)**

  * Pod resources: `requests: cpu=2000m, mem=8Gi`; `limits: cpu=4000m, mem=16Gi` (cross-encoders are heavier).
  * Reranker should be *right-sized* since it runs on small top-k (e.g., 10–50 docs). Consider autoscaling to 0 when idle (KEDA on queue or concurrency) or smaller instance counts.

* **LLM (managed)**

  * No host GPU; pay-per-token. Ensure requests to LPU include strict timeout and token limits.

**Practical SLOs & throttles**

* Per API: enforce max request content length and `*_BATCH_SIZE` as hard guard.
* Set up a circuit-breaker: if embed service p95 latency > X ms for 3 consecutive minutes, progressively reduce batch size or reject new indexing jobs and alert.

---

## Observability & autoscaling wiring (exact metrics & probes)

* Expose `/metrics` (Prometheus) with labels `service, env, endpoint, status_code`. Use:

  * `request_duration_seconds{service="dense"}` for latency-based autoscaling (KEDA / Prometheus Adapter).
  * `model_ready{service="..."} == 1` for readiness probe: K8s readiness endpoint should call `/health` and return Ready only when `model_ready==true`. Liveness should check process-level heartbeat or `/metrics` emission.
* Add an HTTP readiness probe hitting `/health` with 1s initialDelay, 5s period, failureThreshold=3. Liveness: `/metrics` scrape recency or a simple `/health` check every 10s.

---

## Failure modes (precise) + exact mitigations

1. **Dimension mismatch on startup (dense)**

   * Symptom: server exits with code 1 and logs `embedding dimension mismatch`.
   * Mitigation: CI must run container builder prewarm; runtime asserts `DENSE_DIM` vs model-dim. Add pre-deploy check in CI that inspects model metadata (HF/fastembed) and fail the build if mismatch.

2. **Sparse model returns different sparse format**

   * Symptom: downsteam upsert fails (invalid indices/values).
   * Mitigation: embedder must run `to_sparse()` normalizer producing `indices:int[]` and `values:float[]`; add schema unit tests per model.

3. **Qdrant upsert throttling or partial failures**

   * Symptom: `upsert` returns transient HTTP 5xx.
   * Mitigation: exponential backoff + chunked `UPSERT_CHUNK` (env knob); record `batch.embedded` and `index.completed` events.

4. **High latency / slow model warm**

   * Mitigation: Use builder-stage prewarm in CI + mount `/models_cache` via PV or shared cache; add readiness gating in deployment so traffic only goes to Ready pods.

---

## Integration checklist (concrete items to verify before production)

* [ ] CI image build includes prewarm and fails on warmup error.
* [ ] K8s manifests include PV mount for `/models_cache` and resource requests/limits above.
* [ ] Qdrant collection created with `dense.size = DENSE_DIM` and sparse config enabled (IDF/BM25) per collection. ([Qdrant][8])
* [ ] API contracts documented (JSON schema files) and enforced by pydantic models.
* [ ] HPA configured with CPU + custom Prometheus-based latency metric.
* [ ] CI test: roundtrip embed & upsert to a staging Qdrant instance; verify `points.count > 0`.

---

## Minimal config snippet examples (env + collection create)

Dense pod env:

```yaml
env:
  - name: DENSE_MODEL_NAME
    value: "BAAI/bge-small-en-v1.5"
  - name: DENSE_DIM
    value: "384"
  - name: DENSE_BATCH_SIZE
    value: "16"
```

Qdrant collection create (python snippet):

```python
client.create_collection(
  collection_name="docs",
  vectors_config={
    "dense": {"size": 384, "distance": "Cosine"},
    "sparse_minicoil": {"type": "sparse", "config": {"bm25": True}}
  }
)
```

---

## Short rationales / evidence pointers (most load-bearing facts)

* FastEmbed supports BGE and many production embedding models; prewarming reduces runtime cold-start. ([qdrant.github.io][2])
* miniCOIL is Qdrant’s sparse neural retrieval model designed for BM25-style lexical+semantic recall. ([Qdrant][9])
* ms-marco MiniLM cross-encoder is a common light-weight reranker used in production for high-accuracy re-scoring. ([Hugging Face][6])
* Groq-managed Llama 3.1 instances provide high throughput and large context windows; delegating LLMes to managed providers reduces infra GPU complexity. ([GroqCloud][3])
* Qdrant natively supports hybrid (named dense + sparse) collections and server-side BM25/IDF handling to simplify sparse usage. ([Qdrant][1])

---


[1]: https://qdrant.tech/documentation/beginner-tutorials/hybrid-search-fastembed/?utm_source=chatgpt.com "Setup Hybrid Search with FastEmbed"
[2]: https://qdrant.github.io/fastembed/examples/Supported_Models/?utm_source=chatgpt.com "Supported Models - FastEmbed"
[3]: https://console.groq.com/docs/model/llama-3.1-8b-instant?utm_source=chatgpt.com "Llama 3.1 8B - GroqDocs"
[4]: https://huggingface.co/BAAI/bge-small-en-v1.5?utm_source=chatgpt.com "BAAI/bge-small-en-v1.5"
[5]: https://huggingface.co/Qdrant/minicoil-v1?utm_source=chatgpt.com "Qdrant/minicoil-v1"
[6]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2?utm_source=chatgpt.com "cross-encoder/ms-marco-MiniLM-L6-v2"
[7]: https://qdrant.tech/documentation/concepts/collections/?utm_source=chatgpt.com "Collections"
[8]: https://qdrant.tech/documentation/concepts/vectors/?utm_source=chatgpt.com "Vectors"
[9]: https://qdrant.tech/documentation/fastembed/fastembed-minicoil/?utm_source=chatgpt.com "Working with miniCOIL"
[10]: https://qdrant.tech/course/essentials/day-3/sparse-retrieval-demo/?utm_source=chatgpt.com "Demo: Keyword Search with Sparse Vectors"
