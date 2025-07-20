export QDRANT_GRPC_HOST="localhost"
export QDRANT_GRPC_PORT=6334
export ARANGO_ROOT_USERNAME="root"
export ARANGO_ROOT_PASSWORD="superSecret"
export VALKEY_PASSWORD="yourValkeyPass"




**NVIDIA (June 2025)** recommends **page-level chunking** as the most accurate and consistent baseline:

> “Page-level chunking achieved the highest average accuracy (0.648) with the lowest standard deviation (0.107)... It outperformed token- and section-level chunking.”
> — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/?utm_source=chatgpt.com)

**Use page-wise as the default** for structured/unstructured docs.








### indexing_pipeline/
---

| **Component**                         | **Tool(s)**                                          | **Exact Chunking Strategy**                                                                                                                                                        | **Why Chosen for Scalability**                                                                                     |
| ------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Audio Transcription**               | `faster-whisper`, `pydub`, `ffmpeg-python`           | Audio is loaded using `pydub`; sliced into 20–30s segments using silence detection (`pydub.silence.detect_nonsilent`). Each segment becomes a chunk with `start_time`, `end_time`. | `faster-whisper` (CTranslate2) enables fast CPU/GPU transcription. `ffmpeg` ensures universal audio compatibility. |
| **HTML Parsing**                      | `extractous`, `BeautifulSoup`                        | HTML parsed with `BeautifulSoup`; headings (`<h*>`), paragraphs, and sections form logical delimiters.                                                                             | Lightweight, DOM-aware; preserves document structure and metadata during chunking.                                 |
| **PDF Parsing + OCR**                 | `pdfplumber`, `PyMuPDF`, `paddleocr`, `paddlepaddle` | Default: 1 page = 1 chunk. For sparse/visual PDFs: fallback to paragraph chunking using spacing heuristics (`line_gap > 1.5x median`). OCR used for image-heavy scans.             | Multilingual, layout-aware parsing; fallback improves resilience. Runs within Ray pipeline.                        |
| **CSV Chunking**                      | `ray.data.read_csv()` + `.window()`                  | Parsed with `ray.data`. Chunked using `ds.window(bytes_per_window=N)` where `N = max(5120, avg_row_len * CSV_ROWS_PER_CHUNK)`. Adaptive based on content size.                     | Efficient streaming, avoids memory spikes, parallelizable across nodes.                                            |
| **JSON Chunking**                     | `ray.data.read_json()` + `.window()`                 | JSONL: one line = one record. For nested: flatten → explode arrays → chunk using grouping by field depth or record size.                                                           | Robust handling of complex structures. Adaptive chunk size per nesting and token count.                            |
| **Pipeline Orchestration**            | `ray` (core, actors, tasks)                          | Each stage (parsing, chunking, embedding) runs as Ray actor or task. Orchestrated by dispatcher using `ray.remote`.                                                                | Enables parallel processing, distributed execution, and shared memory across stages.                               |
| **Main Parser Entry Point**           | `indexing_pipeline/index.py`                         | Dispatch based on file extension or MIME type via `mimetypes.guess_type()` or `.endswith()`. Handler routes to proper parser, outputs JSONL to `/data/chunked/`.                   | Modular dispatcher. Easy to add custom formats or override handlers.                                               |
| **Content Hashing / Deduplication**   | `hashlib`                                            | Stream full file to compute SHA256 → becomes `document_id`. Chunks get `chunk_{sha256}_{chunk_index}`.                                                                             | Guarantees uniqueness. Streaming avoids full memory usage on large files.                                          |
| **S3 I/O**                            | `boto3`                                              | Uses `s3.download_fileobj()` or `get_object()` with streaming. Output written via `upload_fileobj()` or multipart upload.                                                          | Supports massive files, IAM-secured access, auto-retry, and resumable uploads.                                     |
| **Entity Linking (Multilingual)**     | `ReFinED`                                            | Chunk text passed to `refined.get_entities(text)` → merged into `entities[]`. Handles multi-language linking and coreference.                                                      | Lightweight and accurate. Handles multilingual corpora efficiently on CPU/GPU.                                     |
| **Embedding Generation**              | `elastic/multilingual-e5-small-optimized`            | Chunk text → passed through `AutoModel.from_pretrained()` → CLS vector extracted → stored as `.embedding`. Handles truncation/sliding window.                                      | Small, multilingual, production-ready model. High speed with good quality tradeoff.                                |
| **Vector Index**                      | `qdrant-client`                                      | Vectors inserted into Qdrant with `upload_collections()`. Metadata used for filters and hybrid search. Uses HNSW indexing.                                                         | Fast ANN search. Easy to shard and scale. Supports metadata and full-text hybrid retrieval.                        |
| **Knowledge Graph / Triplet Storage** | `python-arango`                                      | Extracted triplets stored as `UPSERT` into ArangoDB doc and edge collections. Keys derived from `chunk_id`.                                                                        | Graph-native storage with AQL. Scales well for hybrid knowledge-backed retrieval.                                  |

---

### inference_pipeline/ 
---

| **Component**                      | **Tool(s)**                                            | **Exact Runtime Logic**                                                                                                                            | **Why Chosen for Scalability**                                                        |
| ---------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Query Parsing & Expansion**      | `DSPy`, `LangGraph`                                    | Query passed through `DSPy Signature` with prompt tuning and `LangGraph` node chain. Optional: expansion via auto-reformulated query chain.        | Declarative, composable LLM logic. Easily experiment with different flows.            |
| **Sparse Retrieval**               | `SPLADE++`, `SimLM`                                    | Query encoded using `SPLADE` (max-pooling logits over vocab) or `SimLM`. Compared to indexed corpus (BM25, dense, hybrid).                         | Strong performance on zero-shot sparse search. Good contrast to dense-only retrieval. |
| **Dense Retrieval**                | `qdrant-client` (with E5 embeddings)                   | Query encoded using same model (`multilingual-e5-small-optimized`) → `qdrant_client.search(...)` returns dense-matched chunks.                     | Fast ANN search with metadata filtering. Production-proven HNSW under the hood.       |
| **Fusion / Reranking**             | `Reciprocal Rank Fusion (RRF)`, `DSPy reranker module` | Merge sparse + dense hits via RRF (score = Σ 1 / (k + rank\_i)) → reranked using lightweight `DSPy` scoring module or trained reranker model.      | Simple, effective hybrid fusion. Keeps top results balanced across modalities.        |
| **Entity Graph Reasoning**         | `python-arango`, `AQL`, `RGL`                          | If hit contains linked entities: fire AQL query to retrieve neighbors via `RGL` templates → add supporting facts to context.                       | Structured knowledge-aware augmentation. Graph-native fallback when retrieval weak.   |
| **Prompt Template Tuning**         | `DSPy`                                                 | Each step (`rerank`, `generate`, `refine`) uses a `Signature` class with optimizer (e.g. MIPRO, evolutionary search) to tune over training traces. | Declarative tuning. Traceable, reproducible, optimizer-guided.                        |
| **Reasoning & Flow Control**       | `LangGraph`, `DSPy`, `Guardrails`                      | Multi-node LangGraph flow: `Retrieve → Rerank → Answer → Critique → Retry`. Guardrails enforce output format, safety.                              | Modular LLM chaining. Safe and traceable execution.                                   |
| **LLM Inference Runtime**          | `vLLM`                                                 | DSPy-backed prompts are executed via OpenAI-compatible calls to `vLLM` runtime. Supports batching, streaming, KV cache, tensor parallelism.        | High-throughput, low-latency inference. Scales horizontally.                          |
| **Session & Memory Store**         | `ValKeye`                                              | Stores session embeddings, prompt traces, previous messages. Hit if re-query happens within session scope. Supports TTLs and multi-turn lookups.   | Fast, pluggable memory layer. Avoids redundant recompute.                             |
| **Cache Layer (Embedding + Hits)** | `ValKeye`, `qdrant-client`, `ray`                      | Embeddings and chunk hits cached locally (ValKeye) or as `ray.put(...)` objects. TTL tuned based on workload.                                      | Minimizes compute + disk I/O for repeated queries. Supports hybrid memory cache.      |
| **Observability**                  | `Helicone`, `Prometheus`, `Grafana`, `RAGAI-Catalyst`  | LLM calls routed through Helicone. Traces stored and visualized with Grafana dashboards. RAGAI-Catalyst emits retrieval and generation metrics.    | Full traceability, latency monitoring, retrieval performance tracking.                |

---
### infra

| **Component**                    | **Tool(s)**                                  | **Deployment / Runtime Strategy**                                                                                                                 | **Why Chosen for Scalability & Reliability**                                                          |
| -------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Infrastructure as Code (IaC)** | `Pulumi`                                     | All AWS resources (EKS, S3, IAM, AMIs, etc.) provisioned via `Pulumi` in Python. Supports env-based configs and secrets from SOPS.                | Unified infra layer with Git versioning. Easy to compose and reuse components.                        |
| **Kubernetes Platform**          | `AWS EKS`, `KubeRay`, `Karpenter`            | All workloads (Ray, vLLM, DSPy, ArangoDB, Qdrant) deployed on EKS. `KubeRay` handles Ray cluster lifecycle. `Karpenter` handles node autoscaling. | Native autoscaling, GPU-aware scheduling, self-healing. Cost-efficient compute management.            |
| **Ingress & Routing**            | `Traefik`, `Cloudflare`                      | Traefik deployed as ingress controller with HTTPS + routing rules. Cloudflare handles DNS + WAF + CDN.                                            | Secured, cached API routing. Cloudflare protects public endpoints, minimizes latency.                 |
| **Secrets & GitOps**             | `FluxCD`, `SOPS`, `SealedSecrets`            | Secrets encrypted with SOPS (`.yaml` + GPG/AWS KMS). All workloads managed via GitOps using `FluxCD` and synced periodically.                     | Declarative, auditable infra with secure secret rotation. Enables Git-based DR and versioned deploys. |
| **Metrics & Dashboards**         | `Prometheus`, `Grafana`                      | App metrics scraped via Prometheus. Dashboards for indexing, inference latency, chunk count, memory use, request trace count.                     | Realtime insight into system performance and bottlenecks.                                             |
| **LLM Call Tracing**             | `Helicone`                                   | All OpenAI-compatible requests (via DSPy/vLLM) routed through Helicone proxy. Logs prompt/response pairs, costs, durations.                       | Centralized LLM API observability. Enables debugging and optimization of prompt logic.                |
| **RAG Pipeline Observability**   | `RAGAI-Catalyst`                             | Logs and visualizes RAG stages: retrieval, rerank, generation. Integrated with Prometheus for end-to-end trace.                                   | Fine-grained insight into hybrid retrieval and generation quality.                                    |
| **Authentication & RBAC**        | `OIDC`, `Keycloak`, `AWS Cognito` (optional) | OIDC login for dev/admin UI. API protected via OIDC token validation middleware. Role-based access managed via Keycloak groups.                   | Enterprise-grade identity control with fine-grained access management.                                |
| **AMI Optimization**             | `Pulumi` + `Packer`                          | Base AMIs (e.g. vLLM, Ray, DSPy) pre-baked with dependencies. Auto-updated via CI when requirements change.                                       | Reduces cold-start latency. Consistent, reproducible image-based deployments.                         |
| **Cold Start Optimization**      | `Karpenter` Warm Pools, `vLLM` Lazy Load     | Maintain minimum ready nodes via Karpenter warm pools. `vLLM` loads model weights lazily or on-demand for fast ramp-up.                           | Improves SLA, ensures inference latency stays low under bursty traffic.                               |

---
