

# STEP 2/3 - indexing_pipeline

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy



```sh


export PYTHONPATH=$(pwd)
export LOG_LEVEL=INFO                # or DEBUG if not headless
export S3_BUCKET=e2e-rag-system16
export IS_ENGLISH=true
export IS_MULITLINGUAL=true
export OTHER_LANGUAGES=ta,hi         # or (see below table) , paddle2onnx conversion for non english/chinese models. Ignore if IS_MULTILINGUAL=false
export S3_RAW_PREFIX=data/raw/    
export S3_CHUNKED_PREFIX=data/chunked/
export CHUNK_FORMAT=json             # or jsonl for storage efficieny during headless mode       
export MIN_IMG_SIZE_BYTES=3072      # Filter out small images(often unneccessary black images) under 3 KB 
export OCR_RENDER_DPI=300           # Image rendering quality, higher = higher accuracy and cost and higher chance of extracting tiny texts



export EMBEDDING_EL_DEVICE=cpu      # or gpu for embedding and entity linking models
export EMBED_MODEL="elastic/multilingual-e5-small-optimized" # or View recommendations
export LOAD_IN=int8



```

| Language (abbr)        | Language (abbr)           | Language (abbr)   | Language (abbr)                 | Language (abbr)              |
| ---------------------- | ------------------------- | ----------------- | ------------------------------- | ---------------------------- |
| Chinese & English (ch) | Arabic (ar)               | Hindi (hi)        | Uyghur (ug)                     | Persian (fa)                 |
| Urdu (ur)              | Serbian Latin (rs\_latin) | Occitan (oc)      | Italian (it)                    | Marathi (mr)                 |
| Spanish (es)           | Nepali (ne)               | Portuguese (pt)   | Serbian Cyrillic (rs\_cyrillic) | Russian (ru)                 |
| Bulgarian (bg)         | Ukrainian (uk)            | Estonian (et)     | Belarusian (be)                 | Irish (ga)                   |
| Telugu (te)            | Croatian (hr)             | Saudi Arabia (sa) | Hungarian (hu)                  | Tamil (ta)                   |
| Indonesian (id)        | Afrikaans (af)            | Icelandic (is)    | Azerbaijani (az)                | Kurdish (ku)                 |
| Bosnian (bs)           | Lithuanian (lt)           | Czech (cs)        | Latvian (lv)                    | Welsh (cy)                   |
| Maori (mi)             | Danish (da)               | Malay (ms)        | Maltese (mt)                    | Adyghe (ady)                 |
| Dutch (nl)             | Kabardian (kbd)           | Norwegian (no)    | Avar (ava)                      | Polish (pl)                  |
| Dargwa (dar)           | Romanian (ro)             | Ingush (inh)      | Slovak (sk)                     | Lak (lbe)                    |
| Lezghian (lez)         | Slovenian (sl)            | Albanian (sq)     | Tabassaran (tab)                | Swedish (sv)                 |
| Bihari (bh)            | Swahili (sw)              | Maithili (mai)    | Tagalog (tl)                    | Angika (ang)                 |
| Turkish (tr)           | Bhojpuri (bho)            | Uzbek (uz)        | Magahi (mah)                    | Vietnamese (vi)              |
| Nagpur (sck)           | Mongolian (mn)            | Newari (new)      | Abaza (abq)                     | Goan Konkani (gom)           |
| French (fr)            | German (german)           | Japanese (japan)  | Korean (korean)                 | Chinese Trad. (chinese\_cht) |


<details>
  <summary> View embedding and EL models recommendations(Click the triangle)</summary>


## Recommendations:
  ### 1. The same embedding and EL model should be used in both indexing pipeline and in inference pipeline(deployment) so choose wisely.
  ### 2. Smaller models performs closer to larger models, so even if you need slightly higher accuracy, choose smaller models as the inference will have graph based multi hop retreival also. The fusion of multiple smaller and int8 versions is better than fewer large models. 
  ### 3. `elastic/multilingual-e5-small-optimized` is a highly efficient multilingual model supporting 90+ languages but supports dense(similarity) retreival only. 
  ### 4. `Alibaba-NLP/gte-multilingual-base(mGTE-TRM)` have long‑context support and improved multilingual retrieval. It supports both sparse(keyword) and dense(similarity) retreival but there isn't an english only version of `mGTE-TRM`
  ### 5. Use `Alibaba‑NLP/gte-modernbert-base` or `intfloat/e5-small` if the data is english only. If sparse retreival also needed, choose `mGTE‑TRM`
  ### 6. For the env variable `EMBED_MODEL`, kindly choose only the models in these tables as they were tested in RAG8s.

---

### Recommeded models:
| **Model**                                   | **MRR @10 / MTEB**                                  | **Params** | **Size (float32)**    | **Embed Dim** | **Max Tokens** | **VRAM (fp32)** | **VRAM (fp16)** | **VRAM (int8)** |
| ------------------------------------------- | --------------------------------------------------- | ---------- | --------------------- | ------------- | -------------- | --------------- | --------------- | --------------- |
| **Alibaba-NLP/gte-multilingual-base**    | \~ 68–71 MRR\@10 (MIRE M) / \~ 71 nDCG\@10 (MIRACL) | \~ 304 M   | \~ 1.2 GB (est.)      | 768–1024      | 8192           | \~ 5–7 GB       | \~ 3–4 GB       | \~ 1.8–2.2 GB   |
| **elastic/multilingual‑e5‑small‑optimized** | \~ 64.4 MRR\@10 (average)                           | \~ 110 M   | – (int8 quant)        | 384           | 512            | \~ 1–1.5 GB     | n/a             | \~ 1 GB         |
| **Alibaba‑NLP/gte-modernbert-base**                   | \~ 64.38 avg                                        | \~ 149 M   | \~ 0.67 GB (≈ 670 MB) | 768           | 8192           | \~ 5–6 GB       | \~ 3–4 GB       | \~ 2–2.5 GB     |

---

### Other models to compare: 
| **Model**                                   | **MRR\@10 / MTEB**                          | **Params** | **Size (float32)** | **Embed Dim** | **Max Tokens** | **VRAM (fp32)** | **VRAM (fp16)** | **VRAM (int8)** |
| ------------------------------------------- | ------------------------------------------- | ---------- | ------------------ | ------------- | -------------- | --------------- | --------------- | --------------- |
| **elastic/multilingual-e5-small**           | 64.4 MRR\@10 (average)     | \~ 110 M   | \~ 440 MB          | 384           | 512            | \~ 2–3 GB       | \~ 1.5–2 GB     | \~ 1–1.2 GB     |
| **elastic/multilingual-e5-base**            | 65.9 MRR\@10 (average)     | \~ 260 M   | \~ 1.0 GB          | 768           | 512            | \~ 4–6 GB       | \~ 2.5–3.5 GB   | \~ 1.5–2 GB     |
| **elastic/multilingual-e5-large**           | n/a (not published)                         | \~ 500 M   | \~ 2.0 GB          | 1024          | 512            | \~ 8–10 GB      | \~ 4.5–6 GB     | \~ 2.5–3.5 GB   |
| **intfloat/e5-small**                       | 64.4 MRR\@10 (average)     | \~ 110 M   | \~ 440 MB          | 384           | 512            | \~ 2–3 GB       | \~ 1.5–2 GB     | \~ 1–1.2 GB     |
| **intfloat/e5-base**                        | 65.9 MRR\@10 (average)     | \~ 260 M   | \~ 1.0 GB          | 768           | 512            | \~ 4–6 GB       | \~ 2.5–3.5 GB   | \~ 1.5–2 GB     |
| **intfloat/e5-large**                       | n/a (not published)                         | \~ 500 M   | \~ 2.0 GB          | 1024          | 512            | \~ 8–10 GB      | \~ 4.5–6 GB     | \~ 2.5–3.5 GB   |
| **gte‑base‑en‑v1.5**  | \~ 62.39 avg           | \~ 137 M   | \~ 0.22 GB (≈ 220 MB) | 768           | 8192           | \~ 2.5–3.5 GB   | \~ 1.5–2.5 GB   | \~ 1 GB         |
| **gte‑large‑en‑v1.5** | \~ 63.13 avg           | \~ 434 M   | \~ 0.67 GB (≈ 670 MB) | 1024          | 8192           | \~ 5–7 GB       | \~ 3–4 GB       | \~ 2–2.5 GB     |

</details>

---
---


---


```sh

{
  "id": "chunk_<document_sha256>_<chunk_index>",          // Unique global ID for the chunk: includes doc hash and position
  "payload": {
    "document_id": "<document_sha256>",                   // SHA256 of the full file (used for deduplication and lineage)
    "chunk_id": "<document_sha256>_<chunk_index>",        // Local ID for chunk, deterministic within a document
    "chunk_index": 0,                                     // 0-based index, used for ordering chunks during retrieval
    "text": "chunk text content",                         // Actual text of this chunk (paragraph, page, heading, etc.)

    "source_path": "s3://bucket/file.pdf",                // Resolved full URI of the original file (can be local or cloud)
    "source_hash": "<document_sha256>",                   // Redundant doc hash (used for quick reference)

    "file_type": "pdf",                                   // One of: pdf, html, mp3, csv, txt, etc. (prefer MIME-based detection)
    "page_number": 1,                                     // For PDFs/ePubs: the 1-based page index. Null for non-paged types

    "bbox": null,                                         // For visual formats (PDF, HTML): [x0, y0, x1, y1] pixel bbox. Else null

    "metadata": {
      "is_multilingual": false,                           // True if mixed language content detected
      "is_ocr": false,                                    // True if OCR was applied (fallback on image-based PDFs)
      "chunk_type": "paragraph",                          // "paragraph", "heading", "page", "table", "section", etc.
      "timestamp": "2025-07-23T20:30:00Z",                // UTC timestamp of chunk creation (ISO 8601 format)
      "tags": [],                                         // Optional labels: "title", "header", "invoice", etc.
      "layout_tags": ["heading", "paragraph"],            // Structural tags inferred from layout model
      "confidence": null                                  // OCR/ASR confidence score (0.0–1.0); null if not applicable
    },

    "entities": [],                                     // Optional: NER output as list of {text, type, span, confidence}
    "embedding": []                                     // Optional: vector embedding (float list) to be added post-parsing
  },

  "tables": null                                          // Optional: list of extracted tables; null if no tables present
}


```

---


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
| **Secrets & GitOps**             | `argoCD`, `SOPS`, `SealedSecrets`            | Secrets encrypted with SOPS (`.yaml` + GPG/AWS KMS). All workloads managed via GitOps using `argoCD` and synced periodically.                     | Declarative, auditable infra with secure secret rotation. Enables Git-based DR and versioned deploys. |
| **Metrics & Dashboards**         | `Prometheus`, `Grafana`                      | App metrics scraped via Prometheus. Dashboards for indexing, inference latency, chunk count, memory use, request trace count.                     | Realtime insight into system performance and bottlenecks.                                             |
| **LLM Call Tracing**             | `Helicone`                                   | All OpenAI-compatible requests (via DSPy/vLLM) routed through Helicone proxy. Logs prompt/response pairs, costs, durations.                       | Centralized LLM API observability. Enables debugging and optimization of prompt logic.                |
| **RAG Pipeline Observability**   | `RAGAI-Catalyst`                             | Logs and visualizes RAG stages: retrieval, rerank, generation. Integrated with Prometheus for end-to-end trace.                                   | Fine-grained insight into hybrid retrieval and generation quality.                                    |
| **Authentication & RBAC**        | `OIDC`, `Keycloak`, `AWS Cognito` (optional) | OIDC login for dev/admin UI. API protected via OIDC token validation middleware. Role-based access managed via Keycloak groups.                   | Enterprise-grade identity control with fine-grained access management.                                |
| **AMI Optimization**             | `Pulumi` + `Packer`                          | Base AMIs (e.g. vLLM, Ray, DSPy) pre-baked with dependencies. Auto-updated via CI when requirements change.                                       | Reduces cold-start latency. Consistent, reproducible image-based deployments.                         |
| **Cold Start Optimization**      | `Karpenter` Warm Pools, `vLLM` Lazy Load     | Maintain minimum ready nodes via Karpenter warm pools. `vLLM` loads model weights lazily or on-demand for fast ramp-up.                           | Improves SLA, ensures inference latency stays low under bursty traffic.                               |

---




