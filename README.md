

# STEP 2/3 - indexing_pipeline

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy
<details>
  <summary> View chunk stratergies and chunk schema (Click the triangle)</summary>

```sh

{
  "document_id": "abc123xyz",
  "chunk_id": "abc123xyz_1",
  "page_number": 1,
  "source_type": "pdf",               // or "html", "docx", etc.
  "source_path": "s3://bucket/data/raw/doc1.pdf",
  "line_range": None ,  // or [starting_line, ending_line] for txt, csv, spreadsheets,etc
  "start_time": None, 
  "end_time": None,  
  "html_blocks": [
    {
      "block_type": "paragraph",
      "content": "<p>This is a paragraph from HTML source.</p>"
    },
    {
      "block_type": "header",
      "content": "<h2>Section title</h2>"
    }
  ],
  "markdown_blocks": [
    {
      "block_type": "paragraph",
      "content": "This is a paragraph in markdown format."
    }
  ],
  "text": "This is the full extracted text from the chunk, merged from all sources.",
  "tables": [
    [
      ["Header1", "Header2"],
      ["Row1Col1", "Row1Col2"]
    ]
  ],
  "images": [
    "s3://bucket/data/images/doc1/page1_img1.png"
  ],
  "image_ocr": [
    {
      "image_s3": "s3://bucket/data/images/doc1/page1_img1.png",
      "text": "Detected text from OCR on the image"
    }
  ],
  "metadata": {
    "used_ocr": true,
    "is_multilingual": true,
    "num_tables": 1,
    "num_images": 1,
    "timestamp": "2025-07-27T12:00:15Z",
    "parse_chunk_duration": 356 // example in milliseconds
  }
}




```
</details>


### Export the neccessary configs.

```sh



export S3_BUCKET=e2e-rag-system16      
export S3_RAW_PREFIX=data/raw/         
export S3_CHUNKED_PREFIX=data/chunked/ 
export S3_IMAGE_PREFIX=data/images/    
export CHUNK_FORMAT=json             # (OR) 'jsonl' for faster read and storage efficiency for headless use(but not readable)
export DISABLE_OCR=false             # (OR) true = disable ocr and the text in images of docs will not be extracted(but very fast)
export OCR_ENGINE=rapidocr           # (OR) `tesseract` for wide lang and faster (OR) `indicocr` for indian languages (OR) `rapidocr` for complex english
export FORCE_OCR=false               # (OR) true = always OCR; false = skip if text exists(false recommended)
export OCR_RENDER_DPI=300            # higher dpi = high quality image extraction = higher cost and higher chance of extracting tiny texts
export MIN_IMG_SIZE_BYTES=3072       # Filter out tiny images under 3 KB (often unneccessary black empty images)
export IS_MULTILINGUAL=false         # (OR) true. if false, TESSERACT_LANG will be ignored
export TESSERACT_LANG=tam            #  for both `indicocr` `tesseract`. see mapping table below. 


export S3_BUCKET=e2e-rag-system16      
export S3_RAW_PREFIX=data/raw/         
export S3_CHUNKED_PREFIX=data/chunked/ 
export S3_IMAGE_PREFIX=data/images/    
export CHUNK_FORMAT=json             # (OR) 'jsonl' for faster read and storage efficiency for headless use(but not readable)
export DISABLE_OCR=false             # (OR) true = disable ocr and the text in images of docs will not be extracted(but very fast)
export OCR_ENGINE=tesseract           #  (OR) `rapidocr` for complex english (OR) `indicocr` for indian languages
export FORCE_OCR=false               # (OR) true = always OCR; false = skip if text exists(false recommended)
export OCR_RENDER_DPI=300            # higher dpi = high quality image extraction = higher cost and higher chance of extracting tiny texts
export MIN_IMG_SIZE_BYTES=3072       # Filter out tiny images under 3 KB (often unneccessary black empty images)
export IS_MULTILINGUAL=false         # (OR) true. if false, TESSERACT_LANG will be ignored
export TESSERACT_LANG=eng           #  for both `indicocr` and `tesseract` (but only one lang at a time). refer the mapping table below. 


export HF_TOKEN=
export EMBEDDING_EL_DEVICE=cpu      # or gpu for indexing with embedding and entity linking models
export EMBED_MODEL="elastic/multilingual-e5-small-optimized" # or View recommendations
export LOAD_IN=int8



```

<details>
  <summary> View languages table, embedding and EL models recommendations(Click the triangle)</summary>


| Language(Abbr)        | Language(Abbr)            | Language(Abbr)  | Language(Abbr)              | Language(Abbr)       | Language(Abbr)             | Language(Abbr)  |
| --------------------- | ------------------------- | --------------- | --------------------------- | -------------------- | -------------------------- | --------------- |
| Amharic(amh)          | French(fra)               | Oriya(ori)      | Serbian(srp)                | Urdu(urd)            | Azerbaijani(aze)           | Dutch(nld)      |
| Assamese(asm)         | French-Fraktur(fra\_frak) | Pashto(pus)     | Serbian-Cyrillic(srp\_cyrl) | Uzbek(uzb)           | Azerbaijani-Cyr(aze\_cyrl) | Esperanto(epo)  |
| Bosnian(bos)          | Galician(glg)             | Persian(fas)    | Serbian-Latin(srp\_latn)    | Uzbek-Cyr(uzb\_cyrl) | Finnish(fin)               | Latin(lat)      |
| Catalan(cat)          | German(deu)               | Polish(pol)     | Sinhala(sin)                | Vietnamese(vie)      | Tamil(tam)                 | Hebrew(heb)     |
| Cebuano(ceb)          | Greek(ell)                | Portuguese(por) | Slovak(slk)                 | Welsh(cym)           | Telugu(tel)                | Hindi(hin)      |
| Chinese-Sim(chi\_sim) | Gujarati(guj)             | Punjabi(pan)    | Slovenian(slv)              | Yiddish(yid)         | Thai(tha)                  | Hungarian(hun)  |
| Chinese-Tra(chi\_tra) | Haitian(hat)              | Romanian(ron)   | Spanish(spa)                | Yoruba(yor)          | Tibetan(bod)               | Icelandic(isl)  |
| Croatian(hrv)         | Hausa(hau)\*              | Russian(rus)    | Swahili(swa)                | Kurdish(kur)         | Japanese(jpn)              | Indonesian(ind) |
| Czech(ces)            | Khmer(khm)                | Lao(lao)        | Lithuanian(lit)             | Kannada(kan)         | Javanese(jav)              | Italian(ita)    |
| Danish(dan)           | Korean(kor)               | Latvian(lav)    | Nepali(nep)                 | Sinhala(sin)         | Oriya(ori)                 | Malay(may)\*    |
| English(eng)          | Tigrinya(tir)             | Estonian(est)   | Tagalog(tgl)                | Turkish(tur)         | Ukrainian(ukr)             | Uyghur(uig)     |


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


### indexing_pipeline/

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
| **Secrets & GitOps**             | `argoCD`, `SOPS`, `SealedSecrets`            | Secrets encrypted with SOPS (`.yaml` + GPG/AWS KMS). All workloads managed via GitOps using `argoCD` and synced periodically.                     | Declarative, auditable infra with secure secret rotation. Enables Git-based DR and versioned deploys. |
| **Metrics & Dashboards**         | `Prometheus`, `Grafana`                      | App metrics scraped via Prometheus. Dashboards for indexing, inference latency, chunk count, memory use, request trace count.                     | Realtime insight into system performance and bottlenecks.                                             |
| **LLM Call Tracing**             | `Helicone`                                   | All OpenAI-compatible requests (via DSPy/vLLM) routed through Helicone proxy. Logs prompt/response pairs, costs, durations.                       | Centralized LLM API observability. Enables debugging and optimization of prompt logic.                |
| **RAG Pipeline Observability**   | `RAGAI-Catalyst`                             | Logs and visualizes RAG stages: retrieval, rerank, generation. Integrated with Prometheus for end-to-end trace.                                   | Fine-grained insight into hybrid retrieval and generation quality.                                    |
| **Authentication & RBAC**        | `OIDC`, `Keycloak`, `AWS Cognito` (optional) | OIDC login for dev/admin UI. API protected via OIDC token validation middleware. Role-based access managed via Keycloak groups.                   | Enterprise-grade identity control with fine-grained access management.                                |
| **AMI Optimization**             | `Pulumi` + `Packer`                          | Base AMIs (e.g. vLLM, Ray, DSPy) pre-baked with dependencies. Auto-updated via CI when requirements change.                                       | Reduces cold-start latency. Consistent, reproducible image-based deployments.                         |
| **Cold Start Optimization**      | `Karpenter` Warm Pools, `vLLM` Lazy Load     | Maintain minimum ready nodes via Karpenter warm pools. `vLLM` loads model weights lazily or on-demand for fast ramp-up.                           | Improves SLA, ensures inference latency stays low under bursty traffic.                               |

---




