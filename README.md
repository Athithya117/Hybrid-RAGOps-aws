
# RAG8s is a production ready RAG environment to build and ship an E2E RAG system using the SOTA tools, models and strategies as of mid 2025. It is currently under development


## STEP 2/3 - indexing_pipeline

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy
<details>
  <summary> View chunk stratergies and chunk schema (Click the triangle)</summary>

```sh

{
  "chunk_id": "a3f5be12c9d47e09_5",               // Unique chunk ID: <document_hash>_<chunk_index> (1-based)
  "document_id": "a3f5be12c9d47e09",              // Unique document ID (128-bit hash of file path + size)
  "chunk_type": "page",                           // Type of content: "paragraph", "heading", "section", "table", "page", etc.

  "text": "Cleaned markdown-formatted content of the chunk.",  // Final parsed content (Markdown)
  "embedding": null,                              // Optional: vector embedding (array of floats); null if not yet computed

  "source": {
    "file_type": "application/pdf",                  // MIME type preferred (e.g., "application/pdf", "text/html", "audio/mpeg")
    "source_path": "s3://bucket/data/raw/file.pdf",  // Full s3 path to original source
    "page_number": 3,                                // For paged formats like PDF/ePub; null otherwise
    "time": [null, null],                            // [start_time, end_time] in seconds for audio/video; nulls otherwise
    "line_range": null,                              // For plain/tabular text: [start_line, end_line]; null otherwise
    "bbox": null                                     // For visual formats: [x0, y0, x1, y1] in pixel coordinates; null otherwise
  },

  "graph": {
    "graph_node_id": "a3f5be12c9d47e09_5",       // Same as chunk_id (recommended)
    "parent_id": "a3f5be12c9d47e09_page3",       // Parent node ID (e.g., page, section, table)
    "previous_id": "a3f5be12c9d47e09_4",         // Optional: previous chunk
    "next_id": "a3f5be12c9d47e09_6"              // Optional: next chunk
  },

  "metadata": {
    "timestamp": "2025-08-03T12:00:00Z",         // UTC ISO timestamp of chunk creation/parsing
    "tags": ["invoice", "header"],               // High-level content tags (semantic or manual)
    "layout_tags": ["paragraph"]                 // Structural tags (e.g., "heading", "table", etc.)
  },

  "entities": ["Q123", "Q456"],                  // Optional: Linked entity IDs (Wikidata, etc.) or null if not yet computed

  "triplets": [                                  // Extracted subject-predicate-object relations
    {
      "subject": "Invoice",
      "predicate": "hasDate",
      "object": "2025-08-01"
    },
    {
      "subject": "Invoice",
      "predicate": "issuedBy",
      "object": "ACME Corp"
    },
    {
      "subject": "ACME Corp",
      "predicate": "locatedIn",
      "object": "New York"
    }
  ]
}


```


## Component-Level Parsing & Chunking Strategy

| Component            | Tool(s) Used                                                                                                  | Chunking Strategy                                                                                                         | Rationale for Scalability & Fidelity                                                                 |
|----------------------|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **PDF Parsing + OCR** | `PyMuPDF`, `pdfplumber`, `pytesseract`, `RapidOCR` | - **1 page = 1 chunk** (primary granularity)<br>- Extract native text via `pdfplumber`, fallback to OCR if needed<br>- Use `OpenCV` for layout-aware OCR (cropping regions, columnar reading)<br>- Tables extracted as both **structured JSON** and **inline Markdown**<br>- Figures/diagrams extracted to a separate `figures` array | - Page-level chunking aligns with layout and enables **parallelism**<br>- Hybrid OCR improves coverage of low-quality scans<br>- Layout fidelity helps preserve **tables, headers, visual order**                             |
| **DOC/DOCX Conversion** | `LibreOffice` (headless mode), `subprocess`, `tempfile`, `boto3`, `pathlib`, `os`                             | - Convert `.doc` → `.docx` → `.pdf` using `LibreOffice CLI`<br>- Apply **same PDF+OCR pipeline per page** on output PDF<br>- Page alignment maintained between original and converted formats                        | - Avoids unreliable native `.docx` parsing<br>- Ensures **visual and semantic consistency** across systems<br>- Helps in tracing                                                   |
| **Text/HTML Parsing** | `BeautifulSoup`, `html2text`, custom chunkers                                                                 | - Segment by structural tags: `h1–h6`, `p`, `li`, `div`, `table`<br>- Normalize output into **Markdown chunks**<br>- Chunk IDs and parent-child relations inferred from tag hierarchy                         | - Converts semi-structured content into **RAG-ready Markdown**<br>- Retains **hierarchy and inline metadata** for graph linkage<br>- Works well with multi-format source ingestion                      |                                            |
| **HTML Parsing**        | extractous, BeautifulSoup                                                        | - Parse HTML DOM tree  <br> - Chunk by headings (`<h1>`–`<h6>`), paragraphs, and sections                                                                                                                            | - Lightweight, preserves semantic structure  <br> - Works on both web pages and embedded HTML                                     |
| **CSV Chunking**        | ray.data.read\_csv(), `.window()`                                                | - Stream rows  <br> - Chunk based on size heuristics (`max(5120, avg_row_len * ROWS_PER_CHUNK)`)                                                                                                                     | - Efficient streaming for large files  <br> - Memory-safe, scalable via Ray                                                       |
| **JSON/JSONL**          | ray.data.read\_json(), `.window()`                                               | - JSONL: each line = record  <br> - For nested JSON: flatten → explode arrays → chunk by size/depth                                                                                                                  | - Handles deeply nested or irregular structures  <br> - Flexible chunk size based on token count                                  |
| **Audio Transcription** | faster-whisper (CTranslate2), pydub, ffmpeg-python                               | - Audio sliced into 20–30s segments via silence detection (`pydub.silence`)  <br> - Each segment transcribed individually                                                                                            | - Faster-Whisper is GPU/CPU efficient  <br> - Segmentation makes long audio scalable and parallelizable                           |
| **Markdown**            | markdown-it-py, mistune, regex                                                   | - Chunk by heading levels, paragraphs, and code blocks  <br> - Fallback to fixed-token or sentence-based slicing                                                                                                     | - Preserves Markdown structure  <br> - Compatible with LLM indexing and embeddings                                                |
| **PPTX (PowerPoint)**   | python-pptx, Pillow (optional OCR)                                               | - 1 slide = 1 chunk  <br> - Extract text, speaker notes, images  <br> - OCR fallback on image slides                                                                                                                 | - Natural chunking by slide  <br> - Works well with educational or slide-heavy documents                                          |
| **EPUB/eBooks**         | ebooklib, BeautifulSoup, html5lib                                                | - Chunk by chapters/headings from EPUB metadata  <br> - Paragraph or heading-based segmentation within chapters                                                                                                      | - Structure-aware  <br> - Works with long-form content like books                                                                 |
| **Images (Scans)**      | OpenCV, PIL/Pillow, Tesseract or RapidOCR                                        | - 1 image = 1 chunk  <br> - OCR applied to entire image or regions (if detected)                                                                                                                                     | - Useful for form scans, handwritten notes, flyers  <br> - Preserves visual layout                                                |
| **ZIP Archives**        | zipfile, tarfile, custom dispatcher                                              | - Files extracted, routed to correct parsers based on extension (pdf, docx, txt, etc.)                                                                                                                               | - Allows batch ingestion  <br> - Enables unified multi-file upload experience                                                     |
| **Plaintext Files**     | open(), re, nltk, tiktoken (optional)                                            | - Chunk by paragraph, newline gaps (`\n\n`), or fixed line/token window                                                                                                                                              | - Extremely lightweight  <br> - Works well with logs, scraped data, or long articles                                              |

</details>


### Export the neccessary configs.

```sh

# Storage / paths
export S3_BUCKET=e2e-rag-system                      # set per-env/tenant (unique bucket)
export S3_RAW_PREFIX=data/raw/                        # raw ingest prefix (change to isolate datasets)
export S3_CHUNKED_PREFIX=data/chunked/                # chunked output prefix (change to separate processed data)
export CHUNK_FORMAT=json                              # 'json' (readable) or 'jsonl' (stream/space efficient)
export OVERWRITE_DOC_DOCX_TO_PDF=true                 # true to replace docx with PDF, false to keep originals

# OCR & image extraction
export DISABLE_OCR=false                              # true to skip OCR (faster) | false to extract text from images
export OCR_ENGINE=tesseract                           # 'tesseract' (fast/common) or 'rapidocr' (higher accuracy, slower)
export FORCE_OCR=false                                # true to always OCR (use if source text unreliable)
export OCR_RENDER_DPI=300                             # increase for tiny text; lower for speed
export MIN_IMG_SIZE_BYTES=3072                        # ignore images smaller than this (reduce noise)


```

```sh 

# Arango / vector index toggles
export ARANGO_VECTOR_INDEX_ENABLE=true                # false to disable vector ops (read-only or minimal infra)
export ARANGO_VECTOR_INDEX_TYPE="ivf"                 # choose 'hnsw' (<100k docs), 'ivf' (>=100k), 'ivf+pq' for huge corpora
export ARANGO_VECTOR_INDEX_MAX_MEMORY_MB=2048         # soft cap for index memory on node; increase with corpus size

# IVF-specific (only if using ivf)
export ARANGO_VECTOR_INDEX_IVF_NLIST=1000             # set ~sqrt(N_vectors); increase for very large corpora
export ARANGO_VECTOR_INDEX_IVF_NPROBE=10              # raise for recall at cost of latency

# PQ (only if using ivf+pq/pq)
export ARANGO_VECTOR_INDEX_PQ_M=16                    # PQ segments; must divide embedding dim; tune for memory vs accuracy

# HNSW-specific (only if using hnsw)
export ARANGO_VECTOR_INDEX_HNSW_M=32                  # higher => more memory but higher recall
export ARANGO_VECTOR_INDEX_HNSW_EFCONSTRUCTION=200    # raise for better index build quality
export ARANGO_VECTOR_INDEX_HNSW_EFSEARCH=50           # raise for higher query recall (latency ↑)

# FAISS sidecar / local index
export FAISS_INDEX_PATH="/mnt/faiss/index.ivf"        # local index path (empty if not used)
export FAISS_INDEX_DIM=768                            # must match embedding model output
export FAISS_NLIST=256                                # local FAISS nlist; increase for large indices
export FAISS_NPROBE=10                                # raise for recall at latency cost

# Retrieval fusion weights (tune by devset; relative importance)
export W_VEC=0.6                                      # raise if domain embeddings are highly accurate
export W_BM25=0.3                                     # raise if exact keyword matches are critical
export W_GRAPH=0.1                                    # raise if graph/triplet hits are very high precision
export W_RERANK=0.5                                   # meaningful only when reranker enabled

# Candidate fanout & GeAR
export N_VEC=15                                       # top-K vector candidates (raise for recall on large corpora)
export N_BM25=15                                      # top-K BM25 candidates
export N_GRAPH=5                                      # graph neighbor limit (keep small to control DB load)
export MAX_GEARS_HOPS=1                               # 1 default; enable 2 behind feature flag for deeper multi-hop
export GEAR_BEAM_WIDTH=3                              # beam width for GeAR expansion; increase with caution

# Pre-fusion thresholds (filters to reduce noise)
export VEC_SCORE_THRESH=0.20                          # min vector similarity to keep a candidate (raise for precision)
export BM25_SCORE_THRESH=1.50                         # min BM25 to keep (raise to filter weak keyword hits)
export GRAPH_SCORE_THRESH=0.0                         # min graph edge confidence (set >0 if confidences provided)

# Reranker & metadata boosting
export USE_RERANKER=true                              # enable only if you accept added latency/cost for higher precision
export RERANK_BATCH_SIZE=16                           # increase to amortize GPU/CPU when latency allows
export META_BOOST_FIELD="timestamp"                   # metadata key to bias ranking (e.g., timestamp, source_score)
export META_BOOST_WEIGHT=0.20                         # 0.0-1.0; raise if metadata should strongly affect ranking

```

# Timeouts / concurrency / performance
export RETRIEVAL_TIMEOUT=5                            # seconds; increase if backing systems are slower
export RETRIEVAL_BATCH_SIZE=4                         # parallelism for retrieval calls; increase with CPU/network capacity
export MAX_CONCURRENT_QUERIES=32                      # throttle to protect DBs; scale with infra
# Arango general performance / logging
export ARANGO_STORAGE_CACHE_SIZE=2048                 # set ~20-30% host RAM for read-heavy nodes
export ARANGO_QUERY_MEMORY_LIMIT=1024                 # raise if AQL traversals need more memory
export ARANGO_LOG_LEVEL="info"                        # 'debug' only for troubleshooting
export ARANGO_URL="http://arangodb:8529"              # Arango endpoint (use in-cluster svc in prod)
export ARANGO_DB="rag8s"                              # DB name
export ARANGO_USER="root"                             # use non-root least-priv user in prod
export ARANGO_PASS=""                                 # SECRET: inject from Vault/K8s Secret


# Ray / orchestration (node-level)
export RAY_DASHBOARD_PORT=8265                        # Ray dashboard port (change to avoid conflicts)
export RAY_NUM_CPUS=4                                 # set per node; increase for heavier parallel indexing/inference
export RAY_NUM_GPUS=0                                 # set >0 on GPU nodes for model serving/training

# Observability / app logging
export APP_LOG_LEVEL="info"                           # 'debug' only temporarily for troubleshooting



```



```sh


## **RAG8s Inference Flow**

**Goal:** Add lightweight multi-hop reasoning via precomputed triples in ArangoDB without slowing down BM25 + vector + graph retrieval.

---

### **1. Query Processing**

* Embed query (`gte-modernbert-base`).
* Lookup entities via simple dictionary/inverted index from precomputed ReLiK output.

---

### **2. Candidate Generation (Parallel)**

* **Vector (FAISS):** Top `N1` by cosine similarity.
* **BM25 (ArangoSearch):** Top `N2` by keyword relevance.
* **Graph:** If entity match, retrieve related chunks (`N3`) via 1–2 hops.

---

### **3. GeAR Multi-Hop Expansion**

* Seeds = top BM25 + vector results + entity matches.
* Beam search 1–3 hops over precomputed triples in ArangoDB.
* Collect new chunks not in initial set.

```aql
LET seeds = @seed_entity_ids
FOR v, e, p IN 1..2 OUTBOUND seeds GRAPH 'EntityGraph'
  OPTIONS { bfs: true, uniqueVertices: "global" }
  FILTER v.chunk_id != null
  LET path_score = 1 / (1 + LENGTH(p.edges))
  RETURN DISTINCT { chunk_id: v.chunk_id, score: path_score }
```

---

### **4. Fusion**

Weighted sum of normalized scores:

```python
final_score = (
    0.4 * vec_score +
    0.25 * bm25_score +
    0.2 * graph_score +
    0.15 * gear_score
)
```

---

### **5. Dedupe & Sort**

* Keep highest score per `chunk_id`.
* Sort and keep top `K`.

---

### **6. Optional Rerank**

* Apply `gte-modernbert-reranker-base` if latency allows.

---

**Why it works:**

* No extra inference models (triples are precomputed).
* GeAR adds multi-hop reasoning without replacing existing retrieval.
* ArangoDB handles both keyword search and graph traversals efficiently.

---


### The RAG8s platform codebase(currently under development, 20% completed)


```sh

RAG8s/
├── data/                                 # Local directory that syncs with s3://<bucket_name>/data
│   ├── raw/                              # Raw data files
│   └── chunked/                          # Chunks in json/jsonl format
│
├── indexing_pipeline/
│   ├── Dockerfile                        # Docker image for indexing workers
│   ├── index/
│   │   ├── __main__.py                   # CLI entrypoint for indexing jobs  # observe: logs, metrics, traces
│   │   ├── arrangodb_indexer.py          # Indexer: writes chunks/entities into ArangoDB with FAISS integration  # observe: logs, metrics
│   │   ├── config.py                     # Indexing configuration (paths, batch sizes, env)
│   │   └── utils.py                      # Utility helpers used by indexers (parsers, serializers)
│   ├── parse_chunk/
│   │   ├── __init__.py                   # parse_chunk package initializer
│   │   ├── doc_docx_to_pdf.py            # Converts .doc/.docx to PDF (LibreOffice headless flow)
│   │   ├── formats/
│   │   │   ├── __init__.py               # Format module initializer
│   │   │   ├── csv.py                    # CSV reader & chunker logic
│   │   │   ├── doc_docx.py               # Docx parser + fallback handlers
│   │   │   ├── html.py                   # HTML -> Markdown chunker and DOM processing
│   │   │   ├── json.py                   # JSON/JSONL flattening and chunking routines
│   │   │   ├── md.py                     # Markdown chunking and normalization
│   │   │   ├── mp3.py                    # Audio preprocessing wrapper (slicing metadata)
│   │   │   ├── pdf.py                    # PDF page extraction and layout-aware parsing
│   │   │   ├── png_jpeg_jpg.py           # Image OCR pipeline wrapper
│   │   │   ├── ppt_pptx.py               # PPTX slide extractor (n slide = 1 chunk)
│   │   │   ├── spreadsheets.py           # Spreadsheet row/column chunking logic
│   │   │   └── txt.py                    # Plaintext chunkers (paragraph/sentence/window)
│   │   └── router.py                     # Dispatcher to select parser based on MIME/extension
│   ├── relik.sh                          # Helper script to run ReLiK entity/triplet extraction
│   └── requirements-cpu.txt              # Indexing pipeline runtime dependencies (CPU)
│
├── inference_pipeline/
│   ├── Dockerfile                        # Dockerfile for inference server image
│   ├── auth_control.py                   # Authentication, authorization middleware and rate limiting for APIs  # observe: logs, metrics
│   ├── eval.py                           # RAGAI-Catalyst coherence checks, hit@K monitoring, hallucination detection  # observe: logs, metrics
│   ├── frontend/
│   │   ├── Dockerfile                    # Frontend container build file
│   │   ├── main.py                       # Frontend app entry (UI endpoints / static server)  # observe: logs, metrics
│   │   ├── modules                        # Modular UI components / assets
│   │   └── requirements-cpu.txt          # Frontend Python dependencies
│   ├── main.py                           # Inference service entrypoint (REST/gRPC server)  # observe: logs, metrics, traces
│   ├── retreiver.py                      # Retrieval orchestration (hybrid BM25 + vector + graph + GeAR lightweight multihop)  # observe: logs, metrics
│   └── trace_file.py                     # View or download Presigned urls for the raw docs as source link s3://<bucket_name>data/raw/<file_name>.<format>
|
├── infra/
│   ├── charts/
│   │   └── rag8s/
│   │       ├── Chart.yaml                # Helm chart metadata + optional dependencies (Karpenter, Ray, Prometheus)
│   │       ├── values.yaml               # Dynamically created by scripts/dynamic-values.yaml.sh
│   │       ├── templates/                # All rendered Kubernetes manifests
│   │       │   ├── _helpers.tpl          # Shared labels/annotations/name templates
│   │       │   ├── argocd.yaml           # ArgoCD Application definition for GitOps
│   │       │   ├── core/                 # Cluster-wide primitives
│   │       │   │   ├── arangobackup-cronjob.yaml # incremental backups to s3://<bucket_name>/backups/
│   │       │   │   ├── namespaces.yaml   # Namespace creation from values.base.namespaces
│   │       │   │   ├── serviceaccounts.yaml # ServiceAccounts + IRSA
│   │       │   │   ├── rbac.yaml         # Roles, ClusterRoles, Bindings
│   │       │   │   ├── pdb.yaml          # PodDisruptionBudgets from values.base.pdb
│   │       │   │   ├── frontend.yaml     # Frontend deployment, svc, HPA
│   │       │   │   ├── valkeye.yaml      # Deployment for Redis compatible in-memory data storage for rate limiting
│   │       │   │   └── quotas.yaml       # ResourceQuotas from values.base.quotas
│   │       │   ├── monitoring/           # Observability resources
│   │       │   │   ├── servicemonitors.yaml # Prometheus ServiceMonitor CRs
│   │       │   │   ├── grafana.yaml      # Grafana dashboards + datasources
│   │       │   │   └── alerts.yaml       # PrometheusRule alert definitions
│   │       │   ├── networking/           # Ingress, ingress controller, network policies
│   │       │   │   ├── traefik.yaml      # Traefik Helm chart CRDs/config
│   │       │   │   ├── ingress.yaml      # Ingress objects per service
│   │       │   │   └── networkpolicies.yaml # NetworkPolicies for traffic control
│   │       │   ├── rayservices/          # RayServe workloads
│   │       │   │   ├── embedder-reranker.yaml # RayService for embeddings/reranking
│   │       │   │   └── sglang.yaml       # RayService for LLM serving
│   │       │   ├── rayjobs/              # Ray batch jobs
│   │       │   │   └── indexing.yaml     # RayJob for indexing pipeline (CPU provisioner)
│   │       │   └── karpenter/            # Karpenter provisioners
│   │       │       ├── provisioner-cpu.yaml # CPU workloads, Spot + pre-warmed OnDemand
│   │       │       └── provisioner-gpu.yaml # GPU workloads, Spot + pre-warmed OnDemand
│   │       └── README.md                 # Chart-specific README and usage notes
│   │  
│   ├── pulumi-aws/
│   │   ├── __main__.py                    # IAC CLI for EKS cluster provisioning
│   │   ├── cloudflare.py                  # Cloudflare DNS / zone automation helpers
│   │   ├── cloudwatch.py                  # CloudWatch metric/alert helpers
│   │   ├── db_backup.py                   # DB backup/restore automation scripts for local nvmes based ec2s
│   │   ├── eks_cluster.py                 # EKS cluster orchestration code (Pulumi)
│   │   ├── iam_roles.py                   # IAM role & policy creators for services
│   │   ├── indexing_ami.py                # AMI build definitions for indexing nodes
│   │   ├── inference_ami.py               # AMI build definitions for inference nodes
│   │   ├── karpenter.py                   # Karpenter provisioner configuration helpers
│   │   ├── nodegroup.py                   # For arangodb statefulset, valkeye deployment and frontend svc
│   │   ├── pulumi.yaml                    # Pulumi project manifest for infra code
│   │   ├── traefik.py                     # Traefik infrastructure helper code
│   │   └── vpc.py                         # VPC/subnet/networking helper utilities
│   │ 
│   ├── onnx/
│   │   ├── Dockerfile                      # ONNX runtime image for CPU inference services
│   │   ├── grpc.proto                      # gRPC proto definition for ONNX service
│   │   ├── grpc_pb2.py                     # Generated gRPC Python bindings
│   │   ├── grpc_pb2_grpc.py                # Generated gRPC server/client scaffolding
│   │   ├── rayserve-embedder-reranker.py   # Ray Serve wrapper to run embedder + reranker  # observe: logs, metrics
│   │   ├── requirements-cpu.txt            # ONNX service dependencies
│   │   └── run.sh                          # Convenience script to start ONNX gRPC server
│   │  
│   └── sglang/
│       ├── Dockerfile                      # GPU-enabled image for SGLang model serving
│       ├── rayserve-sglang.py              # Ray Serve wrapper for SGLang LLM inference  # observe: logs, metrics
│       └── requirements-gpu.txt            # GPU runtime dependencies (CUDA/pytorch/etc.)
│
├── output.yaml                             # Deployment/output summary produced by infra scripts
│
├── scripts/
│   ├── build_and_push.sh                   # Builds container images and pushes to registry
│   ├── dynamic-values.yaml.sh              # Generates dynamic Helm values (env-specific)
│   ├── helm-deploy.sh                      # Wrapper to deploy Helm charts via CI or locally
│   ├── pulumi-set-configs.sh               # Sets Pulumi configuration and secrets
│   └── pulumi-set-secret.sh                # Stores secrets into Pulumi secret store
│
├── .devcontainer/
│   ├── Dockerfile                          # Devcontainer image build for local development environment
│   ├── devcontainer.json                   # VS Code devcontainer configuration (mounts, settings)
│   └── scripts
│       └── fix-docker-group.sh             # Script to fix Docker group permissions inside devcontainer
│
├── .dockerignore                           # Files/dirs excluded from Docker build context
├── .gitignore                              # Git ignore rules
├── Makefile                                # Convenience targets for build/test/deploy tasks
├── README.md                               # Project overview, setup and usage instructions
├── backups/                                
│   └── dbs/
│       └── arrangodb/                      
│
└── tmp.md                                  # Temporary notes / scratch markdown file

```


# Models overview

---

| Model                        | Language     | Params | Max Tokens              | Efficiency           | Triplet μ-F1 | size (FP32)  | size (INT8 / W4A16) | RAM needed (INT8) | VRAM needed (W4A16) |
| ---------------------------- | ------------ | ------ | ----------------------- | -------------------- | ------------ | ------------ | ------------------- | ----------------- | ------------------- |
| gte-modernbert-base          | English      | 149M   | 8,192                   | High (CPU/ONNX)      | —            | 596 MiB      | 149 MiB (INT8)      | 1.0 GiB           | —                   |
| gte-reranker-modernbert-base | English      | 149M   | 8,192                   | Very high (CPU/ONNX) | —            | 598 MiB      | 149 MiB (INT8)      | 1.0 GiB           | —                   |
| relik-cie-tiny               | English      | 174M   | \~3,000                 | High (CPU)           | 73.8%        | 663.8 MiB    | 165.9 MiB (INT8)    | 0.94 GiB          | —                   |
| relik-cie-small              | English      | 216M   | \~3,000                 | high (CPU)           | 74.3%        | 824.0 MiB    | 206.0 MiB (INT8)    | 1.40 GiB          | —                   |
| relik-cie-large              | English      | 467M   | \~3,000                 | Moderate (CPU)       | 75.6%        | 1,781.5 MiB  | 445.4 MiB (INT8)    | 2.43 GiB          | —                   |
| Qwen3-0.6B-quantized.w4a16   | Multilingual | 600M   | 32,768                  | High (W4A16 AWQ)     | —            | \~2.4 GiB\*  | 860 MiB (W4A16)     | —                 | \~1.1 GiB           |
| Qwen3-1.7B-quantized.w4a16   | Multilingual | 1.7B   | 32,768                  | High (W4A16 AWQ)     | —            | \~6.8 GiB\*  | 2.0 GiB (W4A16)     | —                 | \~2.7 GiB           |
| Qwen3-4B-quantized.w4a16     | Multilingual | 4B     | 32,768 (native)         | High (W4A16 AWQ)     | —            | \~16.0 GiB\* | 3.43 GiB (W4A16)    | —                 | \~5.6 GiB           |
| Qwen3-8B-quantized.w4a16     | Multilingual | 8.2B   | 32,768 / 131,072 (YaRN) | High (W4A16 AWQ)     | —            | \~32.8 GiB\* | \~6.5 GiB (W4A16)†  | —                 | \~11 GiB            |
| Qwen3-14B-quantized.w4a16    | Multilingual | 14.8B  | 32,768 / 131,072 (YaRN) | High (W4A16 AWQ)     | —            | \~59.2 GiB\* | \~10.9 GiB (W4A16)† | —                 | \~18 GiB            |

---

## 🔗 **References & specialties of the default models in RAG8s**

---

### 🔹 **\[1] gte-modernbert-base**

* Embedding-only model for dense retrieval in RAG pipelines
* Long-context support: up to **8192 tokens** (Sufficient for page wise chunking)
* Based on **ModernBERT** (FlashAttention 2, RoPE, no position embeddings)
* Optimized for ONNX export and CPU-inference
* Embedding dimension: **768**
* Parameter size: **149M**
  🔗 [https://huggingface.co/Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)

> **Use case**: Recommended for fast, CPU-efficient **semantic retrieval** in low-latency RAG pipelines with long context document embeddings.

---

### 🔹 **\[2] gte-reranker-modernbert-base**

* **Cross-encoder reranker** for re-ranking retrieved docs
* High BEIR benchmark score (**nDCG\@10 ≈ 90.7%**)
* Same architecture & size as embedding model (149M), supports **8192 tokens**
* Extremely fast CPU inference with ONNX (FlashAttention 2)
  🔗 [https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)

> **Use case**: Ideal for **re-ranking top-k retrieved passages** after dense retrieval to improve precision in RAG answer selection.

---

### 🔹 **\[3] ReLiK-CIE-Tiny**

A compact and efficient **entity + relation extraction** model designed for **Graph-RAG pipelines**. Unlike fast entity-only models (e.g., SpEL, ReFinED), `relik-cie-tiny` can extract both **named entities** and **semantic triplets** (`(head, relation, tail)`), enabling direct construction of **knowledge subgraphs** from raw text.

* Extracts **entities and triplets** in a single pass
* Suitable for **CPU inference** via ONNX
* Balanced for **accuracy and runtime performance**
  🔗 [relik-ie/relik-cie-tiny](https://huggingface.co/relik-ie/relik-cie-tiny)

> **Use case**: Ideal for production Graph-RAG inference pipelines where lightweight models must still generate structured knowledge without relying on large LLM backends.

---

### 🔹 **\[4] RedHatAI/Qwen3-4B-W4A16**

A compact, high-throughput **instruction-tuned LLM** quantized using **W4A16** (4-bit weights + FP16 activations). Built on **Qwen3-4B**, this variant supports **32,768-token context** natively and achieves performance comparable to models 10× its size (e.g., Qwen2.5-72B). Optimized for **SGLang inference**, it balances **speed, memory efficiency, and accuracy**, running seamlessly on GPUs like A10G, L4, and L40S.

* Architecture: **Transformer** (Qwen3 series, multilingual)
* Context Length: **32k tokens** (SGLang-native)
* Quantization: **W4A16 (AWQ)** — 4-bit weights, FP16 activations
* VRAM Usage: **\~4.8–5.2 GiB** (fits on 24 GiB GPUs with headroom)

🔗 [RedHatAI Qwen3-4B-W4A16](https://huggingface.co/RedHatAI/Qwen3-4B-quantized.w4a16)

> “Even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.”
> — [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
> — [Thinking-mode](https://qwenlm.github.io/blog/qwen3/#key-features)

> **Use case**: Smaller models (e.g., Qwen3-4B-W4A16 or 8B) fit on a single VM, making them better suited for data-parallel engines like **SGLang**, rather than tensor-parallel systems like **vLLM**.

---

<details>
 <summary>EC2 (Click the triangle)</summary>

## EC2 instances for spot with fallback pre warmed on-demand scaling with karpenter-gpu

#### **G6f (Fractional L4 GPU) Sizes**

| Instance        | vCPUs | Memory  | GPU Fraction | GPU Memory |
| --------------- | ----- | ------- | ------------ | ---------- |
| **g6f.large**   | 2     | 8 GiB   | 0.125 × L4   | \~3 GiB    |
| **g6f.xlarge**  | 4     | 16 GiB  | 0.25 × L4    | \~6 GiB    |
| **g6f.2xlarge** | 8     | 32 GiB  | 0.5 × L4     | \~12 GiB   |
| **g6f.4xlarge** | 16    | 64 GiB  | 1 × L4       | 24 GiB     |
| **g6f.8xlarge** | 32    | 128 GiB | 2 × L4       | 48 GiB     |

---

#### **G6e (Full L40S GPU) Sizes**

| Instance         | vCPUs | Memory   | GPUs | GPU Memory                               |
| ---------------- | ----- | -------- | ---- | ---------------------------------------- |
| **g6e.xlarge**   | 4     | 32 GiB   | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.2xlarge**  | 8     | 64 GiB   | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.4xlarge**  | 16    | 128 GiB  | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.8xlarge**  | 32    | 256 GiB  | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.12xlarge** | 48    | 384 GiB  | 4    | 178 GiB (4×44)  ([AWS Documentation][1]) |
| **g6e.16xlarge** | 64    | 512 GiB  | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.24xlarge** | 96    | 768 GiB  | 4    | 178 GiB (4×44)  ([AWS Documentation][1]) |
| **g6e.48xlarge** | 192   | 1536 GiB | 8    | 357 GiB (8×44)  ([AWS Documentation][1]) |

---
# Local NVMe based EC2s for hosting vector dbs like qdrant, arrangodb as statefulsets 
| Instance     | vCPU / RAM       | **C8g On-Demand** (USD/hr) | **C8g On-Demand** (₹/hr) | **C8g Spot** (USD/hr)   | **C8g Spot** (₹/hr) | **C8gd On-Demand** (USD/hr) | **C8gd On-Demand** (₹/hr) | **C8gd Spot** (USD/hr)  | **C8gd Spot** (₹/hr) |
| ------------ | ---------------- | -------------------------- | ------------------------ | ----------------------- | ------------------- | --------------------------- | ------------------------- | ----------------------- | -------------------- |
| **medium**   | 1 vCPU/2 GiB     | \$0.044 ([Vantage][1])     | ₹3.7                     | \$0.013 ([Vantage][1])  | ₹1.1                | \$0.054 ([Vantage][2])      | ₹4.5                      | \$0.012 ([Vantage][2])  | ₹1.0                 |
| **xlarge**   | 4 vCPU/8 GiB     | \$0.702 ([Vantage][3])     | ₹58.2                    | \$0.238 ([Vantage][3])  | ₹19.8               | \$0.196 ([Vantage][4])      | ₹16.3                     | \$0.080 ([Vantage][4])  | ₹6.6                 |
| **2xlarge**  | 8 vCPU/16 GiB    | \$0.319 ([Vantage][5])     | ₹26.5                    | \$0.128 ([Vantage][5])  | ₹10.6               | —                           | —                         | —                       | —                    |
| **4xlarge**  | 16 vCPU/32 GiB   | \$0.702 ([Vantage][3])     | ₹58.2                    | \$0.238 ([Vantage][3])  | ₹19.8               | \$0.784 ([Vantage][6])      | ₹65.1                     | \$0.298 ([Vantage][6])  | ₹24.7                |
| **8xlarge**  | 32 vCPU/64 GiB   | \$1.276 ([Vantage][7])     | ₹105.9                   | \$0.522 ([Vantage][7])  | ₹43.3               | \$1.720 ([Vantage][8])      | ₹142.8                    | \$— (spot NA)           | —                    |
| **16xlarge** | 64 vCPU/128 GiB  | \$2.808 ([Vantage][9])     | ₹233.1                   | \$0.731 ([Vantage][9])  | ₹60.7               | \$3.135 ([Vantage][10])     | ₹259.9                    | \$0.796 ([Vantage][10]) | ₹66.1                |
| **24xlarge** | 96 vCPU/192 GiB  | \$4.211 ([Vantage][8])     | ₹349.3                   | \$1.065 ([Vantage][8])  | ₹88.4               | \$5.173 ([Vantage][8])      | ₹429.4                    | \$1.476 ([Vantage][8])  | ₹122.5               |
| **48xlarge** | 192 vCPU/384 GiB | \$7.657 ([Vantage][11])    | ₹635.6                   | \$1.979 ([Vantage][11]) | ₹164.3              | \$9.406 ([Vantage][12])     | ₹780.8                    | \$3.323 ([Vantage][12]) | ₹275.6               |

---

### Notes

* “—” indicates data not available or not listed for that size.
* Spot prices fluctuate; these are **minimums** observed at time of lookup.
* INR values are approximate (USD × 83).
* NVMe-equipped C8gd variants include large local SSD at no extra configuration cost; C8g has no NVMe.
* Source pricing from Instances.Vantage.sh snapshots as of early August 2025.

[1]: https://instances.vantage.sh/aws/ec2/c8g.medium
[2]: https://instances.vantage.sh/aws/ec2/c8gd.medium
[3]: https://instances.vantage.sh/aws/ec2/c8g.4xlarge
[4]: https://instances.vantage.sh/aws/ec2/c8gd.xlarge?cost_duration=monthly&os=linux&region=us-east-1&reserved_term=Standard.noUpfront
[5]: https://instances.vantage.sh/aws/ec2/c8g.2xlarge
[6]: https://instances.vantage.sh/aws/ec2/c8gd.4xlarge
[7]: https://instances.vantage.sh/aws/ec2/c8g.8xlarge
[8]: https://instances.vantage.sh/aws/ec2/c8gd.24xlarge
[9]: https://instances.vantage.sh/aws/ec2/c8g.16xlarge
[10]: https://instances.vantage.sh/aws/ec2/c8gd.16xlarge
[11]: https://instances.vantage.sh/aws/ec2/c8g.48xlarge
[12]: https://instances.vantage.sh/aws/ec2/c8gd.48xlarge

</details>

# EXAMPLES

### Qdrant Payload Object (Point in Vector Database)

* **Purpose:**
  The payload object stores both the **vector embedding** of the text chunk and its **associated metadata** within Qdrant, enabling efficient similarity search combined with rich filtering.

* **Embedding:**
  The numeric array (e.g., `[0.0234, -0.1457, 0.3782, ...]`) is a fixed-length dense vector representing semantic meaning extracted by a transformer model. This vector enables approximate nearest neighbor (ANN) search for relevance ranking.

* **Metadata (Payload):**
  Key-value pairs store important structured information alongside the vector. This includes:

  * `chunk_id`, `document_id` to uniquely identify the source chunk.
  * `chunk_type` describing the nature of the content (e.g., "page").
  * `text` snippet or summary (optional or truncated) for quick reference.
  * Source details like `file_type`, `source_path`, `page_number` to trace origin.
  * Timestamp and tags (`occupation`, `NCO`, `tailor`) for filtering results by content type or category.
  * Linked entity IDs (`Q7531`, `Q1251134`) to connect with knowledge graph nodes or perform hybrid vector-graph queries.

* **Filtering & Search:**
  Payload fields allow **metadata filtering** combined with vector similarity. For example, retrieving points with tag `"occupation"` and embedding closest to the query vector representing "tailor".

* **Use Case in NCO:**
  Storing the NCO chunk as a Qdrant point means you can search for occupations semantically (e.g., "dressmaker" or regional terms for "tailor") even if keyword search fails due to synonyms or multilingual variations, while still filtering precisely by category or document source.

* **Integration:**
  This payload object serves as a **bridge between dense vector search and structured metadata filtering**, enabling hybrid retrieval systems that leverage both unstructured semantic similarity and structured knowledge.

---



```sh
{
  "chunk_id": "e4f72d3a9c5b4f17_5",                           // Unique chunk ID: <document_hash>_<chunk_index> (1-based)
  "document_id": "e4f72d3a9c5b4f17",                         // Unique document ID (128-bit hash of file path + size)
  "chunk_type": "page",                                       // Type of content: "paragraph", "heading", "section", "table", "page", etc.

  "text": "## Page 194\n\nNational Classification of Occupations – 2015 Concordance Table\n\nNCO 2015 NCO 2004\n\n7523.0200 Wood Turner, Machine 7423.15\n\n7523.0300 Wood, Turner Hand 7423.20\n\n7523.0400 Shaper Wood 7423.25\n\n7523.0500 Router, Wood 7423.30\n\n7523.0600 Planer, Wood 7423.35\n\n7523.0700 Four Cutter 7423.40\n\n7523.0800 Wood Sawyer, Hand 7423.42\n\n7523.0900 Moulder, Wood 7423.45\n\n7523.1000 Mortiser Operator 7423.50\n\n7523.1100 Tennoning Machine Operator 7423.55\n\n7523.1200 Jointer Machine Operator 7423.60\n\n7523.1300 Driller Wood 7423.65\n\n7523.1400 Dowell Machine Operator 7423.70\n\n7523.1500 Wood Wool Machine Operator 7423.80\n\n7523.1600 Wood Carver, Machine 8241.10\n\n7523.1700 Fret Saw Machine Operator 8241.20\n\n7523.1800 Sander Operator 8241.30\n\n7523.9900 Wood Working Machine Setters and Setter 8241.90\n\nOperators, Others\n\nGroup 753 Garment and Related Trades Workers\n\nFamily 7531 Tailors, Dressmakers, Furriers and Hatters\n\n7531.0100 Tailor, General 7433.10\n\nVOLUME I 182",

  "embedding": [0.0234, -0.1457, 0.3782, 0.0923, -0.0567,..]  // Vector embedding (array of floats); null if not yet computed

  "source": {
    "file_type": "application/pdf",                           // MIME type preferred (e.g., "application/pdf", "text/html", "audio/mpeg")
    "source_path": "s3://mospi-data/data/raw/nco_2015_occupations.pdf",  // Full s3 path to original source
    "page_number": 194,                                         // For paged formats like PDF/ePub; null otherwise
    "time": [null, null],                                     // [start_time, end_time] in seconds for audio/video; nulls otherwise
    "line_range": null,                                       // For plain/tabular text: [start_line, end_line]; null otherwise
    "bbox": null                                              // For visual formats: [x0, y0, x1, y1] in pixel coordinates; null otherwise
  },

  "graph": {
    "graph_node_id": "e4f72d3a9c5b4f17_5",                   // Same as chunk_id (recommended)
    "parent_id": "e4f72d3a9c5b4f17_page5",                   // Parent node ID (e.g., page, section, table)
    "previous_id": "e4f72d3a9c5b4f17_4",                     // Optional: previous chunk
    "next_id": "e4f72d3a9c5b4f17_6"                          // Optional: next chunk
  },

  "metadata": {
    "timestamp": "2025-08-03T12:15:27Z",                     // UTC ISO timestamp of chunk creation/parsing
    "tags": ["occupation", "tailor", "NCO", "classification"],  // High-level content tags (semantic or manual)
    "layout_tags": ["heading", "paragraph", "list"]           // Structural tags (e.g., "heading", "table", etc.)
  },

  "entities": [
    "Q7531",                                                  // Example Wikidata ID for Tailor occupation (hypothetical)
    "Q1251134"                                                // Wikidata ID for National Classification of Occupations (example)
  ],                                                         // Optional: Linked entity IDs (Wikidata, etc.) or null if not yet computed

  "triplets": [                                              // Extracted subject-predicate-object relations
    {
      "subject": "Tailor (General)", "predicate": "hasNCOCode", "object": "7531.0100"
    },
    {
      "subject": "Tailor (General)","predicate": "belongsToCategory", "object": "Tailors, Dressmakers, Furriers, and Hatters"
    },
    {
      "subject": "Tailor, Tent (Machine)", "predicate": "hasNCOCode", "object": "7534.0100"
    },
    {
      "subject": "Tailor (General)", "predicate": "hasSynonyms", "object": "Seamstress, Dressmaker, Garment Maker, Costume Maker"
    }
  ]
}
```


### Corresponding ArangoDB Graph Data Model Representing NCO Tailor Chunk and Linked Entities
* The **chunk document** holds the main text and metadata with embedding vector stored for similarity search.
* **Entities** are individual occupational concepts or classifications, stored separately to normalize data.
* **Edges** explicitly represent relations such as "mentions", "classifiedBy", or attribute predicates.
* This graph structure enables **multi-hop queries** and semantic traversals beyond keyword matching.

---

#### 1. Chunk Document (`chunks` collection)

```json
{
  "_key": "e4f72d3a9c5b4f17_5",
  "chunk_id": "e4f72d3a9c5b4f17_5",
  "document_id": "e4f72d3a9c5b4f17",
  "chunk_type": "page",
  "text": "## Page 194\n\nNational Classification of Occupations – 2015 Concordance Table\n\nNCO 2015 NCO 2004\n\n7523.0200 Wood Turner, Machine 7423.15\n\n7523.0300 Wood, Turner Hand 7423.20\n\n7523.0400 Shaper Wood 7423.25\n\n7523.0500 Router, Wood 7423.30\n\n7523.0600 Planer, Wood 7423.35\n\n7523.0700 Four Cutter 7423.40\n\n7523.0800 Wood Sawyer, Hand 7423.42\n\n7523.0900 Moulder, Wood 7423.45\n\n7523.1000 Mortiser Operator 7423.50\n\n7523.1100 Tennoning Machine Operator 7423.55\n\n7523.1200 Jointer Machine Operator 7423.60\n\n7523.1300 Driller Wood 7423.65\n\n7523.1400 Dowell Machine Operator 7423.70\n\n7523.1500 Wood Wool Machine Operator 7423.80\n\n7523.1600 Wood Carver, Machine 8241.10\n\n7523.1700 Fret Saw Machine Operator 8241.20\n\n7523.1800 Sander Operator 8241.30\n\n7523.9900 Wood Working Machine Setters and Setter 8241.90\n\nOperators, Others\n\nGroup 753 Garment and Related Trades Workers\n\nFamily 7531 Tailors, Dressmakers, Furriers and Hatters\n\n7531.0100 Tailor, General 7433.10\n\nVOLUME I 182",

  "embedding": [0.0234, -0.1457, 0.3782, 0.0923, -0.0567, ...],
  "source": {
    "file_type": "application/pdf",
    "source_path": "s3://mospi-data/data/raw/nco_2025_occupations.pdf",
    "page_number": 194,
    "time": [null, null],
    "line_range": null,
    "bbox": null
  },
  "metadata": {
    "timestamp": "2025-08-03T12:15:27Z",
    "tags": ["occupation", "tailor", "NCO", "classification"],
    "layout_tags": ["heading", "paragraph", "list"]
  }
}
```

---

#### 2. Entity Documents (`entities` collection)

```json
{
  "_key": "Q7531",
  "wikidata_id": "Q7531",
  "label": "Tailor (General)",
  "aliases": ["Seamstress", "Dressmaker", "Garment Maker", "Costume Maker"]
}
```

```json
{
  "_key": "Q1251134",
  "wikidata_id": "Q1251134",
  "label": "National Classification of Occupations",
  "description": "Standard classification of occupations in India"
}
```

#### 3. Edges Linking Chunk to Entities (`chunkEntityEdges` edge collection)

```json
{
  "_from": "chunks/e4f72d3a9c5b4f17_5",
  "_to": "entities/Q7531",
  "relation": "mentions"
}
```

```json
{
  "_from": "chunks/e4f72d3a9c5b4f17_5",
  "_to": "entities/Q1251134",
  "relation": "sourceClassification"
}
```

---

#### 4. Entity Relations (`entityRelationEdges` edge collection)

```json
{
  "_from": "entities/Q7531",
  "_to": "entities/Q1251134",
  "predicate": "classifiedBy"
}
```

```json
{
  "_from": "entities/Q7531",
  "_to": null,
  "predicate": "hasNCOCode",
  "object": "7531.0100"
}
```

---