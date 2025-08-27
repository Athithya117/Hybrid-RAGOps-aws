## **RAG8s** is a **production-ready** e2e RAG platform with hybrid retrieval (vector + keyword + graph), GeAR multihop reasoning engineered for high throughput, low-latency retrieval, and adaptive scaling. It is organized into **three main components**—`infra/`, `indexing_pipeline/`, and `inference_pipeline/`—each responsible for a distinct set of concerns (platform, data ingestion/indexing, and online query serving + evaluation).

---

## 1) infra/  (Infrastructure & Platform)

**Provides the cloud-native foundation that runs and scales the system. **It's currently aws native** but is extensible to other cloud like Azure and GCP**

**Core responsibilities**
* Fully self hosted infra: **EKS (Kubernetes)**, **Karpenter** for CPU/GPU/spot node pools, autoscaling.  
* Distributed serving & batch: **KubeRay** (RayService, RayJob).  
* IaC & provisioning: **Pulumi**, **AWS Image Builder** (model paths, AMI IDs).  
* GitOps & packaging: **ArgoCD**, **Helm** charts (per-env `values.kind.yaml` / `values.eks.yaml`).  
* Networking & access: **Traefik** ingress, **Cloudflare** DNS, optional **OIDC/Keycloak** for federated auth, **gRPC + rayserve** endpoints.

**Storage & utilities**
* **S3** for raw data, chunk caches, backups; presigned URLs for secure access.  
* **jsonlines / .jsonl** for stage logs/audits and replay.

**Observability & evaluation**
* **Prometheus** + **Alertmanager** → TSDB, alerting.  
* **Grafana** → dashboards and SLO/SLA visualization.  
* **OpenTelemetry Collector** (DaemonSet) → unified pipeline for metrics, logs, traces.  
* **OpenLLMetry** → online + offline evaluation, tracing, guardrails, experiment management (self-hosted in cluster).  

**CI/CD**
* **Makefile** → lint, tests, Docker builds, Helm validation.  
* **ArgoCD** → declarative sync of manifests into cluster.

---

## 2) indexing_pipeline/  (Indexing & Ingestion)

**Ingests raw sources, extracts structured content, produces chunked artifacts and embeddings, and populates graph/vector stores**

* **Parsing:** document ingestion with **PyMuPDF**, **pdfplumber**, **tesserocr/RapidOCR** (OCR), **faster-whisper** (audio), **BeautifulSoup/extractous** (HTML).  
* **Chunking & Preprocessing:** page-level + layout-aware segmentation, silence-based audio slicing, JSON flattening, deduplication, tag-structured HTML/Markdown chunking. 

<details>
  <summary> View chunk stratergies and chunk schema (Click the triangle)</summary>

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy

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
---

* **Embedding generation:** dense vector creation using **gte-modernbert-base** (ONNX) or appropriate sentence/embed models.  
* **Triplet extraction / graph augmentation:** run entity/triplet extractors (e.g., `relik-cie-tiny`) at index time to populate `triplets` and `entity_graph`.  
* **Storage:** store embeddings + metadata in **ArangoDB (with FAISS integration)** for hybrid dense+graph retrieval; persist raw artifacts to S3.  
* **Orchestration & scaling:** RayJobs for parallel ingestion and indexing; cronjobs for backups (arangobackup → S3); modular Docker images for CPU/GPU runtime.

---

## 3) inference_pipeline/  (Retrieval, Reasoning, Generation & Evaluation)

Serves queries end-to-end: retrieval (vector + keyword + graph), multi-hop reasoning, reranking, optional generation, and evaluation/guardrails.

**Model & retrieval primitives**
* Embedding encoder: **gte-modernbert-base** (ONNX).  
* Reranker: **gte-reranker-modernbert-base** (ONNX) — optional cross-encoder reranking.  
* Generator (optional): **Qwen / Qwen3-4b-awq** (quantized model) served via RayService/gRPC.  
* Graph & vector store: **ArangoDB + FAISS**.  
* Cache/fallback store: **ValKeye** (Redis-compatible) for rate limits and LLMLessMode fallback.

**Primary workflow (concise)**
1. **Query intake & normalization** — text normalization, filter parsing.  
2. **Hybrid retrieval** — FAISS vector search + ArangoSearch (BM25) + ArangoDB triplets.  
3. **GeAR multihop** — entity graph traversal (2–3 hops) to surface path-based evidence for multi-hop questions.  
4. **Merge & dedupe** — canonicalize chunks, cluster near-duplicates, normalize scores.  
5. **Ranking** — weighted combination: `vector + bm25 + graph + GeAR_multihop`.  
6. **Answer assembly:**  
   * **If LLM available:** apply reranker, then generate concise NL answer.  
   * **If LLM unavailable (LLM-less fallback):** return deterministic outputs (structured JSON + template text) with full provenance.

**LLM-less fallback (llmless mode) — succinct**
* Deterministic, auditable answers built only from retrieved chunks & triplets; no generative invention.  
* Returns (a) **Structured JSON** (chunks, triplets, graph paths, provenance) and (b) **Concise deterministic text** using templates that cite `(filename.pdf, p.N)`.  
* Ensures graceful degradation and deterministic reproducibility when generation is unavailable.



---

<details>
<summary> RAG8s tree structure/codebase layout for quick overview </summary>

```sh
RAG8s/
├── data/                                 # Local directory that syncs with s3://<bucket_name>/data
│   ├── raw/                              # Raw document files 
│   └── chunked/                          # Chunks in json/jsonl format
│
├── indexing_pipeline/
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
│   └── relik.sh                          # Helper script to run ReLiK entity/triplet extraction
│
├── inference_pipeline/
│   ├── auth_control.py                   # Auth 2.0, authorization middleware and rate limiting for APIs  # observe: logs, metrics
│   ├── eval.py                           # OpenLLMetry coherence checks, hit@K monitoring, hallucination detection  # observe: logs, metrics
│   ├── main.py                           # Inference service entrypoint (REST/gRPC server)  # observe: logs, metrics, traces
│   ├── llm_retrieval.py        # Retrieval orchestration (hybrid BM25 + vector + graph + GeAR lightweight multihop)  # observe: logs, metrics
│   ├── llmless_retrieval.py              # Returns only raw chunks/triplets and source urls from arangodb when rate limit exceeded
│   └── trace_file.py             # View or download Presigned urls for the raw docs as source link s3://<bucket_name>data/raw/<file_name>.<format>
|
├── infra/
│   ├── karpenter-nodepool-cpu/                         
│   │   ├── Chart.lock                        # Helm lockfile: pins exact versions of dependencies (auto-generated)
│   │   ├── Chart.yaml                        # Helm chart metadata & dependencies (kuberay, kube-prometheus-stack)
│   │   ├── charts/
│   │   │   ├── kube-prometheus-stack-76.4.0.tgz   # Vendored Helm chart for monitoring stack (Prometheus, Grafana, Alertmanager)
│   │   │   └── kuberay-operator-1.4.2.tgz         # Vendored Helm chart for Ray cluster/operator
│   │   ├── images/
│   │   │   ├── frontend/
│   │   │   │   ├── Dockerfile                 # Build recipe for RAG frontend container
│   │   │   │   ├── main.py                    # Python app entrypoint for UI; imports deps from requirements-cpu.txt
│   │   │   │   └── cpu-requirements-cpu.txt       # Frontend Python dependencies; installed inside Dockerfile
│   │   │   ├── indexing_rayjob/
│   │   │   │   ├── Dockerfile                 # Build recipe for Ray indexing job container
│   │   │   │   └── cpu-requirements.txt       # Python dependencies for indexing Ray job
│   │   │   └── onnx-embedder-reranker/
│   │   │       ├── Dockerfile                 # Build recipe for ONNX embedder+reranker model server
│   │   │       ├── download_hf.py             # Script to pull HuggingFace models at build/runtime
│   │   │       ├── grpc.proto                 # Protobuf service definition for embedder/reranker API
│   │   │       ├── rayserve_embedder_reranker.py  # RayServe deployment code for ONNX embedder & reranker
│   │   │       ├── rayserve_entrypoint.py     # Entrypoint to launch RayServe app; imports above module
│   │   │       ├── cpu-requirements.txt       # Python dependencies for embedder/reranker service
│   │   │       └── run_and_test_local.sh      # Local test script to validate container works pre-deploy
│   │   ├── rendered-kind.yaml                 # Rendered manifest snapshot for local testing (kind cluster)
│   │   ├── templates/
│   │   │   ├── _helpers.tpl                   # Helm template helpers (common labels, naming functions)
│   │   │   ├── graphana-datasource.yaml       # Configures Grafana datasource (Prometheus + OTEL); depends on values.yaml
│   │   │   ├── karpenter-nodepool.yaml        # Defines Karpenter NodePool/Provisioner for autoscaling CPU nodes
│   │   │   ├── namespace.yaml                 # Creates namespace for this chart; all resources scoped here
│   │   │   ├── networkpolicy.yaml             # NetworkPolicies: secure traffic between frontend, Ray, DB
│   │   │   ├── otel-collector-configmap.yaml  # ConfigMap for OpenTelemetry Collector pipelines (logs, metrics, traces)
│   │   │   ├── otel-collector-daemonset.yaml  # DaemonSet: deploys OTEL Collector agent to all nodes
│   │   │   ├── otel-collector-service.yaml    # Service exposing OTEL Collector for scraping/export
│   │   │   ├── pdbs.yaml                      # PodDisruptionBudgets to ensure availability during upgrades
│   │   │   ├── prometheus-pvc.yaml            # PersistentVolumeClaim for Prometheus TSDB; depends on values.eks.yaml
│   │   │   ├── pvc.yaml                       # PVCs for stateful workloads (embedding/reranker cache, frontend state)
│   │   │   ├── rayservice.yaml                # RayService definition for embedding + reranker workloads
│   │   │   ├── role.yaml                      # RBAC Role for workloads needing cluster-scoped API access
│   │   │   ├── rolebinding.yaml               # RoleBinding linking Role to service account
│   │   │   ├── service-monitor.yaml           # ServiceMonitor CRD for Prometheus to scrape this app’s metrics
│   │   │   ├── service.yaml                   # Kubernetes Services exposing frontend & model containers
│   │   │   ├── serviceaccount.yaml            # ServiceAccount for pods; referenced in RoleBinding and workloads
│   │   │   └── templates/
│   │   │       └── grafana-dashboard-indexing.json  # JSON Grafana dashboard for indexing/embedding job metrics
│   │   ├── values.eks.yaml                    # Helm values for production EKS deployment (CPU/GPU node config, storage)
│   │   └── values.yaml                        # Default Helm values (resource requests, OTEL config, monitoring toggles)
│   ├── karpenter-nodepool-gpu/                # Sglang GPU based similar deployment
│   ├── statefulset-nodegroup/                 # EKS nodegroup for stable statefulsets like arangodb , valkey
│   └── pulumi-aws/                            
│       ├── config.py                 # Global variables & Pulumi config
│       ├── vpc.py                    # Networking must exist before cluster
│       ├── iam_roles_pre_eks.py      # IAM roles required to create EKS
│       ├── eks_cluster.py            # EKS cluster depends on VPC + pre-EKS IAM
│       ├── nodegroups.py             # Nodegroups depend on cluster + IAM
│       ├── iam_roles_post_eks.py     # IAM roles for workloads (Ray, Karpenter, Valkeye)
│       ├── karpenter.py              # Karpenter provisioning depends on cluster + nodegroups + IAM
│       ├── cloudflare.py             # DNS records, depends on cluster endpoint
│       ├── indexing_ami.py           # Indexing AMIs, depends on cluster/nodegroups
│       ├── inference_ami.py          # Inference/GPU AMIs, depends on cluster/nodegroups
│       ├── db_backup.py              # CronJobs or backup jobs, depends on DB running in cluster
│       ├── ___main__.py                  # Orchestrates imports & execution
│       └── pulumi.yaml                   # Pulumi project manifest for infra code
|
└── utils/                                   
|    ├── archive/                           # Files no longer maintained
|    ├── bootstrap-dev.sh                   # Installs all the required tools for development and testing
|    ├── bootstrap-prod.sh                  # Installs minimal tools for prod
|    ├── force_sync_data_with_s3.py         # sync/force sync the data/ in local fs/ with s3://<bucket_name>/data/
|    ├── lc.sh                              # Local kind cluster for testing rag8s
|    └── s3_bucket.py                       # Create/delete s3 bucket 
|    
├── scripts/
│   ├── build_and_push.sh                   # Builds container images and pushes to registry
│   ├── dynamic-values.yaml.sh              # Generates dynamic Helm values (env-specific)
│   ├── helm-deploy.sh                      # Wrapper to deploy Helm charts via CI or locally
│   ├── pulumi-set-configs.sh               # Sets Pulumi configuration and secrets
│   └── pulumi-set-secret.sh                # Stores secrets into Pulumi secret store
│
├── .devcontainer/
│   ├── Dockerfile                          # Devcontainer image build for local development environment
│   └── devcontainer.json                   # VS Code devcontainer configuration (mounts, settings)
|
├── .dockerignore                           # Files/dirs excluded from Docker build context
├── .gitignore                              # Git ignore rules
├── Makefile                                # Convenience targets for build/test/deploy tasks
├── README.md                               # Project overview, setup and usage instructions
└── backups/                                # Local directory that syncs with s3://<bucket_name>/backups
    └── dbs/
        └── arrangodb/

```
</details>

---

# Get started with RAG8

### Prerequesities
 1. Docker enabled on boot and is running
 2. Vscode with `Dev Containers` extension installed
 3. AWS root account or IAM user with admin access for S3, EKS, and IAM role management(free tier sufficient if trying RAG8s locally)

### STEP 0/3 environment setup

#### Clone the repo and build the devcontainer
```sh 
git clone https://github.com/Athithya-Sakthivel/RAG8s.git && cd RAG8s && code .
ctrl + shift + P -> paste `Dev containers: Rebuild Container` and enter
```

#### This will take 20-30 minutes. If the image matches your system, you are ready to proceed.
![alt text](.devcontainer/dev_setup_success.png)

#### Open a new terminal and login to your gh account
```sh
git config --global user.name "Your Name" && git config --global user.email you@example.com
gh auth login

? What account do you want to log into? GitHub.com
? What is your preferred protocol for Git operations? SSH
? Generate a new SSH key to add to your GitHub account? No
? How would you like to authenticate GitHub CLI? Login with a web browser

! First copy your one-time code: <code>
- Press Enter to open github.com in your browser... 
✓ Authentication complete. Press Enter to continue...

```
### Create a private repo in your gh account
```sh
export REPO_NAME="rag8s"

git remote remove origin 2>/dev/null || true
gh repo create "$REPO_NAME" --private >/dev/null 2>&1
REMOTE_URL="https://github.com/$(gh api user | jq -r .login)/$REPO_NAME.git"
git remote add origin "$REMOTE_URL" 2>/dev/null || true
git branch -M main 2>/dev/null || true
git push -u origin main
git pull
git remote -v
echo "[INFO] A private repo '$REPO_NAME' created and pushed. Only visible from your account."

```


## STEP 2/3 - indexing_pipeline


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

# Arango / vector index toggles
export ARANGO_VECTOR_INDEX_ENABLE=true                # range: true|false; false to disable vector ops (read-only or minimal infra)
export ARANGO_VECTOR_INDEX_TYPE="ivf"                 # range: 'hnsw'|'ivf'|'ivf+pq'; choose 'hnsw' (<100k docs), 'ivf' (>=100k), 'ivf+pq' for huge corpora
export ARANGO_VECTOR_INDEX_MAX_MEMORY_MB=2048         # range: 512-65536 MB; soft cap for index memory on node; increase with corpus size

# IVF-specific (only if using ivf)
export ARANGO_VECTOR_INDEX_IVF_NLIST=1000             # range: 256-16384; set ~sqrt(N_vectors); increase for very large corpora
export ARANGO_VECTOR_INDEX_IVF_NPROBE=10              # range: 4-128; raise for recall at cost of latency

# PQ (only if using ivf+pq/pq)
export ARANGO_VECTOR_INDEX_PQ_M=16                    # range: 8-32; PQ segments; must divide embedding dim; tune for memory vs accuracy

# HNSW-specific (only if using hnsw)
export ARANGO_VECTOR_INDEX_HNSW_M=32                  # range: 16-64; higher => more memory but higher recall
export ARANGO_VECTOR_INDEX_HNSW_EFCONSTRUCTION=200    # range: 100-800; raise for better index build quality
export ARANGO_VECTOR_INDEX_HNSW_EFSEARCH=50           # range: 40-300; raise for higher query recall (latency ↑)

# FAISS sidecar / local index
export FAISS_INDEX_PATH="/mnt/faiss/index.ivf"        # range: filesystem path|"empty"; local index path (empty if not used)
export FAISS_INDEX_DIM=768                            # range: embedding dim; must match embedding model output
export FAISS_NLIST=256                                # range: 128-16384; local FAISS nlist; increase for large indices
export FAISS_NPROBE=10                                # range: 1-128; raise for recall at latency cost

# Retrieval fusion weights (tune by devset; relative importance)
export W_VEC=0.6                                      # range: 0.0-1.0; raise if domain embeddings are highly accurate
export W_BM25=0.3                                     # range: 0.0-1.0; raise if exact keyword matches are critical
export W_GRAPH=0.1                                    # range: 0.0-1.0; raise if graph/triplet hits are very high precision
export W_RERANK=0.5                                   # range: 0.0-1.0; meaningful only when reranker enabled

# Candidate fanout & GeAR
export N_VEC=15                                       # range: 5-100; top-K vector candidates (raise for recall on large corpora)
export N_BM25=15                                      # range: 5-100; top-K BM25 candidates
export N_GRAPH=5                                      # range: 1-10; graph neighbor limit (keep small to control DB load)
export MAX_GEARS_HOPS=1                               # range: 1-2; 1 default; enable 2 behind feature flag for deeper multi-hop
export GEAR_BEAM_WIDTH=3                              # range: 1-5; beam width for GeAR expansion; increase with caution

# Pre-fusion thresholds (filters to reduce noise)
export VEC_SCORE_THRESH=0.20                          # range: 0.05-0.40; min vector similarity to keep a candidate (raise for precision)
export BM25_SCORE_THRESH=1.50                         # range: 0.5-3.0; min BM25 to keep (raise to filter weak keyword hits)
export GRAPH_SCORE_THRESH=0.0                         # range: 0.0-0.5; min graph edge confidence (set >0 if confidences provided)

# Reranker & metadata boosting
export USE_RERANKER=true                              # range: true|false; enable only if you accept added latency/cost for higher precision
export RERANK_BATCH_SIZE=16                           # range: 4-64; increase to amortize GPU/CPU when latency allows
export META_BOOST_FIELD="timestamp"                   # range: metadata key name; metadata key to bias ranking (e.g., timestamp, source_score)
export META_BOOST_WEIGHT=0.20                         # range: 0.0-1.0; raise if metadata should strongly affect ranking

# Timeouts / concurrency / performance
export RETRIEVAL_TIMEOUT=5                            # seconds; increase if backing systems are slower
export RETRIEVAL_BATCH_SIZE=4                         # parallelism for retrieval calls; increase with CPU/network capacity
export MAX_CONCURRENT_QUERIES=32                      # throttle to protect DBs; scale with infra
# Arango general performance / logging
export ARANGO_STORAGE_CACHE_SIZE=2048                 # set ~20-30% host RAM for read-heavy nodes
export ARANGO_QUERY_MEMORY_LIMIT=1024                 # raise if AQL traversals need more memory

# Observability / app logging
export APP_LOG_LEVEL="info"                           # 'debug' only temporarily for troubleshooting

```



```sh

User query
    │
    ├─→ arangosearch → candidates
    ├─→ Vector search → semantically similar candidates
    └─→ Graph (ArangoDB) + GeAR → multi-hop linked candidates
            ↓
        Candidate merger & reranker (optional)
            ↓
        LLM generation / RAG output

Graph traversal → “Follow the edges and return everything connected.”

GeAR reasoning → “Use the graph + triplets to infer which paths and nodes are actually relevant to the query, even if not directly connected.”
FAISS handles “meaning in text,” GeAR handles “meaning in structure.” Both are needed for a hybrid RAG.

```


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

---

### 🔹 **\[2] gte-reranker-modernbert-base(Optional)**

* **Cross-encoder reranker** for re-ranking retrieved docs
* High BEIR benchmark score (**nDCG\@10 ≈ 90.7%**)
* Same architecture & size as embedding model (149M), supports **8192 tokens**
* Very fast CPU inference with ONNX (FlashAttention 2)

🔗 [https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)

---

### 🔹 **\[3] ReLiK-CIE-small(For precomputing triplets, not deployed)**

A compact and efficient **entity + relation extraction** model designed for **Graph-RAG pipelines**. Unlike fast entity-only models (e.g., SpEL, ReFinED), `relik-cie-small` can extract both **named entities** and **semantic triplets** (`(head, relation, tail)`), enabling direct construction of **knowledge subgraphs** from raw text.

* Extracts **entities and triplets** in a single pass
* Balanced for **accuracy and runtime performance**

🔗 [relik-ie/relik-cie-small](https://hub.docker.com/r/sapienzanlp/relik#docker-images)

---

### 🔹 **\[4] Qwen3-4B-AWQ**

A compact, high-throughput **instruction-tuned LLM** quantized using **AWQ**. Built on **Qwen3-4B**, this variant supports **32,768-token context** natively and achieves performance comparable to models 10× its size (e.g., Qwen2.5-72B). Optimized for **SGLang inference**, it balances **speed, memory efficiency, and accuracy**, running seamlessly on GPUs like A10G, L4, and L40S.

* Architecture: **Transformer** (Qwen3 series, multilingual)
* Context Length: **32k tokens**
* Quantization: **AWQ** 
* VRAM Usage: **\~4.8–5.2 GiB** (fits on 24 GiB GPUs with headroom)

🔗 [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)

> “Even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.”
> — [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
> — [Thinking-mode](https://qwenlm.github.io/blog/qwen3/#key-features)

> **Use case**: Smaller models (e.g., Qwen3-4B-AWQ or 8B) **fit on a single VM** , making them better suited for data-parallel engines like **SGLang**, rather than tensor-parallel engine like **vLLM**.

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












