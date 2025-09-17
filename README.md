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

```go
{
  "document_id": "doc-12345", // All: global document identifier (filename or UUID)
  "chunk_id": "chunk-0001", // All: unique chunk id (doc-scoped index or UUID)
  "chunk_type": "page", // pdf:page|page_subchunk, pptx:slides, txt:txt_subchunk, md:md_subchunk, jsonl:row_group, csv/html/wav:token_window, images:image
  "text": "Sample extracted text .", // text-bearing formats: extracted canonicalized text; wav/images may have ASR/OCR transcript or empty
  "token_count": 128, // txt,md,wav,html,csv,pdf,pptx,jsonl: tokenizer token count if computed
  "embedding": [0.12, -0.08, 0.44], // All: null before vectorization; numeric array after embedding
  "file_type": "application/pdf", // All: MIME type (e.g. application/pdf, audio/wav, text/plain, image/png, etc)
  "source_url": "s3://my-bucket/docs/report.pdf", // txt,md,jsonl,csv,html,wav: canonical URL (s3://... or https://...) of source object
  "page_number": 5, // pdf: integer page index (parser-defined base); null otherwise
  "slide_range": [1, 3], // pptx: [start_slide, end_slide] inclusive for chunk (replaces slide_range_start/slide_range_end)
  "row_range": [10, 20], // jsonl,csv: [start_row, end_row] inclusive for this chunk (replaces row_range_start/row_range_end)
  "token_range": [0, 127], // html,csv,jsonl: [token_start, token_end] token indices where window starts/ends (replaces token_start/token_end)
  "audio_range": ["00:00:05.000", "00:00:10.000"], // wav: [start_ts, end_ts] segment timestamps; null for non-audio
  "timestamp": "2025-09-14T12:34:56Z", // All: ISO8601 UTC when chunk created (e.g. 2025-09-14T12:34:56Z)
  "parser_version": "v1.2.3", // txt,md,jsonl,csv,html,pdf,pptx,wav: parser semantic version or tag
  "tags": ["financial", "confidential"], // pdf,pptx,images,html,general: producer tags for routing/classification
  "layout_tags": ["page"], // pdf,pptx,images,html: structural labels like ["page","slide","image","table"]
  "used_ocr": true, // pdf,images,pptx,md: whether OCR was applied
  "parse_chunk_duration_ms": 134, // All: integer ms spent parsing this chunk or null for wav
  "heading_path": ["Introduction", "Background"], // md: array representing nested heading hierarchy for the subchunk
  "headings": ["Introduction"], // md: flattened list of headings present inside chunk
  "line_range": [1, 20], // txt,md: [start_line, end_line] inclusive for subchunk (replaces line_range_start/line_range_end)
  "chunk_duration_ms": 5000, // for wav: audio chunk duration in ms since parsing is seperate
}

```

| Field                      | Description                                                                                                                                                                   | Indexed | Filterable |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----: | :--------: |
| document\_id               | Global document identifier (filename or UUID).                                                                                                                                |   Yes   |     Yes    |
| chunk\_id                  | Unique chunk identifier within document or global UUID.                                                                                                                       |   Yes   |     Yes    |
| chunk\_type                | Lexical label of chunk role (page, slide, txt\_subchunk, md\_subchunk, row\_group, token\_window, image).                                                                     |   Yes   |     Yes    |
| text                       | Canonical extracted plain text for text-bearing formats; ASR/OCR transcript for audio/image; empty string otherwise.                                                          |   Yes   |     No     |
| token\_count               | Integer token count from tokenizer when computed.                                                                                                                             |    No   |     Yes    |
| embedding                  | Null before vectorization; numeric vector after embedding for semantic search.                                                                                                |   Yes   |     No     |
| file\_type                 | MIME type of original source (application/pdf, audio/wav, text/plain, image/\*, text/markdown, application/x-ndjson, text/csv, text/html, application/vnd.openxmlformats...). |   Yes   |     Yes    |
| source\_url                | Canonical URL (s3:// or https\://) of source object (txt,md,jsonl,csv,html,wav).                                                                                              |    No   |     Yes    |
| snapshot\_path             | Path to parser snapshot used for provenance (txt,md,html); absent for streaming ingests.                                                                                      |    No   |     Yes    |
| text\_checksum             | SHA256 hex of chunk text for dedupe/integrity (txt,md,jsonl,csv,html,wav).                                                                                                    |   Yes   |     Yes    |
| page\_number               | PDF page index for chunk (parser-specific base).                                                                                                                              |    No   |     Yes    |
| slide\_range\_start        | PPTX starting slide index for chunk.                                                                                                                                          |    No   |     Yes    |
| slide\_range\_end          | PPTX ending slide index for chunk.                                                                                                                                            |    No   |     Yes    |
| row\_range\_start          | JSONL/CSV starting row number for this chunk.                                                                                                                                 |    No   |     Yes    |
| row\_range\_end            | JSONL/CSV ending row number for this chunk.                                                                                                                                   |    No   |     Yes    |
| token\_start               | Token index where window starts in document (html,csv,jsonl).                                                                                                                 |    No   |     Yes    |
| token\_end                 | Token index where window ends in document (html,csv,jsonl).                                                                                                                   |    No   |     Yes    |
| audio\_range\_start        | Audio segment start timestamp (ISO or seconds float) for wav.                                                                                                                 |    No   |     Yes    |
| audio\_range\_end          | Audio segment end timestamp for wav.                                                                                                                                          |    No   |     Yes    |
| timestamp                  | ISO8601 UTC when chunk was produced for recency and auditing.                                                                                                                 |   Yes   |     Yes    |
| parser\_version            | Semantic version or tag of the parser that produced the chunk.                                                                                                                |    No   |     Yes    |
| token\_encoder             | Tokenizer/encoder name used to compute token\_count (e.g. cl100k\_base).                                                                                                      |    No   |     No     |
| tags                       | Producer tags for routing/classification (array of strings).                                                                                                                  |   Yes   |     Yes    |
| layout\_tags               | Structural labels like \["page","slide","image","table"].                                                                                                                     |   Yes   |     Yes    |
| used\_ocr                  | Boolean indicating OCR was applied to generate text.                                                                                                                          |    No   |     Yes    |
| parse\_chunk\_duration\_ms | Integer milliseconds spent parsing this chunk for telemetry.                                                                                                                  |    No   |     No     |
| window\_index              | Sliding window index for token windows (html,csv,jsonl).                                                                                                                      |    No   |     No     |
| heading\_path              | Markdown: array representing nested heading hierarchy for subchunk.                                                                                                           |   Yes   |     Yes    |
| headings                   | Markdown: flattened list of headings present within the chunk.                                                                                                                |   Yes   |     Yes    |
| line\_range\_start         | TXT/MD starting line number for this subchunk.                                                                                                                                |    No   |     Yes    |
| line\_range\_end           | TXT/MD ending line number for this subchunk.                                                                                                                                  |    No   |     Yes    |
| chunk\_duration\_ms        | Duration of audio chunk in milliseconds (wav).                                                                                                                                |    No   |     Yes    |
\

## Component-Level Parsing & Chunking Strategy 

| Component               | Tool(s) Used                                                                      | Chunking Strategy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Rationale for Scalability & Fidelity                                                                                                                                                                                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PDF Parsing + OCR**   | `PyMuPDF`, `pdfplumber`, `pytesseract`, `RapidOCR`                                | - **1 page = 1 chunk** (primary granularity)<br>- Extract native text via `pdfplumber`, fallback to OCR if needed<br>- Use `OpenCV` for layout-aware OCR (cropping regions, columnar reading)<br>- Tables extracted as both **structured JSON** and **inline Markdown**<br>- Figures/diagrams extracted to a separate `figures` array                                                                                                                       | - Page-level chunking aligns with layout and enables **parallelism**<br>- Hybrid OCR improves coverage of low-quality scans<br>- Layout fidelity helps preserve **tables, headers, visual order**                                                                                                                |
| **DOC/DOCX Conversion** | `LibreOffice` (headless mode), `subprocess`, `tempfile`, `boto3`, `pathlib`, `os` | - Convert `.doc` → `.docx` → `.pdf` using `LibreOffice CLI`<br>- Apply **same PDF+OCR pipeline per page** on output PDF<br>- Page alignment maintained between original and converted formats                                                                                                                                                                                                                                                | - Avoids unreliable native `.docx` parsing<br>- Ensures **visual and semantic consistency** across systems<br>- Helps in tracing                                                                                                                                                                                |
| **HTML Parsing**        | `trafilatura`, `tiktoken`, `requests`, `boto3`                                    | - Extract article text + metadata with `trafilatura` → normalize Markdown<br>- Tokenize with `tiktoken` using `TOKEN_ENCODER`<br>- Generate fixed-size windows (`WINDOW_SIZE`, `OVERLAP_TOKENS`), step = `WINDOW_SIZE - OVERLAP_TOKENS`<br>- If ≤ `SPLIT_THRESHOLD`, emit single chunk `"{doc_hash}_1"`, else emit sequential `"{doc_hash}_n"` (1-indexed)                                                                                         | - Token windows give budget control and context continuity<br>- Deterministic chunk IDs enable reproducible joins<br>- Canonical text + token checksums ensure dedupe and idempotent ingestion                                                                                                                  |
| **CSV Chunking**        | `ray.data.read_csv`, `tiktoken`                                                   | `rows_per_chunk = clamp( floor((TARGET_TOKENS_PER_CHUNK - header_tokens) / max(1, sample_row_tokens)), MIN_ROWS_PER_CHUNK, MAX_ROWS_PER_CHUNK )`; accumulate rows until token budget exceeded; split oversized rows with `split_into_token_windows`                                                                                                                                            | Predictable token-bounded chunks; header included if it fits; oversized rows split deterministically                                                                                                                                                                                                            |
| **JSONL Chunking**      | `ray.data.read_json`, `pandas`, `tiktoken`, `boto3`                               | - Infer keys → `header_text` → compute `rows_per_chunk` like CSV<br>- Accumulate records until token budget exceeded<br>- Oversized rows split into deterministic token windows (`TARGET_TOKENS_PER_CHUNK`, 10% overlap)<br>- Chunk IDs: `"{doc_hash}_{n}"`                                                                                                   | - Token-bounded chunks ensure cost control<br>- Deterministic IDs + checksums give idempotence/dedupe<br>- `parse_chunk_duration_ms` gives per-chunk latency observability                                                                                              |
| **Audio Transcription** | `faster-whisper (CTranslate2)`, `pydub`, `ffmpeg-python`                          | - Audio sliced into 20–30s segments via silence detection (`pydub.silence`)<br>- Each segment transcribed individually                                                                                                                                                                                                                                                                                | - Faster-Whisper is CPU efficient<br>- Segmentation makes long audio scalable and parallelizable                                                                                                                                                                                                           |
\| **Markdown**            | `markdown-it-py` (parser), `tiktoken` (tokenizer)                                      | - Parse to heading sections (heading\_path) with `markdown-it-py` → normalized sections.<br>- Merge small adjacent sections up to `MD_MERGE_HEADER_THRESHOLD_TOKENS`.<br>- If section ≤ `MD_MAX_TOKENS_PER_CHUNK` → write `md_section` chunk; otherwise split into `md_subchunk` windows by tokens with overlap `MD_OVERLAP_TOKENS` (preserves heading\_path/headings).<br>- Long single lines are split into char-windows to respect token limits. | - Preserves logical Markdown structure (headings, code blocks) and attributes: `heading_path`, `headings`, `line_range`, `subchunk_index`, `token_count`, `token_encoder`, `parser_version` (snapshot optional). |
\| **PPTX (PowerPoint)**   | `python-pptx`, `Pillow`/`numpy`, optional OCR backends `pytesseract` / `rapidocr_onnxruntime` | - Extract per-slide content: text frames, tables (converted to markdown), images (image blob OCR attempted).<br>- OCR only used when needed or on image-only slides; image OCR uses MIN\_IMG\_BYTES threshold and OCR backend selection; `used_ocr` flag set when OCR contributed text.<br>- Slides grouped into chunks of `PPTX_SLIDES_PER_CHUNK` by default → chunk\_type `slides` with `slide_range` / `slide_range_start`/`slide_range_end` in source/fields. | n slides = 1 chunk, `parse_chunk_duration_ms`, `layout_tags: ["slide"]`, and `snapshot` option. |
\| **Images (Scans / Photos)** | `PIL/Pillow`, `OpenCV` preprocessing, OCR via `pytesseract` or `rapidocr_onnxruntime`         | - 1 image → 1 chunk (image-level extraction).<br>- Preprocess variants (upscale, denoise/sharpen, CLAHE, adaptive thresholding) then run OCR variants and choose best / fallback.<br>- Post-process lines: hyphen-fix, dedupe, filter by alnum ratio. | - Chunk fields: `chunk_type: image`, `file_type: image/*`, `used_ocr` boolean, `layout_tags: ["image"]`, `parse_chunk_duration_ms`, `text` contains OCR lines (if any). Good for scanned pages, forms, photos. |
\| **Plaintext Files (.txt)** | builtin read, canonicalize, `tiktoken` (optional)                                          | - Canonicalize text → split into lines preserving newline semantics.<br>- If total tokens ≤ `TXT_MAX_TOKENS_PER_CHUNK` → single `txt_section` chunk; otherwise produce `txt_subchunk` windows by aggregating lines until token limit, with overlap `TXT_OVERLAP_TOKENS`.<br>- Extremely long single lines are split into char-windows with subchunk indices. | - Lightweight; produces `line_range`, `subchunk_index`, `token_count`, `token_encoder`, `parser_version`, optional `.snapshot.txt` upload. Works well for logs, transcripts, scraped text. |


</details>
---

* **Embedding generation:** dense vector creation using **gte-modernbert-base** (ONNX) or appropriate sentence/embed models.  

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
---
# Get started with RAG8s

## Prerequesities
 1. Docker enabled on boot and is running
 2. Vscode with `Dev Containers` extension installed
 3. AWS root account or IAM user with admin access for S3, EKS, and IAM role management(free tier sufficient if trying RAG8s locally)

## STEP 0/3 environment setup

#### Clone the repo and build the devcontainer
```sh 
git clone https://github.com/Athithya-Sakthivel/RAG8s.git && cd RAG8s && code .
ctrl + shift + P -> paste `Dev containers: Rebuild Container` and enter
```

#### This will take 20-30 minutes. If the image matches your system, you are ready to proceed.
![alt text](.devcontainer/env_setup_success.png)

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
#### Create a private repo in your gh account
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

---

## STEP 2/3 - indexing_pipeline


### Export the neccessary configs.

```sh


export PYTHONUNBUFFERED=1                             # To display logs 
export S3_BUCKET=e2e-rag-system-42                    # Set any globally unique complex name, Pulumi S3 backend -> s3://$S3_BUCKET/pulumi/
export S3_RAW_PREFIX=data/raw/                        # raw ingest prefix (change to isolate datasets)
export S3_CHUNKED_PREFIX=data/chunked/                # chunked output prefix (change to separate processed data)
export CHUNK_FORMAT=json                              # 'json' (readable) or 'jsonl' (stream/space efficient)
export OVERWRITE_DOC_DOCX_TO_PDF=true                 # true to delete and replace docx with PDF, false to keep the originals
export OVERWRITE_ALL_AUDIO_FILES=true                 # true to delete and replace .mp3, .m4a, .aac, etc as .mav 16khz, false to keep the originals
export OVERWRITE_SPREADSHEETS_WITH_CSV=true           # true to delete and replace .xls, .xlsx, .ods, etc as .csv files, false to keep the originals
export OVERWRITE_PPT_WITH_PPTS=true                  # true to delete and replace .ppt files as .pptx, false to keep the originals
export PDF_PAGE_THRESHOLD=1800                   # Threshold to detect large pages and split them into subchunks 
export PDF_WINDOW_SIZE=600                       # Default is page wise chunking, for large page 600 tokens per chunk with 10% token overlap
export PDF_DISABLE_OCR=false                              # true to skip OCR (very fast) or false to extract text from images
export PDF_OCR_ENGINE=rapidocr                            # 'tesseract' (faster) or 'rapidocr' (high accuracy , slightly slower)
export PDF_FORCE_OCR=false                                # true to always OCR(use if source text unreliable but not recommended for scaling)
export PDF_OCR_RENDER_DPI=400                             # increase for detecting tiny text; lower for speed/cost
export PDF_MIN_IMG_SIZE_BYTES=3072                        # ignore images smaller than 3KB (often unneccessary black images)
export IMAGE_OCR_ENGINE=rapidocr                          # or 'tesseract' for image formats .png, .jpeg, .jpg, .tiff, .webp
export HTML_WINDOW_SIZE=600                      # Default is page wise chunking, for large page 800 tokens per chunk with 10% token overlap
export CSV_TARGET_TOKENS_PER_CHUNK=400           # Increase if very large .csv or Decrease if higher precision required
export JSONL_TARGET_TOKENS_PER_CHUNK=600         # Increase if very large .jsonl or Decrease if higher precision required
export MD_MAX_TOKENS_PER_CHUNK=600               # Threshold for split headers in header wise chunking with 10% overlap
export MD_MERGE_HEADER_THRESHOLD_TOKENS=400      # Threshold to cummulatively merge small headers with their next header(s) till MD_MAX_TOKENS_PER_CHUNK
export AUDIO_SLICE_SECONDS=30                    # Audio slices in seconds with 10% overlap. Increase or decrease based on AUDIO_MAX_TOKENS_PER_CHUNK
export AUDIO_MAX_TOKENS_PER_CHUNK=600            # Limit to cummulatively merge text from audio slices with next audio slices
export TXT_MAX_TOKENS_PER_CHUNK=600              # Simple token based chunking with 10% overlap. Increase for cost or decrease for precision
export PPTX_SLIDES_PER_CHUNK=5                   # Number of slides per chunk. Increase for cost or decrease for precision
export PPTX_OCR_ENGINE=rapidocr                  # 'tesseract' (faster), 'rapidocr' (high accuracy , slightly slower)


```



# infra 

```sh

export QDRANT_PRIVATE_IP="10.0.1.10"          # Deterministic private IP for Qdrant that only ray clusters can access
export QDRANT_INSTANCE_TYPE="t3.medium"       # EC2 instance type for Qdrant


export PULUMI_CONFIG_PASSPHRASE=mypassword    # For headless automation

export AWS_ACCESS_KEY_ID="AKIA.."
export AWS_SECRET_ACCESS_KEY=""
export AWS_REGION="ap-south-1"               # AWS region to create resources
export S3_BUCKET=e2e-rag-system-42           # Set any globally unique complex name, Pulumi S3 backend -> s3://$S3_BUCKET/pulumi/
export STACK="prod"                          # Any name for pulumi stack

export MY_SSH_CIDR="203.0.113.42/32"                       # operator SSH CIDR (single IP recommended)
export PUBLIC_SUBNET_CIDRS="10.0.1.0/24,10.0.2.0/24"       # comma-separated public subnets
export VPC_CIDR="10.0.0.0/16"                             # VPC range



# Retrieval fusion weights (tune by devset; relative importance)
export W_VEC=0.6                                      # range: 0.0-1.0; raise if domain embeddings are highly accurate
export W_BM25=0.3                                     # range: 0.0-1.0; raise if exact keyword matches are critical
export W_GRAPH=0.1                                    # range: 0.0-1.0; raise if graph/triplet hits are very high precision

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
* Optimized for ONNX export and CPU-inference(max_length only 50-100). Embedding generation in indexing pipeline is GPU based(for large corpora)
* Embedding dimension: **768**
* Parameter size: **149M**

🔗 [https://huggingface.co/Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)

---

### 🔹 **\[2] Qwen3-4B-AWQ**

A compact, high-throughput **instruction-tuned LLM** quantized using **AWQ**. Built on **Qwen3-4B**, this variant supports **32,768-token context** natively and achieves performance comparable to models 10× its size (e.g., Qwen2.5-72B). Optimized for **SGLang inference**, it balances **speed, memory efficiency, and accuracy**, running seamlessly on GPUs like A10G, L4, and L40S.

* Architecture: **Transformer** (Qwen3 series, multilingual)
* Context Length: **32k tokens**
* Quantization: **AWQ** 
* VRAM Usage: **\~4.8–5.2 GiB** (fits on 24 GiB GPUs with headroom)

🔗 [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)

> “Even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.”
> — [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
> — [Thinking-mode](https://qwenlm.github.io/blog/qwen3/#key-features)

> **Use case**: Smaller models (e.g., Qwen3-4B-AWQ or 30B-A3B) **fit on a single VM** , making them better suited for data-parallel engines like **SGLang**  than tensor-parallel engine like **vLLM**.

---

