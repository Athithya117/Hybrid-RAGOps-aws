# Get started
## Prerequisites
1. **Docker installed, enabled on boot, and running**
2. **Visual Studio Code with the Dev Containers extension installed (for a deterministic environments): [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers)**
3. **An Azure subscription with sufficient permissions (Owner or Contributor) to manage**:
   * Azure Resource Groups
   * Azure Kubernetes Service (AKS)
   * Azure Storage Accounts (Blob)
   * Managed Identities and role assignments
**A free trial or Azure for students subscription is sufficient for development and testing purposes**
4. **A Cloudflare account with a registered domain, with permissions to manage DNS records and create Cloudflare Tunnels (cloudflared)**

## Clone the repo and build the devcontainer(Reproducible). This will take 20-30 minutes. 
```sh 
cd $HOME && rm -rf RAG8s && git clone https://github.com/Athithya-Sakthivel/RAG8s.git && cd RAG8s && code .
```
> ctrl + shift + P -> paste `Dev containers: Rebuild Container Without Cache` and enter

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
export REPO_NAME="rag8s" # or any name

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

#### Run `az login` and export the correct subscription ID

```sh
export AZURE_SUBSCRIPTION_ID="" # Azure subscription hosting AKS, storage, and all RAG platform resources
```

### STEP 1: Provision the Azure Storage Account and required Blob containers by exporting the variables below and running make create-sa (remove with make delete-sa).

```sh
export AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}"   # Azure subscription ID (must already be set after az login)
export AZURE_RESOURCE_GROUP_NAME="rg-e2e-rag"             # Resource group hosting storage account and platform infra
export AZURE_ENDPOINT_SUFFIX="core.windows.net"           # Override this only for sovereign clouds (e.g., Azure China or Gov)
export AZURE_LOCATION="eastus"                            # Azure region for all Azure resources
export AZURE_STORAGE_ACCOUNT_NAME="defaultsa515"          # Azure Storage Account name (3–24 chars, lowercase, globally unique)
export STORAGE_TIER="LRS"                                 # Storage redundancy tier (LRS | ZRS | GRS | RAGRS | GZRS | RAGZRS)
export AZURE_CONTAINER="rag-data-515"                     # Primary Blob container for RAG data
export PULUMI_AZ_CONTAINER="pulumi-state-515"             # Pulumi backend container (state + locking)
export BACKUP_AZ_CONTAINER="backups-515"                  # Backup Blob container (snapshots, archives)
export BACKUP_PREFIX="qdrant/backup"                      # Subdirectory inside backup container (e.g., backups/qdrant/)
export BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS=7              # Move backup blobs to Cool tier after N days
export BACKUP_AZ_CONTAINER_RETENTION_DAYS=30              # Permanently delete backup blobs after N days
export AZURE_STORAGE_CONNECTION_STRING="$(python3 infra/base_infra/get_storage_conn_string.py)"
# make create-sa
export AZURE_DELETE_ACCOUNT=0                             # 1 = delete entire storage account on `make delete-sa`, 0 = containers only
export FORCE_DELETE=1                                     # Skip interactive confirmation prompts
# make delete-sa

```

### STEP 2: Manage platform infrastructure with Pulumi: export all required environment variables, run `make pulumi-preview` to inspect changes, `make pulumi-up` to apply them, or `make pulumi-destroy` to delete resources (destructive; requires PULUMI_FORCE_DESTROY=1). You may refer docs/infra/azure.

```sh
export PULUMI_STACK="staging"                     # Pulumi state scope (infra boundary); PROD: "prod"
export PULUMI_CONFIG_PASSPHRASE="mypassword"      # Pulumi secrets encryption; PROD: strong rotated secret
export AKS_MAX_PODS=60                            # Pod density per node (Azure CNI IP pressure); PROD: raise if subnet allows
export AKS_CLUSTER_NAME="rag-aks"                 # AKS cluster name; change only for parallel clusters/environments
export AKS_SKU="standard"                         # Control-plane SLA tier; PROD: standard (or premium if regulated)
export SYSTEM_NODE_COUNT=1                        # System pool (kube-system, CNI, CoreDNS); PROD: >=2 (prefer 3)
export SYSTEM_NODE_VM_SIZE="Standard_B2s"         # System pool VM (infra-only); PROD: D4s_v5+ for stability
export SYSTEM_NODE_MAX_PODS=60                    # System pod density; must align with AKS_MAX_PODS
export BALANCED_NODE_MIN=0                        # General app pool (APIs, gateways, orchestrators); PROD: >=2 for HA
export BALANCED_NODE_MAX=1                        # General app scale ceiling; raise with QPS/latency targets
export BALANCED_NODE_VM_SIZE="Standard_B2s"       # App/API workloads; PROD: D4s_v5 for concurrency
export CPU_HEAVY_NODE_MIN=0                       # CPU model pool (embeddings, rerankers, tokenizers); PROD: >=1 if hot path
export CPU_HEAVY_NODE_MAX=0                       # CPU burst capacity (batch inference, indexing); PROD: raise as needed
export CPU_HEAVY_NODE_VM_SIZE="Standard_B2s"      # CPU inference placeholder; PROD: F8s_v2 (AVX2, predictable clocks)
export QDRANT_NODE_COUNT=0                        # Vector DB pool (Qdrant, HNSW, WAL); PROD: 1+ for HA/sharding
export QDRANT_NODE_VM_SIZE="Standard_B2s"         # RAM/IO-heavy vector storage; PROD: E8ds_v5 / E16ds_v5
export AKS_LOCATION="${AZURE_LOCATION}"   # Deployment region; change only for latency or quota management
export ACR_NAME=acr49250                                   # Global ACR name; change only if creating a new registry (cannot rename)
export ACR_REPO_PREFIX=rag                              # Logical repo namespace; change when multiple teams/apps share one ACR
export ACR_LOCATION="${AKS_LOCATION:-${AZURE_LOCATION:-eastus}}"  # Region; change only if co-locating with AKS or for compliance
export ACR_SKU=Basic        # All RAG8s images fits into the Basic plan 10GiB. Set `Premium` if geo-replication/private endpoints required.
# make pulumi-up
export PULUMI_FORCE_DESTROY=1                     # Allow destructive changes (staging safety off); PROD: 0
# make pulumi-destroy
```

### STEP 3: Manage kubectl context — run `make set-aks-context` to connect to AKS, or `make delete-aks-context` to fully remove local AKS credentials and kubeconfig entries.
> If locally testing, run `make lc` to create a kind cluster, and use `make set-kind-context` to switch between AKS and kind.

### STEP 4: Rollout the Qdrant StatefulSet and services into AKS by running make rollout-qdrant (remove it with make delete-qdrant).

```sh
export QDRANT__SERVICE__API_KEY="strongpassword1"          # Server-side API auth key; rotate on compromise or scheduled security rotation
export QDRANT_REPLICAS=3                                   # Set to 1 (dev), 3 (min HA), or >=5 (prod quorum); must be >= QDRANT_REPLICATION_FACTOR
export QDRANT_PERSISTENCE_ENABLED=true                     # Set to false when using Azure local NVMe disks (per-pod qdrant backups also supported)
export QDRANT_PERSISTENCE_SIZE="50Gi"                      # Increase to 100Gi+ before disk hits 70%; never shrink an existing PVC
export QDRANT_PERSISTENCE_STORAGE_CLASS="managed-premium"  # Change only when switching cloud, region, or disk tier (e.g. premium → standard)
export QDRANT_CPU_REQUEST="1"                              # Increase to 2+ if sustained CPU >70%; keep <= node allocatable / replicas
export QDRANT_CPU_LIMIT="2"                                # Set equal to request for deterministic perf; raise only to allow controlled bursting
export QDRANT_MEMORY_REQUEST="2Gi"                         # Increase to 4Gi+ when HNSW build or mmap usage grows; must fit node memory
export QDRANT_MEMORY_LIMIT="4Gi"                           # Always >= request; raise only to avoid OOMKills, never rely on overcommit
export QDRANT_IMAGE="qdrant/qdrant:v1.16.0"                # Pin exact version for reproducibility; upgrade only after validating data format compatibility
```

### STEP 5.1: Select the dense embedding model and embedding dimension (FastEmbed-compatible)

Choose a FastEmbed-supported dense text embedding model and its output dimension
(see: [https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models](https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models)).

```sh
export DENSE_MODEL_NAME="jinaai/jina-embeddings-v2-small-en"  # Canonical FastEmbed model ID; changing this requires rebuilding the image
export DENSE_DIM=512                                         # Fixed embedding vector size emitted by the model; must match Qdrant collection schema
export DENSE_IMAGE_TAG=v1                                    # Immutable image tag for this model+dim combination (<registry>/dense:<tag>)
make dense-image
```

### STEP 5.2: Rollout the dense embedding service to AKS

After the image build completes, identify the pushed image (registry URL, e.g. `*.azurecr.io/...`) and export it as `DENSE_IMAGE`. Rollout using `make rollout-dense`; remove with `make delete-dense`.

```sh
export DENSE_IMAGE=""                         # Fully qualified image reference pushed to registry (must match model+dim baked at build time)
export DENSE_REPLICAS=1                       # Desired steady-state pod count; PROD requires >=2 for availability
export DENSE_CPU_REQUEST="250m"               # Guaranteed CPU reserved per pod for scheduler placement and latency stability
export DENSE_CPU_LIMIT="1000m"                # Upper CPU bound per pod to prevent noisy-neighbor contention
export DENSE_MEMORY_REQUEST="512Mi"           # Guaranteed memory reservation; must cover model weights + runtime buffers
export DENSE_MEMORY_LIMIT="1Gi"               # Hard memory cap; exceeding this will OOM-kill the pod

# Optional: Horizontal Pod Autoscaler (HPA)
export DENSE_HPA_ENABLED=false                # Toggles HPA generation; when true, DENSE_REPLICAS becomes the initial replica count
export DENSE_HPA_MIN_REPLICAS=1               # Lower bound for autoscaling to maintain baseline capacity
export DENSE_HPA_MAX_REPLICAS=10              # Upper bound to control cost and prevent runaway scaling
export DENSE_HPA_TARGET_CPU=60                # Target average CPU utilization (%) used by HPA for scaling decisions

make rollout-dense
# make delete-dense
```

### STEP 6.1: Select the sparse embedding model (FastEmbed-compatible)

Choose a FastEmbed-supported **sparse** text embedding model.
Sparse models emit token-weight maps (vocab-based), not fixed-size vectors, so **no dimension variable is required**. [https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-sparse-text-embedding-models](https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-sparse-text-embedding-models)

```sh
export SPARSE_MODEL_NAME="Qdrant/minicoil-v1"   # Canonical FastEmbed sparse model ID; changing this requires rebuilding the image
export SPARSE_IMAGE_TAG=v1                     # Immutable image tag for this model (<registry>/sparse:<tag>)
make sparse-image
```

---

### STEP 6.2: Rollout the sparse embedding service to AKS

After the image build completes, identify the pushed image (registry URL, e.g. `*.azurecr.io/...`) and export it as `SPARSE_IMAGE`.
Rollout using `make rollout-sparse`; remove with `make delete-sparse`.

```sh
export SPARSE_IMAGE=""                        # Fully qualified image reference pushed to registry (must match model baked at build time)
export SPARSE_REPLICAS=1                      # Desired steady-state pod count; PROD requires >=2 for availability
export SPARSE_CPU_REQUEST="250m"              # Guaranteed CPU reserved per pod; sparse models are CPU-light but latency-sensitive
export SPARSE_CPU_LIMIT="1000m"               # Upper CPU bound per pod to prevent noisy-neighbor contention
export SPARSE_MEMORY_REQUEST="512Mi"          # Guaranteed memory; must cover model weights + tokenizer + runtime buffers
export SPARSE_MEMORY_LIMIT="1Gi"              # Hard memory cap; exceeding this will OOM-kill the pod

# Optional: Horizontal Pod Autoscaler (HPA)
export SPARSE_HPA_ENABLED=false               # Toggles HPA generation; when true, SPARSE_REPLICAS becomes the initial replica count
export SPARSE_HPA_MIN_REPLICAS=1              # Lower bound to maintain baseline sparse capacity
export SPARSE_HPA_MAX_REPLICAS=10             # Upper bound to control cost and prevent runaway scaling
export SPARSE_HPA_TARGET_CPU=60               # Target average CPU utilization (%) for autoscaling decisions

make rollout-sparse
# make delete-sparse
```

### STEP 7: Provision and operate the batch indexing CronJob (idempotent RAG indexing)

Creates a Kubernetes CronJob that executes the end-to-end indexing pipeline on a schedule (`make rollout-indexing-cronjob`), supports safe teardown (`make delete-indexing-cronjob`), and allows one-off manual execution for validation and debugging (`make run-indexing-cronjob`).
All RAG source data **must already be present** under `AZURE_CONTAINER/STORAGE_RAW_PREFIX` before the CronJob runs.

> **Testing only:** To seed sample data, you may upload files from the local `data/raw/` directory to `AZURE_CONTAINER/STORAGE_RAW_PREFIX` using
> `python3 infra/base_infra/force_sync_azure_and_local_fs.py --merge-upload`. This performs a merge upload and does not delete existing blobs

```sh
export STORAGE_RAW_PREFIX=data/raw/                 # Blob path prefix containing raw, unprocessed source documents
export STORAGE_CHUNKED_PREFIX=data/chunked/         # Blob path prefix where normalized, chunked outputs are written
export AZURE_STORAGE_CONNECTION_STRING=$(python3 infra/base_infra/get_storage_conn_string.py)  # SA string used by the CronJob for all blob I/O

export OVERWRITE_DOC_DOCX_TO_PDF="true"               # If true, remove originals when converting (.doc/.docx) -> .pdf; set false to keep originals
export OVERWRITE_ALL_AUDIO_FILES="true"               # If true, remove originals when converting (.mp3,.aac/etc) -> (16k wav); false to keep originals
export OVERWRITE_SPREADSHEETS_WITH_CSV="true"         # If true, remove originals when converting (.xls/.xlsx/.ods/etc) to .csv ;false to keep originals
export OVERWRITE_PPT_WITH_PPTS="true"                 # If true, remove orignals when converting .ppt -> .pptx;false to keep originals

export MAX_TOKENS_PER_CHUNK="320"                     # Cummulatively append text sentences of .pdf, .html, .mp3, .png ,etc as a chunk till this token limit  
export MIN_TOKENS_PER_CHUNK="100"                     # Minimum tokens; if chunk < this, append to previous chunk — adjust to avoid tiny fragments
export NUMBER_OF_OVERLAPPING_SENTENCES="2"            # Sentence overlap between adjacent chunks to improve recall; increase for precision 
export PDF_DISABLE_OCR="false"                        # true to skip OCR (fast but miss scanned text); false to enable OCR for scanned/embedded text
export PDF_OCR_ENGINE="rapidocr"                      # 'tesseract' or 'rapidocr' — choose rapidocr for higher accuracy, tesseract for lightweight/multilingual
export PDF_TESSERACT_LANG="eng"                       # (if tess)Language code for tesseract; change only if using tesseract and target language differs
export IMAGE_TESSERACT_LANG="eng"                     # (if tess)tesseract language for image OCR; change only with tesseract and single-language images
export TESSERACT_CONFIG="--oem 1 --psm 6"             # (if tess) Tesseract runtime flags; use --psm 3 for full-page OCR or keep 6 for block segmentation
export PDF_FORCE_OCR="false"                          # true to force OCR even when PDF has text layer (useful for noisy text), false to preserve native text
export PDF_OCR_RENDER_DPI="400"                       # DPI used when rendering PDF pages for OCR; increase for very small text, reduce for speed
export PDF_MIN_IMG_SIZE_BYTES="3072"                  # Skip OCR for images below this size; lower to include smaller images, increase to ignore artifacts
export IMAGE_OCR_ENGINE="tesseract"                   # 'tesseract' or 'rapidocr' for standalone images; choose based on accuracy/speed tradeoff
export IMAGE_MIN_IMG_SIZE_BYTES="3072"                # Skip OCR for images smaller than this; reduce to process thumbnails, increase to avoid noise
export IMAGE_RENDER_DPI="400"                         # DPI when rendering images for OCR; increase for tiny text, lower for better throughput
export IMAGE_UPSCALE_FACTOR="2.0"                     # Upscale small images before OCR; increase for very small/blurred text, decrease for performance
export CSV_TARGET_TOKENS_PER_CHUNK="400"              # Token budget for CSV chunking (including header); increase for wide tables, decrease to split more
export JSONL_TARGET_TOKENS_PER_CHUNK="400"            # Token budget for JSONL chunking; similar guidance as CSV
export PPTX_SLIDES_PER_CHUNK="4"                      # Slides grouped per chunk; increase when slides are short, decrease when slides have lots of text
export PPTX_OCR_ENGINE="rapidocr"                     # OCR engine for PPTX-rendered images; same selection guidance as other OCR engine vars
export PYTHONUNBUFFERED="1"                           # Forces Python stdout/stderr unbuffered so container logs are immediate; keep set in containers

export COLLECTION_NAME="default_rag_collection1"    # Qdrant collection name; change per environment/dataset to avoid collisions
export DENSE_DIM=512                                # Expected dense vector dimensionality; MUST match the model served at DENSE_URL
export BATCH_SIZE="8"                               # Number of chunks sent per embedding batch; increase for throughput, decrease for memory/latency
export UPSERT_CHUNK="500"                           # Points per Qdrant upsert call; larger reduces API overhead but increases request size
export SPARSE_BATCH_FALLBACK="8"                    # Micro-batch size used when sparse service rejects large batches (422)

export QDRANT_SHARD_NUMBER="3"                      # Number of shards per collection; controls horizontal scaling and parallelism (set at collection creation)
export QDRANT_REPLICATION_FACTOR="2"                # Number of replicas per shard; controls availability and durability (collection creation only)
export QDRANT_WRITE_CONSISTENCY_FACTOR="1"          # How many replicas must confirm a write; higher = safer writes, lower availability
export QDRANT_API_KEY=$QDRANT__SERVICE__API_KEY     # API key used by indexer to authenticate to Qdrant
export QDRANT_HNSW_EF_CONSTRUCT="128"               # HNSW build depth; higher improves recall but increases index-build CPU/time
export QDRANT_HNSW_M="32"                           # HNSW max connections per node; higher improves recall but increases RAM per vector
export QDRANT_HNSW_FULL_SCAN_THRESHOLD="10000"      # Vector-count threshold below which brute-force search is preferred over HNSW
export QDRANT_ONDISK="false"                         # TRUE/1/YES => enable on-disk HNSW (saves RAM without much increase in latency if using local NVMe EC2s)

export INDEX_TIMEOUT=1800          # Max allowed runtime (seconds) for index.py before forced termination
export BACKUP_TIMEOUT=300          # Max allowed runtime (seconds) for Qdrant backup execution
export ENABLE_QDRANT_BACKUP=true               # master switch: set false to disable all backups during indexing or if manual backups required
export MIN_INDEXED_POINTS_FOR_BACKUP=100       # minimum newly indexed points required to trigger backup
export MIN_INDEX_DELTA_RATIO_FOR_BACKUP=0.0    # Relative growth threshold (new / existing); 0.0 disables, e.g. 0.05 = 5% growth

export INDEXING_CRONJOB_TIMEZONE="Asia/Kolkata"     # or "Etc/UTC", "Europe/Berlin", "America/New_York" or https://cronjob.live/docs/cron-timezones
export CRON_SCHEDULE="0 */6 * * *"                   # Runs at minute 0 every 6 hours; or adjust fields to alter minute/hour/day
export CRONJOB_CONCURRENCY="Allow"                    # ConcurrencyPolicy (Allow/Forbid/Replace); choose per desired parallel execution behavior
export CRONJOB_BACKOFF_LIMIT="1"                      # Number of retries for failed Jobs; increase for transient failure tolerance
export CRONJOB_MAX_TIME="3600"                         # Kubernetes hard Job runtime limit (activeDeadlineSeconds)
export CRONJOB_PARALLELISM="1"                        # Max parallel pods for a single Job; increase only for partitioned workloads
export CRONJOB_COMPLETIONS="1"                         # Number of successful completions required; default uses parallelism if unset
export CRONJOB_DEBUG_KEEP_POD="true"                 # If true, pods sleep after work to allow debugging; set true only for dev/debug
export INDEXING_BACKUP_CRONJOB_CPU_REQUEST="2"     # CPU request for the CronJob container; raise for CPU-heavy workloads
export INDEXING_BACKUP_CRONJOB_CPU_LIMIT="4"          # CPU limit for the CronJob container; set to cap CPU usage
export INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST="1Gi"   # Memory request for CronJob; set based on worst-case memory used by indexing
export INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT="2Gi"     # Memory limit for CronJob; must be >= request to avoid eviction
export INDEXING_PIPELINE_CPU_IMAGE_REPO="athithya5354/indexing_pipeline_cpu"  # Use the prebuilt docker image or build your own by running `make index-image`
export INDEXING_PIPELINE_CPU_IMAGE_TAG="v17" # Set a consistent tag name for clarity. You may change if building your own image

make rollout-indexing-cronjob 
# make delete-indexing-cronjob 

# if immediate local testing required, run `make run-indexing-cronjob-kind`. If this fails in kind cluster then its almost always due to coreDNS restart in devcontainer, in that case run `make fix-kind-dns`. Increase batch sizes and resources according to your system specs if the pipeline feels slow.  

```

### STEP 8.1: Select the reranker cross-encoder model

Choose a FastEmbed-supported cross-encoder reranker model used to score query–document pairs
(see: [https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-rerank-cross-encoder-models](https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-rerank-cross-encoder-models)).

```sh
export RERANKER_MODEL_NAME="Xenova/ms-marco-MiniLM-L-6-v2"  # FastEmbed cross-encoder model ID; baked into the image at build time
export RERANKER_IMAGE_TAG=v1                               # Immutable image tag for this reranker model (<registry>/reranker:<tag>)
make reranker-image
```

### STEP 8.2: Rollout the reranker service to AKS

After the image build completes, identify the pushed image (registry URL, e.g. `*.azurecr.io/...`) and export it as `RERANKER_IMAGE`. Rollout using `make rollout-reranker`; remove with `make delete-reranker`.

```sh
export RERANKER_IMAGE=""                    # Fully qualified image reference pushed to registry (must match model baked at build time)
export RERANKER_REPLICAS=1                  # Desired steady-state pod count; PROD requires >=2 for availability
export RERANKER_CPU_REQUEST="250m"          # Guaranteed CPU reserved per pod for stable pairwise scoring latency
export RERANKER_CPU_LIMIT="1000m"           # Upper CPU bound per pod to prevent contention during bursty rerank workloads
export RERANKER_MEMORY_REQUEST="512Mi"      # Guaranteed memory reservation; must cover model weights and tokenization buffers
export RERANKER_MEMORY_LIMIT="1Gi"          # Hard memory cap; exceeding this will OOM-kill the pod

# Optional: Horizontal Pod Autoscaler (HPA)
export RERANKER_HPA_ENABLED=false           # Toggles HPA generation; when true, RERANKER_REPLICAS becomes the initial replica count
export RERANKER_HPA_MIN_REPLICAS=1           # Lower bound for autoscaling to maintain baseline reranking capacity
export RERANKER_HPA_MAX_REPLICAS=10          # Upper bound to control cost and prevent unbounded scale-out
export RERANKER_HPA_TARGET_CPU=60            # Target average CPU utilization (%) driving reranker autoscaling decisions

make rollout-reranker
# make delete-reranker
```

### STEP 9: Rollout the retrieval service (online hybrid search, reranking, and LLM prompt assembly)

```sh
export RETRIEVAL_IMAGE="docker.io/athithya5354/retrieval:v23"  # or build your own by running `make retrieval-image`
export RETRIEVER_REPLICAS="1"                                 # number of pods; increase to scale, decrease to save cost
export RETRIEVAL_RES_CPU="200m"                               # cpu request/limit; raise for CPU-heavy workloads
export RETRIEVAL_RES_MEM="256Mi"                              # memory request/limit; raise if OOMs occur

# Optional: Horizontal Pod Autoscaler (HPA)
export RETRIEVER_HPA_ENABLED=false      # Enable/disable HPA for the retriever; when true, pod replicas are auto-scaled based on CPU usage
export RETRIEVER_HPA_MIN=1              # Minimum number of retriever pods to maintain baseline availability under low load
export RETRIEVER_HPA_MAX=5              # Maximum number of retriever pods to cap scale-out and control resource cost
export RETRIEVER_HPA_TARGET_CPU=60      # Target average CPU utilization (%) per pod that drives HPA scaling decisions


export AZURE_STORAGE_CONNECTION_STRING=$(python3 infra/base_infra/get_storage_conn_string.py)
export AZURE_ENDPOINT_SUFFIX="core.windows.net"              # Azure endpoint suffix; change for sovereign clouds

export QUERY_TOPK_DENSE="200"                                 # dense prefetch top-k for RRF fusion; larger=slower but better recall
export QUERY_TOPK_SPARSE="200"                                # sparse prefetch top-k
export RERANKER_MODE="AUTO"                                   # DISABLE|ALWAYS|AUTO; controls when reranker runs
export RERANK_TOP_K="20"                                       # candidates sent to reranker; increase for quality (cost ↑)
export RERANK_AUTO_THRESHOLD="0.75"                           # if fused top score >= this, skip rerank in AUTO (higher => fewer reranks)
export RERANK_THRESHOLD="30"                                  # rank-disagreement threshold (abs(dense_rank - sparse_rank)) to trigger rerank
export RERANK_MARGIN="0.08"                                   # if top-second fused-gap < this, rerank (tie-breaker)
export RERANK_ALPHA="0.6"                                     # weight on reranker vs fused (0..1); higher favors reranker
export RRF_TOP_N="10"                                         # number of fused (deduped) top results returned for reranker/LLM

export GROQ_API_KEY=""                                        # GROQ LLM key; set when using GROQ (highest precedence)
export OPENAI_API_KEY=""                                      # OpenAI API key; set when using OpenAI (fallback if GROQ not set)
export LLM_API_KEY=""                                         # Generic LLM key; set for other providers (lowest precedence)
export LLM_MODEL="llama-3.1-8b-instant"                       # default LLM model id; change to target a different model
export LLM_MAX_TOKENS="512"                                   # max tokens per request; increase for longer outputs (cost ↑)
export LLM_TEMPERATURE="0.1"                                  # sampling temperature 0.0–1.0; higher = creative, lower = deterministic
export MAX_PROMPT_TOKENS="6000"                               # safety cap for prompt construction; raise only if you need larger context

export LLM_SYSTEM_PROMPT="You are an assistant that must base all factual claims ONLY on the provided numbered passages. Each factual sentence MUST end with a citation in the exact format [n], where n corresponds to one of the numbered passage blocks. Use ONLY the provided passage numbers. Do NOT output filenames, URLs, page numbers, or any other metadata. Do NOT invent citations."

export LLM_USER_PROMPT_TEMPLATE="Summarize the following retrieved passages and answer the question in 2-3 sentences.

PASSAGES:
{passages}

QUESTION: {question}

Answer:"
export MAX_CHUNKS_TO_LLM="5"                                  # max docs sent to LLM; raise for more context, lower for cost/perf

make rollout-retriever
# make delete-retriever
```

**STEP 10: Configure Cloudflare DNS and Tunnel (Frontend Edge Exposure)**.

```sh
# One-time local setup: authenticate with Cloudflare, create/reuse tunnel, bind public hostname, and export tunnel credentials

make cloudflare-setup  # follow the printed browser login and authorization steps to export CLOUDFLARE_TUNNEL_TOKEN
export CLOUDFLARE_TUNNEL_TOKEN=

# Rollout cloudflared tunneling agents into Kubernetes
export CLOUDFLARED_VERSION="2025.11.1"
export CLOUDFLARED_TUNNEL_REPLICAS=2
make rollout-cloudflared

# make delete-cloudflared
# optional logout if required `make cloudflare-logout`
```




export VMAGENT_REPLICAS="${VMAGENT_REPLICAS:-1}"                                      # vmagent replica count; increase only for HA if you handle dupes/deduplication.
export VM_RES_CPU="${VM_RES_CPU:-100m}"                                               # victoria container cpu request/limit; raise when ingestion/query CPU saturated.
export VM_RES_MEM="${VM_RES_MEM:-256Mi}"                                              # victoria memory request/limit; raise to avoid OOM for larger TSDB.
export VMAGENT_RES_CPU="${VMAGENT_RES_CPU:-100m}"                                     # vmagent cpu; increase when scraping or remote-write CPU is high.
export VMAGENT_RES_MEM="${VMAGENT_RES_MEM:-256Mi}"                                    # vmagent memory; increase if vmagent OOMs or persistent-queue grows.
# scrape & timing (affects ingestion rate / storage)
export VM_SCRAPE_INTERVAL="${VM_SCRAPE_INTERVAL:-15s}"                                 # global scrape interval; increase (longer) if cardinality/CPU/WAL pressure.
export VM_SCRAPE_TIMEOUT="${VM_SCRAPE_TIMEOUT:-10s}"                                   # per-scrape timeout;
make deploy-vm


make deploy-runbooks  # deploying runbooks will export the env var RUNBOOK_BASE_URL


export ENABLE_SLACK=true                         # master switch for Slack delivery (true/false)
export ENABLE_PAGERDUTY=true                     # master switch for PagerDuty paging (true/false)
export ALERTING_PAGING_SEVERITY_LEVELS=critical  # severities considered paging; routed to Slack if PagerDuty is off
export ALERTING_SLACK_SEVERITY_LEVELS=warning,critical  # severities delivered to Slack when Slack is on

export PAGERDUTY_INTEGRATION_KEY=$PAGERDUTY_INTEGRATION_KEY   # Set when PagerDuty receiver required; empty fully disables PD in build_alertmanager_cm()
export ALERTMANAGER_SLACK_WEBHOOK=$ALERTMANAGER_SLACK_WEBHOOK     # Set when Slack receiver required; empty disables Slack receiver entirely
export ALERT_DEFAULT_CHANNEL="#alerts-prod"                       # (Optional) Change per environment/team; used by Slack templates downstream 
export RUNBOOK_BASE_URL=$RUNBOOK_BASE_URL                         # Set to absolute http(s) URL to enable per-alert runbook links; 
export ALERTING_GROUP_WAIT="30s"                                  # Change to reduce initial fanout latency; wired to Alertmanager global+route group_wait
export ALERTING_GROUP_INTERVAL="5m"                               # Increase to reduce noise for flappy alerts; Alertmanager route group_interval
export ALERTING_REPEAT_INTERVAL="3h"                              # Increase for less reminder spam; decrease for stricter paging policies
export VMALERT_EVAL_INTERVAL="30s"                                # Increase if CPU-bound; decrease for faster detection; passed directly to vmalert
export VMALERT_REPLICAS="2"                                       # Set to 1 for k3s/dev; >=2 for AKS HA; parsed as int with safe fallback
export SLO_SUCCESS_TARGET="0.999"                                 # Change ONLY when SLO policy changes; must be 0<value<1 or validation fails
export SLO_LATENCY_QUANTILE="0.95"                                # Allowed values ONLY: 0.95 or 0.99; controls histogram_quantile in SLO alerts
export SLO_FAST_BURN_MULTIPLIER="2.0"                             # Increase to reduce pages; decrease for aggressive paging; used in fast-burn PromQL
export SLO_SLOW_BURN_MULTIPLIER="1.2"                             # Increase to tolerate long-term degradation; used in slow-burn PromQL
export ALERTMANAGER_REPLICAS="2"                                  # Set >=2 to enable HA gossip; parsed as int with fallback to 1
export ALERTMANAGER_RES_CPU="200m"                                # Increase with high route/template count; no validation, pure manifest pass-through
export ALERTMANAGER_RES_MEM="256Mi"                               # Increase when many alerts or receivers; Alertmanager memory-bound first
export VMALERT_RES_CPU="200m"                                     # Increase with rule count and eval interval; affects vmalert stability
export VMALERT_RES_MEM="256Mi"                                    # Increase with complex PromQL or large rule files


make deploy-alert-manager


```sh

export CLICKHOUSE_REPLICAS=1                  # clickhouse statefulset replicas
export CLICKHOUSE_PVC_SIZE=10Gi              # clickhouse PVC size
export CLICKHOUSE_REQ_CPU=1                   # clickhouse CPU request
export CLICKHOUSE_REQ_MEM=4Gi                 # clickhouse memory request
export CLICKHOUSE_LIMIT_CPU=4                 # clickhouse CPU limit
export CLICKHOUSE_LIMIT_MEM=16Gi              # clickhouse memory limit
export CLICKHOUSE_USER=vector                 # clickhouse user for vector
export CLICKHOUSE_PASSWORD=vectorpass         # clickhouse password (replace with secret manager in prod)
export LOGS_TTL_DAYS=2

make deploy-clickhouse

export K8S_CLUSTER=aks                        # set to aks for production behavior
export VECTOR_REPLICAS=1                      # logical replica control (future-proof)
export VECTOR_REQ_CPU=200m                    # vector CPU request
export VECTOR_REQ_MEM=512Mi                   # vector memory request
export VECTOR_LIMIT_CPU=1000m                 # vector CPU limit
export VECTOR_LIMIT_MEM=1Gi                   # vector memory limit
export VECTOR_DROP_NAMESPACES="kube-system,models,indexing"
export VECTOR_LOG_LEVELS=info,warn,error
make deploy-vector



export GRAFANA_ADMIN_PASSWORD='grafana' # Grafana admin password (secret) — rotate regularly / change per environment
export GRAFANA_ADMIN_USER='admin' # Grafana admin username (secret) — change when onboarding or rotating admin
export GRAFANA_API_KEY='' # Grafana API key — used for API validation; set in CI or automated checks
export GRAFANA_API_URL='' # Grafana API base URL — set for remote API validations or CI post-deploy checks
export GRAFANA_USE_PVC='false' # Feature flag: persist Grafana data on PVC; set true for stateful installations
export GRAFANA_REPLICAS='1' # Grafana replicas (scale horizontally for HA or load); change to scale
export GRAFANA_IMAGE='grafana/grafana:10.3.5' # Grafana container image (pin for deterministic upgrades)
export GRAFANA_CPU_REQ='100m' # Grafana CPU request — tune if Grafana is CPU-starved
export GRAFANA_MEM_REQ='128Mi' # Grafana memory request — increase if OOM or heavy dashboards
export GRAFANA_CPU_LIMIT='500m' # Grafana CPU limit — allow bursts as needed
export GRAFANA_MEM_LIMIT='512Mi' # Grafana memory limit — upper bound to avoid OOM on host
export GRAFANA_PVC_SIZE='5Gi' # PVC size for Grafana data (if GRAFANA_USE_PVC=true) — adjust to retention/plugins
export METRICS_DATASOURCE='VictoriaMetrics' # Logical metrics datasource name used in dashboards
export METRICS_DATASOURCE_URL='http://victoria-metrics.monitoring.svc:8428' # Metrics backend endpoint — change per cluster
export CLICKHOUSE_DATASOURCE='ClickHouse' # Logical ClickHouse datasource name used in dashboard links
export CLICKHOUSE_URL='http://clickhouse.clickhouse.svc:8123' # ClickHouse HTTP endpoint for explore links
export DATASOURCE_URL='http://victoria-metrics.monitoring.svc:8428' # Backend URL used for recording-rule sanity checks
export CI='false' # CI toggle: when true the generator fails hard on validation errors (use in pipelines)
export GRAFANA_NAMESPACE='monitoring' # Namespace where Grafana / ConfigMaps are created
export GRAFANA_PROVISIONING_NAMESPACE='monitoring' # Namespace Grafana expects provisioning ConfigMaps in
export DEFAULT_NAMESPACE='monitoring' # Default namespace injected into dashboard variables
export DASHBOARD_SERVICES='retriever,qdrant' # Comma list of per-service dashboards to render
export GRAFANA_DASHBOARD_UID_PREFIX='platform-' # UID prefix for generated dashboards (avoid collisions)
export RUNBOOK_BASE_URL='https://defaultsa515.z13.web.core.windows.net' # Base runbook URL used in dashboard headers/links
export MAX_PANELS_PER_DASHBOARD='48' # Safety cap to avoid oversized dashboards (prevents runaway renders)
export SLO_SUCCESS_TARGET='0.999' # SLO success target used in dashboard headers/alerts — change for different SLOs
export SLO_LATENCY_QUANTILE='0.95' # Latency quantile used in SLO panels (allowed: 0.95 or 0.99)
export RETRIEVER_LATENCY_THRESHOLD_SECONDS='0.5' # Retriever latency threshold (p95) shown in dashboards
export QDRANT_LATENCY_THRESHOLD_SECONDS='0.8' # Qdrant latency threshold (p95) shown in dashboards





export FRONTEND_HOSTNAME="ui.example.com"                    # public hostname (example: ui.mycompany.com)
export ENABLE_GOOGLE_AUTH="true"                             # enable Google auth (example: true/false)
export GOOGLE_ALLOWED_DOMAINS="company.com,gmail.com"        # allowed domains (example: company.com,gmail.com)
export GOOGLE_CLIENT_ID=""                                   # example: 1234567890-abc.apps.googleusercontent.com
export GOOGLE_CLIENT_SECRET=""                               # example: GOCSPX-xxxxxxxxxxxxxxxx

export JWT_SECRET=""                                         # example: random-32+char-secret
export SESSION_SECRET=""                                     # example: random-32+char-secret
export JWT_EXP_SECONDS=1800                                  # token expiry seconds (example: 1800)
export DISPLAY_SOURCES_IN_UI="true"                          # show sources in UI (example: true)
export DISPLAY_TOPK_IN_UI="true"                             # show top-K results (example: true)

export ENABLE_MICROSOFT_AUTH="true"                          # enable Microsoft auth (example: true/false)
export MS_CLIENT_ID=""                                       # example: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export MS_CLIENT_SECRET=""                                   # example: ~AbCdEfGhIjKlMnOpQrStUvWxYz
export MICROSOFT_ALLOWED_DOMAINS="outlook.com,company.com"   # allowed domains (example: outlook.com,company.com)
export MICROSOFT_ALLOWED_TENANT_IDS=""                       # example: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export MS_TENANT_ID=""                                       # example: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

export ENABLE_GITHUB_AUTH="true"                             # enable GitHub auth (example: true/false)
export GITHUB_CLIENT_ID=""                                   # example: Ov23liFnXdXltUPW34R7
export GITHUB_CLIENT_SECRET=""                               # example: 40-char-github-secret
export GITHUB_ALLOWED_ORGS="my-org"                          # allowed orgs (example: my-org,another-org)

export FRONTEND_AND_AUTH_IMAGE="athithya5354/frontend-and-auth:v10"      # or create by running `make frontend-image`
export FRONTEND_AND_AUTH_REPLICAS=1                          # replica count (example: 1)

export CLOUDFLARED_VERSION="2025.11.1"                       # cloudflared version (example: 2025.11.1)
export CLOUDFLARED_TUNNEL_REPLICAS=1                         # tunnel replicas (example: 1)
make cloudflare-setup
# export CLOUDFLARE_TUNNEL_KEY
make deploy-cloudflared
make deloy-frontend