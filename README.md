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

## STEP 1: Create a storage account: export the following environment variables with appropriate values, then run `make create-sa`. You can delete it using `make delete-sa`. 

```sh
export AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}"   # Azure subscription ID (must already be set after az login)
export AZURE_RESOURCE_GROUP_NAME="rg-e2e-rag"             # Resource group hosting storage account and platform infra
export AZURE_ENDPOINT_SUFFIX="core.windows.net"           # Override this only for sovereign clouds (e.g., Azure China or Gov) to avoid endpoint failures
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

export AZURE_DELETE_ACCOUNT=0                             # 1 = delete entire storage account on `make delete-sa`, 0 = containers only
export FORCE_DELETE=1                                     # Skip interactive confirmation prompts

```

## STEP 2: Manage platform infrastructure with Pulumi: export all required environment variables, run `make pulumi-preview` to inspect changes, `make pulumi-up` to apply them, or `make pulumi-destroy` to delete resources (destructive; requires PULUMI_FORCE_DESTROY=1).

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

export PULUMI_FORCE_DESTROY=1                     # Allow destructive changes (staging safety off); PROD: 0
export AKS_LOCATION="${AZURE_LOCATION}"   # Deployment region; change only for latency or quota management

export ACR_NAME=acr49251                                   # Global ACR name; change only if creating a new registry (cannot rename)
export ACR_REPO_PREFIX=rag                              # Logical repo namespace; change when multiple teams/apps share one ACR
export ACR_LOCATION="${AKS_LOCATION:-${AZURE_LOCATION:-eastus}}"  # Region; change only if co-locating with AKS or for compliance
export ACR_SKU=Standard                                 # SKU; use Premium only for Private Endpoint / geo-replication
```


```sh

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
export IMAGE_RENDER_DPI="600"                         # DPI when rendering images for OCR; increase for tiny text, lower for better throughput
export IMAGE_UPSCALE_FACTOR="2.0"                     # Upscale small images before OCR; increase for very small/blurred text, decrease for performance
export CSV_TARGET_TOKENS_PER_CHUNK="400"              # Token budget for CSV chunking (including header); increase for wide tables, decrease to split more
export JSONL_TARGET_TOKENS_PER_CHUNK="400"            # Token budget for JSONL chunking; similar guidance as CSV
export PPTX_SLIDES_PER_CHUNK="4"                      # Slides grouped per chunk; increase when slides are short, decrease when slides have lots of text
export PPTX_OCR_ENGINE="rapidocr"                     # OCR engine for PPTX-rendered images; same selection guidance as other OCR engine vars
export PYTHONUNBUFFERED="1"                           # Forces Python stdout/stderr unbuffered so container logs are immediate; keep set in containers

export COLLECTION_NAME="default_rag_collection1"    # Qdrant collection name; change per environment/dataset to avoid collisions
export DENSE_DIM="384"                              # Expected dense vector dimensionality; MUST match the model served at DENSE_URL
export BATCH_SIZE="16"                              # Number of chunks sent per embedding batch; increase for throughput, decrease for memory/latency
export UPSERT_CHUNK="500"                           # Points per Qdrant upsert call; larger reduces API overhead but increases request size
export SPARSE_BATCH_FALLBACK="8"                    # Micro-batch size used when sparse service rejects large batches (422)
export QDRANT_HNSW_EF_CONSTRUCT="128"               # HNSW index build depth; higher improves recall but increases index-build time and CPU cost
export QDRANT_HNSW_M="32"                           # HNSW max graph connections per node; higher boosts recall but raises RAM usage per vector
export QDRANT_HNSW_FULL_SCAN_THRESHOLD="10000"      # Point-count threshold affecting full-scan vs HNSW behavior
export QDRANT_ONDISK="TRUE"                         # TRUE/1/YES => enable on-disk HNSW (saves RAM without much increase in latency if using local NVMe EC2s)

export INDEXING_CRONJOB_TIMEZONE="Asia/Kolkata"     # or "Etc/UTC", "Europe/Berlin", "America/New_York" or https://cronjob.live/docs/cron-timezones
export CRON_SCHEDULE="0 */6 * * *"                   # Runs at minute 0 every 6 hours; or adjust fields to alter minute/hour/day
export CRONJOB_CONCURRENCY="Allow"                    # ConcurrencyPolicy (Allow/Forbid/Replace); choose per desired parallel execution behavior
export CRONJOB_BACKOFF_LIMIT="1"                      # Number of retries for failed Jobs; increase for transient failure tolerance
export CRONJOB_PARALLELISM="1"                        # Max parallel pods for a single Job; increase only for partitioned workloads
export CRONJOB_COMPLETIONS="1"                         # Number of successful completions required; default uses parallelism if unset
export CRONJOB_DEBUG_KEEP_POD="true"                 # If true, pods sleep after work to allow debugging; set true only for dev/debug
export INDEXING_BACKUP_CRONJOB_CPU_REQUEST="2"     # CPU request for the CronJob container; raise for CPU-heavy workloads
export INDEXING_BACKUP_CRONJOB_CPU_LIMIT="4"          # CPU limit for the CronJob container; set to cap CPU usage
export INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST="1Gi"   # Memory request for CronJob; set based on worst-case memory used by indexing
export INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT="2Gi"     # Memory limit for CronJob; must be >= request to avoid eviction
export INDEXING_PIPELINE_CPU_IMAGE_REPO="athithya5354/indexing_pipeline_cpu"  # Use the prebuilt docker image or build your own by running `make index-image`
export INDEXING_PIPELINE_CPU_IMAGE_TAG="v12" # Set a consistent tag name for clarity. You may change if building your own image
# optional env if UAI unavailable in local kind cluster
export AZURE_STORAGE_CONNECTION_STRING="$(python3 infra/base_infra/get_storage_conn_string.py)"

make run-indexing-cronjob

export PER_POD="true" # When true, perform per-pod Qdrant backup/restore using individual pod port-forwards; when false, operate via cluster/service endpoint
make backup

export BACKUP_ID="" # Optional explicit backup identifier to restore; leave empty to auto-select the latest backup manifest under the Azure prefix
make restore

export RETRIEVAL_IMAGE="docker.io/athithya5354/retrieval:v11"  # container image:tag; change to deploy a new build/tag
export RETRIEVER_REPLICAS="1"                                 # number of pods; increase to scale, decrease to save cost
export RETRIEVAL_RES_CPU="200m"                               # cpu request/limit; raise for CPU-heavy workloads
export RETRIEVAL_RES_MEM="256Mi"                              # memory request/limit; raise if OOMs occur

export AZURE_STORAGE_CONNECTION_STRING=""                    # Azure connection string (AccountName+AccountKey); set if using presign
export AZURE_ENDPOINT_SUFFIX="core.windows.net"              # Azure endpoint suffix; change for sovereign clouds

export GROQ_API_KEY=""                                        # GROQ LLM key; set when using GROQ (highest precedence)
export OPENAI_API_KEY=""                                      # OpenAI API key; set when using OpenAI (fallback if GROQ not set)
export LLM_API_KEY=""                                         # Generic LLM key; set for other providers (lowest precedence)
export LLM_MODEL="llama-3.1-8b-instant"                       # default LLM model id; change to target a different model
export LLM_MAX_TOKENS="512"                                   # max tokens per request; increase for longer outputs (cost ↑)
export LLM_TEMPERATURE="0.2"                                  # sampling temperature 0.0–1.0; higher = creative, lower = deterministic
export MAX_PROMPT_TOKENS="6000"                               # safety cap for prompt construction; raise only if you need larger context

export LLM_SYSTEM_PROMPT="You are a clear concise assistant. Provide a short explanatory answer in 2-3 sentences. When you cite evidence, use only numeric tags like [1],[2]. Do NOT output filenames, URLs, raw page numbers."  
                                                             # system role prompt; controls assistant policy/tone—change to alter model behavior/safety
export LLM_USER_PROMPT_TEMPLATE="Summarize the following retrieved passages and answer the question in 2-3 sentences.

QUESTION: {question}

PASSAGES:
{passages}

Answer:"                                                           # user prompt template; change task framing but keep {question} and {passages}

export RERANKER_MODE="AUTO"                                   # DISABLE|ALWAYS|AUTO; controls when reranker runs
export RERANK_TOPK="20"                                       # candidates sent to reranker; increase for quality (cost ↑)
export RERANKER_TOP_K="$RERANK_TOPK"                          # backward-compat alias; keep synced
export RERANK_AUTO_THRESHOLD="0.75"                           # if fused top score >= this, skip rerank in AUTO (higher => fewer reranks)
export RERANK_THRESHOLD="30"                                  # rank-disagreement threshold (abs(dense_rank - sparse_rank)) to trigger rerank
export RERANK_MARGIN="0.08"                                   # if top-second fused-gap < this, rerank (tie-breaker)
export RERANK_ALPHA="0.6"                                     # weight on reranker vs fused (0..1); higher favors reranker

export MAX_CHUNKS_TO_LLM="6"                                  # max docs sent to LLM; raise for more context, lower for cost/perf
export QUERY_TOPK_DENSE="200"                                 # dense prefetch top-k for RRF fusion; larger=slower but better recall
export QUERY_TOPK_SPARSE="200"                                # sparse prefetch top-k
export RRF_TOP_N="10"                                         # number of fused (deduped) top results returned for reranker/LLM

make deploy-retriever


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
export K8S_CLUSTER=aks                        # set to aks for production behavior
export VECTOR_REPLICAS=1                      # logical replica control (future-proof)
export VECTOR_REQ_CPU=200m                    # vector CPU request
export VECTOR_REQ_MEM=512Mi                   # vector memory request
export VECTOR_LIMIT_CPU=1000m                 # vector CPU limit
export VECTOR_LIMIT_MEM=1Gi                   # vector memory limit
export VECTOR_DROP_NAMESPACES="kube-system,models,indexing"
export VECTOR_LOG_LEVELS=info,warn,error
make deploy-vector

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
