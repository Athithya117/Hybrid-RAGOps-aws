make lc
make deploy-qdrant
make deploy-models

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


make run-indexing-cronjob-kind:

make deploy-vm 

make deploy-retriever
make deploy-frontend


LOG(){ printf '%s %s\n' "$(date -Iseconds)" "$*"; }
ERR(){ printf '%s ERROR %s\n' "$(date -Iseconds)" "$*" >&2; }

VM_NAMESPACE=${VM_NAMESPACE:-monitoring}
RETRIEVAL_NAMESPACE=${RETRIEVAL_NAMESPACE:-inference}
RETRIEVAL_NAME=${RETRIEVAL_NAME:-retrieval}
QDRANT_NAME=${QDRANT_NAME:-qdrant}

VICTORIA_PORT=${VICTORIA_PORT:-8428}
VMAGENT_PORT=${VMAGENT_PORT:-8429}

RETRIEVAL_METRICS_PORT=${RETRIEVAL_METRICS_PORT:-8001}
QDRANT_METRICS_PORT=${QDRANT_METRICS_PORT:-6333}

LOAD_SECONDS=${LOAD_SECONDS:-20}
PORTFWD_READY_TIMEOUT=${PORTFWD_READY_TIMEOUT:-20}

CURL_BIN=${CURL_BIN:-curl}
PYTHON_BIN=${PYTHON_BIN:-python3}

require(){ command -v "$1" >/dev/null 2>&1 || { ERR "$1 required"; exit 2; }; }
for b in kubectl jq awk sed grep "${CURL_BIN}" "${PYTHON_BIN}"; do require "$b"; done

TMPFILES=()
PFPIDS=()

cleanup(){
  rc=$?
  for pid in "${PFPIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for f in "${TMPFILES[@]:-}"; do
    [ -f "$f" ] && rm -f "$f" || true
  done
  exit "$rc"
}
trap cleanup INT TERM EXIT

find_free_port(){
  "${PYTHON_BIN}" - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
}

start_portforward(){
  ns="$1"; target="$2"; lport="$3"; rport="$4"
  log="$(mktemp /tmp/pf.${target//[^a-zA-Z0-9_.-]/_}.XXXX.log)"
  TMPFILES+=("$log")
  kubectl -n "$ns" port-forward "$target" "$lport:$rport" >"$log" 2>&1 &
  PFPIDS+=("$!")
}

wait_http(){
  url="$1"; timeout="$2"
  end=$((SECONDS+timeout))
  while [ "$SECONDS" -lt "$end" ]; do
    "${CURL_BIN}" -sf --max-time 3 "$url" >/dev/null && return 0
    sleep 1
  done
  return 1
}

promql(){
  q="$1"
  "${CURL_BIN}" -sS -G \
    --data-urlencode "query=$q" \
    "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/api/v1/query"
}

assert_gt0(){
  name="$1"; q="$2"
  LOG "PromQL ${name}: ${q}"
  json="$(promql "$q")"
  echo "$json" | jq .
  val="$(echo "$json" | jq -r '.data.result[0].value[1] // "0"')"
  "${PYTHON_BIN}" - "$val" <<'PY'
import sys
v=float(sys.argv[1])
assert v>0, v
PY
}

assert_eq1(){
  name="$1"; q="$2"
  LOG "PromQL ${name}: ${q}"
  json="$(promql "$q")"
  echo "$json" | jq .
  val="$(echo "$json" | jq -r '.data.result[0].value[1] // "0"')"
  "${PYTHON_BIN}" - "$val" <<'PY'
import sys,math
v=float(sys.argv[1])
assert math.isclose(v,1.0), v
PY
}

LOG "starting VictoriaMetrics port-forward"
LOCAL_VICTORIA_PORT="$(find_free_port)"
start_portforward "$VM_NAMESPACE" svc/victoria-metrics "$LOCAL_VICTORIA_PORT" "$VICTORIA_PORT"
wait_http "http://127.0.0.1:${LOCAL_VICTORIA_PORT}/metrics" "$PORTFWD_READY_TIMEOUT" || { ERR "victoria not ready"; exit 10; }

LOG "starting vmagent port-forward"
LOCAL_VMAGENT_PORT="$(find_free_port)"
start_portforward "$VM_NAMESPACE" svc/vmagent "$LOCAL_VMAGENT_PORT" "$VMAGENT_PORT"
wait_http "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" "$PORTFWD_READY_TIMEOUT" || { ERR "vmagent not ready"; exit 11; }

LOG "vmagent scrape sample"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_VMAGENT_PORT}/metrics" | sed -n '1,80p'

LOG "port-forward retriever"
LOCAL_RETR_PORT="$(find_free_port)"
start_portforward "$RETRIEVAL_NAMESPACE" "svc/${RETRIEVAL_NAME}" "$LOCAL_RETR_PORT" "$RETRIEVAL_METRICS_PORT"
wait_http "http://127.0.0.1:${LOCAL_RETR_PORT}/metrics" 10 || { ERR "retriever metrics unavailable"; exit 12; }

LOG "port-forward qdrant"
LOCAL_QDR_PORT="$(find_free_port)"
start_portforward "$RETRIEVAL_NAMESPACE" "svc/${QDRANT_NAME}" "$LOCAL_QDR_PORT" "$QDRANT_METRICS_PORT"
wait_http "http://127.0.0.1:${LOCAL_QDR_PORT}/metrics" 10 || { ERR "qdrant metrics unavailable"; exit 13; }

LOG "synthetic load ${LOAD_SECONDS}s"
i=0
while [ "$i" -lt "$LOAD_SECONDS" ]; do
  "${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_RETR_PORT}/" >/dev/null || true
  i=$((i+1))
  sleep 1
done

LOG "retriever /metrics head"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_RETR_PORT}/metrics" | sed -n '1,120p'

LOG "qdrant /metrics head"
"${CURL_BIN}" -sS "http://127.0.0.1:${LOCAL_QDR_PORT}/metrics" | sed -n '1,120p'

LOG "VictoriaMetrics label context"
promql 'count by (__name__,service)({service=~".+"})' | jq .

assert_eq1 "up_retriever" "max(up{service=\"${RETRIEVAL_NAME}\"})"
assert_gt0 "retrieval_requests" "sum(increase(retrieval_requests_total[1m]))"
assert_gt0 "qdrant_queries" "sum(increase(qdrant_query_total[1m]))"

LOG "MONITORING E2E PASSED"
