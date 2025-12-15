Assumptions & immediate context (concise)

* You run the indexing pipeline as the repo shows: `pre_conversions.py` (grouping + convert), per-format parsers (pdf/csv/html/wav, etc.) that write chunked parquet + raw `.manifest.json`, and `index.py` which loads chunked parquet, normalizes chunks and upserts embeddings to Qdrant.
* Many environment variables are runtime knobs read directly by those scripts or Dockerfile defaults; I’ll explain each variable precisely, where it is consumed (which file / phase), expected values, and the operational effect.

Azure / cluster / storage identity

* `AZURE_RESOURCE_GROUP_NAME`
  Identifies the Azure Resource Group housing storage/account/AKS resources. Useful for infra scripts and any automation that interacts with the cloud environment. Not consumed by parsing code but required for infra orchestration.

* `AZURE_LOCATION`
  Azure region string (e.g., `centralindia`); used by infra for resource placement and by any region-aware automation. No runtime parser effect.

* `AZURE_STORAGE_ACCOUNT_NAME`
  The storage account (container namespace) used when you run in managed-identity mode or when forming storage URLs. Parsers that use managed identity require this so the SDK can construct `account_url`. If you run with `AZURE_USE_MANAGED_IDENTITY=1`, this variable is mandatory for creating `BlobServiceClient` (see `pre_conversions.py`, parser files). Do not expose account keys if you use MI—prefer workload identity.

* `AZURE_CONTAINER`
  Container name where raw files and chunked outputs live (used everywhere: parsers, pre_conversions, indexer). Parsers build `az://{AZURE_CONTAINER}/{key}` for provenance and uploads raw-manifest and parquet under `STORAGE_CHUNKED_PREFIX`. Must match the container you grant access to (or mount via fsspec).

Conversion / overwrite controls (pre_conversions + Docker defaults)

* `OVERWRITE_DOC_DOCX_TO_PDF`
  When true, after converting `.doc/.docx` to PDF via LibreOffice the original doc/docx blob is deleted. Controlled in `pre_conversions.py`: set to `true` for storage hygiene (only keep canonical PDFs) or `false` to retain originals for audit. If set true, conversions are destructive—ensure manifests/backups if you need originals.

* `OVERWRITE_ALL_AUDIO_FILES`
  Controls whether original audio files (mp3, m4a, etc.) are removed after being converted to normalized WAV (mono, 16k) via FFmpeg. Implemented in `pre_conversions.py`. `true` reclaims storage and forces downstream to rely on canonical `.wav`. `false` keeps raw audio; useful for debugging/transcoding traceability.

* `OVERWRITE_SPREADSHEETS_WITH_CSV`
  If true, spreadsheets converted to CSVs are uploaded and the original spreadsheet object is deleted. Affects downstream behavior because CSVs are split into row groups—use true for normalized ingestion, false to preserve originals.

* `OVERWRITE_PPT_WITH_PPTS`
  Controls whether `.ppt` originals are replaced by `.pptx` after conversion. Similar trade-offs: normalization vs preserving originals.

Tokenization / chunk sizing / overlap

* `MAX_TOKENS_PER_CHUNK`
  Core chunk size budget used by all token-window and sentence-chunking logic (PDF/HTML/TXT/WAV/CVS). Parsers accumulate sentences or rows until adding another sentence/row would exceed this token budget; if a single sentence exceeds it, the sentence is split into token windows. Values are integer token counts (e.g., `320`). Lower => smaller chunks, more chunks and higher indexing cost; higher => bigger chunks, less context switching but larger embedded vectors and more memory.

* `MIN_TOKENS_PER_CHUNK`
  Post-formation merging threshold: if a newly created chunk has fewer tokens than this minimum the parser merges it into the previous chunk to avoid tiny fragments. Keep this moderately lower than `MAX_TOKENS_PER_CHUNK` to avoid too many tiny chunks at document boundaries.

* `NUMBER_OF_OVERLAPPING_SENTENCES`
  The sentence overlap preserved between consecutive chunks to maintain context at chunk boundaries. Parsers compute the new start index as `max(start + 1, end_sentence_idx - overlap)`. Increase overlap for better recall over edges but expect some duplication in embeddings/search.

PDF and OCR configuration

* `PDF_DISABLE_OCR`
  When `true`, OCR is skipped for PDFs even if pages are images; parsing will only use text layers. When `false`, OCR is attempted when a page lacks a reliable text layer. Used in the PDF parser and `pre_conversions` decisions for PDF processing. If you have many scanned docs, set to `false`.

* `PDF_OCR_ENGINE`
  Which OCR backend to try for PDF images: `rapidocr` (higher-accuracy GPU/modern models) or `tesseract` (CPU, widely available). The code will attempt RapidOCR first if configured and installed; fall back to tesseract when chosen or when RapidOCR absent based on environment. The engine choice affects accuracy/performance and dependency footprint.

* `PDF_TESSERACT_LANG`
  If tesseract is the selected engine, this BCP-47-like code instructs Tesseract which language model to use (e.g., `eng`). This must match trained language packs installed into the runtime.

* `PDF_FORCE_OCR`
  When `true`, OCR is applied even if the PDF has selectable text. Use this if the text layer is noisy (scans with misrecognized glyphs) and OCR yields better normalized text. When `false`, native text is preserved to avoid unnecessary OCR cost and potential OCR regressions.

* `PDF_OCR_RENDER_DPI`
  DPI used to rasterize PDF pages before OCR. Higher DPI yields better OCR on small fonts but increases CPU/time and memory. Default in your list is `400`. Use higher fractions for very small, dense text; reduce for performance.

* `PDF_MIN_IMG_SIZE_BYTES`
  When scanning for images on a PDF page, the parser skips images smaller than this threshold to avoid OCRing tiny UI artifacts or icons. Tune this to avoid unnecessary OCR.

Image OCR / rendering

* `IMAGE_OCR_ENGINE`
  Chooses OCR backend for standalone images: `tesseract` vs `rapidocr`. Same tradeoffs as PDFs.

* `IMAGE_TESSERACT_LANG`
  Language code used by Tesseract for image OCR.

* `IMAGE_MIN_IMG_SIZE_BYTES`
  Minimum size to attempt OCR on an image; avoids noisy results on icons/screenshots.

* `IMAGE_RENDER_DPI`
  When converting or upscaling an image prior to OCR, this DPI guides rendering. Higher DPI helps small text.

* `IMAGE_UPSCALE_FACTOR`
  Multiplier used to upsample images prior to OCR to improve recognition of blur/small fonts; increases CPU and memory usage.

CSV / JSONL / tabular chunking

* `CSV_TARGET_TOKENS_PER_CHUNK`
  Target token budget used by the CSV parser to decide how many rows to accumulate into a `row_group` chunk. The CSV parser estimates `sample_row_tokens` and computes `rows_per_chunk = clamp(estimated_rows, MIN_ROWS_PER_CHUNK, MAX_ROWS_PER_CHUNK)` unless explicitly overridden. Larger token budget means more rows per chunk (bigger parquet row) and fewer embedding calls.

* `JSONL_TARGET_TOKENS_PER_CHUNK`
  Same concept for line-delimited JSON parser implementations: budget for grouping JSONL rows into chunk rows.

Presentation / PPTX processing

* `PPTX_SLIDES_PER_CHUNK`
  Number of slides grouped into a single chunk for PPTX parsing. Increase if slides are short and you want more context per chunk; reduce if slides are long and you want finer granularity.

* `PPTX_OCR_ENGINE`
  OCR engine used for images rendered from slides (same choices as PDF/Image OCR).

Container / Python runtime

* `PYTHONUNBUFFERED`
  When set to `1`, python runs with unbuffered stdout/stderr so logs stream immediately to container stdout (Docker / Kubernetes friendly). All Dockerfile and pipeline scripts assume this for predictable logging.

Qdrant / embedding / indexer

* `QDRANT_API_KEY`
  API key for Qdrant. Treated as sensitive. Indexer (`index.py`) uses it when instantiating `QdrantClient(api_key=...)`. In production, prefer putting this value in a Kubernetes Secret and referencing it via `secretKeyRef` instead of inline environment variables.

* `COLLECTION_NAME`
  Name of the Qdrant collection used for upserts. `index.py` will `create_collection_hybrid` or `create_collection_sparse_only` for this collection name. Use different collection names to separate datasets or environments.

* `DENSE_DIM`
  Expected dimensionality of dense vectors supplied by your dense embedding service (DENSE_URL). The indexer validates that vectors returned by `/embed` match this dimension; mismatches fail the pipeline.

* `BATCH_SIZE`
  Number of chunks sent in a single call to dense/sparse embed endpoints. `index.py` builds batches of `BATCH_SIZE` texts for embedding. Larger values increase throughput when the embed service can handle them, but increase memory footprint and potential latency for a single failing batch.

* `UPSERT_CHUNK`
  Number of points sent to Qdrant per upsert call. Larger values reduce API overhead but increase request size and failure blast radius. `index.py` slices `to_upsert` into chunks of this size.

* `SPARSE_BATCH_FALLBACK`
  Micro-batch size used by the sparse embed client when the sparse embed service rejects large batches (422). `index.py` will fall back and split the batch into this smaller size automatically to work around server constraints.

Qdrant HNSW configuration

* `QDRANT_HNSW_EF_CONSTRUCT`
  Parameter for HNSW index construction controlling exploration during build. Higher values improve recall at the cost of higher CPU and slower build time. Applied when `create_collection_hybrid` is invoked.

* `QDRANT_HNSW_M`
  HNSW connectivity parameter: maximum number of connections per node. Higher `M` increases recall and memory usage.

* `QDRANT_HNSW_FULL_SCAN_THRESHOLD`
  Threshold point count used by the client to determine when to perform full scan vs HNSW-style search heuristics. Tuning depends on your Qdrant deployment, dataset size, and latency/recall tradeoffs.

* `QDRANT_ONDISK`
  If truthy (`TRUE`/`1`/`YES`) the collection is created with on-disk HNSW settings, which trades off RAM for disk IO. Useful for large datasets on servers with NVMe-backed storage.

Cronjob / Kubernetes / scheduling knobs

* `INDEXING_CRONJOB_TIMEZONE`
  Timezone string applied to the CronJob manifest when generated by your manifest tool. Controls cron evaluation time zone in Kubernetes 1.25+ CronJob feature gates.

* `CRON_SCHEDULE`
  Cron expression (standard five-field) that determines when the pipeline CronJob runs. The manifest generator places this value into the `schedule` field of the CronJob. Example `"0 */6 * * *"` means every 6 hours at minute 0.

* `CRONJOB_CONCURRENCY`
  Kubernetes CronJob `concurrencyPolicy` (`Allow`, `Forbid`, or `Replace`). `Allow` lets multiple cron Jobs overlap; `Forbid` avoids overlap; `Replace` kills the running one and replaces with a new run.

* `CRONJOB_BACKOFF_LIMIT`
  Backoff limit for failed Jobs spawned by the CronJob. Kubernetes will retry failed jobs up to this count.

* `CRONJOB_PARALLELISM`
  Controls the maximum number of pods a single Job can run in parallel. Only relevant if your Job template is configured for parallelism.

* `CRONJOB_COMPLETIONS`
  How many successful completions the Job should have before considered successful. Usually 1 for batch indexing.

* `CRONJOB_DEBUG_KEEP_POD`
  If `true`, pods intentionally do not exit after work to allow developers to `kubectl exec` into them for debugging; obviously for dev only.

* `INDEXING_BACKUP_CRONJOB_CPU_REQUEST` / `INDEXING_BACKUP_CRONJOB_CPU_LIMIT` / `INDEXING_BACKUP_CRONJOB_MEMORY_REQUEST` / `INDEXING_BACKUP_CRONJOB_MEMORY_LIMIT`
  Resource requests/limits applied to the container in the CronJob manifest generated by the manifest generator. Set requests to what your indexing job needs for normal operation; set limits to cap runaway usage and ensure QoS.

Image / pipeline runtime selection

* `INDEXING_PIPELINE_CPU_IMAGE_REPO` / `INDEXING_PIPELINE_CPU_IMAGE_TAG`
  The Docker image repository and tag used in the CronJob manifest for the indexing container. The manifest generator composes the image string and uses it in the `containers.image` field. Change these to your built image to roll out new versions.

Optional non-managed auth fallback

* `AZURE_STORAGE_CONNECTION_STRING`
  When supplied, code paths prefer connection string auth for `BlobServiceClient.from_connection_string`. This is an alternative to managed identity. It is sensitive; prefer creating a K8s secret and referencing it rather than exposing in plain environment variables. `pre_conversions.py` and parsers accept either connection string or account/key/SAS combos through `build_fs_opts()`.

UAI / Workload Identity (AKS) / Managed identity

* `UAI_RAG_RW_NAME`, `UAI_RAG_RO_NAME`
  Names of the user-assigned identities (or AKS workload identity names) used by Kubernetes resources. They are used only by infra manifests and annotation generation; the runtime code expects `USE_MANAGED_IDENTITY` and client IDs to be present for managed flows.

* `AZURE_ENDPOINT_SUFFIX`
  Endpoint suffix for custom clouds (default `core.windows.net`). Used when constructing `account_url` for `BlobServiceClient`. Change only for sovereign/offline clouds.

* `UAI_RAG_RW_CLIENT_ID` / `UAI_RAG_RW_PRINCIPAL_ID` / `UAI_RAG_RO_CLIENT_ID` / `UAI_RAG_RO_PRINCIPAL_ID`
  User-assigned identity client/principal IDs used to annotate service accounts and request managed identity credentials. When `USE_MANAGED_IDENTITY=1` or `AZURE_USE_MANAGED_IDENTITY=1`, these values are used to pick the exact identity to use for the runtime. `pre_conversions.py` and manifest generation place these into annotations/envs so the pod requests the UAI. Do not store private keys here — these are public IDs.

Optional: parsing / chunker productivity knobs (indexer / parsers)

* `CSV_TARGET_TOKENS_PER_CHUNK`
  Target token budget for CSV chunking, used by the CSV parser to estimate rows per chunk. Doubling means fewer chunks but larger rows per parquet entry.

* `JSONL_TARGET_TOKENS_PER_CHUNK`
  Equivalent for JSONL chunking pipelines.

* `PARSER_VERSION` (per-format envs like `PARSER_VERSION_HTML`, `PARSER_VERSION_CSV`)
  Parser version strings are embedded as `parser_version` per chunk and in parquet metadata; they are vital for traceability and rolling upgrades.

Network / timeouts / retries / backoff

* `HTTP_TIMEOUT` / `REQUEST_TIMEOUT`
  Timeout applied to HTTP calls (fetching HTML, embed services) to avoid hanging. Parsers and embedding clients use these to size calls. Set substantively higher than 1s for heavy remote calls.

* `FETCH_RETRIES` / `FETCH_BACKOFF` / `PUT_RETRIES` / `PUT_BACKOFF`
  Numbers controlling retry attempts and exponential backoff for fetch and storage uploads. Implemented via `retry_call()` wrappers; tuning these affects resiliency to transient network/storage errors.

Misc / dev / helper envs

* `PYTHONUNBUFFERED` (explained above) — ensures logs stream immediately; mandatory for containerized runs.

* `MANIFESTS_DIR`
  Where the manifest generator writes Kubernetes manifests. Not used by parsers directly, but by the infra generator.

* `STORAGE_RAW_PREFIX` / `STORAGE_CHUNKED_PREFIX` (defaults in code)
  Layout prefixes inside the blob container where raw inputs and chunked outputs live. Parsers and `pre_conversions` use these to list, group, and write artifacts. Keep them consistent across the pipeline.

Security & operational guidance embedded in variable behavior

* Secrets (e.g., `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_SAS_TOKEN`, `AZURE_STORAGE_CONNECTION_STRING`, `QDRANT_API_KEY`) should never be committed or exported in scripts in plain text for prod. The manifest generator supports creating Kubernetes Secrets inline from environment literals, but best practice is to store secrets in a secret manager or create them via CI/CD as sealed secrets.

* `USE_MANAGED_IDENTITY` / `AZURE_USE_MANAGED_IDENTITY` toggles the whole credential flow: when enabled, code uses DefaultAzureCredential/ManagedIdentityCredential and requires `AZURE_STORAGE_ACCOUNT_NAME` + optional UAI client id; when disabled, code falls back to fsspec/adlfs or connection string/account key/SAS token. This influences installed dependency requirements: `azure-identity` + `azure-storage-blob` for MI workflows, `fsspec` + `adlfs` for non-MI fsspec flows.

* Resource-request and limit envs in the CronJob manifest are the primary knobs to avoid OOMs and to constrain CPU. They must reflect worst-case memory footprint of parsing + embedding batches.

