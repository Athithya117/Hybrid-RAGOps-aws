# Executive summary (concise)

This document is a **technical reference** of the indexing pipeline when deployed as the CronJob produced by your manifest generator and runtime scripts. It follows the exact runtime control flow implemented in the repository: **manifest generation & apply → Kubernetes CronJob pod start → indexing_pipeline.py runtime → pre_conversions → router → index (embedding & upsert)**. The description below maps to concrete functions, env knobs and behaviors in the codebase and is implementation-accurate.

---

# End-to-end trace (example, simple)

1. Operator `apply`s manifests; CronJob created in namespace.
2. Cron triggers Job; Pod starts and runs `indexing_pipeline.py`.
3. `indexing_pipeline` runs `pre_conversions.py`:

   * Groups `data/raw/foo.pdf` → `data/raw/pdfs/foo.pdf`.
   * Converts `.docx` to `pdfs/`.
   * Normalizes audio to `audio/foo.wav`.
4. Router scans `STORAGE_RAW_PREFIX`, calls `parse_file("data/raw/pdfs/foo.pdf", manifest={})`:

   * `pdf.py` extracts per-page text + figures; chunks into windows; writes `doc_id.parquet` to `STORAGE_CHUNKED_PREFIX`.
   * Writes `data/raw/pdfs/foo.pdf.manifest.json`.
5. `index.py` loads `data/chunked/doc_id.parquet`:

   * Normalizes rows → `chunks` list.
   * Builds embed clients, dedupes already-present point IDs in Qdrant.
   * Embeds new chunks, builds point payloads and upserts to Qdrant in slices.
6. Logs and metadata persisted; subsequent Cron runs skip files with existing raw manifests unless `FORCE_OVERWRITE=true`.

---

# High-level runtime control flow (end → end)

1. **Manifest generation / apply stage (CLI / infra script)**

   * `load_cfg()` collects runtime configuration from `DEFAULTS` + environment and builds `cfg`.
   * `validate_cfg(cfg)` asserts required credentials depending on `USE_MANAGED_IDENTITY`.
   * `cronjob_manifest(cfg, env_map)` builds the Kubernetes CronJob YAML:

     * `concurrencyPolicy` (from cfg) and `jobTemplate.spec` containing:

       * Container `command` set to: `["/bin/sh","-c","/opt/venv/bin/python /indexing_pipeline/indexing_pipeline.py"]`
       * Resource requests/limits from CFG keys (CPU/MEM)
     * Pod `serviceAccountName` set to `SERVICE_ACCOUNT_NAME`.
     * Sensitive envs (e.g., `QDRANT_API_KEY`, `AZURE_*`) are wired as `valueFrom.secretKeyRef` when present at runtime.
   * `apply(cfg)`:

     * Recreates `MANIFESTS_DIR`.
     * Writes YAML manifests: `Namespace`, `ServiceAccount`, `Role`, `RoleBinding`, placeholders for secrets, and `CronJob`.
     * Applies namespace first, waits until it exists.
     * Creates in-cluster secrets inline using `kubectl_create_secret_inline()` for runtime-secret values.
     * Applies remaining manifest files in one `kubectl apply -f -` call.

2. **CronJob execution (Kubernetes)**

   * Cron triggers Job according to `CRON_SCHEDULE`.
   * Job runs a single Pod (by default `parallelism=1` + `completions=1` embedded in `jobTemplate.spec`).
   * Container runs the indexing pipeline command (see above).

3. **indexing_pipeline.py runtime**

   * Entry `main()` sets up signal handlers and calls `run_pipeline(workdir)`.
   * `run_pipeline` steps:

     1. `try_raise_nofile()` (best-effort RLIMIT_NOFILE increase).
     2. `run_pre_conversions(workdir)`:

        * `run_local_and_stream(script, workdir)` starts `/indexing_pipeline/pre_conversions.py` as a subprocess; stdout lines logged at INFO, stderr at WARNING.
        * `pre_conversions.py` groups raw blobs into canonical subfolders (e.g., `audio`, `pdfs`, `csvs`, `others`) and performs conversions (LibreOffice `soffice` for docs/sheets → pdf/csv, `ffmpeg` for audio).
        * Safe move semantics: `safe_move_blob` downloads -> uploads -> deletes with dedupe (size/etag/sha checks). If destination exists and differs, `make_unique_target` appends `-N`.
     3. `connect_or_start_local()` (no-op for local mode in current script).
     4. Start `router` script via `run_local_and_stream` (expected at `parse_chunk/router.py`).

        * The router scans `STORAGE_RAW_PREFIX` and dispatches `parse_file` functions in `parse_chunk/formats/*` to produce chunked parquet and raw manifest files under `STORAGE_CHUNKED_PREFIX`.
     5. Start `index` (`apps/index/index.py`) via `run_local_and_stream`. This script loads chunk parquet(s) and performs embedding & upsert.

4. **Indexing runtime internals (`apps/index/index.py`)**

   * `validate_envs()` asserts required Azure creds or Qdrant connectivity depending on `USE_MANAGED_IDENTITY`.
   * `load_chunks_from_azure(account_name, account_key, container, prefix)`:

     * Uses `BlobServiceClient` (managed identity) or connection string / fsspec mode to list blobs under `AZURE_CHUNKED_PREFIX`.
     * Prefers `.parquet` chunk files; falls back to `.json`.
     * For each parquet: `pyarrow.parquet.read_table()` → `table.to_pydict()` → per-row chunk dict normalized to canonical fields.
     * For JSON chunk lists: `json.loads` and `normalize_chunk()`.
   * `normalize_chunk(chunk)` canonicalizes types: lists, ints, booleans, timestamps; normalizes `headings`, `tags`, `figures`, `row_range`, `token_range`, etc.
   * `validate_and_build_clients()` probes and builds:

     * `DenseClient` (HTTP embed service at `DENSE_URL`) and `SparseClient` (at `SPARSE_URL`) with health checks and smoke `embed` calls.
   * Collection creation:

     * If both dense & sparse healthy → `create_collection_hybrid()`: creates Qdrant collection with `vectors_config={"dense":{size:DENSE_DIM,...}}` and `sparse_vectors_config`.
     * Else `create_collection_sparse_only()`.
   * Embedding & dedupe:

     * `embed_and_upsert()` processes chunks in batches (`BATCH_SIZE`):

       * For each batch, compute point IDs via `id_from_string(chunk_id)`.
       * `existing_point_ids()` retrieves existing points from Qdrant to avoid re-upserting duplicates.
       * `safe_embed_and_points()` produces vectors:

         * Calls `_embed_with_retry_and_split_dense()` to get dense vectors with retries and recursive split on failures.
         * Calls `_embed_sparse_with_retry_and_split()` to get sparse vectors with similar retry/split behavior and fallback chunk sizes if server rejects large batches.
         * `chunk_and_vectors_to_pointstructs()` builds Qdrant payloads with `vectors_payload` (dense and/or sparse) and full chunk `payload` containing `FULL_PAYLOAD_KEYS`.
     * Upsert to Qdrant in `UPSERT_CHUNK` slices with retry wrapper.
   * Logging and graceful shutdown handling for SIGINT/SIGTERM (`SHUTDOWN` flag).

---

# Concise table of supported formats (placed here in the middle)

| Format                              |                                     Input / preconv | Parser module (file)                                     | Chunking granularity                                         | Key chunk fields                                                                                |
| ----------------------------------- | --------------------------------------------------: | -------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| PDF                                 |      `.pdf` (also `.doc/.docx` → pdf via `soffice`) | `parse_chunk/formats/pdf.py` (PDF page extraction & OCR) | Page-level + sentence/token windows                          | `document_id,chunk_id,page_number,text,token_count,figures,used_ocr,source_url,parser_version`  |
| TXT / MD                            |                                       `.txt`, `.md` | `parse_chunk/formats/txt.py`                             | Whole-file or sentence windows                               | `document_id,chunk_id,text,token_count,source_url,parser_version,line_range`                    |
| HTML / remote                       |                         `.html`, `.htm`, remote URL | `parse_chunk/formats/_html.py` (trafilatura if present)  | Token windows (`token_window`) or single page                | `document_id,chunk_id,text,token_count,token_range,source_url,original_manifest,parser_version` |
| CSV / Sheets                        | `.csv` (spreadsheets → multiple csvs via `soffice`) | `parse_chunk/formats/_csv.py`                            | `row_group` (batched rows) or `token_window` (for long rows) | `document_id,chunk_id,text,row_range,token_range,token_count,source_url,parser_version`         |
| Audio                               |  `.wav, .mp3, .m4a, ...` (preconv canonical `.wav`) | `parse_chunk/formats/wav.py` (faster-whisper)            | Speech segments → sentence chunks with `audio_range`         | `document_id,chunk_id,text,token_count,audio_range,parse_ms,source_url,parser_version`          |
| Images (PDF embedded or standalone) |                                 `.png, .jpg, .tiff` | handled in `pdf.py` (and optional image OCR paths)       | Attached to page chunk as `figures` entries                  | `figures` array, `used_ocr`                                                                     |

---

# Universal schema & traceability (runtime-visible semantics)

The indexing pipeline expects chunks to follow a stable, union schema. Important **traceable** fields and their precise semantics:

* `document_id` — canonical id computed from `manifest.file_hash` or `sha256(raw_key + LastModified)`; used to name parquet out_basename and provide stable chunk grouping.
* `chunk_id` — unique id per chunk (parser-specific stable pattern e.g., `{doc_id}_p{page}_{idx}`).
* `source_url` / `raw_key` — canonical AZ path `az://{AZURE_CONTAINER}/{raw_key}` or original remote URL; primary link to source blob.
* `token_count`, `token_range` — measured with `tiktoken` when available (ENC_NAME env) else whitespace fallback; used to size windows.
* `row_range`, `line_range`, `page_number`, `audio_range` — precise offsets locating the chunk inside original content for full-line / row / page / seconds mapping.
* `figures` — array of extracted figure/table text or OCR results attached to the chunk.
* `parser_version`, `timestamp`, parquet-level metadata `{schema_version,producer,created_at}` — essential for lineage and debugging.
* `used_ocr` — boolean flag when OCR was applied and content derived from images.
* `semantic_region`*: a coarse positional label (`intro|early|middle|late|footer|unknown`) derived from the chunk’s starting token index relative to total document tokens, used to preserve where the chunk semantically sits in the document flow.

**Reverse mapping:** from chunk → `source_url` & `token_range`/`page_number`/`row_range`/`audio_range` allow reconstructing the exact original bytes/region.

---

# Runtime control flow details (deep, technical)

## CronJob manifest & kube runtime properties

* CronJob built by `cronjob_manifest()` uses:

  * `schedule`: `CRON_SCHEDULE` / `INDEXING_BACKUP_CRON_EXPRESSION`.
  * `concurrencyPolicy` default `Allow` (so overlapping jobs allowed).
  * Pod template annotations for Workload Identity: `azure.workload.identity/client-id` and `tenant-id` when `USE_MANAGED_IDENTITY=1`.
  * Sensitive envs are configured as `secretKeyRef` only when their env values existed at the time `apply()` was executed; otherwise placeholder manifests remain for operator injection.
* The CronJob `jobTemplate` uses a container `command` string to run the pipeline in sequence. Thus the CronJob is **single-process orchestrator** inside the pod.

## Pod bootstrap (entry command)

* Container runs `/opt/venv/bin/python /indexing_pipeline/indexing_pipeline.py` inside the repo-mounted working dir (`/indexing_pipeline`) and uses the venv copied into the image.
* `indexing_pipeline.py` logs stdout/stderr lines: stdout → INFO, stderr → WARNING.

## pre_conversions (detailed)

* Purpose: canonicalize and normalize raw files before parsing.
* Authentication: `build_blob_service_client()` picks authentication strategy:

  * Managed Identity: `DefaultAzureCredential` or `ManagedIdentityCredential` when `USE_MANAGED_IDENTITY=1`.
  * Else: `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY` or `AZURE_SAS_TOKEN`.
* Grouping:

  * Scans blobs under `STORAGE_RAW_PREFIX` and moves them into canonical subfolders (e.g., `audio/`, `pdfs/`, `csvs/`, `htmls/`, `others/`) via `safe_move_blob()` which downloads to TMP, computes hashes, uploads and dedupes.
* Conversions:

  * Documents → PDF: `run_soffice_convert(src, outdir, "pdf:writer_pdf_Export")`. On conversion failure → upload to `quarantine/` with metadata about error.
  * Spreadsheets → CSV(s): `run_soffice_convert(...,"csv")` and upload resulting CSVs into `csvs/{workbook_name}/`.
  * Audio normalization: `run_ffmpeg_convert(src,dst)` to mono 16k s16 WAV. Upload to `audio/{stem}.wav`. If `OVERWRITE_ALL_AUDIO_FILES=true` original audio is deleted after a successful conversion.
* Metadata:

  * On uploads, `upload_file_to_blob()` sets `ContentSettings` and writes metadata keys produced by `prepare_metadata()` containing base64-encoded `sha256` and provenance info.

## parse_chunk/router.py (router behavior — runtime role)

* Router (launched by `indexing_pipeline.py`) enumerates `STORAGE_RAW_PREFIX`, selects appropriate parser per file extension and invokes `parse_file(s3_key, manifest)` exported by per-format modules under `parse_chunk/formats/*`.
* Each parser ensures idempotency:

  * If `${raw_key}.manifest.json` exists and `FORCE_OVERWRITE` is false → skip.
  * If `${STORAGE_CHUNKED_PREFIX}{doc_id}.parquet` exists and `FORCE_OVERWRITE` is false → add manifest if missing then skip.
* Parsers use `ParquetWriter` abstraction that:

  * Buffers normalized rows into a pyarrow Table and writes a single compressed parquet file (zstd, flavor=spark).
  * Writes parquet metadata `schema_version`, `parser_version`, `producer`, `created_at`.
  * Finally it uploads parquet atomically with `storage_upload_file_atomic()` which writes to a tmp key then renames/moves.

## index.py detailed internal control flow (embedding & upsert)

1. **Load chunks**

   * `load_chunks_from_azure()` lists blobs under `AZURE_CHUNKED_PREFIX` and chooses `.parquet` first, then `.json`.
   * Parquet files: read into pyarrow Table, convert to pydict, iterate rows, build normalized chunk dicts with expected keys.
2. **Normalize**

   * `normalize_chunk()` enforces canonical types and shapes for:

     * Lists: `[headings,tags,layout_tags,figures]`.
     * Numeric ranges: `row_range`,`token_range`,`line_range`.
     * Flags: `used_ocr` normalized to boolean.
     * Strings: `text, file_name, file_type, source_url, parser_version, chunk_type, chunk_id, document_id`.
3. **Prepare embed clients**

   * `DenseClient` & `SparseClient` wrap HTTP embed services at `DENSE_URL` and `SPARSE_URL` with retry wrappers `_post_with_retries()` and `_get_with_retries()` (use `retry_call()` helper).
   * Health checks and small smoke `embed` calls are performed to verify availability.
4. **Deduplicate before embed**

   * For each batch, compute Qdrant IDs via `id_from_string(chunk_id)` (md5 hex → integer from hex slice) and call `existing_point_ids()` (client.retrieve) to find already present points; skip embedding those chunks.
5. **Embedding**

   * `safe_embed_and_points()` does:

     * Dense: `_embed_with_retry_and_split_dense()` tries embedding the whole batch; upon failure it splits recursively into halves until success or single-item failure.
     * Sparse: `_embed_sparse_with_retry_and_split()` does similar but also reacts to `400` server response mentioning a `max=` by splitting to that max; fallback `SPARSE_BATCH_FALLBACK` used when `422`.
   * The embed responses are combined into (chunk, dense_vec, sparse_obj) tuples.
6. **Point struct creation**

   * `chunk_and_vectors_to_pointstructs(items, hybrid)` converts embeddings into Qdrant `points`:

     * `vectors_payload` includes `dense` and/or `sparse`.
     * `payload` includes fields from `FULL_PAYLOAD_KEYS` to preserve chunk metadata in Qdrant payload.
     * Skips points with no vectors in hybrid mode or missing sparse in sparse-only mode.
7. **Upsert**

   * Upserts happen in slices (`UPSERT_CHUNK`) and each `client.upsert()` call wrapped in `retry_call()` with exponential backoff.
   * Logging emits `index.start`, `batch.embedded`, `index.prepared`, `index.completed`.

## Fault handling & safety semantics

* **Idempotency:** router and parsers avoid reprocessing by checking raw-manifest and chunked-parquet existence.
* **Deduplication:** Qdrant `existing_point_ids()` check prevents re-embedding/upserting previously inserted points; point ids derived deterministically.
* **Retries:** Network/backoff/retry helpers `retry_call()` used across storage operations, embed HTTP calls, and upserts.
* **Atomic uploads:** `storage_upload_file_atomic()` writes to a `.tmp.<pid>.<ts>` path then moves to final location (fsspec or azure client).
* **Crash behavior:** If parser fails mid-buffer, parser logs and returns `skipped=True` with `error` field; router will record and continue other blobs.

---

# Important runtime knobs & exact env mappings (extracted from code)

(Only the key runtime knobs used in control flow)

* **Kubernetes / manifest script**

  * `NAMESPACE`, `CRONJOB_NAME`, `CRON_SCHEDULE`, `CONCURRENCY`, `SERVICE_ACCOUNT_NAME`, `MANIFESTS_DIR`, `USE_MANAGED_IDENTITY`, `UAI_RAG_RW_CLIENT_ID`
* **pre_conversions.py**

  * `STORAGE_RAW_PREFIX`, `AZURE_CONTAINER`, `TMP_DIR`, `OVERWRITE_ALL_AUDIO_FILES`, `OVERWRITE_OTHER_TO_PDF`, `OVERWRITE_SPREADSHEETS_WITH_CSV`, `PDF_OCR_ENGINE`, `PDF_FORCE_OCR`
* **parsers / chunking**

  * `STORAGE_CHUNKED_PREFIX`, `MAX_TOKENS_PER_CHUNK`, `MIN_TOKENS_PER_CHUNK`, `NUMBER_OF_OVERLAPPING_SENTENCES`, `TOKEN_ENCODER`/`ENC_NAME`, `PUT_RETRIES`, `PUT_BACKOFF`, `SAVE_SNAPSHOT`
* **index.py**

  * `AZURE_CHUNKED_PREFIX`, `AZURE_CONTAINER`, `QDRANT_URL`, `DENSE_URL`, `SPARSE_URL`, `BATCH_SIZE`, `UPSERT_CHUNK`, `DENSE_DIM`, `SPARSE_BATCH_FALLBACK`, `QDRANT_ONDISK`

---

