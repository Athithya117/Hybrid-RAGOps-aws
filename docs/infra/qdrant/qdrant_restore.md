# docs/infra/qdrant/qdrant_restore.md

## Purpose
This document defines the **exact and deterministic restore process** for Qdrant. It describes required environment variables, manifest resolution, restore execution paths, integrity checks, pod mapping rules, restart behavior, and failure handling. The restore workflow is designed to be idempotent, auditable, and safe to retry.

---

## Prerequisites
- `python3` >= 3.8
- Python packages installed:
  - `requests`
  - `azure-storage-blob`
  - `azure-identity`
- `kubectl` available in PATH **if `PER_POD="true"`**
- Network access to:
  - Kubernetes API
  - Qdrant HTTP API
  - Azure Blob Storage
- Sufficient Azure permissions to download blobs or generate SAS URLs

---

## Required Environment Variables

```text
PER_POD (string)               required values: "true" or "false"
BACKUP_AZURE_CONTAINER         required unless embedded in manifest
BACKUP_AZURE_PREFIX            required default=qdrant/backups
BACKUP_ID (string)             optional; empty selects latest.manifest.json
QDRANT_NAMESPACE               required default=qdrant
QDRANT__SERVICE__API_KEY       optional; sent as Authorization header
TIMEOUT_SECONDS (int)          optional default=300
LOCAL_DIR (path)               optional default=/tmp/qdrant-restore
````

Validation rules:

* `PER_POD` must exactly equal `"true"` or `"false"`.
* If `BACKUP_ID` is empty, `${BACKUP_AZURE_PREFIX}/latest.manifest.json` must exist.
* If `PER_POD="true"`, `kubectl get pods -n ${QDRANT_NAMESPACE}` must succeed.

---

## Manifest Resolution Logic

Restore always begins by resolving a single manifest:

1. If `BACKUP_ID` is provided:

   * Load `${BACKUP_AZURE_PREFIX}/${BACKUP_ID}/manifest.json`
2. If `BACKUP_ID` is empty:

   * Download `${BACKUP_AZURE_PREFIX}/latest.manifest.json`
   * Extract `backup_id`
   * Load the corresponding `manifest.json`
3. Abort immediately if:

   * Manifest download fails
   * `backup_id` field is missing
   * Manifest JSON is invalid

The resolved manifest is the **single source of truth** for restore.

---

## Manifest Expectations (Restore-Critical Fields)

Each restore entry must contain:

* `blob_path`
* `sha256`
* `size`
* `snapshot_name`

Restore will fail fast if any required field is missing.

---

## Restore Execution Order (Strict)

Restore attempts the following strategies **in order**, stopping on first success:

### 1. Server-Side Restore via Presigned Azure SAS (Preferred)

Used when Qdrant can directly pull snapshot data.

Steps:

1. Generate a presigned Azure Blob SAS URL (default expiry ~3600s)
2. Call Qdrant API:

   * `POST /collections/<collection>/snapshots/recover`
3. Poll Qdrant until recovery completes or timeout expires

Conditions:

* Works for service-mode restores
* Works for per-pod restores when port-forwarding is active

---

### 2. Download + Upload via Qdrant Snapshot API (Fallback)

Used when SAS generation or server-side pull fails.

Steps:

1. Download blob to `LOCAL_DIR`
2. Compute SHA256 and compare to manifest
3. Upload snapshot to Qdrant via `/snapshots/upload`
4. Trigger recovery from uploaded snapshot

Restore aborts on checksum mismatch.

---

### 3. Per-Pod Filesystem Restore (`PER_POD="true"` only)

Used for node-local or NVMe-backed storage.

Steps:

1. Discover current Qdrant pods
2. Deterministically map manifest pod entries to live pods
3. For each pod:

   * Copy snapshot file into pod snapshot directory
   * Ensure correct file ownership
4. Optionally restart StatefulSet to force snapshot discovery

---

## Pod Mapping Rules (Per-Pod Restore)

Mapping priority:

1. Exact pod name match
2. StatefulSet ordinal match (e.g., `qdrant-0`)
3. Abort if mapping is ambiguous or incomplete

Restore **fails fast** if pod mapping cannot be resolved deterministically.

---

## Integrity and Safety Checks (Mandatory)

Before applying any restore:

* Verify manifest structure and required fields
* Verify blob size matches manifest
* Verify SHA256 checksum matches manifest
* Abort immediately on mismatch

No partial or corrupted restore is permitted.

---

## Restart and Readiness Behavior

* If `--restart` is used:

  * Perform rollout restart of Qdrant StatefulSet
  * Wait for all pods to reach Ready state
* Readiness criteria:

  * Pod Ready condition true
  * Qdrant health endpoint responsive
* Timeout governed by `TIMEOUT_SECONDS`

---

## Idempotency Guarantees

* Restore operations can be re-run safely
* Snapshots are verified before use
* Partial restores do not corrupt existing data
* Failed restore attempts leave cluster unchanged

---

## Common Failure Scenarios and Actions

| Error               | Action                                             |
| ------------------- | -------------------------------------------------- |
| Manifest not found  | Verify Azure prefix and container                  |
| Checksum mismatch   | Re-download blob or re-run backup                  |
| Pod mapping failure | Verify StatefulSet pod names and ordinals          |
| Qdrant API error    | Inspect Qdrant logs and retry with upload fallback |
| Timeout exceeded    | Increase `TIMEOUT_SECONDS` and retry               |

---

## Standard Restore Invocation

```bash
PER_POD="${PER_POD:-false}" \
BACKUP_AZURE_CONTAINER=my-backups \
BACKUP_AZURE_PREFIX=qdrant/backups \
BACKUP_ID="${BACKUP_ID:-}" \
python3 infra/runners/run_qdrant_restore.py \
  --restore "${BACKUP_ID}" \
  $( [ "${PER_POD}" = "true" ] && echo "--per-pod --restart" ) \
  --azure-container ${BACKUP_AZURE_CONTAINER} \
  --azure-prefix ${BACKUP_AZURE_PREFIX} \
  --namespace ${QDRANT_NAMESPACE:-qdrant}
```

---
