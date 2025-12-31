# docs/infra/qdrant/qdrant_backup.md

Short explanation: Concrete, deterministic runbook to create consistent Qdrant backups and produce verifiable manifests for safe restores. [Technical terms: service-mode snapshot, per-pod snapshot, manifest.json schema, SHA256 integrity, Azure Blob SAS, idempotent invocation]

## Purpose
Define exact backup procedures, environment variables, manifest schema, idempotent CLI invocations, CI checks, and failure remediation for Qdrant snapshots in this platform.

---

## Scope
This document covers:
- How backups are created
- How backup mode is selected (`service` vs `per-pod`)
- What artifacts are produced
- How integrity is guaranteed
- What must be validated before restore

It does **not** cover restore execution details (see `qdrant_restore.md`).

---

## Prerequisites
- `python3` >= 3.8
- `kubectl` available in PATH (required when `PER_POD=true`)
- Python packages:
  - `requests`
  - `azure-storage-blob`
  - `azure-identity`
- Network access to:
  - Qdrant HTTP API
  - Azure Blob Storage

---

## Required Environment Variables

### Execution Control
```text
PER_POD (string)            required values="true"|"false"
LOCAL_DIR (path)            optional default=/tmp/qdrant-backup
TIMEOUT_SECONDS (int)       optional default=300
````

### Qdrant Context

```text
QDRANT_NAMESPACE (string)   required default=qdrant
QDRANT_SERVICE_HOST (url)   optional override for service endpoint
QDRANT__SERVICE__API_KEY    optional API key for secured clusters
```

### Azure Storage

```text
BACKUP_AZURE_CONTAINER      required target container name
BACKUP_AZURE_PREFIX         optional default=qdrant/backups
AZURE_STORAGE_CONNECTION_STRING
  OR
AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
```

Validation rules:

* `PER_POD` must be strictly `"true"` or `"false"`.
* One Azure authentication method must be available.
* When `PER_POD="true"`, `kubectl` must be functional and authorized.

---

## Backup Modes

### Service-Level Backup (`PER_POD="false"`)

Use when Qdrant replicas share network-attached storage.

Characteristics:

* Snapshot created via Qdrant service endpoint
* One snapshot per collection
* Simplest operational path
* No dependency on pod identity

### Per-Pod Backup (`PER_POD="true"`)

Use when Qdrant uses node-local or NVMe storage.

Characteristics:

* Snapshot created independently per pod
* Requires `kubectl port-forward`
* One snapshot per pod per collection
* Preserves pod-local state

**Backup and restore must use the same mode.**

---

## Backup Artifact Layout (Azure Blob)

```text
<container>/
  <prefix>/
    <backup_id>/
      manifest.json
      <collection-or-pod>-<collection>.snapshot
    latest.manifest.json
```

* `backup_id` is generated once per backup run
* `latest.manifest.json` always points to the most recent successful backup

---

## Manifest Schema (Authoritative)

```json
{
  "backup_id": "string",
  "created_at": "ISO8601 string",
  "namespace": "string",
  "mode": "service|per-pod",
  "collections": ["string"],
  "pods": {
    "service": {
      "collections": {
        "<collection>": {
          "snapshot_name": "string",
          "azure_container": "string",
          "blob_path": "string",
          "sha256": "64-hex",
          "size": integer,
          "local_path": "string"
        }
      }
    },
    "<pod-name>": {
      "collections": { "<collection>": { "...": "..." } }
    }
  }
}
```

Constraints:

* `mode` must match the execution mode.
* Every snapshot entry must include `blob_path` and `sha256`.
* Manifest is the single source of truth for restore.

---

## Deterministic Backup Flow

### Common Steps

1. Discover collections via Qdrant API
2. Create snapshot(s)
3. Download snapshot file(s)
4. Compute SHA256 and file size
5. Upload to Azure Blob
6. Write `manifest.json`
7. Update `latest.manifest.json`

### Service-Level Flow

* Snapshot API invoked once per collection
* Files downloaded via service endpoint

### Per-Pod Flow

* Pods discovered and ordered
* One port-forward per pod
* Snapshot invoked via pod-local API

---

## CLI Invocation (Idempotent)

### Service Mode

```bash
PER_POD="false" \
BACKUP_AZURE_CONTAINER=my-backups \
BACKUP_AZURE_PREFIX=qdrant/backups \
LOCAL_DIR=/tmp/qdrant-backup \
python3 infra/runners/run_qdrant_backup.py \
  --backup \
  --azure-container ${BACKUP_AZURE_CONTAINER} \
  --azure-prefix ${BACKUP_AZURE_PREFIX} \
  --local-dir ${LOCAL_DIR}
```

### Per-Pod Mode

```bash
PER_POD="true" \
BACKUP_AZURE_CONTAINER=my-backups \
BACKUP_AZURE_PREFIX=qdrant/backups \
LOCAL_DIR=/tmp/qdrant-backup \
python3 infra/runners/run_qdrant_backup.py \
  --backup \
  --per-pod \
  --azure-container ${BACKUP_AZURE_CONTAINER} \
  --azure-prefix ${BACKUP_AZURE_PREFIX} \
  --local-dir ${LOCAL_DIR}
```

Re-running either command is safe and deterministic.

---

## Integrity Guarantees

* SHA256 is computed locally before upload
* Restore validates SHA256 before applying snapshot
* Manifest and blobs are uploaded atomically per backup_id
* Partial failures are explicitly recorded

---

## Verification Checklist (Post-Backup)

1. `latest.manifest.json` exists in Azure
2. All snapshot entries contain valid SHA256
3. Blob sizes match manifest sizes
4. At least one older backup retained

---

## Common Failures and Remediation

| Error                        | Action                                   |
| ---------------------------- | ---------------------------------------- |
| `--per-pod requires kubectl` | Install kubectl and verify RBAC          |
| Azure auth error             | Validate credentials or managed identity |
| Snapshot API failure         | Check Qdrant logs and collection health  |
| Checksum mismatch            | Re-download snapshot and re-run backup   |

---

## Operational Rules

* Do not mix service-mode and per-pod backups across environments
* Never delete backups without validating restore
* Keep `LOCAL_DIR` on disk with sufficient free space
* Always retain `latest.manifest.json`

---
