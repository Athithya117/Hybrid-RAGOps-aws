# Qdrant — Environment Variables 

Below is a concise, concrete reference for **every environment variable you’ll need to run the backup & restore runners and the deployment generator** (excluding `MANIFESTS_DIR` and `ENV` as requested). For each variable you get: **what it is**, **type**, **exact default (if any)**, **validation rules**, and **when/why to change it** with concrete examples.

---

## Summary table (quick reference)

| Var                                          |                      Type |         Default         |                        Required?                       | Purpose / When to change                                                                                                 |
| -------------------------------------------- | ------------------------: | :---------------------: | :----------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------ |
| `QDRANT_NAMESPACE`                           |                    string |         `qdrant`        |                    no (recommended)                    | Kubernetes namespace where Qdrant runs. Change if you deploy to a different namespace.                                   |
| `QDRANT_IMAGE`                               |       string (`repo:tag`) | `qdrant/qdrant:v1.16.0` |                 yes for deploy scripts                 | Controls Qdrant version. Change to upgrade/downgrade image (must include `:tag`).                                        |
| `QDRANT_REPLICAS`                            |                       int |           `1`           |                           no                           | Number of pod replicas. Increase to scale out for capacity/throughput.                                                   |
| `QDRANT_REPLICATION_FACTOR`                  |                       int |           `1`           |                           no                           | Logical replication factor for data placement. Change to increase redundancy (match cluster size).                       |
| `QDRANT__SERVICE__API_KEY`                   |                    string |          *none*         |             conditional (if cluster locked)            | API key for Qdrant HTTP API. Required if Qdrant is configured to require an API key. **Do not commit**.                  |
| `AZURE_STORAGE_CONNECTION_STRING`            |                    string |          *none*         |         required for non-managed identity flows        | Preferred single-variable Azure auth. Set when not using managed identity.                                               |
| `AZURE_STORAGE_ACCOUNT_NAME`                 |                    string |          *none*         |  required with `AZURE_STORAGE_ACCOUNT_KEY` alternative | Use as pair with `AZURE_STORAGE_ACCOUNT_KEY` when connection string not available.                                       |
| `AZURE_STORAGE_ACCOUNT_KEY`                  |                    string |          *none*         | required with `AZURE_STORAGE_ACCOUNT_NAME` alternative | See above. **Do not commit**.                                                                                            |
| `BACKUP_AZURE_CONTAINER` / `AZURE_CONTAINER` |                    string |   *none* (must supply)  |                     yes at runtime                     | Azure blob container where snapshots/manifests are uploaded. Set to your backups container name.                         |
| `BACKUP_AZURE_PREFIX` / `AZURE_PREFIX`       |                    string |     `qdrant/backups`    |                           no                           | Path prefix inside the container (folder). Change to organize by environment or cluster: e.g. `prod/qdrant/backs`.       |
| `PER_POD`                                    | string `"true"`/`"false"` |        `"false"`        |         yes (mode must match backup manifests)         | Controls per-pod vs service-mode flows. Set `"true"` for node-local NVMe; `"false"` for shared PVs.                      |
| `BACKUP_ID`                                  |                    string |       `""` (empty)      |                           no                           | If empty, runners use the latest manifest. Set explicitly to restore a specific backup.                                  |
| `TIMEOUT_SECONDS` / `--timeout`              |             int (seconds) |          `300`          |                           no                           | Overall operation timeout. Increase for large datasets or slow networks.                                                 |
| `KUBECTL_PROBE_TIMEOUT`                      |             int (seconds) |           `12`          |                           no                           | Timeout for port-forward readiness probing. Increase when cluster is slow to respond.                                    |
| `PORT_BASE` (`--port-base`)                  |                       int |    `7000` (CLI flag)    |                           no                           | Local base port for per-pod port-forward: local port = `PORT_BASE + pod_index`. Change if ports conflict on runner host. |

---

## Detailed descriptions, validation and concrete advice

### `QDRANT_NAMESPACE`

* **What:** Kubernetes namespace where the Qdrant StatefulSet and pods live.
* **Default:** `qdrant`.
* **Validation:** non-empty DNS-label string (k8s namespace rules).
* **When to change:**

  * Use `qdrant` for platform-standard deployments.
  * Change if you run multiple isolated clusters in one k8s cluster (e.g., `qdrant-staging`, `qdrant-prod`).
* **Example:** `export QDRANT_NAMESPACE="qdrant-prod"`

---

### `QDRANT_IMAGE`

* **What:** Container image reference in form `repository:tag`.
* **Default:** `qdrant/qdrant:v1.16.0`.
* **Validation:** must contain a `:` separating repo and tag; scripts will accept a missing tag but will treat it as a risk — always include explicit tag.
* **When to change:**

  * Upgrading/downgrading Qdrant. Set exact version for reproducible deploys.
  * When testing nightly/custom builds use `my-registry/qdrant:ci-20251228`.
* **Concrete rule:** Always pin a tag (no `:latest` in production).
* **Example:** `export QDRANT_IMAGE="qdrant/qdrant:v1.17.2"`

---

### `QDRANT_REPLICAS`

* **What:** Intended number of Qdrant pods (StatefulSet replicas).
* **Default:** `1`.
* **Validation:** positive integer ≤ cluster capacity.
* **When to change:**

  * Increase for capacity or availability; ensure storage class and replication factor are compatible.
  * For single-node test clusters keep `1`. For production multi-node, use `>=3` for fault tolerance (match `QDRANT_REPLICATION_FACTOR`).
* **Example:** `export QDRANT_REPLICAS=3`

---

### `QDRANT_REPLICATION_FACTOR`

* **What:** Logical replication factor used to render cluster peer configuration.
* **Default:** `1`.
* **Validation:** integer ≤ `QDRANT_REPLICAS`.
* **When to change:** increase when you want more copies of vectors across peers (increase durability/availability). Align with cluster topology.
* **Concrete rule:** set `QDRANT_REPLICATION_FACTOR <= QDRANT_REPLICAS`.
* **Example:** `export QDRANT_REPLICATION_FACTOR=2` (only valid if `QDRANT_REPLICAS >= 2`)

---

### `QDRANT__SERVICE__API_KEY`

* **What:** API key included as `api-key` header when calling Qdrant HTTP snapshot/restore endpoints.
* **Default:** none.
* **Validation:** non-empty secret string when Qdrant expects an API key.
* **When to change:** required in locked clusters where Qdrant rejects unauthenticated calls. Not needed for open clusters.
* **Security:** treat as secret; inject as Kubernetes Secret for the service; never write to repo or logs.
* **Example:** set in CI runner secret store, not in plain shell: `export QDRANT__SERVICE__API_KEY="${QDRANT_API_KEY_SECRET}"`

---

### `AZURE_STORAGE_CONNECTION_STRING`

* **What:** Full Azure Storage connection string used by azure-storage-blob client.
* **Default:** none.
* **Validation:** if present, scripts accept it and use for storage ops.
* **When to set:** easiest authentication method for service principals. Use when you cannot use managed identity.
* **Security:** secret — do not commit. Prefer SAS tokens or managed identity in production.
* **Example:** provided via CI secret: `export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=..."`

---

### `AZURE_STORAGE_ACCOUNT_NAME` & `AZURE_STORAGE_ACCOUNT_KEY`

* **What:** Alternative pair to `AZURE_STORAGE_CONNECTION_STRING` (account + key).
* **Default:** none.
* **Validation:** both required together.
* **When to set:** use if you manage account-level credentials separately (older patterns).
* **Security:** secret — do not commit. Consider issuing time-limited SAS tokens instead.
* **Example:**

  ```
  export AZURE_STORAGE_ACCOUNT_NAME="mystorageacct"
  export AZURE_STORAGE_ACCOUNT_KEY="base64-key"
  ```

---

### `BACKUP_AZURE_CONTAINER` / `AZURE_CONTAINER`

* **What:** The name of the Azure Blob container where snapshots and manifests are uploaded. CLI accepts `--azure-container`; the env var is `BACKUP_AZURE_CONTAINER` or `AZURE_CONTAINER` depending on script usage.
* **Default:** none — **you must supply** either env or CLI flag.
* **Validation:** container must exist or script will fail to upload (unless the script creates it; do not rely on auto-create).
* **When to change:** separate containers per environment (e.g., `qdrant-backups-prod` vs `qdrant-backups-staging`) or per team for separation.
* **Concrete example:** `export BACKUP_AZURE_CONTAINER="qdrant-backups-prod"`

---

### `BACKUP_AZURE_PREFIX` / `AZURE_PREFIX`

* **What:** Blob path prefix inside the container; used as folder-like key prefix.
* **Default:** `qdrant/backups`.
* **Validation:** string path (no leading slash recommended).
* **When to change:** organize backups by environment/cluster/date: e.g., `prod/us-east-1/qdrant/backups` or `staging/qdrant/backups`.
* **Concrete example:** `export BACKUP_AZURE_PREFIX="prod/qdrant/backups"`

---

### `PER_POD`

* **What:** Mode selector for backup/restore: `"true"` = per-pod, `"false"` = service-mode.
* **Default:** `"false"`.
* **Validation:** string literal `"true"` or `"false"` (scripts interpret strictly).
* **When to set to `"true"` (per-pod):**

  * Qdrant uses node-local NVMe or ephemeral local disks (each pod stores private data).
  * You require pod-by-pod snapshots (one snapshot artifact per pod).
  * You have `kubectl` access from the runner (port-forward is used).
* **When to leave `"false"` (service-level):**

  * Persistent volumes are network-attached (Azure Disk, EBS) and the service API can return collection-wide snapshots.
  * Simpler network and fewer port-forward requirements.
* **Concrete rule:** backup and restore **must use the same** `PER_POD` value as the manifest was created with. Setting `PER_POD=true` while manifest is service-mode will fail unless special mapping logic is used.
* **Example:** `export PER_POD="true"`

---

### `BACKUP_ID`

* **What:** Explicit identifier of the backup to restore (string).
* **Default:** empty `""`. When empty, the runner fetches `${AZURE_PREFIX}/latest.manifest.json` and uses its `backup_id`.
* **Validation:** if non-empty, the corresponding `${azure_prefix}/${backup_id}/manifest.json` must exist.
* **When to set:**

  * You want to restore a specific historical backup (for rollback or forensic recovery).
  * For automated DR where the latest is not desired.
* **Example:** `export BACKUP_ID="2025-12-28T12:00:00Z-abc123"`

---

### `TIMEOUT_SECONDS` / `--timeout`

* **What:** Global timeout for snapshot/restore operations (seconds). Scripts support `--timeout` flag; environment var may be used as default.
* **Default:** `300` (5 minutes).
* **Validation:** positive integer.
* **When to change:** increase for very large collections, slow networks, or large snapshot downloads/uploads. If timeouts are frequent, raise to 1800 (30 minutes) during maintenance windows.
* **Example:** `export TIMEOUT_SECONDS=1800`

---

### `KUBECTL_PROBE_TIMEOUT`

* **What:** Timeout (seconds) for the port-forward readiness probe (used in per-pod flows).
* **Default:** `12` seconds.
* **Validation:** positive integer.
* **When to change:** increase if cluster control-plane or pods are slow to accept port-forward (high API server load or slow startup). Set to `30` if many pods start slowly.
* **Example:** `export KUBECTL_PROBE_TIMEOUT=30`

---

### `PORT_BASE` (CLI flag `--port-base`)

* **What:** Local base port used for per-pod `kubectl port-forward`. Local port = `PORT_BASE + pod_index`.
* **Default:** `7000` (CLI flag).
* **Validation:** integer, avoid conflicts with other local services.
* **When to change:** if the runner host already has ports in the `7000+` range in use. Use a high ephemeral base such as `17000`.
* **Example CLI:** `--port-base 17000` or set `PORT_BASE=17000` if your scripts read it.

---

## Concrete examples for common scenarios

1. **Single-node test cluster (shared PV)**

   ```
   export QDRANT_NAMESPACE="qdrant-test"
   export QDRANT_IMAGE="qdrant/qdrant:v1.16.0"
   export PER_POD="false"
   export BACKUP_AZURE_CONTAINER="qdrant-backups-staging"
   export BACKUP_AZURE_PREFIX="staging/qdrant/backups"
   export AZURE_STORAGE_CONNECTION_STRING="<SECRET>"
   ```

   Use service-mode; no `kubectl` port-forward required.

2. **Production cluster with node-local NVMe**

   ```
   export QDRANT_NAMESPACE="qdrant-prod"
   export QDRANT_IMAGE="my-registry/qdrant:v1.17.0"
   export QDRANT_REPLICAS=3
   export QDRANT_REPLICATION_FACTOR=2
   export PER_POD="true"
   export BACKUP_AZURE_CONTAINER="qdrant-backups-prod"
   export BACKUP_AZURE_PREFIX="prod/qdrant/backups"
   export AZURE_STORAGE_ACCOUNT_NAME="mystorageacct"
   export AZURE_STORAGE_ACCOUNT_KEY="<SECRET>"
   export KUBECTL_PROBE_TIMEOUT=30
   export TIMEOUT_SECONDS=1800
   ```

   Use per-pod mode; ensure runner has `kubectl` access and no port conflicts.

3. **Restore a specific backup (explicit)**

   ```
   export BACKUP_ID="2025-12-28T12:00:00Z-abc123"
   export PER_POD="true"               # must match original backup mode
   python3 infra/runners/run_qdrant_restore.py --restore "${BACKUP_ID}" --per-pod --azure-container "${BACKUP_AZURE_CONTAINER}" --azure-prefix "${BACKUP_AZURE_PREFIX}" --namespace "${QDRANT_NAMESPACE}" --restart
   ```

---

## Validation rules & guardrails (practical checks)

* `PER_POD` must be `"true"` or `"false"`; otherwise the runner will abort. Validate in shell:

  ```bash
  [[ "${PER_POD}" == "true" || "${PER_POD}" == "false" ]] || { echo "PER_POD must be 'true' or 'false'"; exit 1; }
  ```
* `QDRANT_IMAGE` must include `:`. Quick check:

  ```bash
  [[ "${QDRANT_IMAGE}" =~ : ]] || { echo "QDRANT_IMAGE must include tag (repo:tag)"; exit 1; }
  ```
* Azure auth: one of the following must be present (CI secret injection recommended):

  * `AZURE_STORAGE_CONNECTION_STRING` **or**
  * `AZURE_STORAGE_ACCOUNT_NAME` **and** `AZURE_STORAGE_ACCOUNT_KEY` **or**
  * runner uses managed identity (no env vars; ensure RBAC granted).
* `BACKUP_AZURE_CONTAINER` must point to an existing container: validate with `az storage blob list --container-name $BACKUP_AZURE_CONTAINER` (requires az cli + creds).

---
