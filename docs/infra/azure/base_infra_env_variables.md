# Base Infra — Environment Variables (storage_acc.py)

This file documents every environment variable consumed by `infra/base_infra/storage_acc.py`. Each variable includes: purpose, allowed values / format, default (if any), validation rules, operational impact, and mutability.

---

## Required environment variables

### `AZURE_SUBSCRIPTION_ID`
```bash
export AZURE_SUBSCRIPTION_ID="00000000-0000-0000-0000-000000000000"
````

* **Purpose:** Azure subscription ID to target for all `az` operations.
* **Allowed format:** GUID (string). Must match an accessible subscription for the caller.
* **Default:** none — **required**.
* **Validation:** Non-empty; `az account set --subscription` must succeed.
* **Impact:** Determines where resource group and storage account are created or looked up.
* **Mutability:** Safe to change to operate different subscription (affects state and resources).

### `AZURE_STORAGE_ACCOUNT_NAME`

```bash
export AZURE_STORAGE_ACCOUNT_NAME="mystorageacct"
```

* **Purpose:** Primary storage account name used to create/adopt containers for data, Pulumi state, and backups.
* **Allowed format:** 3–24 characters, lowercase letters and digits only (`^[a-z0-9]{3,24}$`).
* **Default:** none — **required**.
* **Validation:** Regex enforced in script; rejection if invalid.
* **Impact:** Drives where containers are created and where Pulumi backend may live.
* **Mutability:** Changing implies creating/adopting a different storage account.

### `AZURE_CONTAINER` (or `AZURE_DATA_CONTAINER`)

```bash
export AZURE_CONTAINER="rag-data-515"
# or (legacy)
export AZURE_DATA_CONTAINER="rag-data-515"
```

* **Purpose:** Primary blob container for RAG data and application blobs.
* **Allowed format:** 3–63 characters; lowercase letters, digits or `-`; cannot start or end with `-`.
* **Default:** none — **required** (one of the two).
* **Validation:** Script requires either `AZURE_CONTAINER` or legacy `AZURE_DATA_CONTAINER`.
* **Impact:** Container will be created if missing (key-auth path) and used by workloads.
* **Mutability:** Changing requires coordination (existing blobs remain in old container).

---

## Optional environment variables (behavioral / names)

### `AZURE_RESOURCE_GROUP_NAME`

```bash
export AZURE_RESOURCE_GROUP_NAME="rg-e2e-rag"
```

* **Purpose:** Resource group to host storage account and containers.
* **Default:** `rg-e2e-rag`.
* **Validation:** Checked by `az group show`; name length/characters validated by Azure.
* **Impact:** Resource group resolution and creation context.
* **Mutability:** Changing results in creation/adoption in another RG.

### `AZURE_LOCATION`

```bash
export AZURE_LOCATION="centralindia"
```

* **Purpose:** Region for storage account creation.
* **Default:** `centralindia`.
* **Validation:** Must be an Azure region string.
* **Impact:** Affects data residency and available SKUs.
* **Mutability:** Effective at create time only.

### `STORAGE_TIER`

```bash
export STORAGE_TIER="LRS"  # LRS | ZRS | GRS | RAGRS | GZRS | RAGZRS or full SKU token
```

* **Purpose:** High-level token for storage SKU selection.
* **Default:** `LRS`.
* **Validation:** Mapped via `normalize_sku()` to names like `Standard_LRS`.
* **Impact:** Determines redundancy and cost of storage account at creation time.
* **Mutability:** Changing after creation not performed by script (adopt vs create semantics).

### `AZURE_ENDPOINT_SUFFIX`

```bash
export AZURE_ENDPOINT_SUFFIX="core.windows.net"
```

* **Purpose:** Endpoint suffix for Azure cloud (useful for sovereign clouds).
* **Default:** `core.windows.net`
* **Impact:** Affects constructed endpoints used if script ever constructs URLs (present but not heavily used).
* **Mutability:** Use appropriate cloud suffix when targeting non-public clouds.

### `PULUMI_AZ_CONTAINER`

```bash
export PULUMI_AZ_CONTAINER="pulumi-state-515"
```

* **Purpose:** Optional container name reserved for Pulumi backend state.
* **Default:** none (optional).
* **Validation:** Container name rules as above.
* **Impact:** If set, the script ensures the container exists (key auth path).
* **Mutability:** Changing moves where Pulumi state should be stored — coordinate migration.

### `BACKUP_AZ_CONTAINER`

```bash
export BACKUP_AZ_CONTAINER="backups-515"
```

* **Purpose:** Optional container name for backups and archives.
* **Default:** none (optional).
* **Impact:** Used when creating backup containers and applying lifecycle policy.
* **Mutability:** As above — careful when changing.

### `BACKUP_PREFIX`

```bash
export BACKUP_PREFIX="qdrant/backup"
```

* **Purpose:** Subdirectory/prefix in backup container where lifecycle policy will be scoped.
* **Default:** `qdrant`
* **Validation:** free-form string; used to create prefix `container/prefix/` for policy.
* **Impact:** Lifecycle policy applies only to blobs under that prefix path.
* **Mutability:** Safe to change — affects lifecycle targeting for future blobs.

### `BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS`

```bash
export BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS="7"
```

* **Purpose:** Days after modification to move blobs to Cool tier.
* **Allowed format:** Integer string (days).
* **Default:** none — optional.
* **Validation:** Must be integer if provided.
* **Impact:** Used only when `BACKUP_AZ_CONTAINER` is set; applied via management policy.
* **Mutability:** Changing affects future lifecycle behavior.

### `BACKUP_AZ_CONTAINER_RETENTION_DAYS`

```bash
export BACKUP_AZ_CONTAINER_RETENTION_DAYS="30"
```

* **Purpose:** Days after modification to permanently delete blobs under backup prefix.
* **Allowed format:** Integer string (days).
* **Default:** none — optional.
* **Validation:** Must be integer if provided.
* **Impact:** Used in lifecycle policy when backing up artifacts.
* **Mutability:** Deleting/shortening retention can delete blobs when policy triggers.

---

## Deletion / safety controls

### `AZURE_DELETE_ACCOUNT`

```bash
export AZURE_DELETE_ACCOUNT="1"   # "1"/"true" => delete storage account; "0"/"false" => delete containers only
```

* **Purpose:** Controls whether `--delete` removes the entire storage account or only containers.
* **Allowed values:** `1`/`0` or `true`/`false`.
* **Default:** `1` (script default: delete account).
* **Validation:** Interpreted boolean-like.
* **Impact:** `--delete` will remove storage account itself (if true) — destructive.
* **Mutability:** Per-run control; use with care.

### `FORCE_DELETE`

```bash
export FORCE_DELETE="1"
```

* **Purpose:** Skip interactive confirmation prompts when deleting.
* **Allowed values:** `1`/`0`, `true`/`false`.
* **Default:** `1`.
* **Impact:** With `AZURE_DELETE_ACCOUNT=1`, deletion proceeds non-interactively if `FORCE_DELETE=1`.
* **Mutability:** Per-run flag.

---

## Role / identity related (present but not used by default)

### `UAI_RAG_RW_NAME` / `UAI_RAG_RO_NAME`

```bash
export UAI_RAG_RW_NAME="uai-rag-rw"
export UAI_RAG_RO_NAME="uai-rag-ro"
```

* **Purpose:** Identifiers for user-assigned identities if role assignment logic is enabled or extended.
* **Default:** names provided in script but not created by current flow.
* **Impact:** Present for future or optional role assignment flows; script intentionally removes UAI creation logic to simplify bootstrap.
* **Mutability:** N/A for current script; reserved.

---

## Usage summary (script expectations)

* The script requires at minimum:

  * `AZURE_SUBSCRIPTION_ID`
  * `AZURE_STORAGE_ACCOUNT_NAME` (valid 3–24 lowercase alnum)
  * `AZURE_CONTAINER` (or `AZURE_DATA_CONTAINER`)
* Run:

  ```bash
  python infra/base_infra/storage_acc.py --create
  python infra/base_infra/storage_acc.py --delete
  ```
* `--create` will:

  * Ensure resource group exists (create if missing)
  * Create storage account if missing
  * Retrieve account key
  * Ensure `AZURE_CONTAINER` plus optional `PULUMI_AZ_CONTAINER`, `BACKUP_AZ_CONTAINER` exist
  * Optionally apply lifecycle policy when backup envs set
* `--delete` will:

  * If `AZURE_DELETE_ACCOUNT=1` → list containers and delete storage account (destructive)
  * Else → delete only specified containers (key auth required)

---




