# Base Infrastructure — Storage Account & Containers (storage_acc.py)

This document explains the purpose and behavior of the base infra bootstrapper `infra/base_infra/storage_acc.py`. It describes responsibilities, auth model used, lifecycle operations, and the exact sequence the script performs for create and delete operations.

---

## Purpose

`storage_acc.py` is an env-driven bootstrap utility responsible for creating or deleting Azure Storage resources required by the platform:

- Create or adopt a storage account (StorageV2).
- Ensure application data container exists (`AZURE_CONTAINER`).
- Ensure Pulumi backend container exists (if `PULUMI_AZ_CONTAINER`).
- Ensure a backup container exists (if `BACKUP_AZ_CONTAINER`) and optionally apply lifecycle management for backup prefixes.
- Support deletion of containers or the entire storage account in a controlled way.

The script intentionally operates in **key-auth mode** (account key), simplifying initial bootstrap where managed identity role propagation may be restrictive.

---

## Authentication model

- Primary mode: **Azure CLI** (`az`) is required for subscription selection and some operations.
- Container ops are performed using **account key auth**:
  - The script fetches the storage account key via `az storage account keys list` and uses `--auth-mode key --account-key`.
- Fallbacks in other tooling may attempt RBAC role assignment, but this script uses keys to guarantee idempotence and immediate access.

---

## Resource types created

- **Resource Group** (optional creation if missing): `az group create`.
- **Storage Account** (`StorageV2`) with:
  - `--sku` mapped from `STORAGE_TIER` (normalized to `Standard_LRS`, etc.)
  - `--kind StorageV2`
  - `--https-only true`
- **Blob Containers**:
  - Primary data container (`AZURE_CONTAINER` or `AZURE_DATA_CONTAINER`)
  - Pulumi state container (if `PULUMI_AZ_CONTAINER` set)
  - Backup container (if `BACKUP_AZ_CONTAINER` set)

---

## Lifecycle policy (backup container)

- Optional lifecycle policy applied using `az storage account management-policy create`.
- Policy targets a container prefix constructed as: `<BACKUP_AZ_CONTAINER>/<BACKUP_PREFIX>/`
- Actions supported:
  - `tierToCool` after `BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS` (if provided)
  - `delete` after `BACKUP_AZ_CONTAINER_RETENTION_DAYS` (if provided)
- The script writes a temporary JSON policy and applies it; failures in lifecycle application are non-fatal (logged).

---

## Creation flow (what `--create` does, step-by-step)

1. Validate minimum environment variables (subscription, storage account name, container).
2. `az account set --subscription` to target subscription.
3. Ensure resource group exists (`az group create` if missing).
4. Create storage account (if it does not already exist), polling until active.
5. Retrieve storage account resource and account key (`az storage account keys list`).
6. Create/ensure containers:
   - `AZURE_CONTAINER` (required)
   - `PULUMI_AZ_CONTAINER` (optional)
   - `BACKUP_AZ_CONTAINER` (optional)
   All container creation uses key auth to avoid RBAC propagation delays.
7. Optionally apply backup lifecycle policy when backup envs are set.
8. Print the list of ensured containers and finish.

---

## Deletion flow (what `--delete` does, step-by-step)

There are two deletion modes controlled by `AZURE_DELETE_ACCOUNT`:

### A. Storage account deletion (`AZURE_DELETE_ACCOUNT=1`)
1. Inventory containers and sample blob counts using key auth.
2. If `FORCE_DELETE` is not set, prompt for confirmation (`Type 'yes' to confirm`).
3. Delete the storage account with `az storage account delete --yes`.
4. Optionally poll until the account disappears (configurable wait/poll).

### B. Container-only deletion (`AZURE_DELETE_ACCOUNT=0`)
1. Verify storage account exists.
2. For each planned container (`AZURE_CONTAINER`, `PULUMI_AZ_CONTAINER`, `BACKUP_AZ_CONTAINER`), list blobs to estimate impact.
3. If `FORCE_DELETE` not set, prompt for confirmation.
4. Delete specified containers using key auth.

---

## Error handling and logging

- The script uses simple timestamped `INFO` / `WARN` / `ERROR` lines for observability.
- It fails fast on missing required env variables and on unrecoverable CLI errors.
- Role assignment helpers include retries and backoff to tolerate Azure replication delays.
- Lifecycle policy apply is non-fatal: failures are logged and do not abort the create flow.

---

## Assumptions and invariants

- Storage account name must conform to Azure naming rules (3–24 lowercase letters/digits).
- Container names follow blob container constraints and are created with `public_access=none` semantics by default (script uses CLI `storage container create` with `--auth-mode key` or `--auth-mode login` depending on path).
- The script intentionally avoids creating or managing user-assigned identities (UAIs) to simplify bootstrap and avoid role propagation delays.
- Key-auth is used to guarantee immediate access for container operations during bootstrap.

---

## CLI usage

- Create resources (idempotent):
```bash
python infra/base_infra/storage_acc.py --create
````

* Delete resources (containers or storage account depending on env):

```bash
python infra/base_infra/storage_acc.py --delete
```

---

## Minimum required environment for runs

* `AZURE_SUBSCRIPTION_ID`
* `AZURE_STORAGE_ACCOUNT_NAME` (3–24 lowercase letters/digits)
* `AZURE_CONTAINER` or `AZURE_DATA_CONTAINER`

All other environment variables are optional and documented in `base_infra_env_variables.md`.

---
