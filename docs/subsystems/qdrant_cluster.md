# Qdrant Backup Generator — How it Works & Runtime Control Flow

## Overview

This document describes the internal operation and runtime control flow of the `infra/generators/gen_qdrant.py` subsystem. The generator produces deterministic Kubernetes manifests required for Qdrant backups and (when configured) creates a staging Secret in-cluster. It is strictly a manifest generator + optional staging-secret applier; ArgoCD or another GitOps agent is expected to apply the generated manifests to the cluster.

## Purpose and outputs

* Generate deterministic, idempotent YAML files into `infra/manifests/qdrant/`:

  * `values.yaml` — Helm values for Qdrant chart (replica, resources, snapshots S3 config, etc.)
  * `serviceaccount-backup.yaml` — ServiceAccount; IRSA annotation added when `IRSA_ROLE_ARN` is set
  * `backup-cronjob.yaml` — CronJob that snapshots Qdrant pods, compresses archives (`.tar.zst`), uploads to S3, and writes manifest metadata (per-backup and `latest.manifest.json`)
  * `secret-sample.yaml` — placeholder sample (never contains real credentials)
  * `.inputs_hash` — local metadata used to skip regenerating unchanged files (kept outside commit ideally)
* Optionally create/update a Kubernetes Secret `qdrant-backup-aws` in the cluster when `ENV=STAGING` and local AWS creds are present. This secret is **never** written into generated manifests.

## Inputs

The generator reads:

* CLI flags:

  * `--dry-run` — render to stdout only; do not write manifests or create secrets
  * `--delete` — delete generated manifests and remove staging secret from cluster (if present)
* Environment variables (full list omitted here; used to populate values, schedule, image, S3 config, IRSA role, and staging creds)

## High-level runtime control flow (step-by-step)

### 1 — CLI entry and argument parsing

* Program reads CLI args.

  * If `--delete` is passed → proceed to delete flow (section 7).
  * Otherwise continue to normal generate flow.
  * `--dry-run` toggles non-destructive behavior for both manifest generation and staging-secret creation.

### 2 — Load and normalize configuration

* Reads environment variables into a typed `cfg` dictionary.
* Normalizes defaults based on `ENV`:

  * `ENV=STAGING` default values prioritize local/dev friendly settings (1 replica, `emptyDir`, etc.).
  * `ENV=PROD` defaults favor production settings (multiple replicas, `pvc` expected, heavier resources).
* Sensitive items (AWS credentials, Qdrant API key) are recognized and tracked but not embedded into generated manifest files.

### 3 — Compute deterministic inputs hash

* Builds a canonical, JSON-safe representation of all non-sensitive, generator-config inputs.
* Computes SHA256 over canonical JSON with sorted keys and stable separators.
* This hash is stored in `.inputs_hash` and compared on subsequent runs to avoid rewriting unchanged manifests.

### 4 — Build rendering context

* Converts raw config values into a template rendering context:

  * Node selector parsing (`key=value`) if present.
  * Snapshot paths, S3 settings, schedule, image, resource requests, tolerations, and IRSA role annotation flag.

### 5 — Render templates (purely local, deterministic)

* Templates (embedded in the file) are rendered using Jinja2 with strict formatting settings (trim blocks, lstrip blocks).
* Rendered templates are deterministic: same inputs produce identical YAML bytes (subject to env inputs hash check).

### 6 — Dry-run behavior

* If `--dry-run` is active: render outputs are printed to stdout and no files are written and no secrets are created/applied. Execution returns.

### 7 — Manifest write (idempotent)

* If inputs-hash differs from stored `.inputs_hash` (or no hash exists) and not dry-run:

  * Create target manifest directory if missing.
  * Atomically write each generated file (`values.yaml`, `serviceaccount-backup.yaml`, `backup-cronjob.yaml`, `secret-sample.yaml`) using a temporary-file replace pattern to prevent partial writes.
  * Update `.inputs_hash` with the newly computed canonical hash.
* If inputs-hash matches and not dry-run: skip writes and print "No non-secret changes detected; skipping write."

### 8 — Staging secret lifecycle (only when `ENV=STAGING`)

* If `ENV=STAGING` and local AWS credentials (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`) are present:

  * Ensures namespace exists (attempts to create namespace client-side if needed).
  * Builds an in-memory Secret manifest (stringData) with the AWS values and optionally `AWS_SESSION_TOKEN`.
  * If not in `--dry-run`: applies the Secret to the cluster using `kubectl apply -f -`.
  * The generator prints the result (created/updated) but **never** writes these credentials into git-managed files.
* If no local credentials found: the generator skips secret creation and does not create any staging secret.

### 9 — PROD differences (IRSA)

* If `ENV=PROD`:

  * Generator will **not** create any Kubernetes Secret.
  * If `IRSA_ROLE_ARN` is provided, `serviceaccount-backup.yaml` receives `eks.amazonaws.com/role-arn: "<IRSA_ROLE_ARN>"`.
  * CronJob continues to reference AWS credentials via `valueFrom: secretKeyRef` with `optional: true`. In PROD this secret will not exist and IAM role will be used by the container process (pod identity).
  * If `IRSA_ROLE_ARN` is not provided in PROD, generator prints a warning suggesting IRSA should be configured to avoid secrets in production.

### 10 — CronJob runtime expectations (what the generated CronJob will do at run time)

* At scheduled time, CronJob Pod runs backup container; it expects the container image to include `kubectl`, `aws` CLI, `zstd`, `tar`, and `jq`.
* Behavior performed by CronJob script:

  1. Determine timestamp-backed `BACKUP_ID`.
  2. Discover Qdrant pods using label selector (default `app=<release>`).
  3. For each pod:

     * Check that snapshot directory exists inside the pod (`QDRANT__STORAGE__SNAPSHOTS_PATH`).
     * Exec into pod and create tar stream of snapshot directory, compress with `zstd -19 --long=31`, write to a local temp file.
     * Compute archive SHA256 and upload both archive and checksum to S3 under `${BACKUP_PREFIX}/${BACKUP_ID}/${POD}.tar.zst`.
  4. Create a per-backup manifest JSON and upload it to S3 at `${BACKUP_PREFIX}/${BACKUP_ID}/manifest.json`.
  5. Update `latest.manifest.json` at `${BACKUP_PREFIX}/latest.manifest.json` with the new manifest (atomic overwrite).
* The CronJob also performs best-effort partial uploads and logs when snapshot for a pod fails; it continues to other pods rather than aborting the entire job.

### 11 — Delete flow (`--delete`)

* If invoked with `--delete`:

  * Deletes all generated files under `MANIFESTS_DIR`.
  * Removes `.inputs_hash` if present.
  * If `ENV=STAGING`, attempts to delete the `qdrant-backup-aws` Secret from the cluster using `kubectl delete secret ... --ignore-not-found`.
  * Prints results and exits.

## Fault-tolerant and deterministic design choices

* Deterministic hashing ensures manifests are rewritten only when non-sensitive inputs change; this reduces churn in Git and avoids noisy ArgoCD diffs.
* Atomic file writes prevent partially-written YAMLs.
* Secret creation in STAGING is done via `kubectl apply -f -` to avoid writing credentials to disk or Git.
* CronJob uses `optional: true` for secret-based AWS env injection; this produces identical manifests across STAGING and PROD while enabling two different authentication mechanisms:

  * STAGING: explicit Secret injected
  * PROD: IRSA pod identity (no secret present)
* Templates avoid runtime randomness; unique run IDs are generated at job execution time only (inside CronJob script), not in generated manifests.

## Error handling and observability points

* The generator prints warnings for invalid S3 bucket names, missing IRSA in PROD, and detection of local AWS credentials.
* If `kubectl` is missing when attempting to create a staging Secret, the generator logs an informative message and skips secret creation.
* The CronJob script uses per-pod try/catch logic: failures for one pod do not stop the entire backup job.
* Manifest existence and input-hash logic prevents unexpected re-writes; if a manual change is made to generated files outside the generator, the next generator run may overwrite them if inputs indicate change.

## Security considerations

* The generator never writes actual credentials into generated manifest files placed under `MANIFESTS_DIR`. Secrets are applied directly to the cluster in STAGING only when local credentials are present.
* For PROD, IRSA should be used; the ServiceAccount is annotated when `IRSA_ROLE_ARN` is provided.
* `secret-sample.yaml` is intentionally dummy content and safe for Git; real secrets must be managed by secret-management tooling (ExternalSecrets, SOPS + ArgoCD SOPS plugin, SealedSecrets, etc.) if they need to be in version control.

## Outcome

* Generated manifests are GitOps-ready and deterministic.
* STAGING and PROD workflows use a single set of manifests with identical structure; authentication differs by environment (in-cluster Secret vs IRSA) while the manifest schema remains the same, enabling seamless GitOps promotion between environments.
