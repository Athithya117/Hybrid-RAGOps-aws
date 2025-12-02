# Quick plain-English summary

This is the list of environment variables the generator and generated Qdrant backup subsystem use. Each variable controls one small thing (where to put backups, how many replicas, whether to use IRSA, etc.). Below I show each variable, what it does in one line, and two concrete example settings: **STAGING** (kind / dev-like) and **PROD** (EKS with NVMe + IRSA).

**Assumptions**

* ArgoCD will apply the generated manifests; the generator only writes files.
* STAGING runs in a local/kind cluster and may use AWS credentials from env to reach an S3-compatible endpoint (or MinIO).
* PROD runs in EKS; backup CronJob will use IRSA (ServiceAccount annotated with role ARN) and NVMe nodeSelector for Qdrant pods.
* Backup archives use `.tar.zst`.

---

# Environment variables — grouped, explained, STAGING vs PROD examples

## A. Core runtime / placement

* `ENV`

  * What: Mode selector used to pick defaults (affects replica counts, scheduling).
  * STAGING example: `STAGING`
  * PROD example: `PROD`

* `QDRANT_NAMESPACE`

  * What: Kubernetes namespace where Qdrant resources live.
  * STAGING: `qdrant`
  * PROD: `qdrant-prod`

* `QDRANT_RELEASE`

  * What: Helm release name / base label used by manifests and selectors.
  * STAGING: `qdrant`
  * PROD: `qdrant`

* `QDRANT_REPLICAS`

  * What: Number of Qdrant pods (replicas). Generator uses 1 for STAGING, 3 for PROD by default.
  * STAGING: `1`
  * PROD: `3`

* `QDRANT_IMAGE_TAG`

  * What: Qdrant container image tag to deploy.
  * STAGING: `v1.16.1`
  * PROD: `v1.16.1` (pin to same stable tag)

* `CHART_VERSION`

  * What: Helm chart version to reference (keeps chart behavior stable).
  * STAGING: `1.16.0`
  * PROD: `1.16.0`

## B. Node scheduling & storage

* `QDRANT_STORAGE`

  * What: `emptyDir` (no PVC) or `pvc`. Controls persistence mode in values.yaml.
  * STAGING: `emptyDir`
  * PROD: `emptyDir` (or `pvc` if you change strategy)

* `QDRANT_NODE_SELECTOR`

  * What: Node label key=value to schedule Qdrant on NVMe nodes in PROD.
  * STAGING: `role=kind-worker`
  * PROD: `role=qdrant-nvme`

* `QDRANT_TAINT_KEY` / `QDRANT_TAINT_EFFECT`

  * What: Taint key and effect used on qdrant nodegroup; generator adds toleration so qdrant pods can run there.
  * STAGING: `qdrant-dedicated` / `NoSchedule`
  * PROD: `qdrant-dedicated` / `NoSchedule`

## C. Qdrant config essentials

* `QDRANT__SERVICE__API_KEY`

  * What: API key injected into Qdrant env to protect REST endpoints (recommended in PROD).
  * STAGING: `dev-key-REPLACE`
  * PROD: (store in sealed secret / set via external secret) e.g. `prod-secure-api-key`

* `QDRANT__CLUSTER__ENABLED`

  * What: `true`/`false` to enable distributed cluster behavior.
  * STAGING: `true`
  * PROD: `true`

* `QDRANT__CLUSTER__P2P__PORT`

  * What: internal P2P port used by the cluster (default 6335).
  * STAGING: `6335`
  * PROD: `6335`

* `QDRANT__STORAGE__SNAPSHOTS_PATH`

  * What: path inside the pod where snapshots live (used by tar).
  * STAGING: `/qdrant/snapshots`
  * PROD: `/qdrant/snapshots`

* `QDRANT_RESTORE_ON_BOOT`

  * What: if `true`, an init step can restore from S3 on pod start (optional). Usually `false` in STAGING, `true` in PROD if automated restores desired.
  * STAGING: `false`
  * PROD: `true`

## D. Backup subsystem (S3 + compression)

* `BACKUP_S3_BUCKET`

  * What: S3 bucket for backups.
  * STAGING: `dev-qdrant-backups` (or a MinIO bucket)
  * PROD: `e2e-rag-backups`

* `BACKUP_S3_PREFIX`

  * What: S3 path prefix under the bucket where backups live.
  * STAGING: `qdrant/staging`
  * PROD: `qdrant/prod`

* `BACKUP_S3_REGION`

  * What: S3 region for AWS CLI.
  * STAGING: `us-east-1`
  * PROD: `ap-south-1`

* `BACKUP_S3_ENDPOINT`

  * What: Optional custom S3 endpoint (MinIO/CEPH). Empty for AWS.
  * STAGING: `http://minio.local:9000` (if using MinIO)
  * PROD: `` (empty)

* `BACKUP_COMPRESSION`

  * What: `zstd` (preferred) or `none`. Generator uses `zstd` by default.
  * STAGING: `zstd`
  * PROD: `zstd`

* `BACKUP_RETENTION`

  * What: How many backup directories to keep; generator doesn't auto-delete unless you enable it.
  * STAGING: `3`
  * PROD: `30`

* `BACKUP_SCHEDULE`

  * What: Cron schedule for CronJob (default every 6 hours).
  * STAGING: `0 */6 * * *`
  * PROD: `0 */6 * * *`

* `BACKUP_IMAGE`

  * What: The container image used by the CronJob; must include zstd, jq, aws-cli, kubectl. Use pin/digest.
  * STAGING: `registry.example/local-qdrant-backup:sha256:...`
  * PROD: `registry.example/prod-qdrant-backup:sha256:...`

## E. Auth / AWS (STAGING uses static creds; PROD uses IRSA)

* STAGING (env-based):

  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `AWS_SESSION_TOKEN` (optional)
  * Example: set with a MinIO or test AWS user.

* PROD (IRSA):

  * `IRSA_ROLE_ARN`
  * What: IAM role ARN to annotate the backup ServiceAccount. CronJob uses IRSA for S3 writes.
  * Example: `arn:aws:iam::123456789012:role/qdrant-backup-role`

## F. Optional / indexer & collection-level (not cluster envs)

* `QDRANT_SHARD_NUMBER`, `QDRANT_REPLICATION_FACTOR`, `QDRANT_WRITE_CONSISTENCY_FACTOR`

  * What: defaults used by your indexer when creating collections; **these are used at collection creation time (indexing pipeline)**, not as global cluster envs.
  * STAGING: `1`, `1`, `1`
  * PROD: `8`, `3`, `2` (example)

---

# Quick examples: one-liners you can set locally

**STAGING (kind or dev)**

```bash
export ENV=STAGING
export BACKUP_S3_BUCKET=dev-qdrant-backups
export BACKUP_S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export BACKUP_IMAGE=registry.local/qdrant-backup:staging
```

**PROD (EKS + IRSA + NVMe)**

```bash
export ENV=PROD
export BACKUP_S3_BUCKET=e2e-rag-backups
export BACKUP_S3_REGION=ap-south-1
export IRSA_ROLE_ARN=arn:aws:iam::123456789012:role/qdrant-backup-role
export QDRANT_NODE_SELECTOR=role=qdrant-nvme
export BACKUP_IMAGE=registry.prod/qdrant-backup@sha256:<digest>
export QDRANT_REPLICAS=3
export QDRANT_IMAGE_TAG=v1.16.1
```

---

# Short notes / guidance

* **API key** (`QDRANT__SERVICE__API_KEY`) is critical in PROD; use Secrets or external secret operator (do not commit to git).
* **IRSA**: annotate ServiceAccount with `IRSA_ROLE_ARN` in manifests so the CronJob has S3 permissions without embedding creds.
* **Compression**: `BACKUP_COMPRESSION=zstd` is recommended; ensure the backup image includes `zstd`.
* **on_disk**: per-collection `on_disk` setting is configured by your indexer at collection creation time — not via these cluster envs. Make sure your indexing pipeline passes `on_disk:true` when you want disk-backed collections.

---

