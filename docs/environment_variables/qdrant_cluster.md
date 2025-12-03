# Qdrant Backup Generator — Documentation

## Overview

This document describes the environment variables and configuration used by the `gen_qdrant.py` manifest generator. The generator produces deterministic Kubernetes manifests for Qdrant (Helm values, ServiceAccount, CronJob) into `infra/manifests/qdrant/`. It supports two runtime modes:

* **STAGING / local (kind)** — generator will create/update a Kubernetes Secret `qdrant-backup-aws` in the target cluster from local `AWS_*` env values (so you can test backups against a dev S3 or MinIO).
* **PROD / EKS** — generator will not create secrets; instead it emits a ServiceAccount annotated for IRSA (IAM role) and assumes S3 access is provided by pod identity.

Below are two compact, ready-to-copy export blocks (one-line annotation to the right). Each block contains every env variable the generator recognizes.

## STAGING / Local (kind) — Export block (TL;DR)

```bash
export ENV=STAGING                             # tldr: local/dev mode; generator WILL create Kubernetes Secret from local AWS_* envs
export MANIFESTS_DIR=infra/manifests/qdrant    # output directory for generated manifests
export QDRANT_NAMESPACE=qdrant                 # k8s namespace for qdrant resources
export QDRANT_RELEASE=qdrant                   # release name / app label
export QDRANT_IMAGE_TAG=v1.16.1                # qdrant image tag used in values.yaml
export QDRANT_REPLICAS=1                       # replicas for qdrant statefulset in staging
export QDRANT_CPU=1                            # cpu request for qdrant pod
export QDRANT_MEMORY=4Gi                        # memory request for qdrant pod
export QDRANT_STORAGE=emptyDir                 # storage mode: emptyDir or pvc
export QDRANT_NODE_SELECTOR=""                 # nodeSelector key=value (optional)
export QDRANT_TAINT_KEY=qdrant-dedicated       # taint key expected on dedicated nodes
export QDRANT_TAINT_EFFECT=NoSchedule          # taint effect
export QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots  # snapshot path in qdrant pods
export BACKUP_S3_BUCKET=e2e-rag-system-42      # S3 bucket name for backups
export BACKUP_S3_PREFIX=qdrant/backups         # S3 path prefix for backups
export BACKUP_S3_REGION=ap-south-1             # S3 region (or endpoint region)
export BACKUP_S3_ENDPOINT=                      # optional custom S3 endpoint (MinIO)
export BACKUP_COMPRESSION=zstd                 # compression algorithm used for archive
export BACKUP_RETENTION=5                      # number of backups to keep (retention)
export BACKUP_SCHEDULE="0 */6 * * *"           # cron schedule for backups (string)
export BACKUP_IMAGE=your-registry/qdrant-backup:v2  # backup container image (must include zstd, aws, kubectl, jq, tar)
# sensitive for STAGING (used only to create in-cluster Secret; do NOT commit these):
export AWS_ACCESS_KEY_ID=AKIA...               # AWS access key id (staging only)
export AWS_SECRET_ACCESS_KEY=...               # AWS secret access key (staging only)
export AWS_SESSION_TOKEN=...                   # optional session token (staging only)
```

## PROD / EKS — Export block (TL;DR)

```bash
export ENV=PROD                                # tldr: production mode; generator will emit IRSA annotated ServiceAccount and will NOT create secrets
export MANIFESTS_DIR=infra/manifests/qdrant    # output directory for generated manifests
export QDRANT_NAMESPACE=qdrant                 # k8s namespace
export QDRANT_RELEASE=qdrant                   # release name / app label
export QDRANT_IMAGE_TAG=v1.16.1                # qdrant image tag for values.yaml
export QDRANT_REPLICAS=3                       # replicas for prod (recommended 2-3)
export QDRANT_CPU=4                            # cpu request for qdrant pod
export QDRANT_MEMORY=16Gi                       # memory request for qdrant pod
export QDRANT_STORAGE=pvc                       # PROD typically uses PVC (or local NVMe) configured separately
export QDRANT_NODE_SELECTOR=role=qdrant-nvme   # ensure qdrant pods schedule to NVMe nodes
export QDRANT_TAINT_KEY=qdrant-dedicated       # node taint key to isolate qdrant nodes
export QDRANT_TAINT_EFFECT=NoSchedule          # taint effect
export QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots  # snapshot path in qdrant pods
export BACKUP_S3_BUCKET=my-prod-qdrant-backups # S3 bucket for prod backups
export BACKUP_S3_PREFIX=qdrant/backups         # S3 prefix
export BACKUP_S3_REGION=us-east-1              # production AWS region
export BACKUP_S3_ENDPOINT=                      # empty for AWS S3; set for custom S3 compatible
export BACKUP_COMPRESSION=zstd                 # compression algorithm
export BACKUP_RETENTION=30                     # retention count for prod
export BACKUP_SCHEDULE="0 2 * * *"             # nightly backups in prod
export BACKUP_IMAGE=registry.example.com/qdrant-backup@sha256:...  # digest-pinned image recommended
export IRSA_ROLE_ARN=arn:aws:iam::123456789012:role/qdrant-backup-role  # required in PROD
# DO NOT set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in PROD; IRSA provides credentials
```

---

# Environment variable reference (detailed)

For each variable: purpose, valid values, default, staging example, prod example, operational notes.

---

### `ENV`

* **Purpose:** Mode switch for generator behaviour.
* **Valid:** `STAGING`, `PROD` (case-insensitive).
* **Default:** `STAGING`.
* **STAGING example:** `export ENV=STAGING` — generator will create/update `qdrant-backup-aws` Secret in the cluster using local AWS env vars.
* **PROD example:** `export ENV=PROD` — generator emits ServiceAccount with `eks.amazonaws.com/role-arn` annotation and will not create any in-cluster secret.
* **Notes:** Use this to control whether secrets are created and whether IRSA is required.

---

### `MANIFESTS_DIR`

* **Purpose:** Filesystem path where generated manifests are written.
* **Valid:** Any writable path.
* **Default:** `infra/manifests/qdrant`.
* **STAGING example:** `export MANIFESTS_DIR=infra/manifests/qdrant`.
* **PROD example:** same.
* **Notes:** Keep this directory tracked by Git so ArgoCD can pick up and deploy; `.inputs_hash` is created here and should remain gitignored.

---

### `QDRANT_NAMESPACE`

* **Purpose:** Kubernetes namespace used for Qdrant and backup-related resources.
* **Default:** `qdrant`.
* **STAGING:** same.
* **PROD:** same, or set to an environment-specific namespace (e.g., `qdrant-prod`).
* **Notes:** Generator will ensure namespace is present only for STAGING secret creation; ArgoCD will create namespace in cluster when deploying manifests.

---

### `QDRANT_RELEASE`

* **Purpose:** Release/app label used for pod discovery (e.g., `app=<release>`).
* **Default:** `qdrant`.
* **STAGING:** `qdrant`.
* **PROD:** `qdrant`.
* **Notes:** Keep consistent with your Helm release name or Deployment/StatefulSet labels.

---

### `QDRANT_IMAGE_TAG`

* **Purpose:** Qdrant container image tag used in `values.yaml`.
* **Default:** `v1.16.1`.
* **STAGING example:** `export QDRANT_IMAGE_TAG=v1.16.1` (match staging compatibility).
* **PROD example:** `export QDRANT_IMAGE_TAG=v1.16.1` or a stable vetted tag/digest in prod.
* **Notes:** Prefer immutable digests in prod if you require strict reproducibility.

---

### `QDRANT_REPLICAS`

* **Purpose:** Replica count used for generation defaults.
* **Default:** `1` in STAGING; `3` in PROD (generator default logic).
* **STAGING example:** `export QDRANT_REPLICAS=1`.
* **PROD example:** `export QDRANT_REPLICAS=3`.
* **Notes:** Actual StatefulSet replica count is controlled by the Helm chart / manifests ArgoCD applies; generator writes values used by Helm.

---

### `QDRANT_CPU`

* **Purpose:** CPU request for Qdrant pods in generated values.
* **Default:** `1` (STAGING) / `4` (PROD).
* **Examples:** `export QDRANT_CPU=4`.
* **Notes:** Tune based on your expected index size and load; requests and limits should be set conservatively.

---

### `QDRANT_MEMORY`

* **Purpose:** Memory request for Qdrant pods.
* **Default:** `4Gi` (STAGING) / `16Gi` (PROD).
* **Examples:** `export QDRANT_MEMORY=16Gi`.
* **Notes:** Qdrant memory requirements scale with collection size and vector dimensionality.

---

### `QDRANT_STORAGE`

* **Purpose:** Storage mode used by generated values (`emptyDir` or `pvc`).
* **Default:** `emptyDir`.
* **STAGING example:** `export QDRANT_STORAGE=emptyDir` (no PVC, useful for local).
* **PROD example:** `export QDRANT_STORAGE=pvc` (use EBS or local NVMe backed PVC).
* **Notes:** For PROD, prefer PVC backed by local NVMe or high-throughput EBS with appropriate StorageClass.

---

### `QDRANT_NODE_SELECTOR`

* **Purpose:** Optional node selector in `key=value` form to schedule Qdrant on specialized nodes.
* **Default:** empty.
* **STAGING example:** unset (let K8s schedule).
* **PROD example:** `export QDRANT_NODE_SELECTOR="role=qdrant-nvme"` to ensure placement on NVMe nodes.
* **Notes:** When set, generator renders nodeSelector only in PROD values (to avoid local scheduling issues).

---

### `QDRANT_TAINT_KEY` / `QDRANT_TAINT_EFFECT`

* **Purpose:** Taint key and effect used to isolate qdrant nodes; tolerations are added to values.
* **Defaults:** `qdrant-dedicated` and `NoSchedule`.
* **STAGING:** defaults.
* **PROD:** can remain defaults or align with your node taint strategy (e.g., `NoSchedule` or `PreferNoSchedule`).
* **Notes:** Ensure node groups used for Qdrant have matching taints.

---

### `QDRANT__STORAGE__SNAPSHOTS_PATH`

* **Purpose:** Path inside Qdrant pods where snapshots exist (used by backup script).
* **Default:** `/qdrant/snapshots`.
* **STAGING:** same.
* **PROD:** same unless your container layout differs.
* **Notes:** Backup CronJob checks for this path when creating archives.

---

### `BACKUP_S3_BUCKET`

* **Purpose:** S3 bucket name used as the root for uploads.
* **Default:** `e2e-rag-system-42` (placeholder).
* **STAGING example:** `export BACKUP_S3_BUCKET=dev-qdrant-backups` (can point to MinIO or dev S3).
* **PROD example:** `export BACKUP_S3_BUCKET=my-prod-qdrant-backups`.
* **Notes:** Bucket name must conform to S3 naming rules. Generator warns if format looks invalid.

---

### `BACKUP_S3_PREFIX`

* **Purpose:** S3 key prefix under the bucket where backups are written.
* **Default:** `qdrant/backups`.
* **Examples:** `export BACKUP_S3_PREFIX=qdrant/backups`.
* **Notes:** Final backup path uses `${PREFIX}/${BACKUP_ID}/${POD}.tar.zst`. `latest.manifest.json` is written at `${PREFIX}/latest.manifest.json`.

---

### `BACKUP_S3_REGION`

* **Purpose:** Region for S3 uploads (also used for AWS CLI `--region`).
* **Default:** value of `AWS_REGION` or `us-east-1`.
* **STAGING example:** `ap-south-1` (your dev region).
* **PROD example:** `us-east-1`.
* **Notes:** With custom `BACKUP_S3_ENDPOINT`, region may be less relevant for non-AWS endpoints.

---

### `BACKUP_S3_ENDPOINT`

* **Purpose:** Optional custom S3 endpoint (e.g., MinIO, Ceph).
* **Default:** empty (use AWS S3).
* **STAGING example:** `http://minio.local:9000` for local testing.
* **PROD example:** leave empty for AWS S3.
* **Notes:** Ensure the backup image can reach and authenticate against this endpoint.

---

### `BACKUP_COMPRESSION`

* **Purpose:** Compression algorithm used for archives. Current generator uses `zstd`.
* **Default:** `zstd`.
* **Examples:** `zstd`.
* **Notes:** Backup image must include the chosen tool (`zstd` binary). The generator uses `zstd -19 --long=31` for deterministic compression.

---

### `BACKUP_RETENTION`

* **Purpose:** Number of backup snapshots to keep (used by your retention/cleanup logic).
* **Default:** `5` (STAGING) / set higher in PROD.
* **STAGING example:** `export BACKUP_RETENTION=5`.
* **PROD example:** `export BACKUP_RETENTION=30`.
* **Notes:** Generator writes metadata; you should implement lifecycle policies or cleanup jobs as needed.

---

### `BACKUP_SCHEDULE`

* **Purpose:** Cron expression string for the CronJob schedule.
* **Default:** `"0 */6 * * *"` (every 6 hours).
* **STAGING example:** `"*/15 * * * *"` (every 15 minutes for fast testing).
* **PROD example:** `"0 2 * * *"` (nightly at 02:00 UTC).
* **Notes:** Ensure schedule suits snapshot duration and cluster load.

---

### `BACKUP_IMAGE`

* **Purpose:** Backup job container image. Must contain `kubectl`, `aws` CLI, `zstd`, `tar`, `jq`.
* **Default:** `athithya5354/qdrant-backup:v2` (example).
* **STAGING example:** `docker.io/youruser/qdrant-backup:dev` (test).
* **PROD example:** `registry.example.com/qdrant-backup@sha256:<digest>` (digest-pinned).
* **Notes:** Use digest pinned images in PROD for immutability.

---

### `IRSA_ROLE_ARN`

* **Purpose:** IAM Role ARN to annotate the ServiceAccount with for IRSA in EKS. When set, generator emits that annotation and does not create secrets.
* **Default:** empty.
* **STAGING:** leave empty or set; generator will create secret when ENV=STAGING and local AWS creds exist.
* **PROD example:** `export IRSA_ROLE_ARN=arn:aws:iam::123456789012:role/qdrant-backup-role`.
* **Notes:** Role must have policies to allow `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on the configured bucket, and `kms:Decrypt` if using KMS-encrypted objects.

---

### `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (sensitive)

* **Purpose:** These are used **only** in STAGING by the generator to create an in-cluster Kubernetes Secret named `qdrant-backup-aws`. They are **never** written into generated manifests.
* **Default:** none.
* **STAGING example:** set locally to allow generator to create the secret and the CronJob to run against dev S3/MinIO.
* **PROD:** **do not set** these; use IRSA instead.
* **Notes:** Never commit these values to Git. The generator will warn if it detects these env vars and will refuse to embed secrets in file output.

---

## Final operational notes (concise)

* For **STAGING**: set `ENV=STAGING` and have `AWS_*` credentials in your shell if you want the generator to create the in-cluster secret automatically. Use a local or dev S3 endpoint for safe testing. Do not commit real credentials to git.
* For **PROD**: set `ENV=PROD`, set `IRSA_ROLE_ARN` to your IAM role ARN, and **unset** `AWS_*` keys locally. The generated ServiceAccount will contain the IRSA annotation and the CronJob will rely on pod identity for S3 access.
* Always validate the generated YAML with `kubectl apply --dry-run=client` or a validator like `kubeval` before committing. Commit the generated manifests (excluding secrets); ArgoCD will manage deployment.

---
