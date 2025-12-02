# **High-Level Purpose**

`infra/generators/gen_qdrant.py` is a **manifest generator**, not an applier.
Its job:

1. Read env variables + CLI flags (`--dry-run`, `--delete`).
2. Render all YAML deterministically (Qdrant cluster + Service + CronJobs + ConfigMaps).
3. Write them to `infra/manifests/qdrant/`.
4. Never talk to the cluster — ArgoCD will apply them.

It is **fully deterministic**:

* same inputs → identical output bytes
* no randomness
* no network calls
* all logic done in pure Python

---

# **Runtime Control Flow (Step-by-Step)**

Below is the internal control flow precisely as it runs:

---

## **1. Start → Parse CLI arguments**

The script supports:

* `--dry-run` → prints YAML to stdout instead of writing to manifests/
* `--delete` → generates *empty directory* manifest to tell ArgoCD to remove Qdrant
* default → generates full manifest set into manifests/qdrant/

This stage also prohibits conflicting flags (e.g., dry-run + delete).

---

## **2. Load and normalize environment variables**

The generator reads all subsystem env vars:

* **ENV:** `DEV | STAGING | PROD`
* **QDRANT_REPLICAS**
* **QDRANT_CHART_VERSION`
* **QDRANT_IMAGE_TAG**
* **BACKUP_BUCKET`, `BACKUP_REGION`, `BACKUP_PREFIX`
* **BACKUP_IMAGE** (CronJob image that contains zstd, aws-cli, kubectl)
* **S3_AUTH_MODE** (DEV uses local/minio credentials, PROD uses IRSA)
* **QDRANT_STORAGE_PATH`, `QDRANT_SNAPSHOTS_PATH`
* **QDRANT_CONFIG:** `shard_number`, `replication_factor`, `write_consistency_factor`

The script enforces defaults based on ENV:

* DEV → 1 replica, PVC or ephemeral, local S3 (kind)
* STAGING → 1–3 replicas, S3 creds via env
* PROD → 3 replicas, IRSA, S3 backups, NVMe ephemeral volumes optional

It validates:

* required vars present
* bucket names correct format
* Qdrant cluster topology valid

---

## **3. Build an internal deterministic “model”**

The script constructs an **in-memory representation** of all resources:

```
{
  "statefulset": {...},
  "service": {...},
  "backup-cronjob": {...},
  "restore-cronjob": {...},
  "configmap-backup-script": {...},
  "configmap-restore-script": {...}
}
```

Each resource is pure Python dictionaries.

This guarantees:

* No YAML ordering issues
* Deterministic key ordering
* No runtime network or cluster lookups

---

## **4. Render Jinja2 templates → deterministic YAML**

Each resource dictionary is passed through:

```
jinja_env.get_template("resource.yaml.j2").render(model)
```

All templates are embedded **inside the file**, so the generator is truly self-contained.

Jinja2 is configured with:

* sorted keys
* no undefined variables allowed
* newline normalization
* indentation validation

Thus YAML output is stable across machines.

---

## **5. If `--dry-run`** → dump YAML to stdout and exit

No writing occurs.

---

## **6. If `--delete`** → generate “zero object” YAML

The file tree becomes:

```
infra/manifests/qdrant/
    delete.yaml
```

ArgoCD will reconcile → remove Qdrant gracefully.

---

## **7. Else → write YAML files to `manifests/qdrant/`**

The script:

1. Clears the target folder

2. Writes each YAML as its own file:

   * `qdrant-statefulset.yaml`
   * `qdrant-service.yaml`
   * `qdrant-headless.yaml`
   * `qdrant-backup-configmap.yaml`
   * `qdrant-backup-cronjob.yaml`
   * `qdrant-restore-configmap.yaml`
   * `qdrant-restore-cronjob.yaml`
   * `qdrant-rbac.yaml`

3. Performs byte-wise verification that files are exactly reproducible (same hash on rerun).

---

# **Runtime Behavior of Generated System**

Once ArgoCD applies the manifests:

### **Qdrant StatefulSet**

* Deploys N replicas
* Cluster peers auto-discover using headless service
* Storage uses **ephemeral** unless PROD NVMe chosen
* API key injected via environment

### **Backup CronJob**

Runs every X minutes:

1. `exec` into any Qdrant pod and trigger `POST /snapshots`
2. Download the snapshot .tar directory
3. Repack it → `snapshot.tar.zst`
4. Upload to S3 →

   * `prefix/YYYYMMDD-HHMMSS/snapshot.tar.zst`
5. Create `latest.manifest.json` with:

   ```
   { "timestamp": "...", "snapshot": "snapshot.tar.zst", "hash": "...", ... }
   ```
6. Upload that as well

This produces **deterministic, atomic backups**.

### **Restore CronJob**

Triggered manually by setting:

```
restore/enabled = true
restore/target = some-timestamp
```

It:

1. Downloads snapshot.tar.zst
2. Inflates deterministically
3. Pushes snapshot back into Qdrant via `/snapshots/load`
4. Waits for replicas to converge

---

