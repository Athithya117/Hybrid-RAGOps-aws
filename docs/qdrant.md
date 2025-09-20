# Connections, security and backups with qdrant

Qdrant runs as a Docker container on a single EC2 with a fixed private IP (ENI) in a VPC. Ray clusters use that private IP to reach Qdrant. Vertical scaling with local NVMe based VM(c8gd family) is typical for most RAG workloads.

---

## 1) Networking & IPs — deterministic connectivity

* **Deterministic endpoint (ENI):** create an ENI with a fixed private IP (example `10.0.2.15`) and attach it to the Qdrant EC2 via the Launch Template. When ASG replaces the instance the ENI (and its private IP) is re-attached so the endpoint stays stable.
* **Security groups:** Qdrant SG must allow inbound **TCP 6333** only from Ray SG(s) (use SG IDs) and a Qdrant API key is required. No public inbound rules. 
* **VPC/subnet:** Qdrant and Ray indexing cluster are in the same VPC and subnet for lower network latency. 

---

## 2) Docker run & port binding — binding to the private IP

Run Qdrant on the host, mount a persistent host path for storage (NVMe or EBS as your node storage), and bind the host private IP:

```bash
sudo mkdir -p /workspace/qdrant/data
sudo chown 1000:1000 /workspace/qdrant/data

# bind to the deterministic ENI private IP
docker run -d --name qdrant \
  -p 10.0.2.15:6333:6333 \
  -v /workspace/qdrant/data:/qdrant/storage \
  -e QDRANT__HTTP__API_KEY="$QDRANT_API_KEY" \
  qdrant/qdrant:latest
```

Notes:

* `-p <HOST_IP>:6333:6333` binds only that host private IP. `0.0.0.0` would make it listen on all host interfaces (still gated by SG).
* Storage must persist on the host at `/workspace/qdrant/data`. Backups are to S3 (see §6).

---

## 3) Qdrant endpoints & API key usage

* API endpoint: `http://<QDRANT_PRIVATE_IP>:6333`

  * Health: `GET /health`
  * Upsert: `PUT /collections/<col>/points` (or the endpoint matching your Qdrant version)
  * Search: `POST /collections/<col>/search` (or versioned path)
* Always include `api-key` header: `api-key: <QDRANT_API_KEY>`.

Example health check:

```bash
curl -s -H "api-key: $QDRANT_API_KEY" http://10.0.2.15:6333/health
```

---

## 4) How Ray clusters connect

* **Indexing cluster (writes):** call `http://10.0.2.15:6333` with API key to upsert batches. Use deterministic chunk IDs for idempotency.
* **Inference cluster (reads):** same endpoint for searches. Use a read-only discipline or separate API key where possible.

---

## 5) Example upsert payload (batched, deterministic IDs)

(unchanged — keep using SHA256(doc\_id + chunk\_index) style IDs). Example same as before.

---

## 6) Backups → **S3-based snapshots (new plan)**

**Overview:** Use Qdrant’s snapshot API to produce a local snapshot, compress it to `.tar.zst`, upload to S3 with multipart concurrency, compute SHA256, and publish an atomic manifest `latest_qdrant_backup.manifest.json` in the bucket. Don’t rely on EBS snapshots for backups.

**Why:** `.tar.zst` is fast to compress/decompress and efficient for large backups; multipart uploads saturate network and are resumable; manifest ensures atomic pointer to latest snapshot.

### Recommended, robust sequence (atomic & idempotent)

1. **Quiesce writers**: ensure indexing batch finished and no in-flight upserts.
2. **Trigger Qdrant snapshot** via API:

   ```bash
   curl -X POST -H "api-key: $QDRANT_API_KEY" \
     http://127.0.0.1:6333/collections/<collection>/snapshots
   ```

   or Qdrant's global snapshot endpoint if available. Poll `GET /snapshots` or collection snapshot status until `completed`.
3. **Package snapshot**:

   ```bash
   SNAP_DIR=/workspace/qdrant/backups/snapshots/<snapshot_folder>
   OUT=/workspace/qdrant/backups/qdrant_snapshot_$(date -u +%Y%m%dT%H%M%SZ).tar.zst
   tar -C "$SNAP_DIR" -I 'zstd -T0 -19' -cf "$OUT" .
   ```
4. **Compute checksum**:

   ```bash
   sha256sum "$OUT" > "$OUT.sha256"
   ```
5. **Upload to S3 (multipart recommended)**:

   * **Simple** (AWS CLI automatically does multipart for big files):

     ```bash
     aws s3 cp "$OUT" "s3://$S3_BUCKET/qdrant/backups/$(basename $OUT)" --storage-class STANDARD
     ```
   * **Controlled (recommended for large files)** — use Python/boto3 with `TransferConfig` to set `multipart_chunksize` and `max_concurrency`:

     ```python
     import boto3
     from boto3.s3.transfer import TransferConfig
     import hashlib, json, time
     s3 = boto3.client("s3", region_name="ap-south-1")
     cfg = TransferConfig(multipart_chunksize=50*1024*1024, max_concurrency=8)
     s3.upload_file(OUT, S3_BUCKET, S3_KEY, Config=cfg)

     ```
6. **Write manifest locally (manifest.tmp)** including:

   ```json
   {
     "backup_and_restore_path": "/workspace/qdrant/data/",
     "latest_snapshot": "qdrant_snapshot_2025-09-19T15-30-00.snapshot.zst",
     "latest_snapshot_from_path": "/workspace/qdrant/backups/snapshots/",
     "latest_snapshot_to_path": "s3://<bucket>/qdrant/backups/qdrant_snapshot_2025-09-19T15-30-00.snapshot.zst",
     "latest_snapshot_size_in_mb": 20000,
     "multipart_upload_concurrency": 8,
     "latest_snapshot_upload_duration_in_seconds": 120,
   }

  ```

7. **Atomically publish/overwrite manifest**:

   ```bash
   aws s3 cp manifest.tmp s3://$S3_BUCKET/latest_qdrant_backup.manifest.json.tmp
   aws s3 mv s3://$S3_BUCKET/latest_qdrant_backup.manifest.json.tmp s3://$S3_BUCKET/latest_qdrant_backup.manifest.json
   ```

   (or `aws s3 cp manifest.tmp s3://$S3_BUCKET/latest_qdrant_backup.manifest.json && rm manifest.tmp` — but atomic `mv` from a temp key is preferred).
8. **Local cleanup**: rotate or delete local tarball per retention policy.

### Manifest is authoritative

Boot/restore logic should consult `s3://$S3_BUCKET/latest_qdrant_backup.manifest.json` first. It contains the upload path, checksum, concurrency hint and metadata.

### Fallback

If manifest missing, UserData can call Qdrant `GET /snapshots` and pick the newest snapshot by creation\_time or ISO8601 name — but manifest is preferred.

---

## 7) Restore workflow (on first time provisioning / ASG replacement)

1. Instance boots; UserData reads `s3://$S3_BUCKET/latest_qdrant_backup.manifest.json`.
2. If present, verify checksum; **multipart-download** snapshot (or `aws s3 cp`) to `/workspace/qdrant/backups/`.

   * Use boto3 TransferConfig for concurrency if large.
3. Extract:

   ```bash
   tar -I 'zstd -d -T0' -xvf /workspace/qdrant/backups/<snapshot>.tar.zst -C /workspace/qdrant/backups/snapshots/
   # then copy/extract into /workspace/qdrant/data/ as needed
   tar -C /workspace/qdrant/backups/snapshots -xvf ...
   chown -R 1000:1000 /workspace/qdrant/data /workspace/qdrant/backups/snapshots
   ```
4. Start Qdrant container (bind to ENI IP), wait for `/health` OK.
5. Run warm-up queries to populate page-cache; then mark instance healthy for traffic.

---

## 8) Monitoring, alerts & runbook changes

* Alert on: snapshot upload failures, manifest not updated, checksum mismatch, low local disk space, Qdrant `/health` failing, ASG replacement events.
* Restore drills: monthly automated restore-to-staging using the manifest (download → extract → start → verify collection counts & sample queries).
* IAM: Instance role needs read access to snapshot prefix and `latest_qdrant_backup.manifest.json` for restore; snapshot job needs write access to that prefix.

---

## 9) Security & operational guardrails

* Keep `QDRANT_API_KEY` in Secrets Manager/SSM and inject into instances securely.
* S3: use SSE (KMS), bucket policies restricting writes to snapshot prefix, lifecycle rules to age snapshots.
* Atomic manifests: write `manifest.tmp` then `mv` to `latest_qdrant_backup.manifest.json`.

---

## 10) Example scripts (quick)

Create snapshot + tar + upload (bash + boto3 recommended for concurrency):

```bash
# 1. trigger snapshot via API (example)
curl -s -X POST -H "api-key: $QDRANT_API_KEY" http://127.0.0.1:6333/collections/my_collection/snapshots

# 2. wait for completion (polling omitted here)

# 3. tar + zstd
SNAP_DIR=/workspace/qdrant/backups/snapshots/<snap>
OUT=/workspace/qdrant/backups/qdrant_snapshot_$(date -u +%Y%m%dT%H%M%SZ).tar.zst
tar -C "$SNAP_DIR" -I 'zstd -T0 -19' -cf "$OUT" .

# 4. upload with aws cli (or use boto3 for TransferConfig)
aws s3 cp "$OUT" "s3://$S3_BUCKET/qdrant/backups/$(basename $OUT)"
sha256sum "$OUT" > "$OUT.sha256"

# 5. manifest.tmp creation + upload + atomic move (see §6)
```

For high-volume snapshots use the boto3 `TransferConfig` snippet earlier to control `multipart_chunksize` and `max_concurrency`.

---

## Quick checklist (updated)

* [ ] ENI created and attached -> `QDRANT_PRIVATE_IP` exported.
* [ ] Docker Qdrant launched: bound to that IP and mounted host storage `/workspace/qdrant/data`.
* [ ] SG inbound allows only Ray SG(s) to 6333.
* [ ] Indexing job uses deterministic IDs and emits success marker for backup trigger.
* [ ] Snapshot automation: snapshot via Qdrant API → `.tar.zst` → multipart upload to S3 → `latest_qdrant_backup.manifest.json` updated atomically.
* [ ] Restore drill scheduled monthly to verify end-to-end S3 restore works.

---
