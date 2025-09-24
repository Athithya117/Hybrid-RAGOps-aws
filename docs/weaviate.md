# Connections, security and backups with Weaviate

Weaviate runs as a Docker container on a single EC2 with a fixed private IP (ENI) in a VPC. Ray clusters use that private IP to reach Weaviate. Vertical scaling with local NVMe based EC2 (c8gd family) is typical for most RAG workloads. The storage will be lost on EC2 instance stop/termination but the ASG can immediately create a new instance with fast provisioning of the latest backup. ([Weaviate Documentation][1])

---

## 1) Networking & IPs — deterministic connectivity

* **Deterministic endpoint (ENI):** create an ENI with a fixed private IP (example `10.0.2.15`) and attach it to the Weaviate EC2 via the Launch Template. When ASG replaces the instance the ENI (and its private IP) is re-attached so the endpoint stays stable.
* **Security groups:** Weaviate SG must allow inbound **TCP 8080** (HTTP/REST & GraphQL) — and **TCP 50051** if you use gRPC — only from Ray SG(s) (use SG IDs) and any API keys / credentials are required. No public inbound rules. ([Weaviate Documentation][2])
* **VPC/subnet:** Weaviate and Ray clusters in the same VPC and subnet for lower network latency.

---

## 2) Docker run & port binding — binding to the private IP

Run Weaviate on the host, mount a persistent host path for storage (NVMe or EBS as your node storage), and bind the host private IP. Weaviate’s container expects its persistence path to match `PERSISTENCE_DATA_PATH` and the container-side path is typically `/var/lib/weaviate` (host-side many teams map `/var/weaviate` or `/workspace/weaviate/data`). ([Weaviate Documentation][3])

Example (single-node Docker run bound to ENI IP — adjust env vars/modules as needed):

```bash
sudo mkdir -p /workspace/weaviate/data
sudo chown 1000:1000 /workspace/weaviate/data

docker run -d --name weaviate \
  -p 10.0.2.15:8080:8080 \    # REST/GraphQL
  -p 10.0.2.15:50051:50051 \  # gRPC (optional)
  -v /workspace/weaviate/data:/var/lib/weaviate \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="false" \
  -e DEFAULT_VECTORIZER_MODULE="none" \
  cr.weaviate.io/weaviate:latest
```

Notes:

* `-p <HOST_IP>:8080:8080` binds the REST API only to that host private IP. `0.0.0.0` would make it listen on all host interfaces (still gated by SG). ([Weaviate Documentation][1])
* Storage must persist on the host at `/workspace/weaviate/data` (maps to container `/var/lib/weaviate` / `PERSISTENCE_DATA_PATH`). ([Weaviate Documentation][3])

---

## 3) Weaviate endpoints & credential usage

* API endpoint: `http://<WEAVIATE_PRIVATE_IP>:8080`

  * Meta/health/version: `GET /v1/meta` (returns version and module info). Use this for basic health/version checks. ([Weaviate Documentation][4])
  * Search / data operations: GraphQL or REST under `/v1/` (for example `POST /v1/graphql` for GraphQL queries or object CRUD via `/v1/objects`). See the REST docs for exact endpoints. ([Weaviate Documentation][5])
* Use authentication (API keys / OpenID Connect / other auth modules) according to your deployment. Example simple curl health check (replace with secure credential method in production):

```bash
curl -s http://10.0.2.15:8080/v1/meta
```

(For production: enable/require API keys or OIDC and store secrets in Secrets Manager / SSM and inject securely.) ([Weaviate Documentation][6])

---

## 4) How Ray clusters connect

* **Indexing cluster (writes):** call `http://10.0.2.15:8080` with proper API key/auth and use the REST/GraphQL/object APIs to create objects and vectors (use deterministic chunk IDs for idempotency). The Weaviate client libraries (Python/JS) are typically used in ingestion flows. ([Weaviate Documentation][7])
* **Inference cluster (reads):** same endpoint for searches; use GraphQL or REST “search” endpoints that support vector search, filters, and hybrid queries. Use a read-only API key or scoped credentials where possible.

---

## 5) Example upsert payload (batched, deterministic IDs)

(unchanged in principle — keep using SHA256(doc\_id + chunk\_index) style IDs). When using Weaviate, create an object with the ID you computed and include `vector` (if you want to override) or let Weaviate vectorizer populate vectors depending on your modules. Use the batch API for bulk ingestion (Weaviate supports batching via its client libraries). ([Weaviate Documentation][7])

---

## 6) Backups → **S3-backed Weaviate backups (preferred)**

**Overview:** Weaviate has a built-in backup/restore feature that can write backups to cloud storage backends (S3, GCS, Azure) or to the local filesystem (local is intended for testing only). Use the backup API or the official Python client to create backups that are written directly to S3. This is the recommended production pattern rather than relying only on EBS snapshots. ([Weaviate Documentation][8])

**Why:** using the Weaviate backup feature to write directly to S3 provides class-level or whole-instance backups, is resumable by the backup implementation, and is the supported pattern for restore across nodes/instances. You can still keep your own S3 manifest/checksum workflow on top of that for orchestration/atomic pointer semantics if you want (see manifest pattern below). ([Weaviate][9])

### Recommended, robust sequence (atomic & idempotent)

1. **Quiesce writers**: ensure indexing batch finished and no in-flight writes.
2. **Create a backup using Weaviate’s backup API / Python client** (this writes to the configured S3 backend):

Example using the Python client (illustrative — adapt `backup_id`, `include_classes`, and `config` to your needs):

```python
import weaviate
from weaviate.backup import BackupConfigCreate

client = weaviate.Client("http://127.0.0.1:8080")

result = client.backup.create(
    backup_id="weaviate_backup_2025-09-24T153000Z",
    backend="s3",
    include_classes=["MyClass1", "MyClass2"],   # optional
    wait_for_completion=True,
    config=BackupConfigCreate(chunk_size=256)
)
print(result)  # contains status and path information
```

The Python client exposes S3 as a `backend='s3'` option and models to configure compression/chunk sizes. ([Weaviate Python Client][10])

3. **Compute checksum / manifest (optional but recommended):** if you want a single atomic pointer to the latest backup (helpful for ASG/user-data restore), compute a SHA256 of the backup objects (or the main archive path reported by the backup API) and write a `latest_weaviate_backup.manifest.json.tmp` then move it to `latest_weaviate_backup.manifest.json` (atomic rename) in the bucket. Weaviate’s backup API will create S3 objects under the configured bucket/path; your manifest can point at the backup id/path. (You can also use the backup API status endpoints to retrieve the canonical path.) ([MinIO Blog][11])

4. **Upload details:** if you use the Weaviate backup modules, Weaviate writes directly to S3 (no separate `tar.zst` step required unless you prefer an additional packaging layer). If you prefer to produce a `.tar.zst` yourself (for tooling compatibility) you can still do: export objects (via Read API or snapshot), tar+zstd, and upload to S3 with multipart concurrency — but the built-in backup API is designed to simplify/coordinate consistent backups across nodes. ([Weaviate][9])

### Manifest is authoritative (optional)

If you keep a manifest it should reference the Weaviate backup ID / S3 path and a checksum. Your ASG/user-data restore logic should consult the manifest first; otherwise it can query the Weaviate backup listing endpoints to find the latest `backup_id`.

### Fallback

If manifest missing, UserData can either (A) call the Weaviate backup listing endpoint (`GET /v1/backups/s3` or use the Python client) to find the latest backup id/path, or (B) list objects in the S3 snapshot prefix by `LastModified`. Manifest is preferred for atomic pointer semantics. ([Hex Preview][12])

---

## 7) Restore workflow (on first time provisioning / ASG replacement)

1. Instance boots; UserData reads `s3://$S3_BUCKET/latest_weaviate_backup.manifest.json` (if you use the manifest) or queries weaviate `/v1/backups` (via local or remote control plane) for a backup id/path. ([Weaviate][9])
2. If using Weaviate backup API: trigger restore (Python client or REST) for the backup id to restore data into the instance:

Example (Python client, illustrative):

```python
# restore
client = weaviate.Client("http://127.0.0.1:8080")
client.backup.restore(
    backup_id="weaviate_backup_2025-09-24T153000Z",
    backend="s3",
    wait_for_completion=True,
    config=BackupConfigRestore(chunk_size=256)
)
```

Or via REST: `POST /v1/backups/s3/{backup_id}/restore` (Weaviate maps the backend and id in the URL). After restore completes, objects/classes are available on the instance. ([MinIO Blog][11])

3. Start the Weaviate container (bind to ENI IP), wait for `/v1/meta` ok and for the backup restore to report `SUCCESS`.
4. Run warm-up queries to populate caches if desired; then mark instance healthy for traffic.

---

## 8) Monitoring, alerts & runbook changes

* Alert on: backup failures, backup job stuck (`backup` status not progressing), manifest not updated, checksum mismatch, low local disk space, Weaviate health failing (`/v1/meta` unreachable), ASG replacement events. Use the backup status endpoints and Python client to poll backup status programmatically. ([Weaviate][9])
* Restore drills: monthly automated restore-to-staging using the manifest or the backup API (create → restore → verify object counts & sample queries).
* IAM: Instance role needs read access to S3 prefix and `latest_weaviate_backup.manifest.json` for restore; backup job needs write access to that prefix.

---

## 9) Security & operational guardrails

* Keep API keys / OIDC client secrets in Secrets Manager/SSM and inject into instances securely; disable anonymous access unless intentionally used. ([Weaviate Documentation][6])
* S3: use SSE (KMS), bucket policies restricting writes to snapshot prefix, lifecycle rules to age snapshots.
* Atomic manifests: write `manifest.tmp` then `mv` to `latest_weaviate_backup.manifest.json`.

---

## 10) Example scripts (quick)

Create backup via Python client → check status → (optional) write manifest:

```python
import weaviate
from weaviate.backup import BackupConfigCreate

client = weaviate.Client("http://127.0.0.1:8080")

# create backup to S3
res = client.backup.create(
    backup_id="weaviate_backup_2025-09-24T153000Z",
    backend="s3",
    wait_for_completion=True,
    config=BackupConfigCreate(chunk_size=256)
)
print(res)  # contains status/path

# inspect / verify, then write manifest to S3 (uploads via boto3 or aws cli)
```

For large/production snapshots prefer the built-in Weaviate backup modules writing to S3 (they handle distribution and chunking). If you must produce an external tarball or additional checksum/manifest, use `aws s3 cp` (or boto3 TransferConfig) to multipart-upload large artifacts. ([Weaviate Python Client][10])

---

## Quick checklist (updated)

* [ ] ENI created and attached -> `WEAVIATE_PRIVATE_IP` exported.
* [ ] Docker Weaviate launched: bound to that IP and mounted host storage `/workspace/weaviate/data` and `PERSISTENCE_DATA_PATH` set. ([Weaviate Documentation][3])
* [ ] SG inbound allows only Ray SG(s) to 8080 (and 50051 if using gRPC). ([Weaviate Documentation][2])
* [ ] Indexing job uses deterministic IDs and emits success marker for backup trigger.
* [ ] Backup automation: Weaviate backup → S3 (via Weaviate backup API or Python client) → optional manifest + checksum in S3. ([Weaviate][9])
* [ ] Restore drill scheduled monthly to verify end-to-end S3 restore works.

---


[1]: https://docs.weaviate.io/deploy/installation-guides/docker-installation?utm_source=chatgpt.com "Docker"
[2]: https://docs.weaviate.io/weaviate/api/grpc?utm_source=chatgpt.com "gRPC | Weaviate Documentation"
[3]: https://docs.weaviate.io/deploy/configuration/persistence?utm_source=chatgpt.com "Persistence"
[4]: https://docs.weaviate.io/academy/py/zero_to_mvp/hello_weaviate/hands_on?utm_source=chatgpt.com "Getting hands-on | Weaviate Documentation"
[5]: https://docs.weaviate.io/weaviate/api/rest?utm_source=chatgpt.com "RESTful API endpoints"
[6]: https://docs.weaviate.io/deploy/configuration/env-vars?utm_source=chatgpt.com "Environment variables"
[7]: https://docs.weaviate.io/weaviate/client-libraries/python?utm_source=chatgpt.com "Python | Weaviate Documentation"
[8]: https://docs.weaviate.io/deploy/configuration/backups?utm_source=chatgpt.com "Backups"
[9]: https://weaviate.io/blog/tutorial-backup-and-restore-in-weaviate?utm_source=chatgpt.com "Tutorial - Backup and Restore in Weaviate"
[10]: https://weaviate-python-client.readthedocs.io/en/stable/weaviate.backup.html?utm_source=chatgpt.com "weaviate.backup"
[11]: https://blog.min.io/minio-weaviate-integration/?utm_source=chatgpt.com "Backing Up Weaviate with MinIO S3 Buckets"
[12]: https://preview.hex.pm/preview/noizu_weaviate/show/lib/weaviate_api/backups/README.md?utm_source=chatgpt.com "lib/weaviate_api/backups/README.md - Hex Preview"
