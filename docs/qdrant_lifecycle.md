# Qdrant cluster lifecycle 

Short mapping first — what each script does (single sentence each)

1. `gen_qdrant_cluster.py` — renders vendor Helm chart values, installs/updates Qdrant Helm release and (when `ENV=STAGING`) creates an **in-cluster** AWS secret via `kubectl` (no secret YAML written to repo).
2. `run_qdrant_backup.py` — creates **cluster backups**: per-node snapshots → copies snapshot files off pods → uploads to S3 → writes `manifest.json` and `latest.manifest.json`. Deterministic inputs-hash ensures idempotency of manifest generation.
3. `qdrant_restore.py` — restores from a backup manifest in S3: picks a `backup_id` (explicit or `latest.manifest.json`), then for each pod+collection it attempts **recover-from-presigned-URL**; if that fails it streams the S3 object into the node upload endpoint. Optionally restarts the StatefulSet and prints per-collection results.

---
Below: lifecycle phases and exactly how these three files participate.


## 1) Provision / deploy (cluster creation & config)

* Run `gen_qdrant_cluster.py` to produce Helm `values.yaml` and install/upgrade Qdrant (it vendors the chart if missing).
* Behavior by `ENV`:

  * `STAGING`: if AWS creds are present, the script runs `kubectl create secret generic qdrant-backup-aws ... --dry-run=client -o yaml | kubectl apply -f -` to create the secret **inside the cluster** so runtime jobs can access S3 (no secret YAML is committed).
  * `PROD`: IRSA is expected; script does not touch credentials.
* Result: a running Qdrant StatefulSet (`qdrant-0`, `qdrant-1`, ...) with deterministic pod names (StatefulSet semantics), configured persistence paths, and the service exposed (ClusterIP).

Why this matters:

* StatefulSet gives stable pod names which the backup/restore manifests use to map snapshots to nodes.
* SECRET policy avoids committing credentials to gitops manifests.

---

## 2) Normal operation (reads/writes)

* Qdrant handles writes/reads; Raft + replication factor control consistency across shards/replicas.
* Application clients connect to the Qdrant service (ClusterIP or port-forwarded localhost) and perform normal operations.
* Important: backups should consider quiescing or tolerating slight time skew across nodes (no global atomic snapshot primitive).

---

## 3) Backup workflow (`run_qdrant_backup.py`) — what is executed and why

* **Mode selection**

  * **Service-mode (default cluster-level)**: create snapshots via the service URL once per collection and download the resulting snapshot from the service. (Simpler but may be fragile if the service endpoint does not proxy node-local snapshot files.)
  * **Per-pod mode (`--per-pod`)**: port-forward to each pod, call the snapshot API on that pod, and copy the node-local snapshot file from the pod filesystem. (Robust and recommended for local-NVMe.)
* **Discovery**

  * If `--collections` not given, script queries the Qdrant API to enumerate collections.
  * If `--per-pod`, script discovers pods by label and starts port-forwards (7000,7001,... or configured base).
* **Snapshot creation**

  * For each targeted pod and each collection the script requests a snapshot (HTTP POST `/collections/<col>/snapshots?wait=true`).
  * The snapshot is a node-local file. The script then downloads or `kubectl cp`-s it.
* **Upload & manifest**

  * Each snapshot file is checksummed (SHA256) and uploaded to S3 at `s3://<bucket>/<s3_prefix>/<backup_id>/<filename>`.
  * Script writes:

    * `manifest.json` containing `{ backup_id, created_at, namespace, pods: { pod: { collections: { col: { snapshot_name, s3_key, sha256, size, local_path, pod_path }}}}, collections: [...] }`
    * `latest.manifest.json` (same schema, points to the latest backup).
  * Both are uploaded to S3 and a local backup dir is optionally retained.
* **Determinism/idempotency**

  * An inputs-hash is stored `.inputs_hash` so repeated runs with unchanged inputs are detectable (script still performs backup by default; but manifest generation is deterministic).

Why per-pod is important:

* When snapshots live on local NVMe on each node, downloading directly from pods (per-pod) ensures the actual node-local files are captured. Service-mode can return 404 if the public service does not serve node-local files.

---

## 4) Restore workflow (`qdrant_restore.py`) — exact steps performed

* **Select backup**

  * If `--backup-id` provided, use it; otherwise download `s3://<prefix>/latest.manifest.json` and read `backup_id`.
  * Download `s3://<prefix>/<backup_id>/manifest.json`.
* **For each manifest pod entry**

  * Map manifest pods. If manifest has `"service"` entry, discover real pod names in the namespace.
  * For each pod:

    * Start kubectl port-forward `pod/<pod> <local_port>:6333`.
    * For each collection in that pod’s manifest:

      1. Try to generate a presigned S3 GET URL and call the node’s `PUT /collections/<col>/snapshots/recover` with `{"location": "<presigned_url>"}` (primary, efficient method — node fetches from S3).
      2. If recover-from-URL fails, fallback to streaming the S3 object into the node endpoint `POST /collections/<col>/snapshots/upload` (multipart). This streams bytes through the restore client into the node.
    * Stop port-forward.
* **Optional restart**

  * If `--restart` provided: call `kubectl rollout restart statefulset/qdrant -n <ns>` and wait for rollout status.
* **Validation & summary**

  * Script records per-collection restore status and prints a full manifest with `_restore_status` entries.

Why this restores cluster-state deterministically:

* Manifest maps snapshots to pod names. Restoring the exact per-node snapshot files back to the corresponding pod reproduces node-local storage layout. StatefulSet pod identity + Qdrant recover endpoints then let nodes pick up restored data and Raft/replication re-establish cluster consistency.

---

## 5) Secrets and environments

* `STAGING`:

  * `gen_qdrant_cluster.py` and `run_qdrant_backup.py` will, when AWS creds exist locally, **create/update** an in-cluster secret (`qdrant-backup-aws`) via `kubectl apply` (not written to repo). That enables cluster-side actions if needed (jobs or node-level access).
* `PROD`:

  * Use IRSA — scripts do not create or emit secrets. Boto3 on the runner (or node IRSA) must have permissions for S3.

---

## 6) Validation & verification

* After restore you should:

  * Query collection counts (`POST /collections/<col>/points/count`) or `scroll` to validate point data.
  * Check checksums saved in the manifest (script verifies SHA256 when downloading).
  * Use `qdrant_restore.py` summary and optionally `validate_restore.py` (or its logic embedded) to assert cross-pod consistency.
* Good practice: run smoke queries, compare sample points, and check replication/raft status.

---

## 7) Failure modes to be aware of (and what the scripts do)

* **404 when downloading via service** — solved by per-pod snapshot copy (implemented).
* **Presigned-URL unsupported in Qdrant version** — script falls back to streaming upload. Test on your Qdrant release.
* **Pod identity mismatch** — manifests reference pod names; if StatefulSet pods were renamed/recreated with different names, mapping is required. Keep StatefulSet semantics stable.
* **Replication gaps** — if replication factor is low and a node dies before replication finishes, backups will capture only what that node had; restore may lead to data loss. Mitigation: replication_factor ≥ 2 (preferably 3), and quiesce writes when you need a globally consistent snapshot.

---
---
| Aspect                        |                                                                     Per-pod (pod-level) | Cluster-level                                                                                            |
| ----------------------------- | --------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------- |
| **Data locality**             |             Snapshot each pod’s node-local files (captures true NVMe/emptyDir contents) | Snapshot via Qdrant cluster API (assumes access to all data)                                             |
| **Prerequisites**             |       `kubectl` (kubectl cp / port-forward) or pod access; stable StatefulSet pod names | Qdrant cluster API supporting cluster snapshots; durable shared storage (PVC/EBS) recommended            |
| **Implementation complexity** |                          Higher — iterate pods, port-forward/copy, upload per-pod files | Lower — single snapshot per collection/cluster and upload                                                |
| **Backup granularity**        |                                      Fine: per-pod, per-collection files (many objects) | Coarse: single/aggregate archive (fewer objects)                                                         |
| **Restore complexity**        | Higher — push files back to matching pods, restore pod-by-pod, then restart/statefulset | Lower — single restore operation can repopulate cluster                                                  |
| **Consistency / atomicity**   |                    Eventual; needs coordination (quiesce writes or accept small window) | Can be more atomic if cluster snapshot aggregates all shards                                             |
| **Performance**               |                   Best I/O (local NVMe reads) but more network ops to upload many files | Depends on underlying storage (PVC/EBS slower I/O but simpler transfer)                                  |
| **Scalability**               |                     Scales with number of pods (parallelizable), generates many S3 keys | Scales well operationally (single job) but depends on storage backend                                    |
| **Failure modes**             |                     Missing a pod snapshot loses that node’s data; pod identity matters | If storage is durable, fewer failure modes; but if service can’t access node-local data, may miss shards |
| **Cost (S3/object count)**    |    More S3 objects (one per pod/collection/snapshot) → possible higher request overhead | Fewer, larger objects (single archive) → simpler and cheaper to manage                                   |
| **Operational fit**           |                  **Required** for local NVMe / `emptyDir` production (c8gd, i3en, etc.) | **Preferred** for PVC/EBS-backed clusters or where Qdrant cluster snapshot is supported                  |
| **Recommended**               |                                 Use **per-pod** when data is node-local (emptyDir/NVMe) | Use **cluster-level** when volumes are persistent network storage (PVC/EBS)                              |

---

## 8) Typical runbook (commands)
* Deploy / update cluster:

  ```bash
  ENV=STAGING python3 infra/generators/gen_qdrant_cluster.py --apply
  ```

* Default per-pod backup (more robust for local NVMe storage with vectors "on_disk": true, avoids PVC/EBS for faster performance and seamless kind -> eks):

  ```bash
  python3 infra/runners/run_qdrant_backup.py --backup --s3-bucket $S3_BUCKET --per-pod --port-base 7000
  ```

* Cluster-level backup(If not using gen_qdrant_cluster.py or if using qdrant hybrid cloud as operator https://qdrant.tech/documentation/hybrid-cloud/):

  ```bash
  python3 infra/runners/run_qdrant_backup.py --backup --s3-bucket $S3_BUCKET --s3-prefix qdrant/backups
  ```

* Restore latest:

  ```bash
  python3 infra/runners/run_qdrant_restore.py --s3-bucket $S3_BUCKET --s3-prefix qdrant/backups --per-pod --restart
  ```
* Restore a specific backup or if cluster level restore(This does not work with --per-pod backup because cluster restore requires PVC/EBS):

  ```bash
  export BACKUP_ID=20251204T133937Z-2ac5f439
  python3 infra/runners/run_qdrant_restore.py --s3-bucket $S3_BUCKET --s3-prefix qdrant/backups --backup-id $BACKUP_ID --restart
  ```



