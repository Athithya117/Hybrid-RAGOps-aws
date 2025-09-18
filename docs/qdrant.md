# Connections, ports, IPs, endpoints, and backups with qdrant

Qdrant runs as a Docker container on a single EC2 with a fixed private IP in a VPC. Ray cluster does not handles stateful workloads like hosting databases well. Vertical scaling with c8g instance is sufficient for most of the RAG use cases. 

---

# 1) Networking & IPs — how you make connectivity deterministic

* **EC2 private IP:** launch the Qdrant EC2 with a *fixed private IP* (assign at launch or attach an ENI with a fixed private IP). Example: `10.0.2.15`. Your Ray clusters will use that private IP to talk to Qdrant.
* **Security groups:** Qdrant EC2 SG must allow inbound **TCP 6333** only from the Ray clusters’ SG(s) (use SG IDs, not CIDR). No public inbound rules. Example inbound rule:

  * Protocol: TCP, Port: 6333, Source: `sg-INDEXING-RAY` (and/or `sg-INFERENCE-RAY`)
* **VPC/subnet:** Put Qdrant and Ray nodes in the same VPC and (preferably) the same subnet or peered subnets so private IP routing works without NAT.

---

# 2) Docker run & port binding — binding to the private IP

Run Qdrant with the data directory on an EBS-mounted host path and bind the host private IP to port 6333.

Example:

```bash
# assume /mnt/qdrant is your EBS mount on the EC2 host
sudo mkdir -p /mnt/qdrant
sudo chown 1000:1000 /mnt/qdrant   # ensure permissions for Qdrant container user

# run Qdrant, binding to the host private IP
docker run -d --name qdrant \
  -p 10.0.2.15:6333:6333 \                    # bind host private IP -> container port
  -v /mnt/qdrant:/qdrant/storage \            # persistent storage on EBS
  -e QDRANT__HTTP__API_KEY="$QDRANT_API_KEY" \# set API key via env
  qdrant/qdrant:latest
```

Notes:

* `-p <HOST_IP>:6333:6333` binds only the host private IP. If you use `-p 0.0.0.0:6333:6333`, the port is reachable from any interface on the host (still gated by SG).
* Mount host path (`/mnt/qdrant`) to container storage path (`/qdrant/storage`) so data persists on the EBS volume.

---

# 3) Qdrant endpoints & API key usage

* **HTTP API** default: `http://<QDRANT_PRIVATE_IP>:6333`

  * Health: `GET http://10.0.2.15:6333/health`
  * Upsert (vectors): `POST /collections/<collection>/points?wait=true` or the modern `PUT /collections/<col>/points` depending on client library version.
  * Search: `POST /collections/<collection>/point_search` or `POST /collections/<collection>/search` (use the version matching your Qdrant release).
* **API key**: include header `api-key: <QDRANT_API_KEY>` in every request (or `--header "api-key: $QDRANT_API_KEY"` for curl).
* Example curl (health):

```bash
curl -s -H "api-key: $QDRANT_API_KEY" http://10.0.2.15:6333/health
```

* Example curl (search):

```bash
curl -s -H "Content-Type: application/json" -H "api-key: $QDRANT_API_KEY" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "top": 5
  }' \
  http://10.0.2.15:6333/collections/my_collection/points/search
```

(If you use a Qdrant client SDK in Python/Go, configure the host/port and API key accordingly.)

---

# 4) How Ray clusters connect

* **Indexing cluster (writes):** tasks call `http://10.0.2.15:6333` with `api-key` to upsert vectors in batches. Use deterministic chunk IDs in the upsert payload to make operations idempotent.
* **Inference cluster (reads):** tasks call the same endpoint for search — read-only behavior. Keep the same `api-key` that has appropriate permissions.
* **No database modification by inference cluster:** enforce this by issuing only `search` requests from inference code and/or by using a separate API key scoped to read-only (if you can enforce read-only in your environment; otherwise rely on app-level discipline).

---

# 5) Example upsert payload (batched, deterministic IDs)

When you upsert use deterministic id generation (e.g., SHA256 of doc-id+chunk-index). Example JSON payload (simplified):

```json
{
  "points": [
    {
      "id": "chunk::doc123::0",       // deterministic chunk id
      "vector": [0.12, 0.31, ...],
      "payload": {"doc_id":"doc123","chunk_idx":0,"text":"..."}
    },
    {
      "id": "chunk::doc123::1",
      "vector": [0.22, 0.11, ...],
      "payload": {"doc_id":"doc123","chunk_idx":1,"text":"..."}
    }
  ]
}
```

Curl example:

```bash
curl -X PUT "http://10.0.2.15:6333/collections/my_collection/points?wait=true" \
  -H "Content-Type: application/json" -H "api-key: $QDRANT_API_KEY" \
  --data-binary @batch.json
```

Batching: push many points per request (tune for memory/throughput). Deterministic IDs mean rerunning the same batch will upsert/overwrite rather than creating duplicates.

---

# 6) Backups with EBS snapshots — how to make them consistent

Your approach: Qdrant data is stored on an **EBS volume** mounted at `/mnt/qdrant` on the EC2 host. Use **EBS snapshots** for point-in-time full-node backup.

Best-practice procedure for a consistent snapshot:

1. **Pause or stop writers** (recommended):

   * During indexing, at the end of the batch you already control the indexing job — ensure it finishes and no upserts are in flight.
2. **Stop the Qdrant container** (fast and safe):

   ```bash
   docker stop qdrant
   ```

   Stopping the container flushes/cleanly closes WAL and files to disk.
3. **Create EBS snapshot** (AWS CLI example — replace `<VOLUME_ID>`):

   ```bash
   aws ec2 create-snapshot --volume-id vol-0abcdef12345 --description "qdrant backup $(date -u +%Y%m%dT%H%M%SZ)" \
     --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=qdrant-backup},{Key=batch_id,Value=INDEX_BATCH_123}]'
   ```
4. **Restart Qdrant**:

   ```bash
   docker start qdrant
   ```
5. **Verify snapshot success** and tag retention policy (automate deletion after N days if desired).

Why stop the container?

* Stopping ensures all in-memory state and WALs are flushed; snapshotting a live DB risks corruption or partial writes. You *can* snapshot a live volume but then must ensure DB supports consistent online snapshots — Qdrant does not document that as a guaranteed safe approach for hot snapshots, so stopping is the safest.

Automating snapshot:

* Indexing job (at end) can call AWS CLI (with appropriate IAM role) to snapshot the attached volume.
* Alternatively, use an AWS Lambda or Step Function triggered by a message produced by the indexing job.

Restoring from snapshot:

1. Create a new volume from snapshot.
2. Attach volume to new EC2 (or replace old volume).
3. Mount it to `/mnt/qdrant`.
4. Run the Qdrant container with `-v /mnt/qdrant:/qdrant/storage` — you get the full node restored.

Commands:

```bash
# create volume from snapshot
aws ec2 create-volume --snapshot-id snap-0123456789abcdef0 --availability-zone us-east-1a --volume-type gp3

# attach to instance (then mount)
aws ec2 attach-volume --volume-id vol-0abcd... --instance-id i-0123abcd --device /dev/sdf
```

---

# 7) Additional operational tips & hardening

* **IAM & secrets:** Store `QDRANT_API_KEY` in AWS Secrets Manager or SSM Parameter Store; inject into Ray nodes securely (avoid plaintext in user-data). Use instance profiles for snapshot permissions.
* **Health checks & monitoring:** poll `http://<ip>:6333/health` and expose metrics to Prometheus (Qdrant exposes prometheus metrics optionally).
* **Firewall on host:** rely on SGs; optionally install host-level firewall (ufw/iptables) to further restrict access to `10.0.2.0/24` or the Ray SGs.
* **Read-only / principle of least privilege:** if possible consider separate API keys (or app-layer enforcement) for indexing vs inference.
* **Disk capacity planning:** monitor `/mnt/qdrant` usage; keep headroom so Qdrant WAL and merges don’t run out of space.
* **Multiple replicas / HA:** single EC2 + EBS is simple but single-point-of-failure. EBS snapshots are your backup; if you need HA consider Qdrant clustering or multi-node architecture later.
* **Docker restart policy:** set `--restart unless-stopped` so a host reboot restarts Qdrant automatically.
* **Bind to localhost alternatives:** binding to `10.0.2.15` is fine. If you want even stricter control, bind to `127.0.0.1` + use a host proxy with auth — but for Ray cluster access across hosts a private IP is typical.

---

# 8) Example end-to-end script (indexing job triggers snapshot)

Pseudo-steps your indexing job runs after finishing a successful upsert batch:

```bash
# 1) ensure no upserts in flight
# 2) signal Qdrant host or run a command via SSH / SSM to stop container
ssh ec2-user@10.0.2.15 'docker stop qdrant'

# 3) create snapshot of the EBS volume (using instance role or IAM creds)
aws ec2 create-snapshot --volume-id vol-0abcdef... --description "qdrant backup $BATCH_ID" --tag-specifications 'ResourceType=snapshot,Tags=[{Key=batch,Value='$BATCH_ID'}]'

# 4) restart qdrant
ssh ec2-user@10.0.2.15 'docker start qdrant'
```

(Use Systems Manager Run Command instead of SSH for better automation.)

---

# Quick checklist to verify correct setup

* [ ] EC2 has fixed private IP or ENI attached.
* [ ] Docker Qdrant launched with `-p <PRIVATE_IP>:6333:6333` and `-v /mnt/qdrant:/qdrant/storage`.
* [ ] Qdrant responds: `curl -H "api-key: $QDRANT_API_KEY" http://10.0.2.15:6333/health`
* [ ] SG inbound rule allows only Ray clusters SG(s) to 6333.
* [ ] Indexing job uses deterministic IDs and batches upserts.
* [ ] Snapshot automation in place to stop container, snapshot EBS, restart container.
* [ ] Secrets and snapshot IAM permissions are set up securely.

