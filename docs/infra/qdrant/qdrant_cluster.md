# Qdrant Architecture and Runtime Control Flow

## Purpose
This document describes the **Qdrant deployment architecture, configuration model, and runtime control flow** as implemented in this platform. It is authoritative and directly maps to `infra/generators/qdrant_cluster.sh`, generated manifests, and AKS runtime behavior.

---

## Role of Qdrant in the Platform
Qdrant is the **primary stateful vector storage layer** for the RAG platform.

It is responsible for:
- Persisting dense and sparse vector embeddings
- Serving low-latency similarity search queries from the Retrieval API
- Supporting deterministic indexing and re-indexing workflows
- Providing durable state with backup/restore integration to Azure Blob Storage
- Exposing metrics for platform observability

Qdrant is a **hard dependency** for:
- Indexing CronJobs
- Retrieval API
- Backup and restore runners
- Observability dashboards and alerts

---

## Deployment Model Overview

### Deployment Type
- **Helm-managed StatefulSet**
- Namespace: `qdrant`
- Release name: `${QDRANT_RELEASE}` (default: `qdrant`)
- Image: `qdrant/qdrant:v1.16.0`
- Chart version: `1.16.0`

### Control Plane Flow
```sh

qdrant_cluster.sh
├─ generate_manifests()
│    ├─ values.yaml
│    ├─ service-patch.yaml
│    └─ inputs hash (.inputs_hash)
├─ helm upgrade --install
├─ kubectl apply (service patch)
├─ runtime validation
└─ deployment summary (last_deploy_summary.json)

```

Manifests are committed under:
```sh

infra/manifests/qdrant/

```

Flux reconciles the resulting Kubernetes resources.

---

## Runtime Topology

### Kubernetes Resources
- `StatefulSet/qdrant`
- `Service/qdrant` (ClusterIP)
- Optional Secrets:
  - Azure backup credentials
  - Qdrant service API key



### Networking
- API & metrics port: `6333`
- P2P cluster port: `6335`
- Headless service used for peer discovery

### Cluster Mode
- Always enabled (`cluster.enabled = true`)
- Peers auto-generated based on replica count:
```sh

qdrant-0.qdrant-headless:6335
qdrant-1.qdrant-headless:6335
...

```

Replication and sharding are controlled via:
- `QDRANT_REPLICAS`
- `QDRANT_SHARD_NUMBER`
- `QDRANT_REPLICATION_FACTOR`
- `QDRANT_WRITE_CONSISTENCY_FACTOR`

---

## Configuration Generation

### values.yaml
Generated dynamically from environment variables.

Key sections:
- Image repository and tag
- Resource requests and limits
- Cluster peer topology
- Metrics port exposure
- Logging level
- Storage paths
- Replication and shard parameters
- Environment variables sourced from Kubernetes Secrets

No manual editing of `values.yaml` is expected.

---

## Secrets and Credentials

### Backup Credentials (Optional)
If `SECRET_BACKUP_NAME` is set:
- A Kubernetes Secret is created or updated in the `qdrant` namespace
- One of the following must be provided:
- `AZURE_STORAGE_CONNECTION_STRING`
- OR `AZURE_STORAGE_ACCOUNT_NAME` + `AZURE_STORAGE_ACCOUNT_KEY`

These credentials are injected into the Qdrant container via `extraEnv`.

### Service API Key (Optional)
If both are set:
- `QDRANT__SERVICE__API_KEY`
- `SECRET_SERVICE_NAME`

A secret is created and mounted to enable authenticated access to Qdrant.

---

## Persistence Model

### Storage
- Qdrant runs with **local container filesystem storage**
- `persistence.enabled = false` at the Helm level
- Durability is achieved via **explicit backup/restore**, not PVCs

### Paths
- Storage path: `/qdrant/storage`
- Snapshots path: `/qdrant/snapshots`

This design is intentional to:
- Avoid PVC lifecycle complexity
- Enable faster restore workflows
- Support node-side restore from Blob Storage

---

## Backup and Restore Integration

- Qdrant itself does not push backups
- Platform runners trigger snapshot export
- Snapshots are uploaded to Azure Blob Storage
- Restore is performed by:
- Rehydrating snapshots onto nodes
- Restarting Qdrant pods
- Allowing cluster state to converge

This workflow is validated via:
```

make qdrant-backup
make qdrant-restore

```

---

## Metrics and Observability

### Metrics Exposure
- Port: `6333`
- Path: `/metrics`
- Protocol: HTTP

### Pod Annotations
Each pod is annotated with:
```

monitoring.io/scrape: "true"
monitoring.io/port: "6333"
monitoring.io/path: "/metrics"

```

### Validation
Post-install validation enforces:
- Pods exist and are Ready
- Required annotations are present
- Metrics port is declared in container spec
- Safe patching is applied if the port is missing

Failures can be configured to:
- Fail deployment (`FAIL_ON_MISCONFIG=true`)
- Or log warnings and continue

---

## Deployment Idempotency and Safety

### Inputs Hash
- All non-secret inputs are hashed
- If unchanged, manifest generation is skipped
- Prevents unnecessary Flux churn

### Atomic Writes
- All generated files are written atomically
- Prevents partial or corrupted manifests

### Helm Fallback Strategy
1. Vendored chart (if present)
2. Primary Helm repo
3. Fallback Helm repo

Retries with exponential backoff are applied.

---

## Runtime Control Flow (Apply)

1. Validate required tools
2. Generate manifests if inputs changed
3. Ensure namespace exists
4. Create/update secrets (if configured)
5. Helm upgrade/install Qdrant
6. Apply service patch
7. Validate pod readiness and metrics contract
8. Patch StatefulSet ports if required
9. Write deployment summary

---

## Deletion Semantics

`--delete` performs a best-effort cleanup:
- Deletes StatefulSet, Service, Secrets
- Removes generated manifests locally
- Does not delete external backups

---

## Outputs and Artifacts

- `infra/manifests/qdrant/values.yaml`
- `infra/manifests/qdrant/service-patch.yaml`
- `infra/manifests/qdrant/.inputs_hash`
- `infra/manifests/qdrant/last_deploy_summary.json`

These artifacts define the **single source of truth** for Qdrant runtime state.

---

## Operational Guarantees

- Deterministic deployment
- Explicit dependency ordering
- Observable and validated runtime state
- Recoverable without PVCs
- Safe for repeated apply/delete cycles

This architecture is designed for **platform-level reliability**, not ad-hoc experimentation.

