# AKS Environment Variables — Detailed Reference (Compact Scope)

This document explains **only the environment variables that are intentionally exposed** in the compact AKS configuration block.
All other infrastructure parameters are **derived, fixed, or hard-coded** in the Pulumi program to preserve determinism and reduce operator error.

---

## Pulumi / Stack

### `PULUMI_STACK`

```bash
export PULUMI_STACK="staging"
```

**Purpose:** Selects the Pulumi stack representing the environment.
**Allowed values / format:** Alphanumeric identifier (no spaces).
**Operational impact:** Determines which remote state and resources are targeted.
**Mutability:** Safe to switch between stacks; do not rename existing stacks to migrate resources.

---

### `PULUMI_CONFIG_PASSPHRASE`

```bash
export PULUMI_CONFIG_PASSPHRASE="mypassword"
```

**Purpose:** Encrypts Pulumi secrets when using passphrase-based secrets management.
**Allowed values / format:** Any non-empty string; treat as sensitive.
**Operational impact:** Required for headless automation if secrets are present.
**Mutability:** Must remain consistent for a given automation context unless secrets are re-encrypted.

---

## AKS Cluster Core

### `AKS_CLUSTER_NAME`

```bash
export AKS_CLUSTER_NAME="rag-aks"
```

**Purpose:** Name of the AKS managed cluster resource.
**Allowed values / format:** Azure-valid resource name.
**Operational impact:** Changing creates a new cluster.
**Mutability:** Immutable after creation.

---

### `AKS_SKU`

```bash
export AKS_SKU="standard"
```

**Purpose:** Selects AKS control-plane SLA tier.
**Allowed values:** `free`, `standard`, `premium`.

**Operational meaning:**

* `free`: dev / ephemeral only (no SLA)
* `standard`: production default
* `premium`: mission-critical workloads

**Mutability:** Effectively immutable for an existing cluster.

---

### `AKS_LOCATION`

```bash
export AKS_LOCATION="eastus"
```

**Purpose:** Azure region where the AKS cluster is deployed.
**Operational impact:** Determines VM availability, quota, and latency.
**Mutability:** Changing requires full redeploy in a new region.

---

### `AKS_MAX_PODS`

```bash
export AKS_MAX_PODS=60
```

**Purpose:** Maximum pods per node (kubelet + CNI configuration).
**Allowed values:** Positive integer.
**Operational impact:** Affects pod density and subnet pressure.
**Mutability:** Safe for new node pools; existing pools may require replacement.

---

## Node Pools — System (AKS Core)

### `SYSTEM_NODE_COUNT`

```bash
export SYSTEM_NODE_COUNT=1
```

**Purpose:** Number of nodes in the system pool (kube-system, CNI, CoreDNS).
**Operational impact:** Must never scale to zero.
**Recommendation:**

* Staging: `1`
* Production: `>=2` (prefer `3`)

---

### `SYSTEM_NODE_VM_SIZE`

```bash
export SYSTEM_NODE_VM_SIZE="Standard_B2s"
```

**Purpose:** VM SKU for system pool nodes.
**Operational impact:** Stability of control-plane agents.
**Recommendation:**

* Staging: small burstable
* Production: `D4s_v5` or larger

---

### `SYSTEM_NODE_MAX_PODS`

```bash
export SYSTEM_NODE_MAX_PODS=60
```

**Purpose:** Pod density for system nodes.
**Constraint:** Must match `AKS_MAX_PODS`.
**Mutability:** Requires node pool replacement to change.

---

## Node Pools — Balanced (General Purpose)

### `BALANCED_NODE_MIN`

```bash
export BALANCED_NODE_MIN=0
```

**Purpose:** Minimum nodes for general workloads (APIs, gateways, orchestrators).
**Recommendation:**

* Staging: `0–1`
* Production: `>=2` for HA

---

### `BALANCED_NODE_MAX`

```bash
export BALANCED_NODE_MAX=2
```

**Purpose:** Autoscaler ceiling for general workloads.
**Operational impact:** Caps API/frontend scaling.

---

### `BALANCED_NODE_VM_SIZE`

```bash
export BALANCED_NODE_VM_SIZE="Standard_B2s"
```

**Purpose:** VM SKU for general-purpose workloads.
**Recommendation:**

* Staging: small
* Production: `D4s_v5`

---

## Node Pools — CPU-Heavy (Models)

### `CPU_HEAVY_NODE_MIN`

```bash
export CPU_HEAVY_NODE_MIN=0
```

**Purpose:** Minimum nodes for CPU-bound model workloads.
**Workloads:** Embeddings, rerankers, tokenizers, CPU inference.
**Recommendation:**

* Staging: `0`
* Production: `>=1` if latency-sensitive

---

### `CPU_HEAVY_NODE_MAX`

```bash
export CPU_HEAVY_NODE_MAX=0
```

**Purpose:** Maximum nodes for batch or burst CPU inference.
**Operational impact:** Caps parallel model execution.

---

### `CPU_HEAVY_NODE_VM_SIZE`

```bash
export CPU_HEAVY_NODE_VM_SIZE="Standard_B2s"
```

**Purpose:** VM SKU for CPU model pool.
**Recommendation:**

* Production: `F8s_v2` (AVX2, predictable clocks)

---

## Node Pools — Qdrant (Vector Database)

### `QDRANT_NODE_COUNT`

```bash
export QDRANT_NODE_COUNT=0
```

**Purpose:** Fixed-size node pool for Qdrant vector DB.
**Workloads:** HNSW index, WAL, RocksDB storage.
**Recommendation:**

* Staging: `0–1`
* Production: `>=1` (scale via sharding/replication)

---

### `QDRANT_NODE_VM_SIZE`

```bash
export QDRANT_NODE_VM_SIZE="Standard_B2s"
```

**Purpose:** VM SKU for vector database nodes.
**Recommendation:**

* Production: `E8ds_v5` or `E16ds_v5` (RAM + local NVMe)

---

## Safety / Destructive Control

### `PULUMI_FORCE_DESTROY`

```bash
export PULUMI_FORCE_DESTROY=1
```

**Purpose:** Safety gate for destructive Pulumi operations.
**Allowed values:** `0` (safe default), `1` (allow deletes).
**Recommendation:**

* Staging: `1`
* Production: `0`

---

