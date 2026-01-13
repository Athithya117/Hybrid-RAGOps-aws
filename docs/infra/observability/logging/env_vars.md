# Logging Environment Variables

These environment variables control **cluster mode**, **Vector and ClickHouse resource sizing**, **log filtering**, and **retention**.

Change them only when your **deployment environment**, **throughput**, **SLA**, **storage budget**, or **security posture** requires different behavior.

---

## Reading Guide

For each variable, the documentation explains:

- **What** it controls
- **Valid values / type**
- **When and why** to change it
- **Impact** of misconfiguration
- **How to validate** at runtime

---

## Cluster Mode

### `K8S_CLUSTER`

- **What**  
  Selects cluster-specific behavior in infra generators.

- **Values**  
  - `kind` — local development (default)  
  - `aks` — production AKS behavior

- **When to change**  
  - Use `kind` for local or ephemeral clusters  
  - Use `aks` for real Azure Kubernetes Service clusters

- **Impact**  
  Controls generator defaults (images, assumptions, fallbacks).  
  Some manifests are explicitly tuned for `kind` and should not be used in production.

- **Validate**
  ```sh
  echo $K8S_CLUSTER
 
Inspect generated manifests for expected cluster-specific behavior.

---

## Vector (Log Collector)

### `VECTOR_REPLICAS`

* **What**
  Logical replica count for Vector when running outside DaemonSet mode.

* **Values**
  Integer ≥ 1

* **When to change**
  Only relevant if Vector is deployed as a **Deployment or StatefulSet**.
  Leave at `1` when using a DaemonSet (current default).

* **Impact**
  No effect with the current DaemonSet-based deployment.
  Changing this without switching deployment mode does nothing.

* **Validate**

  ```sh
  kubectl -n observability get daemonset,deploy -l app=vector
  ```

---

### `VECTOR_REQ_CPU`, `VECTOR_REQ_MEM`

### `VECTOR_LIMIT_CPU`, `VECTOR_LIMIT_MEM`

* **What**
  Kubernetes resource requests and limits for the Vector container.

* **Values**

  * CPU: millicores or cores (`200m`, `1`)
  * Memory: `Mi` / `Gi` (`512Mi`, `1Gi`)

* **When to change**

  * Increase **requests** if Vector is throttled or backpressure appears
  * Increase **limits** to allow short ingestion bursts
  * Avoid excessive overcommit in shared clusters

* **Sizing guidance**

  * `request_cpu ≈ ingest_events_per_sec * 0.0005` cores
  * `request_mem ≈ ingest_events_per_sec * avg_event_size_bytes * 0.00001` Gi
  * For small clusters, `200m / 512Mi` is conservative and safe

* **Impact**

  * Too small → throttling, dropped logs, OOMKilled
  * Too large → wasted resources, scheduling failures

* **Validate**

  ```sh
  kubectl -n observability top pod -l app=vector
  kubectl -n observability describe pod -l app=vector
  ```

---

### `VECTOR_DROP_NAMESPACES`

* **What**
  Comma-separated list of Kubernetes namespaces whose logs are dropped **before ingestion**.

* **Values**
  CSV string, e.g.:

  ```
  kube-system,models,indexing
  ```

* **When to change**

  * Add noisy, low-value namespaces to reduce cost
  * Remove namespaces when debugging requires full visibility

* **Impact**

  * Reduces ingestion and storage cost
  * Incorrect values cause **silent loss of observability**

* **Validate**

  * Emit logs from a dropped namespace and confirm absence in ClickHouse
  * Inspect Vector transform behavior (`del(.)`)

---

Below is a precise explanation of the **newly added ClickHouse environment variables** in STEP 14, including whether they are **hard limits**, **soft safety knobs**, or **require tuning/testing**.

---

## `CLICKHOUSE_MAX_MEMORY_USAGE`

**What it controls**
Maximum memory a **single query** (including merges triggered by that query) is allowed to allocate.

**Why it exists**
Prevents ClickHouse from consuming all container memory and getting OOMKilled by Kubernetes.

**How it maps internally**
Sets:

* `max_memory_usage`

**Recommended value**
≈ **60–80% of `CLICKHOUSE_LIMIT_MEM`**

**Behavior type**

* Hard limit
* Enforced by ClickHouse

**Testing required?**
Yes, light tuning.

* Too low → large queries fail with `MEMORY_LIMIT_EXCEEDED`
* Too high → risk of Kubernetes OOMKills

---

## `CLICKHOUSE_MAX_MEMORY_USAGE_FOR_USER`

**What it controls**
Maximum **aggregate memory usage per user**, across all concurrent queries.

**Why it exists**
Protects the system from:

* Query storms
* Poorly written dashboards
* Multi-tenant overload (future-proofing)

**How it maps internally**
Sets:

* `max_memory_usage_for_user`

**Recommended value**
Lower than `CLICKHOUSE_MAX_MEMORY_USAGE`

**Behavior type**

* Hard limit
* Enforced at user level

**Testing required?**
Minimal.

* Mostly defensive
* Only tune upward if legitimate workloads are throttled

---

## `CLICKHOUSE_MAX_THREADS`

**What it controls**
Maximum number of execution threads per query.

**Why it exists**
Prevents CPU oversubscription and noisy-neighbor effects inside the pod.

**How it maps internally**
Sets:

* `max_threads`

**Recommended value**
≤ allocatable CPU cores (usually equal to `CLICKHOUSE_REQ_CPU` rounded down)

**Behavior type**

* Hard cap on parallelism
* CPU safety knob

**Testing required?**
Low.

* Increase only if queries are CPU-bound and latency-sensitive
* Lower values favor predictable latency over raw speed

---

## `CLICKHOUSE_BACKGROUND_POOL_SIZE`

**What it controls**
Number of threads used for **background tasks**:

* Part merges
* TTL cleanup
* Mutations

**Why it exists**
Balances foreground query performance vs. merge/cleanup speed.

**How it maps internally**
Sets:

* `background_pool_size`

**Recommended value**

* Start at `1–2`
* Increase only with fast disks (SSD) and spare CPU

**Behavior type**

* Soft performance knob
* Affects merge latency, not correctness

**Testing required?**
Optional.

* Increase only if `system.merges` shows backlog
* No correctness risk

---

## `CLICKHOUSE_TTL_DAYS`

**What it controls**
Retention period for log data at the **table level**.

**Why it exists**
Ensures disk usage remains bounded without manual cleanup.

**How it maps internally**
Used in:

* `TTL ts + INTERVAL <n> DAY DELETE`

**Recommended value**
Same as `LOGS_TTL_DAYS`

**Behavior type**

* Declarative retention policy
* Deterministic

**Testing required?**
No.

* Safe, deterministic
* TTL execution is asynchronous and non-blocking

---

## Summary Table

| Variable                               | Category            | Risk Level | Needs Tuning |
| -------------------------------------- | ------------------- | ---------- | ------------ |
| `CLICKHOUSE_MAX_MEMORY_USAGE`          | Memory safety       | Medium     | Yes          |
| `CLICKHOUSE_MAX_MEMORY_USAGE_FOR_USER` | Multi-tenant safety | Low        | Minimal      |
| `CLICKHOUSE_MAX_THREADS`               | CPU isolation       | Low        | Optional     |
| `CLICKHOUSE_BACKGROUND_POOL_SIZE`      | Merge throughput    | Low        | Optional     |
| `CLICKHOUSE_TTL_DAYS`                  | Data lifecycle      | None       | No           |

---

### Bottom line


## ClickHouse (Log Storage)

### `CLICKHOUSE_REPLICAS`

* **What**
  StatefulSet replica count for **single-node mode**.

* **Values**
  Integer ≥ 1

* **When to change**

  * Leave at `1` for single-node deployments
  * For real HA, switch to **operator mode** instead of increasing this

* **Impact**

  * StatefulSet replicas >1 do **not** create a proper ClickHouse cluster
  * Operator mode is required for real replication and sharding

* **Validate**

  ```sh
  kubectl -n clickhouse get sts
  ```

---

### `CLICKHOUSE_PVC_SIZE`

* **What**
  Persistent volume size for ClickHouse data.

* **Values**
  Storage quantity (e.g. `10Gi`, `100Gi`)

* **When to change**
  Increase when ingestion rate or retention grows.

* **Sizing formula**

  ```
  required_bytes ≈ avg_event_size_bytes
                   * events_per_sec
                   * 86400
                   * retention_days
                   * replication_factor
  ```

  Add **20–50% headroom** for merges and compactions.

* **Impact**

  * Too small → disk pressure, crashes
  * Too large → unnecessary storage cost

* **Validate**

  ```sh
  kubectl -n clickhouse exec <pod> -- df -h
  ```

---

### `CLICKHOUSE_REQ_CPU`, `CLICKHOUSE_REQ_MEM`

### `CLICKHOUSE_LIMIT_CPU`, `CLICKHOUSE_LIMIT_MEM`

* **What**
  Resource requests and limits for ClickHouse.

* **Typical baseline**

  * Request: `1 CPU / 4Gi`
  * Limit: `4 CPU / 16Gi`

* **When to change**
  Scale with:

  * ingestion rate
  * query concurrency
  * retention window
  * heavy aggregations

* **Impact**

  * Under-provisioning → slow ingest, rejected inserts, OOM
  * Over-provisioning → scheduling failures

* **Validate**

  ```sh
  kubectl -n clickhouse top pod
  ```

  Inspect ClickHouse:

  * `system.metrics`
  * `system.asynchronous_metrics`

---

### `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`

* **What**
  Credentials used by Vector to write logs to ClickHouse.

* **Values**
  Strings

* **When to change**

  * Rotate regularly
  * Use a dedicated ingestion user with INSERT-only permissions

* **Impact**

  * Incorrect values → ingestion failures
  * Plain-text env vars are a security risk

* **Validate**

  * Check Vector pod logs for auth errors
  * Test with `clickhouse-client` using the same credentials

---

### `LOGS_TTL_DAYS`

* **What**
  Log retention window in days (ClickHouse TTL).

* **Values**
  Integer ≥ 1

* **When to change**

  * Increase for compliance or longer debugging windows
  * Decrease to reduce storage cost

* **Impact**

  * Longer TTL → higher storage cost
  * Shorter TTL → irreversible data deletion

* **Validate**
  Inspect TTL configuration and oldest parts:

  * `system.parts`
  * Table `TTL` definitions

---

## Operational and Security Rules

* **Secrets**

  * Never hard-code `CLICKHOUSE_PASSWORD` in CI or repo
  * Use Kubernetes Secrets or a cloud secret manager
  * Inject via `valueFrom.secretKeyRef`

* **Vector Deployment Mode**

  * Current setup uses a **DaemonSet**
  * `VECTOR_REPLICAS` has no effect until switching to Deployment/StatefulSet
  * DaemonSet = one collector per node (low latency, simple)
  * Deployment = centralized collectors, horizontal scaling

* **ClickHouse HA**

  * Use `CH_MODE=operator` with Altinity operator
  * StatefulSet replicas alone are not sufficient

---

## Capacity Planning Cheat Sheet

```
daily_bytes ≈ events/sec * avg_event_size_bytes * 86400
required_storage_Gi ≈ daily_bytes
                        * retention_days
                        / (1024^3)
                        * replication_factor
```

Always add **20–50% headroom**.
