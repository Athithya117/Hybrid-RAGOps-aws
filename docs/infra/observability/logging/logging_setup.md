# Logging Setup and Application Log Schema

## 1. Overview

This platform uses a **deterministic, schema-normalized logging pipeline** built on:

* **Vector** as the log collector and normalizer (DaemonSet on Kubernetes)
* **ClickHouse** as the authoritative log storage backend
* **JSON-first application logs** with strict field semantics

The core principle is **log determinism**:
given the same input log event, the resulting stored record in ClickHouse must be identical regardless of runtime environment, pod placement, or source language.

---

## 2. High-Level Architecture

### 2.1 Data Flow

1. **Application containers** write logs to stdout/stderr
2. **Kubernetes** exposes container logs under `/var/log/pods`
3. **Vector DaemonSet**:

   * Collects all Kubernetes logs
   * Normalizes timestamps, levels, and metadata
   * Enforces a fixed schema
   * Drops logs from configured namespaces
4. **ClickHouse**:

   * Stores logs in `logs.kube_logs`
   * Applies optional TTL-based retention
   * Supports deterministic querying and aggregation

---

## 3. Vector Normalization Guarantees

Vector enforces the following invariants before logs are written to ClickHouse:

### 3.1 Timestamp Resolution

* Output column: `ts` (`DateTime64(3)`)
* Resolution: **milliseconds**
* Deterministic rules:

  1. Prefer `parsed.timestamp` if present
  2. Accept either:

     * Unix epoch (seconds)
     * RFC3339 / ISO-8601 string
  3. Fallback to ingestion time (`now()`)

Applications **must not rely** on ingestion time for correctness.

---

### 3.2 Log Level Normalization

* Stored column: `level`
* Canonical values:

  * `DEBUG`
  * `INFO`
  * `WARN`
  * `ERROR`

Accepted input values (case-insensitive):

* `debug`
* `info`
* `warn`, `warning`
* `error`, `err`

Anything else is deterministically coerced to `INFO`.

---

### 3.3 Kubernetes Metadata Mapping

Vector deterministically populates:

| Column      | Source (priority order)                                       |
| ----------- | ------------------------------------------------------------- |
| `service`   | `parsed.service` → `kubernetes.labels.app` → `container_name` |
| `container` | `kubernetes.container_name`                                   |
| `pod`       | `kubernetes.pod_name`                                         |
| `namespace` | `kubernetes.pod_namespace`                                    |

Missing values resolve to empty strings, never `null`.

---

## 4. ClickHouse Storage Schema (Authoritative)

All logs are written to:

```
database: logs
table: kube_logs
```

### 4.1 Table Definition

| Column      | Type                     | Description                     |
| ----------- | ------------------------ | ------------------------------- |
| `ts`        | `DateTime64(3)`          | Event timestamp                 |
| `level`     | `LowCardinality(String)` | Normalized log level            |
| `message`   | `String`                 | Human-readable message          |
| `service`   | `LowCardinality(String)` | Logical service name            |
| `pod`       | `LowCardinality(String)` | Kubernetes pod                  |
| `namespace` | `LowCardinality(String)` | Kubernetes namespace            |
| `container` | `LowCardinality(String)` | Container name                  |
| `trace_id`  | `String`                 | Distributed trace ID            |
| `span_id`   | `String`                 | Distributed span ID             |
| `fields`    | `String`                 | JSON-encoded structured payload |

This schema is **stable and versioned implicitly**. Backward-incompatible changes are not allowed.

---

## 5. Deterministic Application Log Contract (MANDATORY)

### 5.1 Required Log Format

All application logs **must be valid JSON objects** written to stdout.

Example (canonical):

```json
{
  "timestamp": "2025-01-12T10:42:31.123Z",
  "level": "info",
  "message": "user login succeeded",
  "service": "auth-api",
  "trace_id": "9f3a2c8e1b4d",
  "span_id": "a1b2c3d4",
  "user_id": "12345",
  "ip": "203.0.113.10"
}
```

---

### 5.2 Mandatory Fields (Application Layer)

| Field       | Type         | Rules                     |
| ----------- | ------------ | ------------------------- |
| `timestamp` | string | int | RFC3339 or Unix seconds   |
| `level`     | string       | One of accepted values    |
| `message`   | string       | Must be human-readable    |
| `service`   | string       | Stable service identifier |

If these are missing, Vector will still ingest the log, but **semantic guarantees are lost**.

---

### 5.3 Optional but Strongly Recommended

| Field            | Purpose                      |
| ---------------- | ---------------------------- |
| `trace_id`       | Cross-service correlation    |
| `span_id`        | Distributed tracing          |
| Any other fields | Become part of `fields` JSON |

All non-reserved keys are preserved verbatim inside `fields`.

---

### 5.4 Prohibited Patterns

Applications **must not**:

* Emit plain-text (non-JSON) logs
* Change key meanings dynamically
* Embed structured data inside `message`
* Emit multi-line log entries
* Use locale-dependent timestamps
* Emit arrays or top-level non-object JSON

Violations reduce observability guarantees and may be rejected in the future.

---

## 6. Namespace-Based Log Dropping

Vector deterministically drops logs from configured namespaces:

* Default drop list:

  * `kube-system`

Controlled via:

```sh
VECTOR_DROP_NAMESPACES="models,indexing,qdrant" # model logs are redundant as retrieval service includes them
```

Dropping happens **before** ClickHouse ingestion.

---

## 7. Operational Guarantees

* Idempotent deployment (generators re-runnable)
* Deterministic schema enforcement
* Safe retries during ClickHouse bootstrap
* No reliance on mutable runtime ordering
* Environment-driven configuration only

---

## 8. Summary for Application Engineers

**If you remember one thing:**

> Emit one JSON object per line, with a stable timestamp, level, message, and service name. Everything else is deterministic infrastructure behavior.

This contract is non-negotiable.

---

