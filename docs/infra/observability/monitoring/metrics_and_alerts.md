# Monitoring and Metrics – Actual Implementation (VictoriaMetrics)

## 1. Monitoring Architecture (What Exists)

The monitoring stack consists of:

* **vmagent**

  * Kubernetes-native scraper
  * Discovers pods via `kubernetes_sd_configs`
  * Scrapes `/metrics` endpoints
  * Remote-writes samples
* **VictoriaMetrics (single-node)**

  * Time-series storage
  * Query API compatible with PromQL
* **Application-level Prometheus exposition**

  * `prometheus_client` used **only** to expose metrics
  * No Prometheus server is deployed

There is **no**:

* Prometheus server
* ServiceMonitor / PodMonitor CRDs
* Alertmanager
* kube-prometheus-stack

All monitoring resources run in the **`monitoring`** namespace.

---

## 2. Scraping & Ingestion Behavior (vmagent)

### 2.1 Target Discovery

vmagent discovers scrape targets using **pod annotations**:

```yaml
monitoring.io/scrape: "true"
monitoring.io/path: "/metrics"
monitoring.io/port: "8001"
```

Discovery mechanism:

```yaml
kubernetes_sd_configs:
  - role: pod
```

Relabeling constructs scrape addresses as:

```
<podIP>:<annotated-port>
```

This is enforced by:

```yaml
replacement: '$1:$2'
```

(no escaping, no mutation)

---

### 2.2 Scraping Parameters

* Scrape interval: **15s**
* Scrape timeout: **10s**
* Scrapes only:

  * ready pods
  * explicitly annotated pods

---

### 2.3 Remote Write (Single Source of Truth)

vmagent writes **all samples** to VictoriaMetrics:

```
http://victoria-metrics.monitoring.svc.cluster.local:8428/api/v1/write
```

This URL is:

* **hardcoded**
* **passed only via args**
* **not overridable via env**
* **not injected via secrets**

This eliminates configuration drift.

---

## 3. Storage & Query (VictoriaMetrics)

### 3.1 VictoriaMetrics Node

* Mode: **single-node**
* HTTP listen: `:8428`
* Retention: **1d** (dev / local default)
* Storage: `emptyDir` (PVC optional)

### 3.2 Query API

VictoriaMetrics exposes PromQL-compatible endpoints:

```
/api/v1/query
/api/v1/query_range
```

Example:

```bash
curl 'http://victoria-metrics.monitoring.svc.cluster.local:8428/api/v1/query?query=up'
```

> Note: `up{}` is **not a reliability signal** in VM pipelines; ingestion counters are authoritative.

---

## 4. Retrieval Service – Metrics Specification

Metrics are exposed on:

```
http://<pod-ip>:8001/metrics
```

via `prometheus_client`.

Common labels:

* `service="retrieval"`
* `env="<environment>"`

### 4.1 Counters

| Metric                        | Description            |
| ----------------------------- | ---------------------- |
| `retrieval_requests_total`    | Total HTTP requests    |
| `retrieval_errors_total`      | Error count            |
| `dense_embed_requests_total`  | Dense embedding calls  |
| `sparse_embed_requests_total` | Sparse embedding calls |
| `qdrant_query_total`          | Qdrant queries         |
| `llm_calls_total`             | LLM invocations        |
| `presign_requests_total`      | Presign URL requests   |

### 4.2 Histograms

| Metric                               | Purpose                  |
| ------------------------------------ | ------------------------ |
| `retrieval_request_duration_seconds` | End-to-end latency       |
| `dense_embed_duration_seconds`       | Dense embed latency      |
| `sparse_embed_duration_seconds`      | Sparse embed latency     |
| `qdrant_query_duration_seconds`      | Vector DB latency        |
| `llm_call_duration_seconds`          | LLM latency              |
| `rerank_duration_seconds`            | Reranking latency        |
| `retrieved_docs_count`               | Result size distribution |

Buckets are statically defined and bounded.

### 4.3 Gauges

| Metric          | Meaning         |
| --------------- | --------------- |
| `service_ready` | Readiness (1/0) |

---

## 5. Cardinality Rules (Enforced)

Allowed labels:

* `service`
* `env`
* `endpoint`
* `status_code`

Disallowed:

* request IDs
* user IDs
* query text
* dynamic dimensions

Histogram growth is bounded and intentional.

---

## 6. Qdrant Metrics

Qdrant exposes metrics on:

```
/metrics (port 6333)
```

vmagent scrapes Qdrant using the **same pod-annotation mechanism**.

Key metrics:

* request counts
* latency histograms
* resource usage
* cluster health

Collection-level labels are used sparingly to avoid cardinality blowups.

---

## 7. Alerting Status (Current Reality)

There is **no alerting system deployed**.

No:

* Alertmanager
* Rule evaluation
* Notifications

All alerting logic described previously is **aspirational**, not active.

Future alerting options:

* VictoriaMetrics alerting (vmalert)
* External systems querying VM

---

## 8. Determinism Guarantees (What Is Enforced)

The system guarantees determinism through:

* Single discovery mechanism (pod annotations)
* Single scraper (vmagent)
* Single writer (remote write)
* Single storage backend (VictoriaMetrics)
* Hardcoded ingestion endpoint
* No env-based overrides
* No CRD-based indirection
* Explicit validation against escaping / mutation

---

## 9. How E2E Validity Is Proven

The system is considered **healthy** when:

1. vmagent reports:

   ```
   vm_promscrape_targets{status="up"} >= 1
   ```
2. VictoriaMetrics reports:

   ```
   vm_rows_inserted_total{type="promremotewrite"} > 0
   ```

`up{}` is **observational only**, not a correctness gate.

---

## 10. Summary (Truthful)

* Metrics are **Prometheus-format**, not Prometheus-operated
* vmagent is the scraper
* VictoriaMetrics is the TSDB
* Discovery is annotation-based
* Ingestion is the source of truth
* Tests validate ingestion, not convenience metrics

This document now reflects **exactly what runs in-cluster**, with no legacy Prometheus assumptions.
