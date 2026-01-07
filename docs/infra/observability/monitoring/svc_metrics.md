# Overview

Below are two compact, production-proven reference tables for **Retriever** and **Qdrant** metrics intended for direct use in dashboards and alert rules.

The content is operationally validated against **Qdrant v1.16.x** behavior and Prometheus semantics.

---

## How to read the tables and scope guarantees

### How to read each row

* **Metric name and type**
* **Labels**
* **Cardinality note**
* **Recommended for alerting** with a concrete **PromQL example**
* **Recommended for dashboard** with a concrete **PromQL panel expression**

### Scope and guarantees

* Only metrics that have been **observed end-to-end by test scripts** (exported → scraped → stored → queryable) are included.
* Experimental, undocumented, or inconsistently ingested metrics are intentionally excluded.
* PromQL expressions are written to be **production-safe** (division guards, low-cardinality joins).

---

# Retriever — emitted by `apps/inference/retrieval/query.py`

| Metric                                                           |                Type | Labels                                      | Cardinality note                        | Recommended for alerting? (example)                                                                                         | Recommended for dashboard? (example)                     |
| ---------------------------------------------------------------- | ------------------: | ------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `retrieval_requests_total`                                       |             Counter | `service`, `env`, `endpoint`, `status_code` | Low                                     | **Yes** — traffic drop: `sum(rate(retrieval_requests_total[5m])) < 0.1` (for 2m)                                            | **Yes** — RPS: `sum(rate(retrieval_requests_total[1m]))` |
| `retrieval_request_duration_seconds_bucket` (+ `_count`, `_sum`) |           Histogram | `service`, `env`, `endpoint`, `status_code` | Moderate (buckets × endpoints)          | **Yes** — p95 latency: `histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)) > 1`     | **Yes** — p95 latency timeseries                         |
| `retrieval_errors_total`                                         |             Counter | `service`, `env`, `endpoint`, `error_type`  | **Caution** — keep `error_type` bounded | **Yes** — error rate: `sum(rate(retrieval_errors_total[5m])) / clamp_min(sum(rate(retrieval_requests_total[5m])),1) > 0.01` | **Yes** — same expression                                |
| `service_ready`                                                  |               Gauge | `service`, `env`                            | Very low                                | **Yes** — service down: `min_over_time(service_ready{service="retrieval"}[5m]) == 0`                                        | **Yes** — readiness stat                                 |
| `dense_embed_requests_total`                                     |             Counter | `service`, `env`                            | Low                                     | Optional                                                                                                                    | **Yes** — RPS                                            |
| `dense_embed_duration_seconds_bucket`                            |           Histogram | `service`, `env`                            | Low × buckets                           | Optional                                                                                                                    | **Yes** — p95 latency                                    |
| `sparse_embed_requests_total` / `sparse_embed_duration_seconds`  | Counter / Histogram | `service`, `env`                            | Low                                     | Optional                                                                                                                    | Yes                                                      |
| `qdrant_query_total` / `qdrant_query_duration_seconds`           | Counter / Histogram | `service`, `env`                            | Low                                     | Optional                                                                                                                    | Yes                                                      |
| `llm_calls_total` / `llm_call_duration_seconds`                  | Counter / Histogram | `service`, `env`                            | Low                                     | Optional                                                                                                                    | Yes                                                      |
| `presign_requests_total` / `presign_duration_seconds`            | Counter / Histogram | `service`, `env`                            | Low                                     | No                                                                                                                          | Optional                                                 |
| `retrieved_docs_count`                                           |           Histogram | `service`, `env`                            | Low                                     | No                                                                                                                          | **Yes** — distribution                                   |
| `rerank_duration_seconds`                                        |           Histogram | `service`, `env`                            | Low                                     | Optional                                                                                                                    | Optional                                                 |
| `process_*`, `python_*`                                          |     Gauge / Counter | minimal                                     | Very low                                | **Yes** — process health                                                                                                    | **Yes**                                                  |

**Notes (Retriever)**

* Alert on `retrieval_errors_total` only if `error_type` values are normalized.
* Use `histogram_quantile()` only with `le` aggregation.
* Always scope by `service` and `env`.

---

# Qdrant — common metrics (Qdrant v1.16.x)

## **Important semantic clarification (authoritative)**

* **HTTP success and failure semantics live exclusively in**
  `rest_responses_total{status="…"}`.
* **`rest_responses_fail_total` does NOT represent HTTP 4xx/5xx responses.**
  It tracks **internal execution failures** and is typically zero in normal operation.
* Therefore:

  * **Do not** use `rest_responses_fail_total` to compute REST error rates.
  * **Always derive REST error rate from `rest_responses_total{status}`.**

---

| Metric                                                        |            Type | Labels                                     | Cardinality note                                               | Recommended for alerting? (example)                                                               | Recommended for dashboard? (example) |
| ------------------------------------------------------------- | --------------: | ------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `rest_responses_total`                                        |         Counter | `endpoint`, `method`, `status`             | **Watch cardinality** — endpoints may include collection names | **Yes** — traffic & error alerts                                                                  | **Yes** — RPS                        |
| `rest_responses_fail_total`                                   |         Counter | `endpoint`, `method`                       | Same as above                                                  | **Yes (binary only)** — internal failure detected: `sum(rate(rest_responses_fail_total[5m])) > 0` | Optional (debug only)                |
| `rest_responses_duration_seconds_bucket` (+ `_count`, `_sum`) |       Histogram | `le`, often `endpoint`, `method`, `status` | Potentially high                                               | **Yes** — p95 latency SLO                                                                         | **Yes** — p95 latency                |
| `collections_total`                                           |           Gauge | none / `cluster`                           | Low                                                            | Optional                                                                                          | Yes                                  |
| `collections_vector_total`                                    |           Gauge | optional `id`                              | Per-collection                                                 | Optional                                                                                          | Yes                                  |
| `collection_hardware_metric_*`                                |           Gauge | `id`                                       | Per-collection                                                 | Optional                                                                                          | Yes                                  |
| `cluster_pending_operations_total`                            | Gauge / Counter | optional `peer_id`                         | Low                                                            | **Yes**                                                                                           | Yes                                  |
| `cluster_peers_total`, `cluster_*`                            | Gauge / Counter | optional `peer_id`                         | Low                                                            | Optional                                                                                          | Yes                                  |
| `process_*`                                                   | Gauge / Counter | minimal                                    | Low                                                            | **Yes**                                                                                           | Yes                                  |
| `up`                                                          |           Gauge | `instance`, `job`                          | Low                                                            | **Yes**                                                                                           | Yes                                  |

---

## Canonical Qdrant PromQL (correct for v1.16.x)

### REST request rate

```
sum(rate(rest_responses_total[1m]))
```

### REST HTTP error rate (authoritative)

```
sum(rate(rest_responses_total{status=~"4..|5.."}[5m]))
/
clamp_min(sum(rate(rest_responses_total[5m])), 1)
```

### Optional splits

Client errors (4xx):

```
sum(rate(rest_responses_total{status=~"4.."}[5m]))
```

Server errors (5xx):

```
sum(rate(rest_responses_total{status=~"5.."}[5m]))
```

### Internal failures only (rare)

```
sum(rate(rest_responses_fail_total[5m])) > 0
```

---

## Important Qdrant operational notes

* The `endpoint` label may include collection names unless metrics anonymization is enabled.
* To control cardinality:

  * enable metrics anonymization, or
  * prefer per-collection metrics with `id` labels, or
  * normalize endpoints at scrape time.
* For histograms, aggregate **only by `le`** when computing quantiles.

---

# Finalized plan

## 1. Minimal proven dashboard panels — Retriever

* Service Ready
  `max(service_ready{service="retrieval"})`

* Requests/s
  `sum(rate(retrieval_requests_total[1m]))`

* P95 latency
  `histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))`

* Error rate
  `sum(rate(retrieval_errors_total[5m])) / clamp_min(sum(rate(retrieval_requests_total[5m])),1)`

---

## 2. Minimal proven dashboard panels — Qdrant (corrected)

* Target Up
  `max(up{job=~"qdrant.*"})`

* Requests/s
  `sum(rate(rest_responses_total[1m]))`

* P95 latency
  `histogram_quantile(0.95, sum(rate(rest_responses_duration_seconds_bucket[5m])) by (le))`

* **REST HTTP error rate (correct)**

  ```
  sum(rate(rest_responses_total{status=~"4..|5.."}[5m]))
  /
  clamp_min(sum(rate(rest_responses_total[5m])),1)
  ```

* Total vectors
  `sum(collections_vector_total)`

---

## 3. Alert rules to start with (corrected)

### Retriever

* **RetrieverDown**
  `min_over_time(service_ready{service="retrieval"}[5m]) == 0`
  `for: 2m`

* **RetrieverHighErrors**
  `sum(rate(retrieval_errors_total[5m])) / clamp_min(sum(rate(retrieval_requests_total[5m])),1) > 0.01`
  `for: 5m`

* **RetrieverHighLatency_p95**
  `histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)) > 1`
  `for: 5m`

### Qdrant

* **QdrantDown**
  `up{job=~"qdrant.*"} == 0`
  `for: 2m`

* **QdrantHighHttpErrors (correct)**

  ```
  sum(rate(rest_responses_total{status=~"4..|5.."}[5m]))
  /
  clamp_min(sum(rate(rest_responses_total[5m])),1) > 0.01
  ```

* **QdrantInternalFailures**
  `sum(rate(rest_responses_fail_total[5m])) > 0`

---

