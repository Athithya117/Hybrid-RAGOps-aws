# Monitoring workflow — VictoriaMetrics + vmagent

## Simple summary

A lightweight telemetry pipeline scrapes Prometheus-format metrics from annotated pods and forwards them via Prometheus remote-write to a single-node VictoriaMetrics TSDB. Discovery is annotation-driven; vmagent performs scraping and remote-write, VictoriaMetrics stores and serves PromQL queries for validation and analysis.

## Precise description

### Goals

* Provide reliable, low-friction observability for application pods (retriever, qdrant).
* Keep the pipeline minimal and deterministic: single discovery mechanism, single scraper, single writer, single storage backend for development and CI validation.
* Make ingestion the authoritative signal of health (tests assert ingestion, not just exposure).

---

## Components and responsibilities

### vmagent (scraper)

* Discovers pod targets via `kubernetes_sd_configs` (`role: pod`).
* Uses a ConfigMap (`scrape.yml`) to define scrape jobs and relabeling.
* Scrapes `/metrics` on the target pod IP + annotated port.
* Remote-writes scraped samples to VictoriaMetrics (`/api/v1/write`).
* Exposes its own `/metrics` for operational telemetry (`vm_promscrape_*`, `vmagent_*`, persistent queue metrics).

Key vmagent flags (example):

```
-promscrape.config=/config/scrape.yml
-remoteWrite.url=http://victoria-metrics.<namespace>.svc.cluster.local:<port>/api/v1/write
-remoteWrite.tmpDataPath=/vmagent-remotewrite-data
-promscrape.suppressDuplicateScrapeTargetErrors=true
-promscrape.suppressScrapeErrors=true
```

### VictoriaMetrics (single-node)

* Receives remote-write requests, indexes and stores time-series.
* Exposes PromQL-compatible query endpoints:

  * `/api/v1/query`
  * `/api/v1/query_range`
* Dev defaults: 1d retention and `emptyDir` storage. Production must use durable storage and/or a clustered VM architecture.

### Instrumented applications (examples: retriever, qdrant)

* Expose Prometheus-format metrics at `/metrics`.
* Must conform to pod annotation contract so vmagent can discover them.

Annotation contract (per-pod):

```yaml
metadata:
  annotations:
    monitoring.io/scrape: "true"
    monitoring.io/port: "8001"
    monitoring.io/path: "/metrics"
```

---

## Scrape configuration (how vmagent builds targets)

vmagent uses pod discovery and relabeling to synthesize scrape targets. Important bits from `scrape.yml`:

* Discovery:

  ```yaml
  kubernetes_sd_configs:
    - role: pod
  ```

* Relabeling to keep only annotated pods and ready pods:

  ```yaml
  - source_labels: [__meta_kubernetes_namespace]
    action: keep
    regex: <namespace>
  - source_labels: [__meta_kubernetes_pod_annotation_monitoring_io_scrape]
    action: keep
    regex: "true"
  - source_labels: [__meta_kubernetes_pod_ready]
    action: keep
    regex: "true"
  ```

* Build scrape address from pod IP and annotation port:

  ```yaml
  - source_labels: [__meta_kubernetes_pod_ip,__meta_kubernetes_pod_annotation_monitoring_io_port]
    action: replace
    regex: (.+);(.+)
    replacement: '$1:$2'
    target_label: __address__
  ```

> The `replacement: '$1:$2'` literal must be preserved exactly in rendered manifests (no stray backslashes). This string instructs vmagent to format the scrape target as `<podIP>:<port>`.

---

## Metrics of interest (what apps and vmagent expose and why they matter for RAG)

### Retriever / Application metrics (examples)

* `retrieval_requests_total` — total HTTP requests handled (request volume).
* `retrieval_request_duration_seconds` — latency distribution (end-to-end).
* `retrieval_errors_total` — error count (failure rate).
* `dense_embed_requests_total`, `sparse_embed_requests_total` — embedding subsystem call counts.
* `qdrant_query_total`, `qdrant_query_duration_seconds` — vector DB usage and latency.
* `llm_calls_total`, `llm_call_duration_seconds` — LLM invocation counts and latency.
* `retrieved_docs_count` — result set sizes (useful for quality and cost analysis).
* `service_ready` (gauge) — readiness signal.

Why these matter for RAG:

* Request rates, error rates and latencies directly map to user experience and cost (LLM calls, vector DB calls).
* Embedding and retrieval metrics show where bottlenecks or excess cost occur.
* Monitoring both application- and infra-level metrics enables root-cause: is slowness in retriever, qdrant, or LLM?

### vmagent / ingestion metrics (operational)

* `vm_promscrape_discovery_kubernetes_objects` — discovery object counts (presence of pods).
* `vm_promscrape_scrape_pool_targets{status="up"}` — per-job scrape target counts (up/down target counts).
* `vmagent_remotewrite_sent_bytes_total` / `vm_persistentqueue_bytes_written_total` — evidence of remote-write traffic.
* `vm_promscrape_scraped_samples_sum` and related scrape duration counters.

Why these matter:

* They prove the scraper discovered and successfully scraped targets and that remote-write is being attempted. They are the primary signals when data fails to appear in the TSDB.

---

## Ingestion verification (what tests should assert)

Prefer authoritative signals over `up{}`:

1. **vmagent local metrics** — verify vmagent reports discovery activity and non-zero remote-write counters:

   * `vm_promscrape_discovery_kubernetes_objects{role="pod"} > 0`
   * `vm_persistentqueue_bytes_written_total > 0` or `vmagent_remotewrite_sent_bytes_total > 0`

2. **VictoriaMetrics query** — use PromQL and inspect `.stats.seriesFetched` in the JSON response:

   * Example queries used for validation:

     * `count({__name__=~"app_info|retrieval_requests_total|retrieval_errors_total"})`
     * `max(up{namespace="<namespace>"})`
   * Check the HTTP JSON response `.data.result` and `.stats.seriesFetched`; `seriesFetched > 0` confirms the query engine read series.

---

## RBAC and permissions (critical infra requirement)

vmagent performs cluster-level discovery. Provide a dedicated ServiceAccount and bind a ClusterRole with minimum discovery permissions:

Minimal ClusterRole rules:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata: { name: vmagent-clusterrole }
rules:
- apiGroups: [""]
  resources: ["pods", "endpoints", "services", "nodes"]
  verbs: ["get", "list", "watch"]
```

Bind to the vmagent ServiceAccount:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata: { name: vmagent-clusterrolebinding }
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: vmagent-clusterrole
subjects:
- kind: ServiceAccount
  name: vmagent
  namespace: monitoring
```

---

## Configuration knobs (recommended env vars / generator-grade knobs)

Expose these variables at generator/runtime so infra is predictable and overridable in CI:

* `VM_NAMESPACE` — namespace running monitoring (default `monitoring`)
* `VMAGENT_PORT` — vmagent metrics port (default `8429`)
* `VICTORIA_PORT` — VictoriaMetrics HTTP port (default `8428`)
* `VMAGENT_IMAGE`, `VM_IMAGE` — images and pinned versions
* `VM_SCRAPE_INTERVAL`, `VM_SCRAPE_TIMEOUT` — scraping behavior
* `RETRIEVAL_NAMESPACE`, `QDRANT_NAMESPACE` — application namespaces
* `RETRIEVAL_METRICS_PORT`, `QDRANT_METRICS_PORT` — expected metrics ports
* `REMOTE_WRITE_URL` — remote-write destination (default to cluster-local Victoria)

Generators should validate numeric ports, preserve string literals required by vmagent relabeling, and refuse to render manifests with unintended escape characters.

---

## Operational checks & quick diagnostics

1. Check monitoring objects and vmagent config:

   ```bash
   kubectl -n monitoring get svc,deployment,sa,clusterrolebinding -l app=vmagent
   kubectl -n monitoring get configmap vmagent-config -o yaml | sed -n '1,200p'
   ```

2. Inspect vmagent logs for discovery or RBAC errors:

   ```bash
   kubectl -n monitoring logs deploy/vmagent --tail=200
   # Look for: "pods is forbidden" => RBAC missing
   ```

3. Probe vmagent and Victoria locally (port-forward):

   ```bash
   kubectl -n monitoring port-forward svc/vmagent 8429:8429 &
   kubectl -n monitoring port-forward svc/victoria-metrics 8428:8428 &
   curl -sS http://127.0.0.1:8429/metrics | grep vm_promscrape
   curl -sS -G --data-urlencode 'query=count(up)' http://127.0.0.1:8428/api/v1/query | jq .
   ```

4. PromQL verification used by CI:

   * `count({__name__=~"app_info|retrieval_requests_total|retrieval_errors_total"})` — checks for application metric series.
   * `max(up{namespace="<namespace>"})` — ensures some scrape presence (but prefer `.stats.seriesFetched` for ingestion confirmation).

---

## Troubleshooting priority checklist

1. vmagent logs show `pods is forbidden` → fix RBAC (apply ClusterRole + ClusterRoleBinding for vmagent ServiceAccount).
2. vmagent `/metrics` shows discovery counters = 0 but pods exist → check `vmagent-config` relabel rules and ensure pods have monitoring annotations.
3. vmagent reports scrapes but Victoria query returns empty → verify `-remoteWrite.url` is correct and Victoria `/metrics` shows write/insert counters.
4. Rendered `scrape.yml` contains backslashes or an escaped `'$1:$2'` → fix generator templating so `replacement: '$1:$2'` appears exactly.
5. If pod metrics are visible by direct port-forward but ingestion is 0 → check network policies, DNS (service lookup), and vmagent logs for write errors.

---

## Repository conventions (where to look / what to change)

* Generator script: `infra/generators/monitoring_and_alerts.sh`

  * Renders `infra/manifests/00-monitoring.yaml` with namespace, configmap, vmagent and Victoria deployments and services.
  * Must create `serviceaccount`, `clusterrole`, `clusterrolebinding` for vmagent.
  * Must validate `replacement: '$1:$2'` literal presence and absence of backslashes.
* Application generators:

  * `infra/generators/retriever.sh`
  * `infra/generators/qdrant_cluster.sh`
  * These must ensure pods have the annotation contract and declare metrics ports (or patch the pod spec to declare them).
* Tests:

  * `infra/tests/monitoring/test_retriever.sh`
  * `infra/tests/monitoring/test_qdrant.sh`
  * Test harnesses port-forward vmagent and Victoria, optionally generate load, and assert ingestion based on PromQL results and `.stats.seriesFetched`.

---

## Minimal example PromQL checks for CI

* Application metrics present:

  ```
  count({__name__=~"app_info|retrieval_requests_total|retrieval_errors_total"})
  ```
* Namespace-level scrape presence:

  ```
  max(up{namespace="inference"})
  ```

Interpretation:

* Query HTTP JSON `.data.result` non-empty → series exist.
* Prefer to assert `.stats.seriesFetched > 0` to avoid false positives from empty query parsing.

---
