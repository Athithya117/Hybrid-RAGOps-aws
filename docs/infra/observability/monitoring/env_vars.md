# Prometheus environment variables — exact meaning, default, when to change (concrete)

Below are the Prometheus-related environment variables used by the generator and Helm values. For each variable I show:

* **What it is** (effect / mapping)
* **Default value used in this repo**
* **Valid format / type**
* **When to change it** (concrete criteria)
* **Risk / consequence** of changing it

Use these variables to make Prometheus behavior deterministic across environments. Where appropriate I include the exact Helm value path that the generator writes.

---

## `PROM_SCRAPE_INTERVAL`

* **What it is:** Global scrape interval Prometheus uses for target endpoints (how often Prometheus requests `/metrics` from targets).
* **Helm mapping:** `prometheus.prometheusSpec.scrapeInterval` and ServiceMonitor `endpoints[].interval`.
* **Default in this repo:** `15s`
* **Type / format:** Prometheus duration string, e.g. `15s`, `30s`, `1m`.
* **When to change:**

  * Increase (longer interval) when total metric **series cardinality** or scrape load causes Prometheus resource pressure (high CPU, WAL churn, or scraping timeouts). Example: move to `30s` or `1m` if series count > 200k or Prometheus CPU saturated.
  * Decrease (shorter interval) only if you need finer-grained alerting or SLO detection (e.g., SLO requires detection <15s). Example: change to `10s` only if you have tight SLOs and Prometheus sizing to match.
* **Risk:** Shorter intervals increase CPU, memory, WAL writes, metrics cardinality impact; longer intervals increase detection latency for alerts and SLO violations.

**Example**

```bash
export PROM_SCRAPE_INTERVAL="15s"
```

---

## `PROM_EVALUATION_INTERVAL`

* **What it is:** How often Prometheus evaluates recording and alerting rules.
* **Helm mapping:** `prometheus.prometheusSpec.evaluationInterval`
* **Default in this repo:** unset explicitly → generator sets it equal to `PROM_SCRAPE_INTERVAL` by default (so both are the same).
* **Type / format:** Prometheus duration string, same format as scrape interval.
* **When to change:**

  * Keep equal to `PROM_SCRAPE_INTERVAL` in almost all cases. Change only when you want rules evaluated less often than scrapes (reduce alert noise/resource usage) or more often (rare).
  * Example: `scrapeInterval=15s`, `evaluationInterval=30s` reduces rule eval CPU while retaining data fidelity.
* **Risk:** If evaluationInterval is much longer than scrapeInterval, alerts will have extra latency; if shorter, rules may evaluate on stale data or more often than new samples arrive.

**Example**

```bash
export PROM_EVALUATION_INTERVAL="15s"
```

---

## `PROM_RETENTION`

* **What it is:** How long Prometheus retains raw time-series data on disk (e.g., `7d` = 7 days).
* **Helm mapping:** `prometheus.prometheusSpec.retention`
* **Default in this repo:** `7d`
* **Type / format:** Duration string such as `7d`, `30d`, `24h`.
* **When to change:**

  * Increase retention when you need historical troubleshooting/alerts beyond default window (e.g., compliance or long-term analysis). Example: 30d for production business analytics.
  * Decrease retention to conserve disk if cardinality is high and storage is constrained (e.g., 3d).
* **Risk:** Longer retention requires more PVC size and IO; shorter retention reduces ability to investigate past incidents.

**Example**

```bash
export PROM_RETENTION="7d"
```

---

## `PROM_REPLICAS`

* **What it is:** Number of Prometheus replicas managed by the chart (operator/HA setup).
* **Helm mapping:** used to set `prometheus.prometheusSpec.replicaCount` (chart/operator behaviour).
* **Default in this repo:** `1` for `kind` (local), `2` recommended for AKS/prod (generator default logic).
* **Type / format:** Integer ≥ 1
* **When to change:**

  * Set to `1` for development or single-node clusters.
  * Set to `>1` for production only if you handle deduplication (e.g., Thanos or Alertmanager route dedupe) and understand HA implications (alerts may duplicate).
  * Use `2` for basic HA; use >2 only for special multi-instance topologies and with dedup tooling.
* **Risk:** Multiple Prometheus replicas without deduplication will cause duplicate alerts and duplicate remote-write data; more replicas increase resource usage.

**Example**

```bash
export PROM_REPLICAS=1
```

---

## `PROM_STORAGE_SIZE`

* **What it is:** PVC size requested for Prometheus TSDB storage.
* **Helm mapping:** `prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage`
* **Default in this repo:** `50Gi` in Helm generator examples; for dev you may use `2Gi`.
* **Type / format:** Kubernetes storage quantity (e.g., `50Gi`, `10Gi`)
* **When to change:**

  * Increase with higher scrape frequency, higher retention, or higher cardinality (more series).
  * Estimate: approximate retention * ingestion rate * cardinality multiplier. If cardinality doubles, storage needed roughly doubles.
  * Use `2Gi` for small local clusters (short retention), `50Gi`+ for production depending on retention and cardinality.
* **Risk:** Underprovisioned PVC will cause Prometheus to run out of disk; expansion may be possible but slower.

**Example**

```bash
export PROM_STORAGE_SIZE="50Gi"
```

---

## `PROM_STORAGE_CLASS`

* **What it is:** StorageClass used by the Prometheus PVC.
* **Helm mapping:** `prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName`
* **Default in this repo:** empty (cluster default)
* **Type / format:** StorageClass name string
* **When to change:**

  * Set to a high-performance class (`managed-premium`, `premium-rwo`) for production to reduce IO latency for WAL/TSDB.
  * Use default or ephemeral storage for CI/kind.
* **Risk:** Using slow storage will impact WAL replay and query performance; using ephemeral `emptyDir` loses data on pod restart.

**Example**

```bash
export PROM_STORAGE_CLASS="managed-premium"
```

---

## `PROM_CPU_REQUEST` / `PROM_CPU_LIMIT`

* **What they are:** CPU request/limit for the Prometheus pod.
* **Helm mapping:** `prometheus.prometheusSpec.resources.requests.cpu` and `.limits.cpu`
* **Defaults in this repo:** `PROM_CPU_REQUEST=500m`, `PROM_CPU_LIMIT=2000m` (examples)
* **Type / format:** Kubernetes CPU quantities (e.g., `500m`, `2000m`)
* **When to change:**

  * Increase requests/limits when CPU is consistently high (scraping, rule evaluation, queries). Example: increase to `2000m` request for heavy workloads.
  * Matches expected traffic and cardinality; large cardinality systems require more CPU.
* **Risk:** Too low request leads to throttling; too low limit causes OOMKill or OOMIf memory bound.

**Example**

```bash
export PROM_CPU_REQUEST="500m"
export PROM_CPU_LIMIT="2000m"
```

---

## `PROM_MEM_REQUEST` / `PROM_MEM_LIMIT`

* **What they are:** Memory request/limit for the Prometheus pod.
* **Helm mapping:** `prometheus.prometheusSpec.resources.requests.memory` and `.limits.memory`
* **Defaults in this repo:** `PROM_MEM_REQUEST=1Gi`, `PROM_MEM_LIMIT=8Gi` (examples)
* **Type / format:** Kubernetes memory quantities (e.g., `1Gi`, `8Gi`)
* **When to change:**

  * Increase memory when Prometheus OOMs, when query latency rises, or when WAL memory pressure is observed.
  * For high-cardinality or long-retention, increase to multiple tens of GB (production often uses 16Gi–64Gi depending on scale).
* **Risk:** Insufficient memory results in OOMs and restart loops, data gaps and failed queries.

**Example**

```bash
export PROM_MEM_REQUEST="1Gi"
export PROM_MEM_LIMIT="8Gi"
```

---

## `PROM_WAL_COMPRESSION` (or `walCompression` controlled by generator)

* **What it is:** Enables compression of the Prometheus WAL (reduces WAL IO and disk usage).
* **Helm mapping:** `prometheus.prometheusSpec.walCompression`
* **Default in this repo:** forced **true** (generator sets `walCompression: true`).
* **Type / format:** boolean (`true`/`false`)
* **When to change:**

  * Keep **true** for almost all environments. Do not disable unless debugging a rare compression bug or running an extremely old Prometheus that lacks support.
* **Risk:** Disabling increases WAL write amplification and disk consumption; enabling has negligible downside on modern versions.

**Example**

```bash
export PROM_WAL_COMPRESSION="true"
```

---

## `FAIL_ON_MISCONFIG` (monitoring generator helper)

* **What it is:** Controls whether the generator should fail the apply if ServiceMonitor → Service bindings are invalid.
* **Default in this repo:** `false`
* **Type / format:** boolean (`true`/`false`)
* **When to change:**

  * Set to `true` in CI or production deploy jobs to fail early on misconfiguration.
  * Keep `false` locally to allow iterative testing.
* **Risk:** If `true`, `apply` will abort on binding mismatches; good for gatekeeping.

**Example**

```bash
export FAIL_ON_MISCONFIG="true"
```

---

## Practical guidance / tuning checklist (concrete)

1. **Dev / local (kind):**

   * `PROM_REPLICAS=1`
   * `PROM_SCRAPE_INTERVAL=15s`
   * `PROM_RETENTION=1d` or `2d`
   * `PROM_STORAGE_SIZE=2Gi`
   * `PROM_WAL_COMPRESSION=true`
   * Resource requests small: `PROM_CPU_REQUEST=250m`, `PROM_MEM_REQUEST=512Mi`.

2. **Staging / small cluster:**

   * `PROM_REPLICAS=1`
   * `PROM_SCRAPE_INTERVAL=15s`
   * `PROM_RETENTION=7d`
   * `PROM_STORAGE_SIZE=50Gi` (estimate based on cardinality)
   * `PROM_CPU_REQUEST=500m`, `PROM_MEM_REQUEST=2Gi`

3. **Production / high-cardinality:**

   * `PROM_REPLICAS=2` (with deduplication or Thanos)
   * `PROM_SCRAPE_INTERVAL=15s` or `30s` (if cardinality huge)
   * `PROM_RETENTION=14d` or `30d`
   * `PROM_STORAGE_SIZE=200Gi+` (depends on ingestion; use sizing calculator)
   * `PROM_CPU_REQUEST=2000m+`, `PROM_MEM_REQUEST=16Gi+`
   * `PROM_WAL_COMPRESSION=true`
   * Use a fast `PROM_STORAGE_CLASS` (SSD/premium)

4. **If Prometheus shows load/slow queries:**

   * First: increase `PROM_CPU_REQUEST` & `PROM_MEM_REQUEST`.
   * Second: increase `PROM_SCRAPE_INTERVAL` from `15s` → `30s`.
   * Third: reduce label cardinality (remove dynamic labels from metrics/histograms or add relabeling to drop `collection`).

---

## Exact Helm keys to check / set in generated values file

```yaml
prometheus:
  prometheusSpec:
    replicaCount: <PROM_REPLICAS>
    scrapeInterval: "<PROM_SCRAPE_INTERVAL>"
    evaluationInterval: "<PROM_EVALUATION_INTERVAL>"
    walCompression: <PROM_WAL_COMPRESSION>        # boolean
    retention: "<PROM_RETENTION>"
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: "<PROM_STORAGE_CLASS or omitted>"
          resources:
            requests:
              storage: "<PROM_STORAGE_SIZE>"
    resources:
      requests:
        cpu: "<PROM_CPU_REQUEST>"
        memory: "<PROM_MEM_REQUEST>"
      limits:
        cpu: "<PROM_CPU_LIMIT>"
        memory: "<PROM_MEM_LIMIT>"
```

---

Use the above variables to make Prometheus configuration explicit and deterministic. Change values only for one of these three reasons: (a) capacity / resource pressure, (b) SLO / detection latency requirements, (c) cardinality or storage retention requirements. Adjust pairing of interval ↔ retention ↔ storage_size together (changing one typically requires re-evaluating the others).
