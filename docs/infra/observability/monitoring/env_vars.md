# Monitoring environment variables (what to change, when, and why)

This document describes **only** the environment variables that are intentionally exposed by `infra/generators/monitoring_and_alerts.sh`.
They are ordered by **leverage**: the ones most likely to break or unlock observability come first; cosmetic or static knobs are last.

The defaults are safe for **local k3s / dev**. Change values **only** when the conditions below apply.

---

## 1. `REMOTE_WRITE_URL` (highest leverage)

**What it is**
The endpoint where `vmagent` sends all scraped metrics (VictoriaMetrics remote-write API).

**Default**
`http://victoria-metrics.monitoring.svc.cluster.local:8428/api/v1/write`

**When to change**

* When exporting metrics to a **remote / centralized TSDB**
* When switching from in-cluster VictoriaMetrics to:

  * VM Cluster
  * Managed VictoriaMetrics
  * Thanos Receive
  * Any HTTPS / authenticated endpoint

**Concrete examples**

* Local dev / k3s: **leave default**
* Central metrics cluster:

  ```
  https://vm-remote.company.internal/api/v1/write
  ```

**Failure mode if wrong**

* TSDB stays empty even though `/metrics` endpoints work
* All PromQL tests fail with “no results”

---

## 2. `VMAGENT_REPLICAS`

**What it is**
Number of `vmagent` pods scraping and remote-writing metrics.

**Default**
`1`

**When to change**

* Keep `1` for dev, CI, single-node, or non-deduplicated setups
* Increase **only** if:

  * You need HA for scraping **and**
  * Your TSDB or downstream system handles duplicate samples

**Concrete values**

* Dev / k3s: `1`
* Production with dedupe: `2`

**Failure mode if misused**

* Duplicate metrics
* Inflated series counts
* Duplicate alerts

---

## 3. `VM_RES_CPU`, `VM_RES_MEM` (VictoriaMetrics resources)

**What they are**
CPU and memory requests/limits for the VictoriaMetrics TSDB container.

**Defaults**

* CPU: `100m`
* Memory: `256Mi`

**When to change**
Increase when **VictoriaMetrics** shows:

* Slow PromQL queries
* High CPU usage
* OOM kills
* Large retention or high cardinality

**Concrete guidance**

* Small dev cluster: defaults are fine
* Medium cluster: `500m` CPU, `2–4Gi` memory
* Large / prod: `1000–4000m` CPU, `8–32Gi` memory

**Failure mode if too small**

* OOM restarts
* Slow or failing PromQL queries
* Data gaps during WAL replay

---

## 4. `VMAGENT_RES_CPU`, `VMAGENT_RES_MEM` (vmagent resources)

**What they are**
CPU and memory for the scraper + remote-write agent.

**Defaults**

* CPU: `100m`
* Memory: `256Mi`

**When to change**
Increase when:

* Scraping many pods/services
* Remote-write backlog grows
* vmagent logs show queue pressure or OOMs

**Concrete guidance**

* Dev / few targets: defaults
* 100–500 targets: `250–500m` CPU, `512Mi–1Gi` memory
* Large clusters: scale further based on scrape count

**Failure mode if too small**

* Dropped samples
* Delayed ingestion
* Remote-write queue growth

---

## 5. `VM_SCRAPE_INTERVAL`

**What it is**
How often vmagent scrapes `/metrics` from targets.

**Default**
`15s`

**When to change**

* Increase interval if:

  * Cardinality is high
  * Victoria CPU or WAL IO is under pressure
* Decrease only if you *need* higher resolution and can afford the cost

**Concrete values**

* Standard observability: `15s`
* High cardinality / cost control: `30s` or `1m`
* Tight SLOs (rare): `10s`

**Failure mode if misconfigured**

* Too short → high CPU, WAL churn
* Too long → slow alert detection

---

## 6. `VM_SCRAPE_TIMEOUT`

**What it is**
Maximum time allowed for a single scrape request.

**Default**
`10s`

**When to change**

* Increase if targets are slow or return large payloads
* Leave unchanged for normal services

**Concrete values**

* Normal services: `10s`
* Slow exporters: `20s`

**Failure mode if too low**

* Frequent scrape failures
* Missing samples

---

## 7. `VMAGENT_IMAGE`, `VM_IMAGE` (lowest leverage)

**What they are**
Pinned container images for vmagent and VictoriaMetrics.

**Defaults**

* `victoriametrics/vmagent:v1.99.0`
* `victoriametrics/victoria-metrics:v1.99.0`

**When to change**

* Version upgrades
* Security patches
* Controlled testing in staging

**Rules**

* Always pin exact versions
* Never use `:latest`
* Upgrade in staging before prod

---

## Summary (mental model)

* **Empty TSDB?** → check `REMOTE_WRITE_URL`
* **Metrics duplicated?** → check `VMAGENT_REPLICAS`
* **Queries slow / OOMs?** → raise `VM_RES_MEM` first
* **Samples delayed or dropped?** → raise vmagent resources
* **Too expensive?** → increase `VM_SCRAPE_INTERVAL`

---
