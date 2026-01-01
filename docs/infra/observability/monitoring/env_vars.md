# Monitoring — Environment Variables (generator inputs and policy)

## Overview

This document enumerates the environment variables intentionally consumed by observability files in`infra/generators/`. These variables are the knobs the generator uses to render deterministic manifests, alert rules, and dashboards. Variables are ordered by operational leverage (most impactful first). Change values only with an approved PR and CI validation.


---

## Usage model

* The generator reads these variables at render time and produces Kubernetes manifests, vmalert rules, and Grafana dashboard JSON.
* Variables are **policy knobs** (SLO/alert thresholds, resource sizing, image pins) or **platform tuning** (scrape interval/timeouts, replica counts).
* Secrets (webhooks, credentials) must be stored in Kubernetes Secrets or your secret manager; the generator references Secrets rather than embedding plaintext values.
* CI responsibilities:

  * Validate variable formats and presence where required.
  * Render templates (`envsubst` or equivalent) and attach rendered artifacts to PRs.
  * Run PromQL/alert rule syntax checks (e.g., `promtool` or `vmalert` dry-run).
  * Optionally run an ingestion verification job against a staging cluster.

---

## Environment variables (ordered by operational leverage)

### 1. `VMAGENT_REPLICAS`

**Description:** Number of `vmagent` pods performing discovery, scraping, and remote-write.
**Default:** `1`
**When to change:** Increase to `2` for HA scraping if the TSDB or remote-write endpoint supports deduplication. Keep at `1` for dev/CI.
**Failure modes:** Duplicated series, inflated cardinality, duplicate alerts.

---

### 2. VictoriaMetrics resource sizing

**Variables:** `VM_RES_CPU`, `VM_RES_MEM`
**Description:** CPU and memory `resources.requests`/`limits` for the VictoriaMetrics container.
**Defaults:**

* `VM_RES_CPU=100m`
* `VM_RES_MEM=256Mi`
  **When to change:** Increase when queries are slow, OOMs occur, retention or cardinality grows.
  **Guidance:**
* Dev: `100m` / `256Mi`
* Medium: `500m` / `2Gi`
* Prod: `1000m–4000m` / `8Gi–32Gi`
  **Failure modes:** OOM restarts, slow queries, data gaps.

---

### 3. vmagent resource sizing

**Variables:** `VMAGENT_RES_CPU`, `VMAGENT_RES_MEM`
**Description:** CPU and memory requests/limits for vmagent.
**Defaults:**

* `VMAGENT_RES_CPU=100m`
* `VMAGENT_RES_MEM=256Mi`
  **When to change:** Increase when scraping many targets or vmagent logs show queue pressure.
  **Guidance:** For 100–500 targets use `250–500m` CPU and `512Mi–1Gi` memory.
  **Failure modes:** Dropped samples, remote-write backlog growth.

---

### 4. `VM_SCRAPE_INTERVAL`

**Description:** Global scrape interval used by vmagent.
**Default:** `15s`
**When to change:** Increase to `30s` or `60s` to reduce cost when cardinality is high. Decrease only if higher-resolution metrics are required and resources permit.
**Failure modes:** Too short → CPU/WAL pressure; too long → slower alert detection.

---

### 5. `VM_SCRAPE_TIMEOUT`

**Description:** Per-scrape timeout applied by vmagent.
**Default:** `10s`
**When to change:** Increase for slow exporters or large payloads (e.g., `20s`).
**Failure modes:** Scrape failures and missing samples.

---

### 6. Image pins for monitoring components

**Variables:** `VMAGENT_IMAGE`, `VM_IMAGE`, `VMALERT_IMAGE`, `GRAFANA_IMAGE`
**Description:** Exact container image tags used by the generator to render Deployments. Pinning ensures deterministic, reviewable changes.

**Recommended pinned defaults:**

* `VMAGENT_IMAGE=victoriametrics/vmagent:v1.132.0`
* `VM_IMAGE=victoriametrics/victoria-metrics:v1.132.0`
* `VMALERT_IMAGE=victoriametrics/vmalert:v1.132.0`
* `GRAFANA_IMAGE=grafana/grafana:12.3.1`

**Guidance:** Update images via staged rollouts and CI validation only. Do not use `:latest`.

---

### 7. SLO and alerting policy knobs

These variables define SLO targets and burn thresholds. They are policy knobs used to render static rule templates into environment-specific rules.

**`SLO_SUCCESS_TARGET`**
*Purpose:* Success target (example `0.999` for 99.9% success).
*Default:* `0.999`
*Validation:* `^0\.[0-9]{1,6}$`. Production policy typically requires `>= 0.99`.

**`SLO_LATENCY_QUANTILE`**
*Purpose:* Quantile used for p-percentile latency SLOs (e.g., `0.95` or `0.99`).
*Default:* `0.95`
*Validation:* Allowed values `0.95` or `0.99`.

**`SLO_FAST_BURN_MULTIPLIER`**
*Purpose:* Multiplier for fast-burn (page) alerts.
*Default:* `2`
*Validation:* Numeric > 1.

**`SLO_SLOW_BURN_MULTIPLIER`**
*Purpose:* Multiplier for slow-burn (warning) alerts.
*Default:* `1.2`
*Validation:* Numeric > 1.

---

### 8. Service-specific latency thresholds

**`RETRIEVER_LATENCY_THRESHOLD_SECONDS`**
*Purpose:* p95 latency threshold for retriever SLO alerts (seconds).
*Default:* `0.5`
*Validation:* Numeric float.

**`QDRANT_LATENCY_THRESHOLD_SECONDS`**
*Purpose:* p95 latency threshold for Qdrant SLO alerts (seconds).
*Default:* `0.8`
*Validation:* Numeric float.

---

### 9. Notification webhook (secret)

**`ALERTMANAGER_SLACK_WEBHOOK`**
*Purpose:* Slack webhook URL used for non-paging notification receivers. Store as a Kubernetes Secret, not as a plaintext env var.
*Validation:* Must be a valid URL when provided.
*When to set:* Required to enable Slack non-paging notifications.
```sh
https://www.svix.com/resources/guides/how-to-get-slack-webhook-url/
```
---

### 10. **`PAGERDUTY_INTEGRATION_KEY`**

*Purpose:* PagerDuty service integration key used by Alertmanager to create and resolve incidents for critical alerts. Store as a **Kubernetes Secret**, not as a plaintext env var.

*Validation:* Must be a non-empty integration key string issued by PagerDuty for a Prometheus or Events API v1 integration.

*When to set:* **Required** to enable paging for critical alerts (`severity: critical`).

```sh
https://support.pagerduty.com/main/docs/services-and-integrations#create-a-service
```

---


## CI validation requirements

Every PR that modifies templates or these variables must run the following checks:

1. **Format & presence**

   * Validate SLO and threshold formats:

   ```bash
   echo "$SLO_SUCCESS_TARGET" | grep -Eq '^0\.[0-9]{1,6}$' || echo "INVALID SLO_SUCCESS_TARGET"
   echo "$SLO_LATENCY_QUANTILE" | grep -Eq '^0\.(95|99)$' || echo "INVALID SLO_LATENCY_QUANTILE"
   echo "$RETRIEVER_LATENCY_THRESHOLD_SECONDS" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || echo "INVALID RETRIEVER_LATENCY_THRESHOLD_SECONDS"
   ```
2. **Render & syntax**

   * Render templates with `envsubst` (or the configured template engine) and attach the rendered artifacts to the PR.
   * Run `promtool check rules` or `vmalert` dry-run on the rendered rules.
3. **Label and annotation lint**

   * Lint rendered alert rules for required labels and annotations: `plane`, `severity`, `runbook`, and `service` where applicable.
4. **Optional ingestion check (integration)**

   * CI integration jobs should optionally verify ingestion by querying the TSDB and asserting `.data.result` non-empty or `.stats.seriesFetched > 0` for representative metrics in staging.

---

## Per-environment recommended configurations

**Development**

```
VMAGENT_REPLICAS=1
VM_SCRAPE_INTERVAL=15s
SLO_SUCCESS_TARGET=0.99
SLO_LATENCY_QUANTILE=0.95
RETRIEVER_LATENCY_THRESHOLD_SECONDS=1.0
QDRANT_LATENCY_THRESHOLD_SECONDS=1.0
VMALERT_IMAGE=victoriametrics/vmalert:v1.132.0
GRAFANA_IMAGE=grafana/grafana:12.3.1
DEFAULT_NAMESPACE=monitoring
```

**Production**

```
VMAGENT_REPLICAS=2
VM_RES_CPU=1000m
VM_RES_MEM=8Gi
VMAGENT_RES_CPU=500m
VMAGENT_RES_MEM=1Gi
VM_SCRAPE_INTERVAL=15s
SLO_SUCCESS_TARGET=0.999
SLO_LATENCY_QUANTILE=0.95
SLO_FAST_BURN_MULTIPLIER=2
SLO_SLOW_BURN_MULTIPLIER=1.2
RETRIEVER_LATENCY_THRESHOLD_SECONDS=0.5
QDRANT_LATENCY_THRESHOLD_SECONDS=0.8
ALERTMANAGER_SLACK_WEBHOOK=<stored-in-k8s-secret>
VMALERT_IMAGE=victoriametrics/vmalert:v1.132.0
GRAFANA_IMAGE=grafana/grafana:12.3.1
DEFAULT_NAMESPACE=monitoring
```

---

## Failure-mode quick reference

* **Empty TSDB / missing series:** check `VMAGENT_REPLICAS`, `VM_SCRAPE_INTERVAL`, `VM_SCRAPE_TIMEOUT`, vmagent logs, and vmagent resources.
* **No alerts firing when expected:** verify rendered rules include the intended SLO values (`SLO_SUCCESS_TARGET`, thresholds) and that CI render artifacts match applied manifests.
* **Excessive paging:** inspect SLO targets and burn multipliers; consider adjusting `SLO_FAST_BURN_MULTIPLIER` or thresholds.
* **Dashboards with incorrect links:** ensure `DEFAULT_NAMESPACE` matches Grafana provisioning and that the generator URL-encodes SQL links.

---

## Governance

* The generator is the authoritative source for rendered monitoring artifacts; direct edits to rendered files are prohibited.
* Secrets (webhooks, API keys) must be stored in Kubernetes Secrets. 
* Changes to SLOs or alerting knobs require a PR that includes: a one-line justification, rendered artifacts, and CI passing render/syntax/ingestion checks.
* Treat SLO env vars as policy: avoid frequent churn and document business rationale when changing.

---
