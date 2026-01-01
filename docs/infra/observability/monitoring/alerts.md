# Alerts — Platform alerting design and runtime control flow

## Simple summary

Alerts convert observability signals into deterministic operational actions. They are organized into three planes — **ingestion (monitoring truth)**, **safety (structural correctness)**, and **slo (user-facing reliability)**. Ingestion and safety alerts prevent noisy or misleading paging by gating SLO alerts. Critical alerts page (PagerDuty); warnings post to Slack. Alerts are evaluated by **vmalert**, routed and inhibited by **Alertmanager**, and authored as a single source of truth under `infra/manifests/alerts/` (rendered `slo.rules.yaml`, `vmalert-deployment.yaml`, `alertmanager-config.yaml`).

---

## Precise specification

### Goals

1. Provide authoritative evidence about system health and user impact.
2. Page only when human intervention is required now; otherwise notify (Slack).
3. Ensure alerts are actionable and link to runbooks.
4. Make alert generation, validation, and deployment deterministic and CI-gated.

### Actors and components

* **Instrumented services** (retriever, qdrant, vmagent, VictoriaMetrics). Metrics exposed on `/metrics`.
* **vmagent** — scrapes pod metrics and remote-writes to VictoriaMetrics.
* **VictoriaMetrics** — TSDB and PromQL endpoint.
* **vmalert** — evaluates alert rules (reads `slo.rules.yaml`, posts to Alertmanager).
* **Alertmanager** — routes, groups and inhibits notifications; configured with `alertmanager.yml`.
* **PagerDuty** & **Slack** — notification endpoints (PagerDuty for paging, Slack for async).
* **CI** — renders templates, runs rule validation (`promtool` or `vmalert` dry-run), lints labels, and publishes manifests.

---

## Alert classification (planes)

* **ingestion**: proves observability pipeline integrity. Must fire before trusting any other alerts.
* **safety**: detects structural, binary failures that threaten correctness or availability.
* **slo**: burn-rate or latency alerts reflecting user impact; SLO alerts drive paging policy.

---

## Required labels and annotations for every alert

Every rule must include these labels/annotations:

Labels:

```
labels:
  plane: {ingestion|safety|slo}
  severity: {critical|warning|info}
  service: <service-name>          # where applicable: retriever, qdrant, vmagent, etc.
  runbook: <runbook-id>            # optional label if used by tooling
```

Annotations:

```
annotations:
  summary: "<one-line human summary>"
  description: "<short reproduction/verification steps>"
  runbook: "https://runbooks.internal/<service>#<anchor>"
```

Policy:

* `plane` and `severity` are mandatory.
* `service` is required for service-scoped rules.
* `runbook` annotation must be a stable URL served from the runbooks site (not GitHub).

---

## Concrete alert rules (canonical list with PromQL)

All rules grouped under `groups:` in `slo.rules.yaml`. Use exact expressions below unless intentionally variant.

### Ingestion (monitoring truth)

1. **VmagentDiscoveryEmpty**

```yaml
alert: VmagnetDiscoveryEmpty
expr: vm_promscrape_discovery_kubernetes_objects{role="pod"} == 0
for: 2m
labels: { severity: "critical", plane: "ingestion" }
annotations:
  summary: "vmagent sees zero discovery objects"
  runbook: "<RUNBOOK_INGESTION_URL>"
```

2. **VmagentNoRemoteWrite**

```yaml
alert: VmagentNoRemoteWrite
expr: increase(vm_persistentqueue_bytes_written_total[5m]) == 0
for: 5m
labels: { severity: "critical", plane: "ingestion" }
annotations:
  summary: "No remote-write bytes recorded by vmagent"
  runbook: "<RUNBOOK_INGESTION_URL>"
```

### Safety (structural)

1. **RetrieverNotReady**

```yaml
alert: RetrieverNotReady
expr: service_ready{service="retrieval"} == 0
for: 2m
labels: { severity: "critical", plane: "safety", service: "retriever" }
annotations:
  summary: "Retriever service reports not ready"
  runbook: "<RUNBOOK_RETRIEVER_URL>"
```

2. **QdrantDeadReplicas**

```yaml
alert: QdrantDeadReplicas
expr: collection_dead_replicas > 0
for: 2m
labels: { severity: "critical", plane: "safety", service: "qdrant" }
annotations:
  summary: "Qdrant reports dead replicas"
  runbook: "<RUNBOOK_QDRANT_URL>"
```

3. **QdrantSnapshotStuck**

```yaml
alert: QdrantSnapshotStuck
expr: snapshot_creation_running > 0
for: 30m
labels: { severity: "warning", plane: "safety", service: "qdrant" }
annotations:
  summary: "Qdrant snapshot running > 30m"
  runbook: "<RUNBOOK_QDRANT_URL>"
```

### SLO (burn-rate and latency)

All SLO rules use `SLO_SUCCESS_TARGET`, burn multipliers, and `SLO_LATENCY_QUANTILE` injected from env vars.

1. **RetrieverErrorBudgetFastBurn**

```yaml
alert: RetrieverErrorBudgetFastBurn
expr: (
  sum(rate(retrieval_errors_total[1h]))
  / sum(rate(retrieval_requests_total[1h]))
) / (1 - __SLO_SUCCESS_TARGET__) > __SLO_FAST_BURN_MULTIPLIER__
for: 10m
labels: { severity: "critical", plane: "slo", service: "retriever" }
annotations:
  summary: "Retriever error budget fast burn (1h)"
  runbook: "<RUNBOOK_RETRIEVER_URL>#error-budget-fast-burn"
```

2. **RetrieverErrorBudgetSlowBurn**

```yaml
alert: RetrieverErrorBudgetSlowBurn
expr: (
  sum(rate(retrieval_errors_total[6h]))
  / sum(rate(retrieval_requests_total[6h]))
) / (1 - __SLO_SUCCESS_TARGET__) > __SLO_SLOW_BURN_MULTIPLIER__
for: 30m
labels: { severity: "warning", plane: "slo", service: "retriever" }
annotations:
  summary: "Retriever error budget slow burn (6h)"
  runbook: "<RUNBOOK_RETRIEVER_URL>#error-budget-slow-burn"
```

3. **RetrieverHighP95Latency**

```yaml
alert: RetrieverHighP95Latency
expr: histogram_quantile(__SLO_LATENCY_QUANTILE__, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)) > __RETRIEVER_LATENCY_THRESHOLD_SECONDS__
for: 5m
labels: { severity: "warning", plane: "slo", service: "retriever" }
annotations:
  summary: "Retriever p95 latency above threshold"
  runbook: "<RUNBOOK_RETRIEVER_URL>#latency"
```

4. **QdrantErrorBudgetFastBurn**

```yaml
alert: QdrantErrorBudgetFastBurn
expr: (
  sum(rate(rest_responses_fail_total[1h]))
  / sum(rate(rest_responses_total[1h]))
) / (1 - __SLO_SUCCESS_TARGET__) > __SLO_FAST_BURN_MULTIPLIER__
for: 10m
labels: { severity: "critical", plane: "slo", service: "qdrant" }
annotations:
  summary: "Qdrant error budget fast burn (1h)"
  runbook: "<RUNBOOK_QDRANT_URL>#error-budget-fast-burn"
```

5. **QdrantHighP95Latency**

```yaml
alert: QdrantHighP95Latency
expr: histogram_quantile(__SLO_LATENCY_QUANTILE__, sum(rate(rest_responses_duration_seconds_bucket[5m])) by (le)) > __QDRANT_LATENCY_THRESHOLD_SECONDS__
for: 5m
labels: { severity: "warning", plane: "slo", service: "qdrant" }
annotations:
  summary: "Qdrant p95 latency above threshold"
  runbook: "<RUNBOOK_QDRANT_URL>#latency"
```

> Implementation note: use `yaml.safe_dump(..., sort_keys=False)` to preserve human order when rendering rules.

---

## Paging policy and routing (Alertmanager)

### PagerDuty vs Slack

* **Page (PagerDuty)**: any alert with `severity: critical` and `plane` in `{ingestion, safety, slo}`.
* **Notify (Slack)**: alerts with `severity: warning` or `info` (SLO warnings, long-running noncritical conditions).

### Example Alertmanager route (deterministic)

```yaml
route:
  group_by: ["alertname", "service", "plane"]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 3h
  receiver: default
  routes:
    - match:
        plane: ingestion
        severity: critical
      receiver: pagerduty
    - match:
        plane: safety
        severity: critical
      receiver: pagerduty
    - match:
        plane: slo
        severity: critical
      receiver: pagerduty
    - match:
        plane: slo
        severity: warning
      receiver: slack
receivers:
  - name: pagerduty
    pagerduty_configs:
      - service_key: ${PAGERDUTY_SERVICE_KEY}
  - name: slack
    webhook_configs:
      - url: ${ALERTMANAGER_SLACK_WEBHOOK}
  - name: default
    webhook_configs:
      - url: ${DEFAULT_WEBHOOK}
```

### Inhibition rules (must be present)

```yaml
inhibit_rules:
  - source_match: { plane: "ingestion", severity: "critical" }
    target_match: { plane: "slo" }
    equal: ["service"]
  - source_match: { plane: "safety", severity: "critical" }
    target_match: { plane: "slo" }
    equal: ["service"]
  - source_match: { plane: "slo", severity: "critical" }
    target_match: { plane: "slo", severity: "warning" }
    equal: ["service"]
```

Rationale: ingestion/safety critical alerts suppress downstream SLO warnings to avoid alert storms and misdirection.

---

## vmalert deployment and config (runtime control flow)

* vmalert reads rules from a mounted `ConfigMap` (key `slo.rules.yaml`) and evaluates against `DATASOURCE_URL` (VictoriaMetrics).
* vmalert sends alerts to `NOTIFIER_URL` (Alertmanager).
* Key deployment args:

  ```
  -rule=/etc/vmalert/slo.rules.yaml
  -datasource.url=${DATASOURCE_URL}
  -notifier.url=${NOTIFIER_URL}
  -evaluationInterval=${VMALET_EVAL_INTERVAL}
  ```
* Use a single replica in dev. For HA, run `replica=2` only if Alertmanager and downstream deduplication handle duplicates and you accept potential duplicated notifications.

---

## Environment variables (control knobs)

The generator consumes these env vars (deterministic names):

Policy & SLO:

* `SLO_SUCCESS_TARGET` — e.g., `0.999`
* `SLO_LATENCY_QUANTILE` — `0.95` or `0.99`
* `SLO_FAST_BURN_MULTIPLIER` — numeric (e.g., `2`)
* `SLO_SLOW_BURN_MULTIPLIER` — numeric (e.g., `1.2`)
* `RETRIEVER_LATENCY_THRESHOLD_SECONDS`
* `QDRANT_LATENCY_THRESHOLD_SECONDS`

Runtime wiring:

* `DATASOURCE_URL` — `http://victoria-metrics.monitoring.svc:8428`
* `NOTIFIER_URL` — `http://alertmanager.monitoring.svc:9093`
* `VMALERT_IMAGE` — pinned image, e.g. `victoriametrics/vmalert:v1.132.0`
* `VMALERT_REPLICAS`, `VMALERT_EVAL_INTERVAL`

Notification secrets (store in K8s Secret; generator references them, does not embed):

* `PAGERDUTY_SERVICE_KEY` (Secret)
* `ALERTMANAGER_SLACK_WEBHOOK` (Secret)
* `DEFAULT_WEBHOOK` (Secret or config)

---

## CI validation and rendering workflow (deterministic)

1. Developer updates template or env vars in source.
2. CI runs generator: `python infra/generators/alerting.py --generate` → writes `infra/manifests/alerts/`:

   * `slo.rules.yaml` (plain rules)
   * `vmalert-deployment.yaml` (ConfigMap + Deployment + Service)
   * `alertmanager-config.yaml`
3. CI performs checks:

   * YAML parse
   * `promtool check rules slo.rules.yaml` (if `promtool` present) or `vmalert` dry run
   * Lint that each rule has required `labels` and `annotations` (`plane`, `severity`, `runbook`)
   * Ensure `runbook` URLs are resolvable (HTTP HEAD) for staging or validate known base URL mapping
4. CI attaches rendered artifacts and diffs to PR.
5. On merge, promotion job runs `--apply` to staging. Staging `--validate` includes smoke PromQL checks to confirm `seriesFetched > 0` for one sample metric before enabling paging receivers.
6. Production promotion follows after staging smoke passes.

---

## Operational runbook and ownership requirements

* Every alert must reference a runbook URL (stable site served outside cluster). Runbook must include: intent, impact, verification queries, immediate actions, escalation steps.
* Ownership: the owning service team (retriever, qdrant) owns SLO/runbook maintenance for their alerts.
* Alert changes require PR that updates runbook or references existing runbook anchor; CI enforces this.

---

## Troubleshooting, failure modes and mitigations

* **No series in TSDB**: first check ingestion alerts. If ingestion critical fired, stop paging on SLOs and fix vmagent/remote-write.
* **Duplicate alerts**: check `VMALET_REPLICAS` and dedupe logic in downstream notification pipeline.
* **Missing runbook links**: CI rejects PRs that add alerts without runbook annotation.
* **Excessive paging**: adjust `SLO_FAST_BURN_MULTIPLIER` or review `for:` windows; consider adding recording rules to stabilize rates.
* **PromQL slowness**: use recording rules for expensive expressions used by multiple alerts; scale VictoriaMetrics resources if query latency high (`VM_RES_MEM`).

---

## Files and locations (canonical)

* Source templates / generator: `infra/generators/alerting.py`, `infra/generators/templates/`
* Rendered manifests: `infra/manifests/alerts/`

  * `slo.rules.yaml`
  * `vmalert-deployment.yaml`
  * `alertmanager-config.yaml`
* Runbooks source: `docs/runbooks/*.md`
* Runbooks deployed static site: `RUNBOOK_BASE_URL/*`

---

## Change-management / release rules

* Any change to `SLO_*` env vars requires a PR with justification and CI pass.
* Any change to `slo.rules.yaml` must include:

  * updated runbook or proof existing runbook still applies
  * unit test or PromQL check proving the rule compiles
* Enable `--validate --smoke` in staging promotion job; require green before production apply.

---

## Minimal acceptance tests for an alert rule PR

* `slo.rules.yaml` rendered with no leftover placeholders.
* `promtool check rules` exits zero (or `vmalert --dryRun` passes).
* Each rule contains required `labels` and `annotations`.
* Runbook URL HEAD returns 200 (staging) or matches allowed `RUNBOOK_BASE_URL` pattern.

---

## Enforcement & governance

* Generator is single source of truth; do not hand-edit rendered manifests.
* Secrets must be kept in K8s Secret or secret manager; generator references secrets but does not embed secrets in repo.
* Keep SLO env vars stable; do not treat them as experiment flags — changes must follow the SLO governance flow.

---

## Summary (operational invariant)

1. Do not trust SLO or service alerts until ingestion plane is green.
2. Page only on `severity: critical` and when the alert maps to immediate human action.
3. Ensure every alert has an executable runbook reachable at a stable external URL.
4. Render, validate, and deploy alerts via the generator and CI pipeline; never hand-edit production rules.
