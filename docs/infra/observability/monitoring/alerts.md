# Alerts — Architecture and runtime control flow (precise & concrete)

## 1. System overview (single paragraph)

Alerts are an automated control loop that converts telemetry into human decisions. The loop has three distinct functional layers: (A) **metric ingestion** (truth collection and transport), (B) **rule evaluation** (policy → decisions), and (C) **notification and routing** (operator action). CI produces the artifacts (rules, deployment manifests, Alertmanager config) and enforces correctness before any artifact can be promoted to staging/production. At runtime vmalert continuously evaluates the CI-rendered rules against VictoriaMetrics; Alertmanager enforces grouping, routing and inhibition; external endpoints (PagerDuty, Slack) convert routed alerts into human work.

---

## 2. Logical components and responsibilities

1. **Instrumented services (retriever, qdrant, vmagent, etc.)**

   * Responsibility: expose application and subsystem metrics at `/metrics`.
   * Additionally export explicit health/readiness gauges and domain counters/histograms used by SLO logic.
   * Owner teams must ensure stable labels and bucket boundaries.

2. **vmagent**

   * Responsibility: scrape pods, perform SD, optionally buffer (persistent queue), remote-write to VictoriaMetrics.
   * Key runtime concerns: discovery correctness, remote-write throughput, queue fill state.

3. **VictoriaMetrics (TSDB / PromQL)**

   * Responsibility: accept remote-writes, store time series, answer PromQL queries used by vmalert.
   * Key runtime concerns: query latency, ingestion throughput, disk I/O.

4. **vmalert**

   * Responsibility: mount the CI-rendered `slo.rules.yaml`, evaluate rule expressions on a fixed evaluation interval, emit alerts to Alertmanager.
   * Key runtime knobs: `-evaluationInterval`, `-datasource.url`, `-notifier.url`.
   * Operation: single binary reading a ConfigMap and polling VictoriaMetrics.

5. **Alertmanager**

   * Responsibility: group, inhibit, route, and repeat notifications according to deterministic routing rules; deliver to receivers (PagerDuty, Slack, default webhook).
   * Key runtime knobs: `group_wait`, `group_interval`, `repeat_interval`, `receivers`.

6. **CI**

   * Responsibility: authoritative rendering of rules and configs (no hand edits), run validation checks, and gate promotion to staging/prod.
   * Outputs: `infra/manifests/alerts/` containing `slo.rules.yaml`, `vmalert-deployment.yaml`, `alertmanager-config.yaml`, optional notifier secret.

7. **Notification endpoints**

   * Responsibility: convert routed alerts into human actions (PagerDuty pages; Slack messages for warnings/ops channels).

---

## 3. Concrete runtime control flow (step-by-step)

This is the canonical runtime sequence from metric emission to human action.

1. **Metric emission**

   * Instrumented services expose metrics at `/metrics`.
   * vmagent discovers targets (SD) and scrapes metrics at configured intervals.

2. **Remote write**

   * vmagent buffers and remote-writes scrapes to VictoriaMetrics via `vm_persistentqueue*` pipeline.
   * If remote-write fails, vmagent persistent metrics change (used by ingestion alerts).

3. **TSDB storage & query**

   * VictoriaMetrics stores series and serves PromQL queries from vmalert.
   * vmalert queries relevant recording rules and alert expressions at every evaluation interval.

4. **vmalert evaluation**

   * vmalert reads the mounted `slo.rules.yaml` from its ConfigMap every eval interval (e.g., 30s).
   * For each rule:

     * Compute expression result using VictoriaMetrics.
     * If expression becomes true, enter `pending` state.
     * If expression remains true for `for:` window, transition to `firing` and POST alert to Alertmanager.
   * vmalert attaches rule labels and annotations to alert payload.

5. **Alertmanager routing & inhibition**

   * On alert receipt, Alertmanager:

     * Groups related alerts per `group_by` and waits `group_wait` before sending the first notification.
     * Applies configured `routes` to select a receiver (PagerDuty/Slack/default).
     * Applies `inhibit_rules`: e.g., ingestion `critical` → inhibit SLO alerts with the same `service` label;
       safety `critical` → inhibit SLO alerts; slo `critical` → inhibit slo `warning`.
     * Sends notifications to receivers, respecting `group_interval` and `repeat_interval`.

6. **Receiver actions**

   * PagerDuty pages on `critical` (per routing); Slack receives `warning` or `info` notifications.
   * Receivers surface runbook links and context from alert annotations.

7. **Operator/On-call**

   * PagerDuty on-call follows the runbook referenced in the alert annotation (stable external URL).
   * If the on-call confirms ingestion/safety issue, SLO alerts are considered suppressed per inhibition rules; if not, they triage SLOs.

8. **Post-incident**

   * CI gated artifact changes (if needed) follow the governance workflow (PR → validate → smoke in staging → promote to prod).

---

## 4. Deterministic gates and decision points (explicit)

List of hard decision gates that determine whether an alert will page or not:

1. **Ingestion gate (monitoring truth)**

   * If `VmagentDiscoveryEmpty` or `VmagentNoRemoteWrite` is `critical` and firing, **do not trust SLO alerts** for the impacted `service` due to `inhibit_rules`. This prevents paging on downstream symptoms when telemetry is broken.

2. **Safety gate (structural correctness)**

   * If a `safety` `critical` alert (e.g., `RetrieverNotReady`, `QdrantDeadReplicas`) is firing for a service, **inhibit SLO alerts** for that `service`. Structural correctness is prioritized over downstream throttles.

3. **SLO escalation gate**

   * `SLO critical` alerts suppress SLO `warning` alerts via inhibition so paged alerts represent immediate human action.

4. **Traffic floor gate (implemented in rules)**

   * Error-budget alerts require a minimum request rate (`SLO_MIN_REQUEST_RATE`) to avoid false positives at near-zero traffic. If traffic is below floor, burn alerts do not page.

5. **Evaluation→For gating**

   * vmalert’s evaluation cadence + `for:` windows create time stabilization: alerts only fire after sustained conditions.

6. **Staging smoke gate**

   * CI promotion includes a smoke check: ensure essential recording rules produce series (`*_rate_1h > 0`) before enabling PagerDuty receivers in prod.

---

## 5. File & artifact ownership (concrete locations)

* Generator (single source): `infra/generators/alerting.py`
* Rendered artifacts (authoritative output): `infra/manifests/alerts/`

  * `slo.rules.yaml` (plain rules)
  * `vmalert-deployment.yaml` (ConfigMap + Deployment + Service)
  * `alertmanager-config.yaml` (Alertmanager ConfigMap)
  * optional: notifier secret manifest
* Runbooks: `docs/runbooks/*.md` (deployed to `RUNBOOK_BASE_URL/*`)
* CI: must run `python infra/generators/alerting.py --generate/--validate` and attach artifacts to PR.

---

## 6. Key runtime parameters and their operational meaning

(These are the knobs that determine timing, grouping and paging behavior.)

* `VMALERT_EVAL_INTERVAL` — how often vmalert re-evaluates rules (default: 30s). Shorter = faster detection, higher CPU.
* Rule `for:` — time a condition must hold before firing (e.g., 10m fast burn). Stabilizes transient flaps.
* `ALERTING_GROUP_WAIT` — time Alertmanager waits to group related alerts (e.g., 30s) before first notification.
* `ALERTING_GROUP_INTERVAL` — grouping window between repeats (e.g., 5m).
* `ALERTING_REPEAT_INTERVAL` — how often an unresolved alert is re-sent (e.g., 3h).
* `SLO_SUCCESS_TARGET` — policy constant; used in burn-rate math to convert raw error rate to burn rate.
* `SLO_FAST_BURN_MULTIPLIER` / `SLO_SLOW_BURN_MULTIPLIER` — map burn rate to days-to-exhaust and decide page vs warn.
* `SLO_MIN_REQUEST_RATE` — traffic floor to avoid triggers at near-zero traffic.
* `PAGERDUTY_ROUTING_KEY`, `ALERTMANAGER_SLACK_WEBHOOK` — secrets referenced at runtime by Alertmanager.

---

## 7. Failure modes, how they propagate, and deterministic mitigations

1. **Metric ingestion failure**

   * Symptom: `vm_promscrape_discovery_kubernetes_objects==0` or `increase(vm_persistentqueue_bytes_written_total[5m])==0`.
   * Propagation: VictoriaMetrics has stale/no data → vmalert may not detect real SLO violations → false calm.
   * Mitigation: Ingestion `critical` alerts inhibit SLO alerts; ops must fix vmagent/remote-write ASAP.

2. **VictoriaMetrics query slowness or OOM**

   * Symptom: vmalert query latency increases or timeouts.
   * Propagation: vmalert may miss evaluations or return partial results, causing false negatives/positives.
   * Mitigation: Monitor vmalert/victoria metrics; use recording rules to reduce expensive queries.

3. **vmalert binary misconfiguration**

   * Symptom: vmalert fails to start or repeatedly reloads ConfigMap.
   * Propagation: No alert evaluations until fixed.
   * Mitigation: CI promtool / vmalert dry-run; readiness probe on vmalert container; minimal replica in dev.

4. **Alertmanager misrouting**

   * Symptom: Alerts dropped or not routed to PagerDuty/Slack.
   * Propagation: Pages not delivered.
   * Mitigation: Alertmanager metrics + test notifications during staging promotion; ensure secrets are present and valid.

5. **Duplicate pages from HA vmalert**

   * Symptom: Multiple pages for same alert.
   * Propagation: Paging fatigue.
   * Mitigation: Run single vmalert replica in dev; if HA required, accept duplicate notifications or add dedupe in downstream pipeline; ensure Alertmanager dedup/inhibit is tuned.

6. **Runbook or annotation missing**

   * Symptom: Alert lacks remediation steps.
   * Propagation: On-call confusion and delay.
   * Mitigation: CI enforces runbook annotations and HEAD checks (staging).

---

## 8. CI rendering, validation and promotion (exact control flow)

1. **Developer PR**

   * Change templates / envs in generator source.

2. **CI: generation**

   * `python infra/generators/alerting.py --generate` → writes `infra/manifests/alerts/`.

3. **CI: static checks**

   * YAML parse
   * `promtool check rules slo.rules.yaml` (mandatory)
   * Linter: ensure labels `plane` and `severity` present; `service` where required; `runbook` annotation present or PR blocked.
   * Runbook HEAD check for `RUNBOOK_BASE_URL` mapping (staging required).

4. **CI: attach artifacts**

   * Rendered artifacts and diffs attached to PR.

5. **On merge: staging promotion**

   * `--apply --staging` deploys manifests into staging cluster.
   * Staging `--validate --smoke` runs live PromQL checks:

     * Essential recording rules produce series (e.g., `retrieval_requests_rate_1h > 0`) *before paging receivers are enabled*.
     * Optionally send test notifications.

6. **Production promotion**

   * After staging smoke green, promote to prod; ensure PagerDuty routing keys are present in Secrets.

---

## 9. Operational playbooks (concise)

### When an ingestion `critical` alert fires

1. PagerDuty pages on-call ops.
2. On-call: check vmagent logs, API access, network between vmagent and VictoriaMetrics.
3. If confirmed, keep SLO paging inhibited and focus on restoring metric flow.
4. After restore, verify series freshness and re-enable SLO paging.

### When a safety `critical` alert fires (e.g., RetrieverNotReady)

1. PagerDuty pages service owner.
2. Owner: validate pod readiness, check k8s events, check dependency health (datastore, network), examine service logs.
3. If service is down by design (deploy), annotate and silence or escalate per governance.

### When an SLO `critical` page fires (fast burn)

1. PagerDuty page includes:

   * burn rate, implied time to exhaustion (SLO_WINDOW / burn_rate)
   * runbook link
   * top correlated errors and recent deploys (from dashboards)
2. On-call executes runbook, mitigates immediately (rollback, reduce traffic, scale, patch).
3. Postmortem: update SLO vars or runbook if needed.

---

## 10. Observability for the alerting system itself (must-have metrics)

Monitor these internal metrics continuously and add dashboards for them:

* **vmagent**

  * `vm_promscrape_discovery_kubernetes_objects`
  * `vm_persistentqueue_bytes_written_total` and queue depth
  * scrape errors

* **VictoriaMetrics**

  * query latency, query failures, memory pressure, ingestion rate

* **vmalert**

  * evaluation duration per rule
  * rules reloaded, last load success/failure
  * alerts emitted rate
  * vmalert process health, pod restarts

* **Alertmanager**

  * alerts received, notifications sent, silences, inhibited alerts
  * routing decision counts

* **Receivers**

  * PagerDuty/Slack delivery failures (from webhook responses)

If any of these falls outside expected ranges, trigger a monitoring alert that directly pages the platform/oncall owning monitoring infra.

---

## 11. Hard requirements & governance (enforced in CI)

* **Single source generator**: all rules must be generated by `infra/generators/alerting.py`; rendered manifests are not hand-edited.
* **Runbook requirement**: each alert must include a `runbook` annotation referencing the stable runbook site.
* **PromQL validation**: `promtool check rules` must pass in CI.
* **Traffic floors**: `SLO_MIN_REQUEST_RATE` or equivalent gating exported and validated to avoid low-traffic false positives.
* **Change justification**: any change to `SLO_*` envs must include justification in PR and pass CI smoke tests.

---
