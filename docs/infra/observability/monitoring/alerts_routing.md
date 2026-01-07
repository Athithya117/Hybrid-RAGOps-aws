# Alert Routing and Inhibition Documentation

## Summary of Routing Decisions
- Generator assigns **severity** and **plane** to each alert.
- **Fast burn** (RetrieverErrorBudgetFastBurn, QdrantErrorBudgetFastBurn) → severity: `critical`, plane: `slo`.
- **Slow burn** (RetrieverErrorBudgetSlowBurn) → severity: `warning`, plane: `slo`.

### Alertmanager Route Construction
- `ALERTING_PAGING_SEVERITY_LEVELS` (default `"critical"`) → PagerDuty if `PAGERDUTY_ROUTING_KEY` is set.
- `ALERTING_SLACK_SEVERITY_LEVELS` (default `"warning,critical"`) → Slack if `ALERTMANAGER_SLACK_WEBHOOK` is set.

### Effective Receiver
1. If severity ∈ paging list and `PAGERDUTY_ROUTING_KEY` exists → PagerDuty.
2. Else if severity ∈ slack list and `ALERTMANAGER_SLACK_WEBHOOK` exists → Slack.
3. Else → base route receiver (fallback order: configured slack → pagerduty → default/noop).

### Defaults
- Both PagerDuty & Slack configured: `critical` → PagerDuty, `warning` → Slack.
- PagerDuty not configured but Slack is: `critical` → Slack.
- Neither configured: notifications → default/noop.

---

## Alerts Assessment (finalized metric names and PromQL)

> **Note:** use the explicit recording rules (below) for SLO alerts where shown. Recording rules improve performance and avoid expression drift.

### Ingestion Plane

1. **VmagentDiscoveryEmpty**  
   - Expr:
     ```
     vm_promscrape_discovery_kubernetes_objects{role="pod"} == 0
     ```
   - For: `2m`  
   - Labels: `ingestion, critical, vmagent`  
   - Fires when vmagent reports zero discovered pod targets for 2 minutes.  
   - Routing: PagerDuty if configured, else Slack, else default.  
   - Inhibition: suppresses SLO alerts for the same service.

2. **VmagentNoRemoteWrite**  
   - Expr (preferred):
     ```
     sum(increase(vmagent_remotewrite_bytes_sent_total[5m])) == 0
     ```
     or alternative:
     ```
     sum(increase(vmagent_remotewrite_rows_pushed_after_relabel_total[5m])) == 0
     ```
   - For: `5m`  
   - Labels: `ingestion, critical, vmagent`  
   - Fires when vmagent is not remote-writing any bytes/rows for 5 minutes (indicates remote-write break).  
   - Routing/Inhibition: same as VmagentDiscoveryEmpty.

---

### Safety Plane

3. **RetrieverNotReady**  
   - Expr:
     ```
     service_ready{service="retrieval"} == 0
     ```
   - For: `2m`  
   - Labels: `safety, critical, retriever`  
   - Routing: PagerDuty if configured, else Slack.  
   - Inhibition: suppresses SLO alerts for the same service.

4. **QdrantDeadReplicas**  
   - Preferred Expr (if metric exists in your Qdrant exporter):
     ```
     collection_dead_replicas > 0
     ```
   - Fallback pattern if your exporter uses different name (discover metric name with `/api/v1/series`):
     ```
     sum(collections_replicas_dead_total) > 0
     ```
   - For: `2m`  
   - Labels: `safety, critical, qdrant`  
   - Routing/Inhibition: same as RetrieverNotReady.

5. **QdrantSnapshotStuck**  
   - If exporter exposes `snapshot_creation_running`:
     ```
     snapshot_creation_running > 0
     ```
   - For: `30m`  
   - Labels: `safety, warning, qdrant`  
   - Routing: Slack if configured, else default.  
   - Inhibition: none (only critical safety alerts inhibit).

---

### SLO Plane (recording rules + alerts)

#### Required recording rules (recommended to add to vmalert / rules files)
Add the following recording rules to your rules file (YAML) and load them into your alert evaluator (vmalert or equivalent):

```

groups:

* name: service_recording_rules
  rules:

  * record: retrieval:request:rate:1h
    expr: sum(rate(retrieval_requests_total[1h]))
  * record: retrieval:errors:rate:1h
    expr: sum(rate(retrieval_errors_total[1h]))
  * record: retrieval:request:rate:6h
    expr: sum(rate(retrieval_requests_total[6h]))
  * record: retrieval:errors:rate:6h
    expr: sum(rate(retrieval_errors_total[6h]))
  * record: qdrant:requests:rate:1h
    expr: sum(rate(rest_responses_total[1h]))
  * record: qdrant:errors:rate:1h
    expr: sum(rate(rest_responses_total{status=~"4..|5.."}[1h]))

```

> Use these `record:` names directly in SLO alerts to ensure short, stable expressions and efficient evaluation.

#### Alerts (use recording rules above)

6. **RetrieverErrorBudgetFastBurn**  
   - Expr:
     ```
     ( retrieval:errors:rate:1h / max(retrieval:request:rate:1h, 1) ) / (1 - SLO_SUCCESS_TARGET)
       > SLO_FAST_BURN_MULTIPLIER
     ```
   - For: `10m`  
   - Labels: `slo, critical, retriever`  
   - Routing: PagerDuty if configured, else Slack.  
   - Inhibition: suppressed if ingestion/safety critical for retriever.

7. **RetrieverErrorBudgetSlowBurn**  
   - Expr:
     ```
     ( retrieval:errors:rate:6h / max(retrieval:request:rate:6h, 1) ) / (1 - SLO_SUCCESS_TARGET)
       > SLO_SLOW_BURN_MULTIPLIER
     ```
   - For: `30m`  
   - Labels: `slo, warning, retriever`  
   - Routing: Slack if configured.  
   - Inhibition: suppressed if critical slo alert firing.

8. **RetrieverHighP95Latency**  
   - Expr:
     ```
     histogram_quantile(SLO_LATENCY_QUANTILE,
       sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le)
     ) > RETRIEVER_LATENCY_THRESHOLD_SECONDS
     ```
   - For: `5m`  
   - Labels: `slo, warning, retriever`  
   - Routing: Slack if configured.  
   - Inhibition: suppressed if critical slo alert firing.

9. **QdrantErrorBudgetFastBurn**  
   - Expr (using Qdrant recording rules):
     ```
     ( qdrant:errors:rate:1h / max(qdrant:requests:rate:1h, 1) ) / (1 - SLO_SUCCESS_TARGET)
       > SLO_FAST_BURN_MULTIPLIER
     ```
   - For: `10m`  
   - Labels: `slo, critical, qdrant`  
   - Routing: PagerDuty if configured.  
   - Inhibition: suppressed if ingestion/safety critical for qdrant.

10. **QdrantHighP95Latency**  
    - Expr:
      ```
      histogram_quantile(SLO_LATENCY_QUANTILE,
        sum(rate(rest_responses_duration_seconds_bucket[5m])) by (le)
      ) > QDRANT_LATENCY_THRESHOLD_SECONDS
      ```
    - For: `5m`  
    - Labels: `slo, warning, qdrant`  
    - Routing: Slack if configured.  
    - Inhibition: suppressed if critical slo alert firing.

---

## Alert semantics and derivations (explicit final PromQL snippets)

- **Retriever — Total Failures (counter)** (use as stat):
```

sum(retrieval_errors_total)

```

- **Retriever — Error Rate** (use in Grafana / SLO or alert if not using recording rules):
```

sum(rate(retrieval_errors_total[5m]))
/
clamp_min(sum(rate(retrieval_requests_total[5m])), 1)

```
*Multiply by 100 if expressing as percent in dashboards.*

- **Qdrant — Total Failures (counter)** (stat):
```

sum(rest_responses_total{status=~"4..|5.."})

```
(Fallback to `sum(rest_responses_fail_total)` only if that series is proven to be incrementing in your build.)

- **Qdrant — Error Rate** (final expression):
```

sum(rate(rest_responses_total{status=~"4..|5.."}[5m]))
/
clamp_min(sum(rate(rest_responses_total[5m])), 1)

```
*Multiply by 100 if expressing as percent in dashboards.*

- **Why use `rest_responses_total{status=~"4..|5.."}`**: some Qdrant builds expose `rest_responses_fail_total` but leave it zero; deriving failures from status-coded `rest_responses_total` is robust across versions.

---

## Inhibition rules (explicit)

- Ingestion critical alerts (`ingestion, critical`) for a given service → inhibit SLO alerts (plane: slo) for the same service.
- Safety critical alerts (`safety, critical`) for