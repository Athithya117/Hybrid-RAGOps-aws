## Alert: VmagentDiscoveryEmpty

## Intent
Verify and restore the observability ingestion pipeline. While this alert is firing, downstream metrics and SLO alerts are not trustworthy.

## Impact
Metrics may be missing or stale in VictoriaMetrics. Dashboards and SLO-based alerts may be misleading.

## Verification
- Inspect vmagent pods:
  - `kubectl -n monitoring get pods -l app=vmagent`
- Inspect logs:
  - `kubectl -n monitoring logs deploy/vmagent --tail=200`
- Inspect scrape targets:
  - `kubectl -n monitoring port-forward deploy/vmagent 8429`
  - Open `/targets` or `/metrics`
- PromQL:
  - `vm_promscrape_discovery_kubernetes_objects{role="pod"} == 0`

## Immediate actions
- Restart vmagent:
  - `kubectl -n monitoring rollout restart deploy/vmagent`
- Verify VictoriaMetrics is healthy and reachable.
- Check RBAC permissions and network/DNS connectivity between vmagent and VictoriaMetrics.

## Escalation
Escalate if vmagent repeatedly fails after restarts, or if WAL, disk, or persistent queue errors are observed.

## Resolution criteria
- `vm_promscrape_discovery_kubernetes_objects{role="pod"} > 0`
- `increase(vm_persistentqueue_bytes_written_total[5m]) > 0`
- Recent metrics are visible in VictoriaMetrics.
