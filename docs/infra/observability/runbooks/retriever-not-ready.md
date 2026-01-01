## Alert: RetrieverNotReady

## Intent
Restore retriever service readiness so user requests can be served successfully.

## Impact
User-facing retrieval requests fail or time out. Error budget consumption is likely increasing.

## Verification
- PromQL: `service_ready{service="retrieval"} == 0`
- Inspect pods:
  - `kubectl -n inference get pods -l app=retrieval`
- Inspect logs:
  - `kubectl -n inference logs -l app=retrieval --since=10m`

## Immediate actions
- Restart the retriever deployment:
  - `kubectl -n inference rollout restart deploy/retrieval`
- If the issue followed a recent release, roll back to the previous version.
- Temporarily scale replicas if CPU or memory pressure is observed.

## Escalation
Escalate if the service does not become ready after restart or rollback, or if a dependency outage is identified.

## Resolution criteria
- `service_ready{service="retrieval"} == 1`
- A successful end-to-end retrieval request is observed.
