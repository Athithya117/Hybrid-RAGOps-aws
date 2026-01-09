# QdrantDeadReplicas — Runbook (critical)

**Goal:** Restore all Qdrant replicas to healthy state without causing data loss.

**Expected RTO:**

* 5–20 minutes if recovery requires only restarting a single replica.
* Immediate escalation if quorum (required replicas count) is lost or more than one replica is unhealthy.
* Never restart more than one replica at a time unless explicitly directed by an owner.

---

## Quick checks

```sh
# REQUIRED
export CH_FROM="${CH_FROM:-20M}"     # start window (e.g. 60M, 6H)
export CH_TO="${CH_TO:-0M}"          # end window (0M = now)
export CH_LIMIT="${CH_LIMIT:-5000}"  # max rows to return
export CH_LEVELS="${CH_LEVELS:-err}" # levels csv (e.g. info,warn,error) or empty for all

# OPTIONAL
export CH_FORMAT="${CH_FORMAT:-PrettyCompact}" # PrettyCompact|TSV|JSONEachRow
export NAMESPACE="${NAMESPACE:-qdrant}"         # k8s namespace to inspect
export POD_LABEL="${POD_LABEL:-app.kubernetes.io/name=qdrant,app.kubernetes.io/instance=qdrant}" # pod selector
export CH_QUERY_SCRIPT="${CH_QUERY_SCRIPT:-infra/setup/clickhouse_query.sh}" # helper path

# cluster state
kubectl -n "$NAMESPACE" get deploy -l "$POD_LABEL" -o wide || true
kubectl -n "$NAMESPACE" get pods -l "$POD_LABEL" -o wide || true
kubectl -n "$NAMESPACE" get pvc -o wide || true
kubectl -n "$NAMESPACE" get events --sort-by='.lastTimestamp' --field-selector type=Warning || true

# compact table: pod | phase | ready
kubectl -n "$NAMESPACE" get pods -l $POD_LABEL \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}' || true

# Identify candidate dead pod by readiness (first not-ready); will be empty if none
DEAD_POD="$(kubectl -n "$NAMESPACE" get pods -l $POD_LABEL \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>/dev/null | awk '$2=="false"{print $1; exit}' || true)"

[ -n "$DEAD_POD" ] && kubectl -n "$NAMESPACE" describe pod "$DEAD_POD" || echo "[info] no single not-ready pod identified"
[ -n "$DEAD_POD" ] && kubectl -n "$NAMESPACE" logs "$DEAD_POD" --tail=300 || true

# Optional ClickHouse enrichment (post-incident; do not block immediate actions)
if [ -x "$CH_QUERY_SCRIPT" ]; then
  bash "$CH_QUERY_SCRIPT" --service=qdrant --from="$CH_FROM" --to="$CH_TO" \
    --limit="$CH_LIMIT" --format="$CH_FORMAT" ${CH_LEVELS:+--levels="$CH_LEVELS"} || true
else
  echo "[info] ClickHouse query skipped (helper not present)"
fi
```

---

## On-Call Quick Decision Matrix (Appendix)

### 1. Confirm (≤2m)

Proceed **only if all are true**:

* `collection_dead_replicas > 0`
* Exactly **one** replica is unhealthy
* All other Qdrant pods are `Ready`
* PVCs are `Bound`

If the alert auto-resolves, stop.

---

### 2. Classify and Act (choose one)

#### A. Single dead replica (safe to act)

* Signals:

  * `dead_replicas == 1`
  * One clearly identified unhealthy pod (see `DEAD_POD`)
  * No quorum instability signals
* Action: restart **only the identified dead pod**

```sh
kubectl -n "$NAMESPACE" delete pod "$DEAD_POD"
```

If `DEAD_POD` is empty or ambiguous → **do not restart**, escalate.

---

#### B. Multiple dead replicas or quorum risk

* Signals:

  * `dead_replicas >= 2`
  * Repeated leader elections or consensus churn
* Action:

  * Escalate immediately
  * Do **not** restart pods
  * Do **not** scale the workload

---

#### C. Storage failure

* Signals:

  * PVC `Pending`, `Lost`, or `Failed`
  * I/O errors or corruption messages
* Action:

  * Escalate to storage owner
  * Do **not** restart
  * Do **not** delete PVCs

---

#### D. Node-level failure

* Signals:

  * Node `NotReady`
  * Disk pressure or eviction events
* Action:

  * Escalate to infrastructure owner
  * Restart **only after** node is healthy

---

### 3. Hard Guardrails

Do not:

* Delete PVCs
* Scale Qdrant to zero
* Restart multiple replicas simultaneously
* Restart a pod unless it is **clearly identified as unhealthy**
* Run manual recovery commands without owner approval

---

### 4. Resolution Criteria

All must be true before closing:

* `collection_dead_replicas == 0`
* No new Qdrant warnings for 10 minutes
* Pods stable with no restarts
* Alert auto-resolves

If not resolved within 20 minutes, escalate.

---

### 5. Log Interpretation Hints

* `replica is dead`, `failed to sync shard` → replica failure
* `I/O error`, `corruption` → storage failure
* Repeated leader changes → quorum instability

---

## Post-incident (optional)

* Use ClickHouse logs and queries for root-cause analysis and trend investigation, not for immediate triage.
* Produce a short incident report: trigger time, actions taken, root cause, and follow-ups.

---
