# RetrieverNotReady — Runbook (critical)

**Goal:** Restore Retriever readiness without causing data corruption, secret exposure, or unnecessary disruption.

**Expected RTO:**

* 2–10 minutes if caused by a single pod issue or transient dependency failure.
* Immediate escalation if *all replicas* are NotReady or core dependencies are unavailable.
* Never restart more than one pod at a time.

---

## What “Not Ready” Means

Retriever is considered **Not Ready** when the readiness probe (`/readyz`) returns non-200. Common causes:

* Missing or invalid ConfigMap or Secret.
* A dependent service is unreachable (Qdrant, Dense, Sparse, Reranker).
* Startup probe / cold start completion pending.
* Resource pressure (OOM, CPU throttling).
* Bad deployment / image or environment variable mismatch.

---

## Quick checks (REQUIRED)

```sh
# REQUIRED
export NAMESPACE="${NAMESPACE:-inference}"
export APP_NAME="${APP_NAME:-retrieval}"
export APP_LABEL="app.kubernetes.io/name=${APP_NAME}"
export SINCE="${SINCE:-10m}"

# show deployment(s), replicasets, pods
kubectl -n "$NAMESPACE" get deploy -l "$APP_LABEL" -o wide || true
kubectl -n "$NAMESPACE" get rs -l "$APP_LABEL" -o wide || true
kubectl -n "$NAMESPACE" get pods -l "$APP_LABEL" -o wide || true

# compact table: pod | phase | ready
kubectl -n "$NAMESPACE" get pods -l "$APP_LABEL" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}' || true

# identify a single not-ready pod (first match); will be empty if none
DEAD_POD="$(kubectl -n "$NAMESPACE" get pods -l "$APP_LABEL" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>/dev/null | awk '$2=="false"{print $1; exit}' || true)"

kubectl -n "$NAMESPACE" get cm,secret -l "$APP_LABEL" || true
kubectl -n "$NAMESPACE" get events --sort-by='.lastTimestamp' --field-selector type=Warning || true

# describe / logs for the candidate pod (if identified)
[ -n "$DEAD_POD" ] && kubectl -n "$NAMESPACE" describe pod "$DEAD_POD" || echo "[info] no single not-ready pod identified"
for p in $(kubectl -n "$NAMESPACE" get pods -l "$APP_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
  [ -n "$p" ] || continue
  echo "---- $p ----"
  kubectl -n "$NAMESPACE" logs "$p" --since="$SINCE" --tail=200 || true
done
```

---

## Optional — ClickHouse diagnostics (Post-incident deep dive)

Use ClickHouse only for post-mortem or when additional log context is required. Do not make CH queries part of the immediate required checks.

```sh
# OPTIONAL
export CH_FROM="${CH_FROM:-20M}"
export CH_TO="${CH_TO:-0M}"
export CH_LIMIT="${CH_LIMIT:-5000}"
export CH_LEVELS="${CH_LEVELS:-err}"
export CH_FORMAT="${CH_FORMAT:-PrettyCompact}"
export CH_QUERY_SCRIPT="${CH_QUERY_SCRIPT:-infra/setup/clickhouse_query.sh}"

if [ -x "$CH_QUERY_SCRIPT" ]; then
  bash "$CH_QUERY_SCRIPT" --service=retriever --from="$CH_FROM" --to="$CH_TO" \
    --limit="$CH_LIMIT" --format="$CH_FORMAT" ${CH_LEVELS:+--levels="$CH_LEVELS"} || true
else
  echo "[info] ClickHouse query skipped (helper not present)"
fi
```

---

## Decision Matrix

### 1) Confirm (≤2 minutes)

Proceed only if **all** of the following are true:

* Deployment exists for Retriever.
* At least one Pod exists and at least one Replica is Running.
* A not-ready condition is present (`/readyz` non-200 or container ready=false).

If the alert auto-resolves, stop.

---

### 2) Classify and Act (choose one)

#### A. Single pod NotReady (safe to act)

**Signals**

* Exactly one pod shows `ready=false` and other pods are Ready (or replicas = 1).
* Logs indicate transient dependency timeout, connection refused, or startup timeout.

**Action**

```sh
# restart only the identified not-ready pod
kubectl -n "$NAMESPACE" delete pod "$DEAD_POD"
```

If `DEAD_POD` is empty or ambiguous, do not restart—escalate.

---

#### B. All pods NotReady (systemic)

**Signals**

* All replicas show `ready=false` or readiness failures across pods.

**Action**

* Verify dependent services (Qdrant, Dense, Sparse, Reranker) and Secrets.
* Do not restart pods. Escalate immediately to platform/owner teams.

---

#### C. Dependency failure

**Signals**

* Logs contain `connection refused`, `timeout`, or mentions of dependency URLs.
* Upstream services reported NotReady.

**Action**

* Confirm upstream service health.
* Do not restart retriever until dependency is healthy.
* Escalate to the owning team of the dependency.

---

#### D. Config / Secret error

**Signals**

* CrashLoopBackOff, KeyError, missing env errors in logs.

**Action**

* Verify presence and correctness of ConfigMap and Secret:

  ```sh
  kubectl -n "$NAMESPACE" get cm "${APP_NAME}-config" -o yaml || true
  kubectl -n "$NAMESPACE" get secret "${APP_NAME}-secret" -o yaml || true
  ```
* If using ExternalSecret / KeyVault, escalate to platform owner.
* Do not patch secrets in-cluster without owner approval.

---

#### E. Resource pressure / bad rollout

**Signals**

* OOMKilled, frequent restarts, CPU throttling, or a recent image change.
  **Action**
* Do not scale replicas arbitrarily.
* Roll back to the previous image via the deployment owner if confirmed.
* Escalate to infra/owner for node issues.

---

## Hard guardrails (non-negotiable)

Do not:

* Edit ConfigMaps or Secrets ad-hoc without approval.
* Restart more than one pod at a time.
* Scale the deployment to zero.
* Apply manifests from an unreviewed working tree.
* Restart a pod unless it is unambiguously identified as the failing instance.

---

## Resolution criteria (ALL required)

Close the incident only when all are true:

* All pods `READY` (containerStatuses[*].ready == true).
* Readiness endpoint returns HTTP 200 on each pod.
* No new warning events for 10 minutes.
* No pod restarts observed during that 10-minute window.
* Alert auto-resolves.

If the incident is not resolved within 15 minutes, escalate.

---

## Log interpretation hints

* `failed to connect to qdrant` → Qdrant is unavailable.
* `timeout contacting dense` → model/encoder service down or overloaded.
* `missing env` / `KeyError` → ConfigMap or Secret issue.
* `OOMKilled` → memory resource pressure.
* Readiness probe failures only → likely dependency or readiness logic (do not restart blindly).

---

## Escalation map

* Retriever application / image issues → Retrieval owner
* Qdrant → Vector DB owner
* Dense / Sparse / Reranker → Model serving owner
* Secrets / KeyVault → Platform / Infra owner
* Node issues (NotReady, eviction) → Infrastructure owner

---

## Post-incident (optional)

* Run ClickHouse diagnostics for trend and root-cause analysis.
* Create a short incident report with:

  * Trigger time and duration
  * Actions taken and by whom
  * Root cause and fix
  * Follow-up actions (playbooks, tests, monitoring)

---
