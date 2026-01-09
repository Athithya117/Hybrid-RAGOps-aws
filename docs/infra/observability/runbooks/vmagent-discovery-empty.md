# VmagentDiscoveryEmpty — Runbook (critical)

**Goal:** Restore vmagent Kubernetes discovery so it can list and scrape pod/endpoints targets. Avoid config or RBAC changes without owner approval; prefer safe, reversible actions.

**Expected RTO**

* 2–15 minutes for common cases (RBAC or transient API connectivity).
* Immediate escalation if cluster API is degraded or clusterrole/clusterrolebinding is missing.
* Never make irreversible RBAC or ConfigMap edits without owner approval.

**Note:** ClickHouse is **not** required for vmagent discovery troubleshooting. ClickHouse or log-aggregation backends may be useful for post-incident RCA only.

---

## Quick checks (REQUIRED)

/* This block is the single diagnostic script to run during triage. Keep outputs bounded. */

```sh
# REQUIRED: environment
export NAMESPACE="${NAMESPACE:-monitoring}"
export VMAGENT_SVC="${VMAGENT_SVC:-vmagent}"
export VMAGENT_LABEL="${VMAGENT_LABEL:-app=vmagent}"
export SINCE="${SINCE:-10m}"
export VMAGENT_CONFIGMAP="${VMAGENT_CONFIGMAP:-vmagent-config}"

# 1) basic cluster view
kubectl -n "$NAMESPACE" get pods -l "$VMAGENT_LABEL" -o wide || true
kubectl -n "$NAMESPACE" get svc "$VMAGENT_SVC" -o wide || true
kubectl -n "$NAMESPACE" get configmap "$VMAGENT_CONFIGMAP" -o yaml || true
kubectl get clusterrole vmagent-clusterrole -o yaml 2>/dev/null || true
kubectl get clusterrolebinding vmagent-clusterrolebinding -o yaml 2>/dev/null || true

# 2) vmagent pod readiness & events
kubectl -n "$NAMESPACE" get pods -l "$VMAGENT_LABEL" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}' || true
kubectl -n "$NAMESPACE" get events --sort-by='.lastTimestamp' --field-selector type=Warning || true

# 3) inspect vmagent logs for discovery/authorization errors (bounded)
for p in $(kubectl -n "$NAMESPACE" get pods -l "$VMAGENT_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
  [ -n "$p" ] || continue
  echo "---- $p ----"
  kubectl -n "$NAMESPACE" logs "$p" --since="$SINCE" --tail=200 | egrep -i "discov|list|forbidden|unauthorized|error|kubernetes" || true
done

# 4) quick metrics check (vmagent /metrics) via port-forward (local, timed)
PF_PORT=$(python3 - <<'PY'
import socket,sys
s=socket.socket()
s.bind(('',0))
p=s.getsockname()[1]; s.close()
print(p)
PY)
kubectl -n "$NAMESPACE" port-forward svc/"$VMAGENT_SVC" "$PF_PORT":8429 >/dev/null 2>&1 & PF_PID=$!
sleep 1
curl -sS "http://127.0.0.1:${PF_PORT}/metrics" | egrep -i "vm_promscrape_discovery_kubernetes_objects|vm_promscrape_targets_scraped|vm_promscrape_series_fetched" || true
kill $PF_PID 2>/dev/null || true
```

---

## Interpretation — common root causes

1. **RBAC/Permissions**

   * vmagent cannot `list`/`watch` pods/endpoints/namespaces. Logs contain `forbidden`/`unauthorized`/`cannot list pods`.

2. **Kubernetes API connectivity**

   * vmagent cannot reach the kube-apiserver (network policy, proxy, DNS). Logs show connection/refused/timeouts.

3. **ConfigMap misconfiguration**

   * `vmagent-config` scrape job relabel rules or namespace filters may limit discovery; e.g., relabel keeping only `monitoring` namespace or incorrect regex.

4. **ServiceAccount / ClusterRole / ClusterRoleBinding missing or incorrect**

   * Missing `clusterrole` or `clusterrolebinding` entries prevent discovery.

5. **Pod/node networking or CNI issues**

   * API reachable but pod annotations or endpoints unavailable.

6. **Transient control plane degradation**

   * API server overloaded, etcd issues.

---

## Decision matrix & actions

### 1) Confirm (≤2 minutes)

Proceed only if **vm_promscrape_discovery_kubernetes_objects{role="pod"} == 0** (alert condition) and at least one vmagent pod exists. If vmagent auto-resolves, stop.

### 2) Investigate & act (ordered gates)

#### A — RBAC / permission errors (fast path)

**Signals**

* vmagent logs contain `forbidden` / `permission denied` / `cannot list pods`.
* `kubectl get clusterrole vmagent-clusterrole` or clusterrolebinding absent.

**Action**

1. Do **not** edit ClusterRole/Binding yourself unless authorized.
2. Gather evidence:

   ```sh
   kubectl -n monitoring logs <vmagent-pod> --since=10m | egrep -i "forbidden|unauthorized|cannot list|rbac"
   kubectl get clusterrole vmagent-clusterrole -o yaml
   kubectl get clusterrolebinding vmagent-clusterrolebinding -o yaml
   ```
3. Escalate immediately to platform owner with logs and RBAC manifests. Provide the clusterrole and binding YAML (copy output).
4. If instructed by owner, apply the approved RBAC fix and then restart vmagent pods one at a time:

   ```sh
   kubectl -n monitoring rollout restart deployment/vmagent
   ```

   Verify discovery metric becomes >0.

#### B — Kubernetes API connectivity (network/DNS)

**Signals**

* Logs show connection refused, TLS handshake, or timeouts to `kubernetes.default.svc`.
* Port-forward to vmagent `/metrics` shows no discovery but service account seems present.

**Action**

1. Check kube-apiserver reachability from a vmagent pod (if shell available) or from a debug pod:

   ```sh
   kubectl -n monitoring run --rm -i --restart=Never curltest --image=curlimages/curl -- sh -c \
     "curl -sS --max-time 5 https://kubernetes.default.svc || echo APISERVER_FAIL"
   ```
2. If API unreachable → escalate to infrastructure / control-plane owner.
3. Do not restart vmagent until API connectivity is resolved.

#### C — ConfigMap / scrape config filters

**Signals**

* `vmagent-config` scrape.yml contains relabel rules or namespace filters that drop targets.
* Recent config changes / commits.

**Action**

1. Inspect `vmagent-config` for relabel_rules or `action: keep` filters that may drop pods:

   ```sh
   kubectl -n monitoring get configmap vmagent-config -o yaml
   ```
2. If config appears incorrect, do not edit in-place without owner approval. Escalate to monitoring owner or revert via approved git manifests.
3. If owner approves a safe rollback, apply and then restart vmagent deployment (one rollout).

#### D — Transient / pod-level issue

**Signals**

* One vmagent pod shows OOM or CrashLoopBackOff but others healthy.
* Logs show transient errors followed by normal behavior on other pods.

**Action**

1. Describe the failing pod and inspect logs (bounded):

   ```sh
   kubectl -n monitoring describe pod <pod>
   kubectl -n monitoring logs <pod> --since=10m --tail=200 || true
   ```
2. If isolated, delete only the failing pod to force restart:

   ```sh
   kubectl -n monitoring delete pod <pod>
   ```
3. Verify `vm_promscrape_discovery_kubernetes_objects` becomes >0 and targets are scraped.

#### E — Control plane or cluster-wide degradation

**Signals**

* Multiple core services failing, apiserver errors, etc.
  **Action**
* Escalate immediately to infrastructure / cluster SRE.

---

## Hard guardrails (non-negotiable)

* Do not modify ClusterRole/ClusterRoleBinding, ConfigMap, or other cluster-level RBAC without owner approval.
* Do not delete PVCs, Victoria, or Prometheus components as part of vmagent discovery remediation.
* Do not restart all vmagent replicas at once; restart one pod only if isolated.
* Do not change scrape config live unless an approved rollback is available.

---

## Resolution criteria (ALL required)

Close the incident only when all of the following are true:

* `vm_promscrape_discovery_kubernetes_objects{role="pod"}` > 0.
* `vm_promscrape_targets_scraped` > 0 (vmagent reports at least one target scraped).
* vmagent pods are `Ready` and no repeated CrashLoopBackOff/errors for 10 minutes.
* No new warning events for 10 minutes.
* Alert auto-resolves.

If unresolved after 20 minutes, escalate.

---

## Log interpretation hints

* `forbidden` / `RBAC` / `user ... is not authorized` → RBAC misconfiguration.
* `dial tcp ... connection refused` or `i/o timeout` → network/API reachability.
* `no endpoints` / relabel `keep` filters → scrape config filters are excluding targets.
* Repeated `watch closed` errors → API server instability.

---

## Escalation map

* RBAC/ClusterRole issues → Platform / Security owner (RBAC team)
* Kubernetes API / control plane → Infrastructure / Cluster SRE
* vmagent config errors → Monitoring owner
* Network / CNI → Infrastructure / Network owner

---

## Post-incident

* Run post-incident queries (VictoriaMetrics PromQL) to confirm historical discovery anomalies and trend.
* Produce a short incident report: trigger, root cause, actions, and follow-ups (including automated tests or CI checks for vmagent RBAC/config).
* Consider adding CI validation to ensure `vmagent` clusterrole/rolebinding and `vmagent-config` format are present before deploy.

---
