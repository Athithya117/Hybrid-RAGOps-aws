## Alert: QdrantDeadReplicas

## Intent
Restore Qdrant replica health and redundancy to prevent data unavailability or loss.

## Impact
One or more Qdrant replicas are unavailable. Query reliability and data durability may be degraded.

## Verification
- PromQL: `collection_dead_replicas > 0`
- Inspect pods:
  - `kubectl -n qdrant get pods -l app=qdrant`
- Inspect logs:
  - `kubectl -n qdrant logs statefulset/qdrant --tail=200`
- Check node and PersistentVolume health for the affected replica.

## Immediate actions
- Restart the failed Qdrant pod(s).
- If the underlying node is unhealthy, cordon and drain it so the pod reschedules.
- If a replica cannot recover, restore it from the most recent snapshot.

## Escalation
Escalate immediately if replicas cannot be restored, multiple collections are affected, or data loss is suspected.

## Resolution criteria
- `collection_dead_replicas == 0`
- All Qdrant pods are Ready.
- Queries succeed across all collections.
