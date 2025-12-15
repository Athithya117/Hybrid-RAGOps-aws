#!/usr/bin/env bash
# utils/fix_kind_cluster_dns.sh
#
# Idempotent helper to restore a known-good CoreDNS config for Kind / local clusters,
# restart CoreDNS and wait for it to become Ready.
#
# Usage:
#   utils/fix_kind_cluster_dns.sh [--timeout SECONDS]
#
# Notes:
# - Requires `kubectl` in PATH and cluster access.
# - Will not attempt node-level sysctl changes; it only patches CoreDNS ConfigMap and restarts.


TIMEOUT=60

usage() {
  cat <<EOF
Usage: $0 [--timeout SECONDS] [--help]

Patches kube-system/coredns ConfigMap with a safe Corefile (forward via /etc/resolv.conf),
restarts the coredns deployment and waits up to TIMEOUT seconds for rollout.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)
      shift
      TIMEOUT="${1:-$TIMEOUT}"
      shift
      ;;
    -t)
      shift
      TIMEOUT="${1:-$TIMEOUT}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2
      ;;
  esac
done

command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not in PATH"; exit 3; }

echo "[fix-kind-dns] Using kubectl context: $(kubectl config current-context 2>/dev/null || echo '(unknown)')"
echo "[fix-kind-dns] Applying CoreDNS ConfigMap (safe default)"

# Use a heredoc to generate the ConfigMap. Keep this Corefile conservative:
# - forward . /etc/resolv.conf avoids plugin/forward policy parsing issues
# - minimal plugins: errors, ready, kubernetes, forward, cache, loop, reload, loadbalance
kubectl -n kube-system apply -f - <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
EOF

echo "[fix-kind-dns] Restarting CoreDNS deployment"
kubectl -n kube-system rollout restart deployment coredns

echo "[fix-kind-dns] Waiting up to ${TIMEOUT}s for CoreDNS rollout to complete..."
if ! kubectl -n kube-system rollout status deployment coredns --timeout="${TIMEOUT}s"; then
  echo "[fix-kind-dns] ERROR: CoreDNS rollout did not finish within ${TIMEOUT}s" >&2
  echo "[fix-kind-dns] Dumping coredns pods and recent events for debugging:"
  kubectl -n kube-system get pods -l k8s-app=kube-dns -o wide || true
  kubectl -n kube-system get pods -l k8s-app=kube-dns -o yaml || true
  kubectl -n kube-system get events --sort-by='.metadata.creationTimestamp' | tail -n 50 || true
  exit 4
fi

echo "[fix-kind-dns] Verifying CoreDNS pod readiness"
# Wait for all coredns pods to be Ready
END=$(( $(date +%s) + TIMEOUT ))
while true; do
  NOT_READY=$(kubectl -n kube-system get pods -l k8s-app=kube-dns -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>/dev/null || true)
  if [[ -z "$NOT_READY" ]]; then
    # try alternate label selector if above returned nothing
    NOT_READY=$(kubectl -n kube-system get pods -l k8s-app=coredns -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>/dev/null || true)
  fi

  all_ready=true
  if [[ -n "$NOT_READY" ]]; then
    while IFS= read -r line; do
      name=$(printf "%s" "$line" | cut -d'|' -f1)
      ready=$(printf "%s" "$line" | cut -d'|' -f2)
      if [[ "$ready" != "true" ]]; then
        all_ready=false
        break
      fi
    done <<< "$NOT_READY"
  else
    # If we couldn't query pods, break with failure
    all_ready=false
  fi

  if $all_ready ; then
    echo "[fix-kind-dns] All CoreDNS pods Ready"
    break
  fi

  if [[ $(date +%s) -ge $END ]]; then
    echo "[fix-kind-dns] Timeout waiting for CoreDNS pods to be Ready" >&2
    kubectl -n kube-system get pods -l k8s-app=kube-dns -o wide || true
    kubectl -n kube-system get events --sort-by='.metadata.creationTimestamp' | tail -n 50 || true
    exit 5
  fi
  sleep 2
done

echo "[fix-kind-dns] CoreDNS fixed and healthy"
exit 0
