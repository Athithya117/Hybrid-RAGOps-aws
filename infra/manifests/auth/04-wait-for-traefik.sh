#!/usr/bin/env bash
set -euo pipefail
TIMEOUT=${1:-60}
INTERVAL=2
echo "Waiting for Traefik CRD ingressroutes..."
for i in $(seq 1 $((TIMEOUT / INTERVAL))); do
  if kubectl api-resources | grep -q ingressroutes; then
    echo "ingressroutes CRD present"; break
  fi
  sleep $INTERVAL
done
echo "Checking traefik service in kube-system..."
for i in $(seq 1 $((TIMEOUT / INTERVAL))); do
  if kubectl -n kube-system get svc traefik >/dev/null 2>&1; then
    echo "traefik svc present"; exit 0
  fi
  sleep $INTERVAL
done
echo "Timed out waiting for traefik" >&2
exit 2
