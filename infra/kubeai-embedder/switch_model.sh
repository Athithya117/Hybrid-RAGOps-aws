#!/usr/bin/env bash
set -euo pipefail
NAMESPACE=${NAMESPACE:-kubeai}
if [ "${1:-}" = "" ]; then echo "usage: $0 <gte-micro|gte-small|gte-modernbert>" >&2; exit 2; fi
case "$1" in
  gte-micro) FILE=models/gte-micro-cpu.yaml; NAME=gte-micro-cpu ;;
  gte-small) FILE=models/gte-small-cpu.yaml; NAME=gte-small-cpu ;;
  gte-modernbert) FILE=models/gte-modernbert-gpu.yaml; NAME=gte-modernbert-gpu ;;
  *) echo "unknown model: $1" >&2; exit 2 ;;
esac
kubectl -n "$NAMESPACE" apply -f "$FILE" \
&& for f in models/*.yaml; do if [ "$f" != "$FILE" ]; then kubectl -n "$NAMESPACE" delete -f "$f" --ignore-not-found=true; fi; done \
&& echo "applied $FILE; waiting up to 5m for model to report loaded" \
&& for i in $(seq 1 60); do s=$(kubectl -n "$NAMESPACE" get model "$NAME" -o jsonpath='{.status.cache.loaded}' 2>/dev/null || echo "false"); if [ "$s" = "true" ]; then echo "model $NAME loaded"; exit 0; fi; sleep 5; done; echo "timeout waiting for model to load" >&2; kubectl -n "$NAMESPACE" get pods -l kubeai.k8s.io/model="$NAME" -o wide || true; exit 1
