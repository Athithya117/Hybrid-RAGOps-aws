#!/usr/bin/env bash
set -euo pipefail
NAMESPACE=auth
echo "Port-forward oauth2-proxy-google-svc -> http://localhost:18080 (Host header if needed: google.auth.127.0.0.1.nip.io)"; kubectl -n $NAMESPACE port-forward svc/oauth2-proxy-google-svc 18080:80 &
echo "Port-forward oauth2-proxy-azure-svc  -> http://localhost:18081 (Host header if needed: azure.auth.127.0.0.1.nip.io)"; kubectl -n $NAMESPACE port-forward svc/oauth2-proxy-azure-svc 18081:80 &
echo "Port-forward frontend deploy -> http://localhost:8000"; kubectl -n $(kubectl get ns --no-headers -o custom-columns=:metadata.name | grep -E "inference|default|frontend" | head -n1 || echo inference) port-forward deploy/frontend-deployment 8000:8000 &
