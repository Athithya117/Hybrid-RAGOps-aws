#!/usr/bin/env bash
set -euo pipefail
NAMESPACE=auth
echo "Port-forwarding oauth2-proxy-google-svc -> http://localhost:18080 (Host: google.auth.127.0.0.1.nip.io)"; kubectl -n $NAMESPACE port-forward svc/oauth2-proxy-google-svc 18080:80 &
echo "Port-forwarding oauth2-proxy-azure-svc  -> http://localhost:18081 (Host: azure.auth.127.0.0.1.nip.io)"; kubectl -n $NAMESPACE port-forward svc/oauth2-proxy-azure-svc 18081:80 &
