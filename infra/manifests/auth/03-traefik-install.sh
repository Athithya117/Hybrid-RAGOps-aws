#!/usr/bin/env bash
set -euo pipefail
helm repo add traefik https://helm.traefik.io/traefik
helm repo update
if [ "kind" = "kind" ]; then
  helm upgrade --install traefik traefik/traefik -f 01-traefik-values-kind.yaml --version 20.3.1 --create-namespace --namespace kube-system
else
  helm upgrade --install traefik traefik/traefik -f 02-traefik-values-aks.yaml --version 20.3.1 --create-namespace --namespace kube-system
fi
