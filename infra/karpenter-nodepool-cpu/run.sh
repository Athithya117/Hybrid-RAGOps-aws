#!/usr/bin/env bash
# set -euo pipefail
retry() {
  local retries=6 delay=8
  for i in $(seq 1 "$retries"); do
    "$@" && return 0
    echo "[$i/$retries] Command failed; retrying in ${delay}s..."
    sleep "$delay"
  done
  echo "Command failed after $retries attempts: $*" >&2
  return 1
}
add_repo() {
  local name="$1" url="$2"
  local found_by_url found_by_name
  found_by_url=$(helm repo list -o yaml 2>/dev/null | yq -r '.[].url' 2>/dev/null || true | grep -xF "$url" || true)
  found_by_name=$(helm repo list -o yaml 2>/dev/null | yq -r '.[].name' 2>/dev/null || true | grep -xF "$name" || true)
  if [ -n "$found_by_url" ]; then
    echo "repo URL already present for $url"
    retry helm repo update
    return 0
  fi
  if [ -n "$found_by_name" ]; then
    local current_url
    current_url=$(helm repo list -o yaml | yq -r ".[] | select(.name==\"$name\") | .url" || true)
    if [ "$current_url" != "$url" ]; then
      retry helm repo remove "$name"
      retry helm repo add "$name" "$url"
    else
      retry helm repo update
    fi
    return 0
  fi
  retry helm repo add "$name" "$url"
}

ensure_bin() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required binary '$1' not found. Install it and re-run."; exit 1; }
}

backup_resource() {
  local kind_name="$1" file
  file="/tmp/kubebackup-$(echo "$kind_name" | sed 's#[/:]#_#g').yaml"
  kubectl get "$kind_name" -o yaml > "$file" 2>/dev/null || true
}

annotate_and_label() {
  local kind_name="$1" release_name="$2" release_ns="$3"
  backup_resource "$kind_name"
  kubectl annotate "$kind_name" "meta.helm.sh/release-name=${release_name}" --overwrite >/dev/null 2>&1 || true
  kubectl annotate "$kind_name" "meta.helm.sh/release-namespace=${release_ns}" --overwrite >/dev/null 2>&1 || true
  kubectl label "$kind_name" "app.kubernetes.io/managed-by=Helm" --overwrite >/dev/null 2>&1 || true
}

adopt_cert_manager() {
  local release="cert-manager" ns="cert-manager"
  local crds clusterroles crbindings
  crds=$(kubectl get crd -o name 2>/dev/null | grep -E 'certificates\.cert-manager\.io|certificaterequests\.cert-manager\.io|challenges\.acme\.cert-manager\.io|clusterissuers\.cert-manager\.io|issuers\.cert-manager\.io|orders\.acme\.cert-manager\.io' || true)
  clusterroles=$(kubectl get clusterrole -o name 2>/dev/null | grep -E 'cert-manager|cainjector|cert-manager-webhook' || true)
  crbindings=$(kubectl get clusterrolebinding -o name 2>/dev/null | grep -E 'cert-manager|cainjector|cert-manager-webhook' || true)
  if [ -n "$crds" ] || [ -n "$clusterroles" ] || [ -n "$crbindings" ]; then
    echo "Adopting existing cert-manager resources (will backup to /tmp)..."
    for r in $crds; do
      annotate_and_label "$r" "$release" "$ns"
    done
    for r in $clusterroles; do
      annotate_and_label "$r" "$release" "$ns"
    done
    for r in $crbindings; do
      annotate_and_label "$r" "$release" "$ns"
    done
    echo "Adoption annotations/labels applied (CRD/ClusterRole/ClusterRoleBinding)."
  else
    echo "No cert-manager-related cluster-scoped resources found to adopt."
  fi
}

adopt_prometheus_crds_if_any() {
  local release="prometheus-operator-crds" ns="monitoring"
  if kubectl get crd prometheuses.monitoring.coreos.com >/dev/null 2>&1; then
    echo "Prometheus CRD exists; ensure Helm metadata for prometheus-operator-crds (backup to /tmp)..."
    for c in $(kubectl get crd -o name | grep -E 'prometheuses|alertmanagers|prometheusrules|servicemonitors|podmonitors|probes' || true); do
      annotate_and_label "$c" "$release" "$ns"
    done
  fi
}

main() {
  ensure_bin helm
  ensure_bin kubectl
  ensure_bin yq
  ensure_bin dos2unix
  dos2unix templates/otel-collector.yaml 2>/dev/null || true

  cd "${HOME}/RAG8s/infra/karpenter-nodepool-cpu" || { echo "cannot cd to chart dir"; exit 1; }

  rm -f Chart.lock || true
  [ -f values.schema.json ] && mv values.schema.json values.schema.json.broken || true

  add_repo jetstack https://charts.jetstack.io
  add_repo kuberay https://ray-project.github.io/kuberay-helm/
  add_repo traefik https://traefik.github.io/charts
  add_repo prometheus-community https://prometheus-community.github.io/helm-charts
  add_repo open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
  add_repo karpenter https://charts.karpenter.sh/ || true

  retry helm repo update

  set +e
  retry helm dependency update . || true
  retry helm dependency build . || true
  set -e

  helm lint --debug . -f values.yaml || true

  kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
  kubectl create namespace onnx-serving --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
  kubectl create namespace cert-manager --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true

  if ! kubectl get crd prometheuses.monitoring.coreos.com >/dev/null 2>&1; then
    retry helm upgrade --install prometheus-operator-crds prometheus-community/prometheus-operator-crds -n monitoring --wait || true
  else
    echo "Prometheus CRDs already present; skipping install."
    adopt_prometheus_crds_if_any
  fi

  if kubectl get crd certificaterequests.cert-manager.io >/dev/null 2>&1; then
    echo "Detected existing cert-manager CRDs; adopting annotations/labels for Helm release cert-manager/cert-manager"
    adopt_cert_manager
  fi

  retry helm upgrade --install cert-manager jetstack/cert-manager -n cert-manager --set installCRDs=true --wait || true

  retry helm upgrade --install kuberay-operator kuberay/kuberay-operator -n onnx-serving --wait || true
  retry helm upgrade --install traefik traefik/traefik -n onnx-serving --wait || true

  retry helm upgrade --install rag8s . -n onnx-serving -f values.yaml --create-namespace --wait || {
    echo "Primary rag8s install failed; trying again with --force"
    retry helm upgrade --install rag8s . -n onnx-serving -f values.yaml --create-namespace --wait --force
  }

  retry helm template rag8s . -n onnx-serving -f values.yaml \
    --set opentelemetry-collector.image.repository=otel/opentelemetry-collector-contrib \
    --set opentelemetry-collector.mode=deployment > tmp.yaml

  python3 - <<'PY' 2>/dev/null
import yaml,sys
try:
    docs=list(yaml.safe_load_all(open('tmp.yaml')))
    print('parsed_docs=',len(docs))
except Exception as e:
    print('YAML_PARSE_ERROR:',e)
    sys.exit(1)
PY
  kubectl apply --dry-run=client -f tmp.yaml || true
  kubectl apply --dry-run=server -f tmp.yaml || true
  kubectl apply -n onnx-serving -f tmp.yaml || true

  kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager -n cert-manager || true
  kubectl wait --for=condition=Available --timeout=300s deployment/kuberay-operator -n onnx-serving || true
  kubectl wait --for=condition=Available --timeout=300s deployment/traefik -n onnx-serving || true

  kubectl get pods -n cert-manager || true
  kubectl get pods -n onnx-serving || true

  kubectl logs -n onnx-serving deploy/kuberay-operator --tail=100 || true
  echo "run.sh finished"
}
main "$@"


