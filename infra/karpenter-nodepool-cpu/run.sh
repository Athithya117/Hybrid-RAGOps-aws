#!/usr/bin/env bash
set -euo pipefail

# ---------- pinned chart versions (keep in sync with Chart.yaml) ----------
KUBERAY_OPERATOR_CHART_VERSION="1.4.2"
KUBE_PROM_STACK_CHART_VERSION="77.0.0"
OTEL_COLLECTOR_CHART_VERSION="0.131.0"
TRAEFIK_CHART_VERSION="37.0.0"
CERT_MANAGER_CHART_VERSION="v1.18.2"

# Optional: separate CRDs for Prometheus Operator (recommended by project)
# This version aligns with Operator v0.85.0 commonly paired with recent stacks.
PROM_OPERATOR_CRDS_CHART="prometheus-community/prometheus-operator-crds"
PROM_OPERATOR_CRDS_VERSION="23.0.0"

# ---------- optional Karpenter (EKS only) ----------
: "${ENABLE_KARPENTER:=false}"
# OCI chart since the legacy repo is deprecated.
KARPENTER_OCI_CHART="oci://ghcr.io/karpenter/karpenter"
KARPENTER_VERSION="${KARPENTER_VERSION:-v1.4.0}"

# For EKS use: export these before running (only used when ENABLE_KARPENTER=true)
: "${CLUSTER_NAME:=}"
: "${KARPENTER_CONTROLLER_ROLE_ARN:=}"
: "${INTERRUPTION_QUEUE:=}"   # e.g. karpenter-events
: "${AWS_REGION:=}"

# ---------- other knobs ----------
NAMESPACE="onnx-serving"
CM_NAMESPACE="cert-manager"
MON_NS="monitoring"
KARP_NS="karpenter"
HELM_TIMEOUT="15m0s"

retry() {
  local retries="${2:-6}" delay="${3:-8}" i=1
  while true; do
    if eval "$1"; then return 0; fi
    if (( i >= retries )); then
      echo "command failed after $retries attempts: $1" >&2
      return 1
    fi
    echo "retry $i/$retries failed; sleeping $delay"
    sleep "$delay"; i=$((i+1))
  done
}

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing binary: $1" >&2; exit 1; }; }

add_repo() {
  local name="$1" url="$2"
  if helm repo list -o yaml 2>/dev/null | yq -r '.[].url' 2>/dev/null | grep -qxF "$url"; then
    retry "helm repo update"
    return
  fi
  if helm repo list -o yaml 2>/dev/null | yq -r '.[].name' 2>/dev/null | grep -qxF "$name"; then
    local cur
    cur="$(helm repo list -o yaml | yq -r ".[] | select(.name==\"$name\") | .url")"
    if [[ "$cur" != "$url" ]]; then
      retry "helm repo remove $name"
      retry "helm repo add $name $url"
    else
      retry "helm repo update"
    fi
  else
    retry "helm repo add $name $url"
  fi
}

is_kind() { kubectl get nodes -o jsonpath='{.items[0].metadata.name}' 2>/dev/null | grep -qi 'kind'; }

main() {
  cd "${HOME}/RAG8s/infra/karpenter-nodepool-cpu" || { echo "cannot cd to chart dir" >&2; exit 1; }
  need helm; need kubectl; need yq
  # Avoid stale locks from dev edits
  rm -f Chart.lock || true
  [ -f values.schema.json ] && mv values.schema.json values.schema.json.broken || true

  echo "==> adding/pinning Helm repos"
  add_repo jetstack https://charts.jetstack.io
  add_repo kuberay https://ray-project.github.io/kuberay-helm/
  add_repo traefik https://traefik.github.io/charts
  add_repo prometheus-community https://prometheus-community.github.io/helm-charts
  add_repo open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
  retry "helm repo update"

  echo "==> namespaces"
  kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - || true
  kubectl create namespace "$CM_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - || true
  kubectl create namespace "$MON_NS" --dry-run=client -o yaml | kubectl apply -f - || true

  echo "==> cert-manager (pinned ${CERT_MANAGER_CHART_VERSION})"
  retry "helm upgrade --install cert-manager jetstack/cert-manager \
    --namespace ${CM_NAMESPACE} \
    --version ${CERT_MANAGER_CHART_VERSION} \
    --set installCRDs=true \
    --timeout ${HELM_TIMEOUT} \
    --wait"

  echo "==> prometheus-operator CRDs (pinned ${PROM_OPERATOR_CRDS_VERSION})"
  retry "helm upgrade --install prometheus-operator-crds ${PROM_OPERATOR_CRDS_CHART} \
    --namespace ${MON_NS} \
    --version ${PROM_OPERATOR_CRDS_VERSION} \
    --timeout ${HELM_TIMEOUT} \
    --wait"

  echo "==> KubeRay operator (pinned ${KUBERAY_OPERATOR_CHART_VERSION})"
  retry "helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --namespace ${NAMESPACE} \
    --version ${KUBERAY_OPERATOR_CHART_VERSION} \
    --timeout ${HELM_TIMEOUT} \
    --wait"

  echo "==> Traefik (pinned ${TRAEFIK_CHART_VERSION})"
  TRAEFIK_EXTRA=""
  if is_kind; then
    # Avoid LoadBalancer pending on kind
    TRAEFIK_EXTRA="--set service.type=NodePort"
  fi
  retry "helm upgrade --install traefik traefik/traefik \
    --namespace ${NAMESPACE} \
    --version ${TRAEFIK_CHART_VERSION} \
    ${TRAEFIK_EXTRA} \
    --timeout ${HELM_TIMEOUT} \
    --wait || true"  # don't hard fail if waiting on LB

  echo "==> kube-prometheus-stack (pinned ${KUBE_PROM_STACK_CHART_VERSION})"
  retry "helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    --namespace ${MON_NS} \
    --version ${KUBE_PROM_STACK_CHART_VERSION} \
    --timeout ${HELM_TIMEOUT} \
    --wait"

  echo "==> (optional) OpenTelemetry Collector (pinned ${OTEL_COLLECTOR_CHART_VERSION})"
  # If you enable this in your values, the umbrella chart can also manage it. Here we keep it as a separate release.
  if [[ "${ENABLE_OTEL_COLLECTOR:-false}" == "true" ]]; then
    retry "helm upgrade --install otel-collector open-telemetry/opentelemetry-collector \
      --namespace ${MON_NS} \
      --version ${OTEL_COLLECTOR_CHART_VERSION} \
      --set mode=deployment \
      --timeout ${HELM_TIMEOUT} \
      --wait"
  fi

  if [[ "${ENABLE_KARPENTER}" == "true" ]]; then
    echo "==> Karpenter (EKS only) via OCI (pinned ${KARPENTER_VERSION})"
    if [[ -z "${CLUSTER_NAME}" || -z "${KARPENTER_CONTROLLER_ROLE_ARN}" || -z "${INTERRUPTION_QUEUE}" || -z "${AWS_REGION}" ]]; then
      echo "Missing one of: CLUSTER_NAME, KARPENTER_CONTROLLER_ROLE_ARN, INTERRUPTION_QUEUE, AWS_REGION" >&2
      exit 1
    fi
    kubectl create namespace "${KARP_NS}" --dry-run=client -o yaml | kubectl apply -f - || true
    retry "helm upgrade --install karpenter ${KARPENTER_OCI_CHART} \
      --namespace ${KARP_NS} \
      --version ${KARPENTER_VERSION} \
      --set serviceAccount.annotations.\"eks\.amazonaws\.com/role-arn\"=${KARPENTER_CONTROLLER_ROLE_ARN} \
      --set settings.clusterName=${CLUSTER_NAME} \
      --set settings.interruptionQueueName=${INTERRUPTION_QUEUE} \
      --set settings.aws.defaultInstanceProfile=KarpenterNodeInstanceProfile-${CLUSTER_NAME} \
      --timeout ${HELM_TIMEOUT} \
      --wait"
    # Your NodePool/EC2NodeClass manifests should be applied separately once IAM/IAM Roles for Service Accounts are in place.
  fi

  echo "==> lint, template, and install the umbrella chart (with deps disabled to avoid double install)"
  helm lint --debug . -f values.yaml || true

  # Disable subcharts (managed as standalone releases above)
  EXTRA_SET="--set kuberay-operator.enabled=false \
             --set prometheus.enabled=false \
             --set opentelemetry-collector.enabled=${ENABLE_OTEL_COLLECTOR:-false} \
             --set ingress.controller.traefik.enabled=false \
             --set cert-manager.enabled=false"

  retry "helm upgrade --install rag8s . \
    --namespace ${NAMESPACE} \
    --values values.yaml \
    ${EXTRA_SET} \
    --timeout ${HELM_TIMEOUT} \
    --atomic"

  echo "==> waits and quick status"
  kubectl wait --for=condition=Available --timeout=300s -n ${CM_NAMESPACE} deployment/cert-manager || true
  kubectl wait --for=condition=Available --timeout=300s -n ${NAMESPACE} deployment/kuberay-operator || true
  kubectl wait --for=condition=Available --timeout=300s -n ${NAMESPACE} deployment/traefik || true
  kubectl wait --for=condition=Available --timeout=300s -n ${MON_NS} deployment/kube-prometheus-stack-operator || true

  echo "==> installed components"
  helm list -A
  kubectl get pods -A -o wide
  echo "run.sh finished ✓"
}

main "$@"
