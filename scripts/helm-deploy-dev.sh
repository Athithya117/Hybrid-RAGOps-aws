
sudo apt install dos2unix
dos2unix templates/otel-collector.yaml

kubectl delete -f tmp.yaml
clear
cd ~/RAG8s/infra/karpenter-nodepool-cpu

helm repo update
helm dependency update .
clear

helm lint --debug .
helm install rag8s . -n onnx-serving -f values.yaml --create-namespace --dry-run --debug

helm template rag8s . -n onnx-serving -f values.yaml > tmp.yaml
kubectl create namespace onnx-serving --dry-run=client -o yaml | kubectl apply -f -



kubectl apply --dry-run=client -f tmp.yaml
kubectl apply --dry-run=server -f tmp.yaml
kubectl apply -n onnx-serving -f tmp.yaml
kubectl describe raycluster -n inference $(kubectl get rayclusters -n inference -o jsonpath='{.items[0].metadata.name}') && \
kubectl logs -n ray-system deploy/kuberay-operator --tail=10


CHART_DIR=cd ~/RAG8s/infra/karpenter-nodepool-cpu
RELEASE_NAME="${1:-rag8s}"
VALUES_FILE="${2:-$CHART_DIR/values.yaml}"
NAMESPACE="$(yq e '.namespace' "$VALUES_FILE" 2>/dev/null || echo 'onnx-serving')"
MON_NS="$(yq e '.network.prometheusNamespace' "$VALUES_FILE" 2>/dev/null || echo 'monitoring')"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.crds.yaml
kubectl apply -f https://raw.githubusercontent.com/karpenter/karpenter/main/config/crds/karpenter.k8s.aws_provisioners.yaml || true
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace "$MON_NS" --dry-run=client -o yaml | kubectl apply -f -
helm dependency update "$CHART_DIR"
helm upgrade --install "$RELEASE_NAME" "$CHART_DIR" -f "$VALUES_FILE" --namespace "$NAMESPACE" --create-namespace --wait --timeout 10m
