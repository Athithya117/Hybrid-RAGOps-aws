export CHART_DIR=./infra/charts/rag8s-aws        
export VALUES=$CHART_DIR/values.yaml
export RELEASE=my-rag8s
export NAMESPACE=rag8s-system


helm lint $CHART_DIR --values $VALUES

cd infra/charts/rag8s-aws
helm dependency update

helm template rag8s-embedder-reranker-dev infra/charts/rag8s-aws -n inference -f infra/charts/rag8s-aws/values.dev.yaml \
  --set rayservices.onnxEmbedderReranker.enabled=true \
  --set arangodb.enabled=false \
  --set monitoring.enabled=false \
  --set 'monitoring.serviceMonitors=[]' \
  --set rayservices.vllm.enabled=false \
  --set valkey.enabled=false \
  --set frontend.enabled=false \
  --set argocd.enabled=false \
  --set arangobackup.enabled=false \
  --set rayjobs.indexing.enabled=false \
  --show-only templates/core/namespaces.yaml \
  --show-only templates/core/serviceaccounts.yaml \
  --show-only templates/core/rbac.yaml \
  --show-only templates/rayservices/embedder-reranker.yaml \
> infra/charts/rag8s-dev/1_embedder-reranker.yaml


kubectl describe raycluster -n inference $(kubectl get rayclusters -n inference -o jsonpath='{.items[0].metadata.name}') && \
kubectl logs -n ray-system deploy/kuberay-operator --tail=10

