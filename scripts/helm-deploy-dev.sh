export CHART_DIR=./infra/eks-manifests   
export VALUES=$CHART_DIR/values.yaml
export RELEASE=my-rag8s
export NAMESPACE=rag8s-system


helm template ./infra/eks-manifests -f values.eks.yaml and -f values.kind.yaml

helm lint $CHART_DIR --values $VALUES
kubectl apply --dry-run=client -f
kubectl apply --dry-run=server -f


kubectl describe raycluster -n inference $(kubectl get rayclusters -n inference -o jsonpath='{.items[0].metadata.name}') && \
kubectl logs -n ray-system deploy/kuberay-operator --tail=10

