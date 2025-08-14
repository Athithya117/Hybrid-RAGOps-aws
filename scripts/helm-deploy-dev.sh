

mkdir -p infra/charts/rag8s-aws/dev

helm template rag8s-aws \
  infra/charts/rag8s-aws \
  -f infra/charts/rag8s-aws/values/base.yaml \
  -f infra/charts/rag8s-aws/values/argocd.yaml \
  -f infra/charts/rag8s-aws/values/karpenter.yaml \
  -f infra/charts/rag8s-aws/values/monitoring.yaml \
  -f infra/charts/rag8s-aws/values/networking.yaml \
  -f infra/charts/rag8s-aws/values/ray.yaml \
  --show-only templates/core/namespaces.yaml \
  --namespace arangodb \
  > infra/charts/rag8s-aws/dev/00_namespaces.yaml

