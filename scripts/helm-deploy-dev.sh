


helm upgrade --install arangodb-release \
  infra/charts/rag8s-aws \
  --namespace arangodb \
  --create-namespace \
  --values values.yaml \
  --set arangodb.enabled=true \
  --set core.arangodb.enabled=true \
  --set core.arangodb.image.repository=arangodb/arangodb \
  --set core.arangodb.image.tag=3.12.5 \
  --set ray.enabled=false \
  --set rayservices.sglang.enabled=false \
  --set rayservices.onnxEmbedderReranker.enabled=false \
  --set argocd.enabled=false \
  --set monitoring.enabled=false \
  --set karpenter.enabled=false \
  --set network.traefik.enabled=false \
  --set serviceAccounts={}

helm upgrade --install arangodb-release \
  infra/charts/rag8s-aws \
  --namespace arangodb \
  --create-namespace \
  --values infra/charts/rag8s-aws/values.yaml \
  --set arangodb.enabled=true \
  --set core.arangodb.enabled=true \
  --set ray.enabled=false \
  --set rayservices.sglang.enabled=false \
  --set rayservices.onnxEmbedderReranker.enabled=false \
  --set argocd.enabled=false \
  --set monitoring.enabled=false \
  --set karpenter.enabled=false \
  --set network.traefik.enabled=false \
  --set serviceAccounts={}
