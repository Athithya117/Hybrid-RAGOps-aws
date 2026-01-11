RG=acr
LOC="$(az account list-locations --query "[0].name" -o tsv)" && \
ACR=acr49251 && \
az group create --name "$RG" --location "$LOC" --output none && \
az acr create --name "$ACR" --resource-group "$RG" --location "$LOC" --sku Basic --admin-enabled false

export ACR_NAME=acr49250
export AKS_ACR_ATTACH=true

az graph query -q "Resources | where type =~ 'microsoft.containerservice/managedclusters' | project id" --first 100 -o json | jq -r '.data[].id' | xargs -r az resource delete --ids


bash -c '
set -u
ACR_NAME="${ACR_NAME:-acr49251}"

repos="$(az acr repository list --name "$ACR_NAME" -o tsv 2>/dev/null || true)"

for r in $repos; do
  az acr repository delete \
    --name "$ACR_NAME" \
    --repository "$r" \
    --yes || true
done

az acr delete \
  --name "$ACR_NAME" \
  --yes || true
'
