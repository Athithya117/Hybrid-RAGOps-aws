# CONFIGURATION A — minimal baseline (defaults exercised, CREATE_NAT=false, generated storage account)
export RESOURCE_NAME_PREFIX=rag
export AZURE_LOCATION=eastus
export VNET_CIDR=10.1.0.0/16
export AKS_SUBNET_PREFIX=10.1.1.0/24
export APPGW_SUBNET_PREFIX=10.1.2.0/24
export CREATE_NAT=false

# CONFIGURATION B — alternate control paths (CREATE_NAT=true, provided storage account, AZ_REGION fallback)
export RESOURCE_NAME_PREFIX=rag
export AZ_REGION=centralindia
export AZURE_STORAGE_ACCOUNT_NAME=ragstoracct123
export AZURE_CONTAINER=rag-data-custom
export VNET_CIDR=10.2.0.0/16
export AKS_SUBNET_PREFIX=10.2.1.0/24
export APPGW_SUBNET_PREFIX=10.2.2.0/24
export CREATE_NAT=true
