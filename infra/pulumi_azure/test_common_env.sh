#######################################################################
# CASE 1 — Baseline / Default Platform (Explicit User-Assigned NAT)
# Purpose:
# - Golden path
# - Deterministic egress
# - Explicit NAT attached to AKS subnet
# - Matches your current successful preview
#######################################################################
export VNET_NAME="rag-vnet"
export VNET_CIDR="10.1.0.0/16"
export AKS_SUBNET_PREFIX="10.1.1.0/24"
export APPGW_SUBNET_PREFIX="10.1.2.0/24"

export CREATE_NAT=true
export AKS_OUTBOUND_TYPE=userAssignedNATGateway

export AKS_CLUSTER_NAME="rag-aks"
export AKS_MAX_PODS=110

export SYSTEM_NODE_COUNT=1
export SYSTEM_NODE_VM_SIZE=Standard_D4ds_v4

export APP_NODE_COUNT_MIN=1
export APP_NODE_COUNT_MAX=3
export APP_NODE_VM_SIZE=Standard_D4ds_v4

export QDRANT_NODE_COUNT=1
export QDRANT_NODE_VM_SIZE=Standard_D4dsv5

export FORCE=0


#######################################################################
# CASE 2 — Managed NAT (AKS-Owned NAT, No User-Created NAT)
# Purpose:
# - Validate managedNATGateway path
# - Ensure platform works when AKS owns outbound infra
# - No explicit NAT gateway resources expected
#######################################################################
export VNET_NAME="rag-vnet"
export VNET_CIDR="10.2.0.0/16"
export AKS_SUBNET_PREFIX="10.2.1.0/24"
export APPGW_SUBNET_PREFIX="10.2.2.0/24"

export CREATE_NAT=false
export AKS_OUTBOUND_TYPE=managedNATGateway

export AKS_CLUSTER_NAME="rag-aks-managednat"
export AKS_MAX_PODS=110

export SYSTEM_NODE_COUNT=1
export SYSTEM_NODE_VM_SIZE=Standard_D4ds_v4

export APP_NODE_COUNT_MIN=1
export APP_NODE_COUNT_MAX=2
export APP_NODE_VM_SIZE=Standard_D2s_v5

export QDRANT_NODE_COUNT=1
export QDRANT_NODE_VM_SIZE=Standard_D4dsv5

export FORCE=0


#######################################################################
# CASE 3 — No NAT / LoadBalancer Outbound (Edge / Regression Case)
# Purpose:
# - Validate explicit non-NAT path
# - Catch assumptions that NAT always exists
# - Useful for internal-only or firewall-managed clusters
#######################################################################
export VNET_NAME="rag-vnet"
export VNET_CIDR="10.3.0.0/16"
export AKS_SUBNET_PREFIX="10.3.1.0/24"
export APPGW_SUBNET_PREFIX="10.3.2.0/24"

export CREATE_NAT=false
export AKS_OUTBOUND_TYPE=loadBalancer

export AKS_CLUSTER_NAME="rag-aks-lb"
export AKS_MAX_PODS=80

export SYSTEM_NODE_COUNT=1
export SYSTEM_NODE_VM_SIZE=Standard_D2s_v5

export APP_NODE_COUNT_MIN=1
export APP_NODE_COUNT_MAX=1
export APP_NODE_VM_SIZE=Standard_D2s_v5

export QDRANT_NODE_COUNT=1
export QDRANT_NODE_VM_SIZE=Standard_D4dsv5

export FORCE=0


#######################################################################
# CASE 4 — Scale / IP-Pressure Validation
# Purpose:
# - Validate subnet sizing + maxPods interactions
# - Ensure CIDR math and autoscaling remain safe
#######################################################################
export VNET_NAME="rag-vnet"
export VNET_CIDR="10.4.0.0/16"
export AKS_SUBNET_PREFIX="10.4.1.0/23"   # intentionally larger
export APPGW_SUBNET_PREFIX="10.4.3.0/24"

export CREATE_NAT=true
export AKS_OUTBOUND_TYPE=userAssignedNATGateway

export AKS_CLUSTER_NAME="rag-aks-scale"
export AKS_MAX_PODS=150

export SYSTEM_NODE_COUNT=2
export SYSTEM_NODE_VM_SIZE=Standard_D4ds_v4

export APP_NODE_COUNT_MIN=2
export APP_NODE_COUNT_MAX=6
export APP_NODE_VM_SIZE=Standard_D4ds_v4

export QDRANT_NODE_COUNT=2
export QDRANT_NODE_VM_SIZE=Standard_E8ds_v5

export FORCE=0


#######################################################################
# CASE 5 — Destructive / Safety-Gate Validation
# Purpose:
# - Explicitly exercise FORCE=1 logic
# - Used only when validating destroy/replace behavior
# - NEVER run casually
#######################################################################
export VNET_NAME="rag-vnet"
export VNET_CIDR="10.5.0.0/16"
export AKS_SUBNET_PREFIX="10.5.1.0/24"
export APPGW_SUBNET_PREFIX="10.5.2.0/24"

export CREATE_NAT=true
export AKS_OUTBOUND_TYPE=userAssignedNATGateway

export AKS_CLUSTER_NAME="rag-aks-force"
export AKS_MAX_PODS=110

export SYSTEM_NODE_COUNT=1
export SYSTEM_NODE_VM_SIZE=Standard_D4ds_v4

export APP_NODE_COUNT_MIN=1
export APP_NODE_COUNT_MAX=2
export APP_NODE_VM_SIZE=Standard_D4ds_v4

export QDRANT_NODE_COUNT=1
export QDRANT_NODE_VM_SIZE=Standard_D4dsv5

export FORCE=1
