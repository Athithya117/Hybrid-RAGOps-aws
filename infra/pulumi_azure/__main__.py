#!/usr/bin/env python3
# infra/pulumi_azure/__main__.py
# Deterministic, idempotent Pulumi program for AKS platform.
# NO quota checks — user requested removal of quota logic.

import os
import pulumi
import pulumi_azure_native as azure_native
from pulumi_azure_native import resources, network, containerservice

# -----------------------------
# Helpers
# -----------------------------
def env(name: str, default: str = None) -> str:
    v = os.getenv(name)
    return v if v is not None else default

def as_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        raise SystemExit(f"Environment variable {name} must be an integer; got: {v}")

def as_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in ("0", "false", "False", "")

# -----------------------------
# Config / env knobs (minimal)
# -----------------------------
STACK = env("PULUMI_STACK", "staging")
LOCATION = env("AKS_LOCATION", env("AZURE_LOCATION", "eastus"))

AKS_NETWORK_PLUGIN = env("AKS_NETWORK_PLUGIN", "kubenet")  # kubenet default for staging to avoid VNet delegation complexity
AKS_NETWORK_POLICY = env("AKS_NETWORK_POLICY", "calico")
AKS_OUTBOUND_TYPE = env("AKS_OUTBOUND_TYPE", "managedNATGateway")
CREATE_NAT = as_bool("CREATE_NAT", False)

AKS_MAX_PODS = as_int("AKS_MAX_PODS", 60)
AKS_CLUSTER_NAME = env("AKS_CLUSTER_NAME", "rag-aks")
AKS_SKU = env("AKS_SKU", "standard")

SYSTEM_NODE_COUNT = as_int("SYSTEM_NODE_COUNT", 1)
SYSTEM_NODE_VM = env("SYSTEM_NODE_VM_SIZE", "Standard_B2s")
SYSTEM_NODE_MAX_PODS = as_int("SYSTEM_NODE_MAX_PODS", AKS_MAX_PODS)

BALANCED_MIN = as_int("BALANCED_NODE_MIN", 0)
BALANCED_MAX = as_int("BALANCED_NODE_MAX", 1)
BALANCED_VM = env("BALANCED_NODE_VM_SIZE", "Standard_B2s")

CPU_HEAVY_MIN = as_int("CPU_HEAVY_NODE_MIN", 0)
CPU_HEAVY_MAX = as_int("CPU_HEAVY_NODE_MAX", 0)
CPU_HEAVY_VM = env("CPU_HEAVY_NODE_VM_SIZE", "Standard_B2s")

QDRANT_COUNT = as_int("QDRANT_NODE_COUNT", 0)
QDRANT_VM = env("QDRANT_NODE_VM_SIZE", "Standard_B2s")

PULUMI_FORCE_DESTROY = as_bool("PULUMI_FORCE_DESTROY", False)

# Normalize strings
AKS_NETWORK_PLUGIN = AKS_NETWORK_PLUGIN.lower()
AKS_OUTBOUND_TYPE = AKS_OUTBOUND_TYPE

# -----------------------------
# Deterministic resource names
# -----------------------------
rg_name = "rg-e2e-rag"
vnet_name = f"rag-vnet-{STACK}"
snet_aks_name = "snet-aks"
snet_appgw_name = "snet-appgw"
nat_pip_name = f"rag-nat-pip-{STACK}"
nat_name = f"rag-nat-{STACK}"

# -----------------------------
# Idempotent RG reference
# -----------------------------
rg = resources.get_resource_group_output(resource_group_name=rg_name)

# -----------------------------
# Networking: VNet + Subnets
# - If using azure (Azure CNI) we create delegated AKS subnet
# - If using kubenet we create plain subnets (no delegation)
# -----------------------------
vnet = network.VirtualNetwork(
    vnet_name,
    resource_group_name=rg.name,
    address_space=network.AddressSpaceArgs(address_prefixes=["10.1.0.0/16"]),
    location=LOCATION,
    tags={"env": STACK, "managedBy": "pulumi", "project": "rag", "stack": STACK},
)

# AppGW / general subnet
snet_appgw = network.Subnet(
    snet_appgw_name,
    resource_group_name=rg.name,
    virtual_network_name=vnet.name,
    address_prefix="10.1.2.0/24",
    private_endpoint_network_policies="Disabled",
    private_link_service_network_policies="Enabled",
)

# AKS subnet: delegated only when Azure CNI is selected
if AKS_NETWORK_PLUGIN == "azure":
    snet_aks = network.Subnet(
        snet_aks_name,
        resource_group_name=rg.name,
        virtual_network_name=vnet.name,
        address_prefix="10.1.1.0/24",
        delegations=[network.DelegationArgs(name="delegation-aks", service_name="Microsoft.ContainerService/managedClusters")],
        private_endpoint_network_policies="Disabled",
        private_link_service_network_policies="Enabled",
        # nat_gateway attached only if user requests userAssignedNATGateway and CREATE_NAT is true
    )
else:
    snet_aks = network.Subnet(
        snet_aks_name,
        resource_group_name=rg.name,
        virtual_network_name=vnet.name,
        address_prefix="10.1.1.0/24",
        private_endpoint_network_policies="Disabled",
        private_link_service_network_policies="Enabled",
    )

# -----------------------------
# NAT (user-assigned) — create only when requested for prod style
# If AKS_OUTBOUND_TYPE == userAssignedNATGateway and CREATE_NAT true, create pip+natgw and attach to subnet
# For staging default (managedNATGateway) we create nothing here
# -----------------------------
natgw = None
if AKS_OUTBOUND_TYPE == "userAssignedNATGateway" and CREATE_NAT:
    pip = network.PublicIPAddress(
        nat_pip_name,
        resource_group_name=rg.name,
        public_ip_allocation_method="Static",
        sku=network.PublicIPAddressSkuArgs(name="Standard"),
        location=LOCATION,
        tags={"env": STACK, "managedBy": "pulumi", "project": "rag", "stack": STACK},
    )
    natgw = network.NatGateway(
        nat_name,
        resource_group_name=rg.name,
        location=LOCATION,
        sku=network.NatGatewaySkuArgs(name="Standard"),
        public_ip_addresses=[network.SubResourceArgs(id=pip.id)],
        tags={"env": STACK, "managedBy": "pulumi", "project": "rag", "stack": STACK},
    )
    # attach NAT to subnet by re-creating subnet with natGateway if AKS_PLUGIN is azure (delegation required)
    # (Azure requires natGateway property be present on subnet resource; here we perform an idempotent update by creating a new subnet resource with same name)
    snet_aks = network.Subnet(
        snet_aks_name,
        resource_group_name=rg.name,
        virtual_network_name=vnet.name,
        address_prefix="10.1.1.0/24",
        delegations=[network.DelegationArgs(name="delegation-aks", service_name="Microsoft.ContainerService/managedClusters")] if AKS_NETWORK_PLUGIN == "azure" else None,
        private_endpoint_network_policies="Disabled",
        private_link_service_network_policies="Enabled",
        nat_gateway=network.SubResourceArgs(id=natgw.id),
    )

# -----------------------------
# Build agent pool profiles
# -----------------------------
def agent_profile_args(name: str, vm_size: str, count: int, mode: str = "User", enable_autoscaling: bool = False, min_count: int = None, max_count: int = None):
    # Compose args for ManagedCluster.agent_pool_profiles
    args = containerservice.ManagedClusterAgentPoolProfileArgs(
        name=name,
        count=count,
        vm_size=vm_size,
        os_type="Linux",
        type="VirtualMachineScaleSets",
        mode=mode,
        max_pods=AKS_MAX_PODS,
        vnet_subnet_id=snet_aks.id if AKS_NETWORK_PLUGIN == "azure" else None,
    )
    if enable_autoscaling:
        args.enable_auto_scaling = True
        args.min_count = min_count if min_count is not None else 0
        args.max_count = max_count if max_count is not None else count
    return args

# system pool (system)
system_pool = agent_profile_args("systempool", SYSTEM_NODE_VM, SYSTEM_NODE_COUNT, mode="System")

# app pool (user, autoscaled)
apppool_count = BALANCED_MIN if BALANCED_MIN > 0 else 0
app_pool = agent_profile_args("apppool", BALANCED_VM, apppool_count, mode="User", enable_autoscaling=True, min_count=BALANCED_MIN, max_count=BALANCED_MAX)

# qdrant pool
qdrant_pool = agent_profile_args("qdrant", QDRANT_VM, QDRANT_COUNT, mode="User")

# cpuheavy pool
cpu_pool = agent_profile_args("cpuheavy", CPU_HEAVY_VM, CPU_HEAVY_MIN if CPU_HEAVY_MIN>0 else 0, mode="User", enable_autoscaling=True, min_count=CPU_HEAVY_MIN, max_count=CPU_HEAVY_MAX)

agent_pools = [system_pool, app_pool, qdrant_pool, cpu_pool]

# Remove vnet_subnet_id if using kubenet (AKS will handle networking)
for p in agent_pools:
    if AKS_NETWORK_PLUGIN != "azure":
        # ensure vnet_subnet_id is not set (kubenet or other plugin)
        try:
            setattr(p, "vnet_subnet_id", None)
        except Exception:
            pass

# -----------------------------
# Network profile (no docker_bridge_cidr)
# -----------------------------
network_profile = containerservice.ContainerServiceNetworkProfileArgs(
    network_plugin=AKS_NETWORK_PLUGIN,
    network_policy=AKS_NETWORK_POLICY,
    service_cidr="10.0.0.0/16",
    dns_service_ip="10.0.0.10",
    pod_cidr="10.244.0.0/16",
    outbound_type=AKS_OUTBOUND_TYPE,
)

# -----------------------------
# ManagedCluster creation (do NOT pass kubernetesVersion to avoid pinning)
# -----------------------------
cluster = containerservice.ManagedCluster(
    f"{AKS_CLUSTER_NAME}-pulumi",
    resource_group_name=rg.name,
    location=LOCATION,
    dns_prefix=AKS_CLUSTER_NAME,
    enable_rbac=True,
    network_profile=network_profile,
    agent_pool_profiles=agent_pools,
    identity=containerservice.ManagedClusterIdentityArgs(type="SystemAssigned"),
    sku=containerservice.ManagedClusterSKUArgs(name="Base", tier="Standard") if (AKS_SKU and AKS_SKU.lower() == "standard") else None,
    tags={"env": STACK, "managedBy": "pulumi", "project": "rag", "stack": STACK},
)

# -----------------------------
# Export outputs
# -----------------------------
pulumi.export("aks_cluster_name", cluster.name)
pulumi.export("vnet_name", vnet.name)
pulumi.export("blob_container_name", "rag-data-515")
pulumi.export("system_pool_name", "systempool")
pulumi.export("note", f"Region={LOCATION}. If apply fails, check region quota, VM size availability, RG permissions and outbound type (AKS_OUTBOUND_TYPE).")
