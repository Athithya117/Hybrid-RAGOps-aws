from __future__ import annotations
from typing import Optional
import pulumi
from pulumi_azure_native import containerservice
def agent_profile_args(name: str, vm_size: str, count: int, max_pods: int, mode: str = "User", enable_autoscaling: bool = False, min_count: Optional[int] = None, max_count: Optional[int] = None, vnet_subnet_id: Optional[str] = None):
    args = containerservice.ManagedClusterAgentPoolProfileArgs(name=name, count=count, vm_size=vm_size, os_type="Linux", type="VirtualMachineScaleSets", mode=mode, max_pods=max_pods, vnet_subnet_id=vnet_subnet_id)
    if enable_autoscaling:
        args.enable_auto_scaling = True
        args.min_count = min_count if min_count is not None else 0
        args.max_count = max_count if max_count is not None else count
    return args
def create_aks_cluster_from_ids(resource_group_name: str, location: str, stack: str, aks_cluster_name: str, aks_sku: str, system_node_vm: str, system_node_count: int, max_pods: int, balanced_vm: str, balanced_min: int, balanced_max: int, cpu_vm: str, cpu_min: int, cpu_max: int, qdrant_vm: str, qdrant_count: int, aks_network_plugin: str, outbound_type: str, snet_aks_id: str | None, control_uami_id: str, kubelet_uami_id: str, nat_gateway_id: str | None = None):
    vnet_subnet = snet_aks_id if snet_aks_id else None
    system_pool = agent_profile_args("systempool", system_node_vm, system_node_count, max_pods, mode="System", vnet_subnet_id=vnet_subnet)
    apppool_count = balanced_min if balanced_min > 0 else 0
    app_pool = agent_profile_args("apppool", balanced_vm, apppool_count, max_pods, mode="User", enable_autoscaling=True, min_count=balanced_min, max_count=balanced_max, vnet_subnet_id=vnet_subnet)
    qdrant_pool = agent_profile_args("qdrant", qdrant_vm, qdrant_count, max_pods, mode="User", vnet_subnet_id=vnet_subnet)
    cpu_pool = agent_profile_args("cpuheavy", cpu_vm, cpu_min if cpu_min>0 else 0, max_pods, mode="User", enable_autoscaling=True, min_count=cpu_min, max_count=cpu_max, vnet_subnet_id=vnet_subnet)
    agent_pools = [system_pool, app_pool, qdrant_pool, cpu_pool]
    network_profile = containerservice.ContainerServiceNetworkProfileArgs(network_plugin=aks_network_plugin, network_policy="calico", service_cidr="10.0.0.0/16", dns_service_ip="10.0.0.10", pod_cidr="10.244.0.0/16", outbound_type=outbound_type)
    if outbound_type == "managedNATGateway":
        network_profile.nat_gateway_profile = containerservice.ManagedClusterNATGatewayProfileArgs(managed_outbound_ip_profile=containerservice.ManagedClusterManagedOutboundIPProfileArgs(count=1))
    mc_args = {"resource_group_name": resource_group_name, "location": location, "dns_prefix": aks_cluster_name, "enable_rbac": True, "network_profile": network_profile, "agent_pool_profiles": agent_pools, "tags": {"env": stack, "managedBy": "pulumi", "project": "rag", "stack": stack}}
    if aks_sku and aks_sku.lower() == "standard":
        mc_args["sku"] = containerservice.ManagedClusterSKUArgs(name="Base", tier="Standard")
    mc_args["identity"] = containerservice.ManagedClusterIdentityArgs(type="UserAssigned", user_assigned_identities={control_uami_id: {}})
    mc_args["identity_profile"] = {"kubeletidentity": {"resourceId": kubelet_uami_id}}
    cluster = containerservice.ManagedCluster(f"{aks_cluster_name}-pulumi", resource_group_name=mc_args["resource_group_name"], location=mc_args["location"], dns_prefix=mc_args["dns_prefix"], enable_rbac=mc_args["enable_rbac"], network_profile=mc_args["network_profile"], agent_pool_profiles=mc_args["agent_pool_profiles"], identity=mc_args["identity"], identity_profile=mc_args["identity_profile"], sku=mc_args.get("sku"), tags=mc_args.get("tags"))
    return cluster
