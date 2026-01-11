"""
AKS cluster creation module.

Exports:
  create_aks_cluster(rg, location, stack, params...)
Returns:
  cluster (ManagedCluster resource)
Notes:
  - Mirrors previous cluster creation in your monolith __main__.py.
  - Agent pool helper keeps same defaults; does not modify behavior.
"""

from __future__ import annotations
import pulumi
from pulumi_azure_native import containerservice

def agent_profile_args(name: str, vm_size: str, count: int, max_pods: int, mode: str = "User",
                       enable_autoscaling: bool = False, min_count: int = None, max_count: int = None,
                       vnet_subnet_id: str | None = None):
    args = containerservice.ManagedClusterAgentPoolProfileArgs(
        name=name,
        count=count,
        vm_size=vm_size,
        os_type="Linux",
        type="VirtualMachineScaleSets",
        mode=mode,
        max_pods=max_pods,
        vnet_subnet_id=vnet_subnet_id,
    )
    if enable_autoscaling:
        args.enable_auto_scaling = True
        args.min_count = min_count if min_count is not None else 0
        args.max_count = max_count if max_count is not None else count
    return args

def create_aks_cluster(rg, location: str, stack: str, aks_cluster_name: str, aks_sku: str,
                       system_node_vm: str, system_node_count: int, max_pods: int,
                       balanced_vm: str, balanced_min: int, balanced_max: int,
                       cpu_vm: str, cpu_min: int, cpu_max: int,
                       qdrant_vm: str, qdrant_count: int,
                       aks_network_plugin: str, outbound_type: str, snet_aks_id: str | None):
    """
    Create the ManagedCluster and return it.
    Keep resource names and tags consistent with previous behavior.
    """
    system_pool = agent_profile_args("systempool", system_node_vm, system_node_count, max_pods, mode="System", vnet_subnet_id=snet_aks_id)
    apppool_count = balanced_min if balanced_min > 0 else 0
    app_pool = agent_profile_args("apppool", balanced_vm, apppool_count, max_pods, mode="User", enable_autoscaling=True, min_count=balanced_min, max_count=balanced_max, vnet_subnet_id=None if aks_network_plugin != "azure" else snet_aks_id)
    qdrant_pool = agent_profile_args("qdrant", qdrant_vm, qdrant_count, max_pods, mode="User", vnet_subnet_id=None if aks_network_plugin != "azure" else snet_aks_id)
    cpu_pool = agent_profile_args("cpuheavy", cpu_vm, cpu_min if cpu_min>0 else 0, max_pods, mode="User", enable_autoscaling=True, min_count=cpu_min, max_count=cpu_max, vnet_subnet_id=None if aks_network_plugin != "azure" else snet_aks_id)

    agent_pools = [system_pool, app_pool, qdrant_pool, cpu_pool]

    # For non-azure plugin, ensure vnet_subnet_id is omitted
    if aks_network_plugin != "azure":
        for p in agent_pools:
            try:
                setattr(p, "vnet_subnet_id", None)
            except Exception:
                pass

    network_profile = containerservice.ContainerServiceNetworkProfileArgs(
        network_plugin=aks_network_plugin,
        network_policy="calico",
        service_cidr="10.0.0.0/16",
        dns_service_ip="10.0.0.10",
        pod_cidr="10.244.0.0/16",
        outbound_type=outbound_type,
    )

    cluster = containerservice.ManagedCluster(
        f"{aks_cluster_name}-pulumi",
        resource_group_name=rg.name,
        location=location,
        dns_prefix=aks_cluster_name,
        enable_rbac=True,
        network_profile=network_profile,
        agent_pool_profiles=agent_pools,
        identity=containerservice.ManagedClusterIdentityArgs(type="SystemAssigned"),
        sku=containerservice.ManagedClusterSKUArgs(name="Base", tier="Standard") if (aks_sku and aks_sku.lower() == "standard") else None,
        tags={"env": stack, "managedBy": "pulumi", "project": "rag", "stack": stack},
    )

    return cluster
