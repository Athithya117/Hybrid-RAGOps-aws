from __future__ import annotations
from typing import Optional
import pulumi
from pulumi import Output
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

def _mk_cluster(args):
    """
    Internal helper to construct the ManagedCluster resource object.
    args is a dict with keys matching ManagedCluster constructor inputs.
    Returns the containerservice.ManagedCluster resource instance.
    """
    return containerservice.ManagedCluster(
        args["name"],
        resource_group_name=args["resource_group_name"],
        location=args["location"],
        dns_prefix=args["dns_prefix"],
        enable_rbac=args["enable_rbac"],
        network_profile=args["network_profile"],
        agent_pool_profiles=args["agent_pool_profiles"],
        identity=args["identity"],
        identity_profile=args.get("identity_profile"),
        sku=args.get("sku"),
        tags=args.get("tags"),
    )

def create_aks_cluster(rg,
                       location: str,
                       stack: str,
                       aks_cluster_name: str,
                       aks_sku: str,
                       system_node_vm: str,
                       system_node_count: int,
                       max_pods: int,
                       balanced_vm: str,
                       balanced_min: int,
                       balanced_max: int,
                       cpu_vm: str,
                       cpu_min: int,
                       cpu_max: int,
                       qdrant_vm: str,
                       qdrant_count: int,
                       aks_network_plugin: str,
                       outbound_type: str,
                       snet_aks_id: str | None,
                       kubelet_identity = None):
    """
    Create the AKS ManagedCluster.

    If `kubelet_identity` (a managedidentity.UserAssignedIdentity resource) is provided,
    we resolve its resource id before constructing the cluster so that `userAssignedIdentities`
    map uses a *literal* ARM resource id string as key. This avoids passing an Output as a dict key
    which causes Pulumi/serializer errors.

    Returns either:
      - a containerservice.ManagedCluster resource (if kubelet_identity not provided), or
      - an Output[containerservice.ManagedCluster] if cluster creation is created inside an apply.
    """

    system_pool = agent_profile_args("systempool", system_node_vm, system_node_count, max_pods, mode="System", vnet_subnet_id=snet_aks_id)
    apppool_count = balanced_min if balanced_min > 0 else 0
    app_pool = agent_profile_args("apppool", balanced_vm, apppool_count, max_pods, mode="User", enable_autoscaling=True, min_count=balanced_min, max_count=balanced_max, vnet_subnet_id=None if aks_network_plugin != "azure" else snet_aks_id)
    qdrant_pool = agent_profile_args("qdrant", qdrant_vm, qdrant_count, max_pods, mode="User", vnet_subnet_id=None if aks_network_plugin != "azure" else snet_aks_id)
    cpu_pool = agent_profile_args("cpuheavy", cpu_vm, cpu_min if cpu_min>0 else 0, max_pods, mode="User", enable_autoscaling=True, min_count=cpu_min, max_count=cpu_max, vnet_subnet_id=None if aks_network_plugin != "azure" else snet_aks_id)

    agent_pools = [system_pool, app_pool, qdrant_pool, cpu_pool]

    network_profile = containerservice.ContainerServiceNetworkProfileArgs(
        network_plugin=aks_network_plugin,
        network_policy="calico",
        service_cidr="10.0.0.0/16",
        dns_service_ip="10.0.0.10",
        pod_cidr="10.244.0.0/16",
        outbound_type=outbound_type,
    )

    base_args = {
        "name": f"{aks_cluster_name}-pulumi",
        "resource_group_name": rg.name,
        "location": location,
        "dns_prefix": aks_cluster_name,
        "enable_rbac": True,
        "network_profile": network_profile,
        "agent_pool_profiles": agent_pools,
        "tags": {"env": stack, "managedBy": "pulumi", "project": "rag", "stack": stack},
    }

    if aks_sku and aks_sku.lower() == "standard":
        base_args["sku"] = containerservice.ManagedClusterSKUArgs(name="Base", tier="Standard")

    # If kubelet_identity is provided, we MUST map userAssignedIdentities with a literal resource-id key.
    # To avoid Output-as-dict-key serialization problems, create the ManagedCluster inside an apply that receives the resolved id string.
    if kubelet_identity is not None:
        # Expect kubelet_identity to be a resource with .id and .principal_id
        def _make_cluster(kid_id):
            # kid_id is a plain string here (ARM resource id)
            mc_args = dict(base_args)
            # identity must include the userAssigned identity in userAssignedIdentities map
            mc_args["identity"] = containerservice.ManagedClusterIdentityArgs(
                type="SystemAssigned,UserAssigned",
                user_assigned_identities={ kid_id: {} },
            )
            # set identity_profile.kubeletidentity.resourceId to the same literal id
            mc_args["identity_profile"] = {"kubeletidentity": {"resourceId": kid_id}}
            return _mk_cluster(mc_args)

        # Resolve the id then create the cluster resource inside the apply.
        return kubelet_identity.id.apply(lambda kid: _make_cluster(kid))

    # Default path: system-assigned only
    base_args["identity"] = containerservice.ManagedClusterIdentityArgs(type="SystemAssigned")
    return _mk_cluster(base_args)
