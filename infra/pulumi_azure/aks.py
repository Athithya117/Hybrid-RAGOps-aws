# aks.py
"""
AKS provisioning helper for Pulumi (azure-native).
 - Creates an AKS managed cluster with:
   - system agent pool (scale/size configurable)
   - app/user pool (autoscaling)
   - qdrant dedicated pool (stateful)
 - Uses SystemAssigned identity for AKS control plane.
 - Accepts resource_group_name and aks_subnet_id from core_network.
 - Returns dict with cluster, kubeconfig (if requested), and agent pool names.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Any

import pulumi
import pulumi_azure_native as azure_native
from pulumi import Output

# -------------------------
# Helpers (fail-fast)
# -------------------------
def _cfg_attr(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        import config as cfg  # type: ignore
        if hasattr(cfg, "cfg") and hasattr(cfg.cfg, name):
            return getattr(cfg.cfg, name)
    except Exception:
        pass
    return os.environ.get(name.upper(), default)


def parse_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name.upper())
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


def parse_int(name: str, default: int) -> int:
    v = os.environ.get(name.upper())
    return int(v) if v is not None else default


def default_tags() -> Dict[str, str]:
    stack = pulumi.get_stack()
    env = os.environ.get("ENV", "STAGING")
    prefix = os.environ.get("RESOURCE_NAME_PREFIX", "rag")
    return {"managedBy": "pulumi", "project": prefix, "stack": stack, "env": env}


# -------------------------
# Primary create function
# -------------------------
def create_aks_cluster(resource_group_name: str,
                       aks_subnet_id: str,
                       *,
                       cluster_name: Optional[str] = None,
                       location: Optional[str] = None,
                       system_node_count: Optional[int] = None,
                       system_vm_size: Optional[str] = None,
                       app_node_min: Optional[int] = None,
                       app_node_max: Optional[int] = None,
                       app_vm_size: Optional[str] = None,
                       qdrant_node_count: Optional[int] = None,
                       qdrant_vm_size: Optional[str] = None,
                       enable_log_analytics: bool = False,
                       log_analytics_workspace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create AKS ManagedCluster and AgentPools. Returns dict with:
      cluster, kube_admin_config (secret Output, optional), system_pool, app_pool, qdrant_pool
    """

    # Resolve defaults from env or cfg
    cluster_name = cluster_name or _cfg_attr("aks_cluster_name") or os.environ.get("AKS_CLUSTER_NAME") or f"aks-{pulumi.get_stack()}"
    location = location or _cfg_attr("azure_location") or os.environ.get("AZURE_LOCATION", "eastus")

    system_node_count = system_node_count or int(_cfg_attr("system_node_count") or os.environ.get("SYSTEM_NODE_COUNT", "1"))
    system_vm_size = system_vm_size or _cfg_attr("system_vm_size") or os.environ.get("SYSTEM_NODE_VM_SIZE", "Standard_D4ds_v4")

    app_node_min = app_node_min or int(_cfg_attr("app_node_min") or os.environ.get("APP_NODE_COUNT_MIN", "1"))
    app_node_max = app_node_max or int(_cfg_attr("app_node_max") or os.environ.get("APP_NODE_COUNT_MAX", "3"))
    app_vm_size = app_vm_size or _cfg_attr("app_vm_size") or os.environ.get("APP_NODE_VM_SIZE", "Standard_D4ds_v4")

    qdrant_node_count = qdrant_node_count or int(_cfg_attr("qdrant_node_count") or os.environ.get("QDRANT_NODE_COUNT", "1"))
    qdrant_vm_size = qdrant_vm_size or _cfg_attr("qdrant_vm_size") or os.environ.get("QDRANT_NODE_VM_SIZE", "Standard_E8s_v5")

    # Basic cluster resource
    pulumi.log.info(f"[aks] Creating managed cluster: {cluster_name} in {location}")

    cluster = azure_native.containerservice.ManagedCluster(
        cluster_name,
        resource_group_name=resource_group_name,
        resource_name=cluster_name,
        location=location,
        dns_prefix=cluster_name,
        enable_rbac=True,
        identity=azure_native.containerservice.ManagedClusterIdentityArgs(type="SystemAssigned"),
        network_profile=azure_native.containerservice.ContainerServiceNetworkProfileArgs(
            network_plugin="azure",
            network_policy="calico",
            outbound_type="loadBalancer"
        ),
        tags=default_tags()
    )

    # Create system pool (as AgentPool resource)
    system_pool = azure_native.containerservice.AgentPool(
        f"{cluster_name}-systempool",
        resource_group_name=resource_group_name,
        cluster_name=cluster.name,
        agent_pool_name="systempool",
        count=max(1, system_node_count),
        vm_size=system_vm_size,
        os_type="Linux",
        mode="System",
        vnet_subnet_id=aks_subnet_id,
        type="VirtualMachineScaleSets",
        enable_auto_scaling=True,
        min_count=max(1, system_node_count),
        max_count=max(system_node_count, 3),
        max_pods=110,
        tags=default_tags()
    )

    # App/user pool with autoscaling
    app_pool = azure_native.containerservice.AgentPool(
        f"{cluster_name}-apppool",
        resource_group_name=resource_group_name,
        cluster_name=cluster.name,
        agent_pool_name="apppool",
        count=max(1, app_node_min),
        vm_size=app_vm_size,
        os_type="Linux",
        mode="User",
        vnet_subnet_id=aks_subnet_id,
        type="VirtualMachineScaleSets",
        enable_auto_scaling=True,
        min_count=app_node_min,
        max_count=app_node_max,
        tags=default_tags()
    )

    # Qdrant dedicated pool (no autoscale recommended)
    qdrant_pool = azure_native.containerservice.AgentPool(
        f"{cluster_name}-qdrantpool",
        resource_group_name=resource_group_name,
        cluster_name=cluster.name,
        agent_pool_name="qdrant",
        count=max(1, qdrant_node_count),
        vm_size=qdrant_vm_size,
        os_type="Linux",
        mode="User",
        vnet_subnet_id=aks_subnet_id,
        type="VirtualMachineScaleSets",
        enable_auto_scaling=False,
        tags=default_tags()
    )

    # Optionally collect kubeconfig admin credentials (secret). This causes a call to get_cluster_admin_credentials which is allowed in apply phase.
    kube_admin_config = cluster.name.apply(lambda _:
                                           azure_native.containerservice.list_managed_cluster_user_credentials(
                                               resource_group_name=resource_group_name,
                                               resource_name=cluster_name
                                           ).kubeconfigs[0].value
                                           if True else None)

    # Export some values in a lightweight consistent shape (consumer can call cluster.name etc.)
    pulumi.export("aks_cluster_name", cluster.name)
    pulumi.export("aks_kube_admin_config_base64", Output.secret(kube_admin_config))

    return {
        "cluster": cluster,
        "system_pool": system_pool,
        "app_pool": app_pool,
        "qdrant_pool": qdrant_pool,
        "kube_admin_config_base64": kube_admin_config
    }
