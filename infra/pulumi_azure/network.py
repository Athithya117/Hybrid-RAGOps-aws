"""
Network primitives for the RAG platform Pulumi program.

Exports:
  create_network(rg, location, stack, aks_network_plugin)
Returns:
  dict with keys: vnet, snet_appgw, snet_aks
"""

from __future__ import annotations
import pulumi
from pulumi_azure_native import network

def create_network(rg, location: str, stack: str, aks_network_plugin: str = "kubenet"):
    vnet_name = f"rag-vnet-{stack}"
    snet_appgw_name = "snet-appgw"
    snet_aks_name = "snet-aks"

    vnet = network.VirtualNetwork(
        vnet_name,
        resource_group_name=rg.name,
        address_space=network.AddressSpaceArgs(address_prefixes=["10.1.0.0/16"]),
        location=location,
        tags={"env": stack, "managedBy": "pulumi", "project": "rag", "stack": stack},
    )

    snet_appgw = network.Subnet(
        snet_appgw_name,
        resource_group_name=rg.name,
        virtual_network_name=vnet.name,
        address_prefix="10.1.2.0/24",
        private_endpoint_network_policies="Disabled",
        private_link_service_network_policies="Enabled",
    )

    if aks_network_plugin == "azure":
        snet_aks = network.Subnet(
            snet_aks_name,
            resource_group_name=rg.name,
            virtual_network_name=vnet.name,
            address_prefix="10.1.1.0/24",
            delegations=[network.DelegationArgs(name="delegation-aks", service_name="Microsoft.ContainerService/managedClusters")],
            private_endpoint_network_policies="Disabled",
            private_link_service_network_policies="Enabled",
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

    return {"vnet": vnet, "snet_appgw": snet_appgw, "snet_aks": snet_aks}
