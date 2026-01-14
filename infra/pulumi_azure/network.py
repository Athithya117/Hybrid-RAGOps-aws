from __future__ import annotations
import pulumi
from pulumi_azure_native import network
def create_network(rg, location: str, stack: str, aks_network_plugin: str = "kubenet", create_nat: bool = True, nat_public_ip_count: int = 1):
    vnet_name = f"rag-vnet-{stack}"
    snet_appgw_name = "snet-appgw"
    snet_aks_name = "snet-aks"
    vnet = network.VirtualNetwork(vnet_name, resource_group_name=rg.name, address_space=network.AddressSpaceArgs(address_prefixes=["10.1.0.0/16"]), location=location, tags={"env": stack, "managedBy": "pulumi", "project": "rag", "stack": stack})
    snet_appgw = network.Subnet(snet_appgw_name, resource_group_name=rg.name, virtual_network_name=vnet.name, address_prefix="10.1.2.0/24", private_endpoint_network_policies="Disabled", private_link_service_network_policies="Enabled")
    nat_gateway = None
    if create_nat:
        pubips = []
        for i in range(max(1, nat_public_ip_count)):
            pip = network.PublicIPAddress(f"nat-pip-{i}-{stack}", resource_group_name=rg.name, location=location, sku=network.PublicIPAddressSkuArgs(name="Standard"), public_ip_allocation_method="Static", tags={"managedBy": "pulumi", "project": "rag"})
            pubips.append(network.SubResourceArgs(id=pip.id))
        nat_gateway = network.NatGateway(f"natgw-{stack}", resource_group_name=rg.name, location=location, public_ip_addresses=pubips, sku=network.NatGatewaySkuArgs(name="Standard"), tags={"managedBy": "pulumi", "project": "rag"})
        snet_aks = network.Subnet(snet_aks_name, resource_group_name=rg.name, virtual_network_name=vnet.name, address_prefix="10.1.1.0/24", delegations=[network.DelegationArgs(name="delegation-aks", service_name="Microsoft.ContainerService/managedClusters")] if aks_network_plugin == "azure" else None, private_endpoint_network_policies="Disabled", private_link_service_network_policies="Enabled", nat_gateway=network.SubResourceArgs(id=nat_gateway.id))
    else:
        snet_aks = network.Subnet(snet_aks_name, resource_group_name=rg.name, virtual_network_name=vnet.name, address_prefix="10.1.1.0/24", delegations=[network.DelegationArgs(name="delegation-aks", service_name="Microsoft.ContainerService/managedClusters")] if aks_network_plugin == "azure" else None, private_endpoint_network_policies="Disabled", private_link_service_network_policies="Enabled")
    return {"vnet": vnet, "snet_appgw": snet_appgw, "snet_aks": snet_aks, "nat_gateway": nat_gateway}
