# core_network.py
"""
Core network & storage provisioning helpers for pulumi_azure.
Creates or reuses:
 - Resource Group
 - Storage Account (or lookup)
 - Blob Container
 - Virtual Network + AKS subnet + optional appgw subnet
 - Optional NAT Gateway creation helper
Deterministic outputs (returned as a dict):
 - resource_group (Resource or get result)
 - resource_group_name (primitive)
 - storage_account (Resource or get result)
 - storage_account_name (primitive)
 - blob_container (Resource)
 - vnet (Resource)
 - vnet_id (primitive)
 - aks_subnet (Resource)
 - aks_subnet_id (primitive)
 - appgw_subnet (Resource)
 - nat (dict) optional
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Any

import pulumi
import pulumi_azure_native as azure_native


# -------------------------
# Local helpers (fail-fast)
# -------------------------
def _cfg_attr(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Try to read from a central config `cfg` if present, else environment var.
    This library deliberately does not import a central config module to keep modules decoupled.
    """
    try:
        # avoid a hard import failure if config module missing
        import config as cfg  # type: ignore
        if hasattr(cfg, "cfg") and hasattr(cfg.cfg, name):
            return getattr(cfg.cfg, name)
    except Exception:
        pass
    return os.environ.get(name.upper(), default)


def required_env(name: str) -> str:
    v = _cfg_attr(name.lower())
    if v is None:
        raise RuntimeError(f"Missing required configuration: {name} (set env var {name})")
    return v


def validate_storage_account_name(name: str) -> str:
    if not name:
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME is required")
    if not re.match(r"^[a-z0-9]{3,24}$", name):
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME must be 3-24 lowercase alphanumeric characters")
    return name


def default_tags() -> Dict[str, str]:
    stack = pulumi.get_stack()
    env = os.environ.get("ENV", "STAGING")
    prefix = os.environ.get("RESOURCE_NAME_PREFIX", "rag")
    return {
        "managedBy": "pulumi",
        "project": prefix,
        "stack": stack,
        "env": env,
    }


# -------------------------
# Core functions
# -------------------------
def ensure_resource_group(resource_group_name: Optional[str] = None,
                          location: Optional[str] = None) -> Any:
    """
    Reuse an existing RG (lookup) if present, otherwise create it.
    Returns either azure_native.resources.GetResourceGroupResult (lookup) or ResourceGroup resource.
    """
    rg_name = resource_group_name or _cfg_attr("resource_group_name") or os.environ.get("AZURE_RESOURCE_GROUP_NAME")
    location = location or _cfg_attr("azure_location") or os.environ.get("AZURE_LOCATION", os.environ.get("AZ_REGION", "eastus"))

    if not rg_name:
        raise RuntimeError("Resource group name not provided. Set AZURE_RESOURCE_GROUP_NAME or pass parameter.")

    try:
        rg = azure_native.resources.get_resource_group(resource_group_name=rg_name)
        pulumi.log.info(f"[core_network] Found existing resource group: {rg_name}")
        return rg
    except Exception:
        pulumi.log.info(f"[core_network] Creating resource group: {rg_name} in {location}")
        rg_res = azure_native.resources.ResourceGroup(
            rg_name,
            resource_group_name=rg_name,
            location=location,
            tags=default_tags()
        )
        return rg_res


def ensure_storage_account(resource_group_name: str,
                           storage_account_name: Optional[str] = None) -> Any:
    """
    Reuse storage account if present, else create one.
    Accepts resource_group_name as either primitive string or lookup result with .name.
    Returns get_storage_account result or StorageAccount Resource.
    """
    sa_name = storage_account_name or _cfg_attr("storage_account_name") or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    validate_storage_account_name(sa_name)

    # extract RG name primitive for lookup calls
    rg_raw = resource_group_name if isinstance(resource_group_name, str) else (getattr(resource_group_name, "name", None) or None)
    if not rg_raw:
        raise RuntimeError("resource_group_name must be a string or a resource with .name attribute")

    try:
        sa = azure_native.storage.get_storage_account(resource_group_name=rg_raw, account_name=sa_name)
        pulumi.log.info(f"[core_network] Found existing storage account: {sa_name}")
        return sa
    except Exception:
        pulumi.log.info(f"[core_network] Creating storage account: {sa_name} in RG {rg_raw}")
        sa_res = azure_native.storage.StorageAccount(
            sa_name,
            resource_group_name=rg_raw,
            account_name=sa_name,
            location=_cfg_attr("azure_location") or os.environ.get("AZURE_LOCATION", "eastus"),
            sku=azure_native.storage.SkuArgs(name=azure_native.storage.SkuName.STANDARD_LRS),
            kind=azure_native.storage.Kind.STORAGE_V2,
            minimum_tls_version="TLS1_2",
            allow_blob_public_access=False,
            tags=default_tags()
        )
        return sa_res


def ensure_blob_container(resource_group_name: str, storage_account_name: str, container_name: Optional[str] = None) -> azure_native.storage.BlobContainer:
    """
    Always creates blob container resource (idempotent) for the storage account.
    """
    cont_name = container_name or _cfg_attr("blob_container") or os.environ.get("AZURE_CONTAINER", f"rag-data-{pulumi.get_stack()}")
    rg_raw = resource_group_name if isinstance(resource_group_name, str) else (getattr(resource_group_name, "name", None) or None)
    sa_raw = storage_account_name if isinstance(storage_account_name, str) else (getattr(storage_account_name, "name", None) or None)

    if not rg_raw or not sa_raw:
        raise RuntimeError("Both resource_group_name and storage_account_name are required for create_blob_container")

    container = azure_native.storage.BlobContainer(
        f"blobcontainer-{cont_name}",
        resource_group_name=rg_raw,
        account_name=sa_raw,
        container_name=cont_name,
        public_access=azure_native.storage.PublicAccess.NONE
    )
    return container


def create_virtual_network(resource_group_name: str,
                           vnet_name: Optional[str] = None,
                           vnet_prefix: Optional[str] = "10.1.0.0/16") -> Dict[str, Any]:
    """
    Create VNet and subnets. Returns dict with vnet, vnet_id, aks_subnet, aks_subnet_id, appgw_subnet.
    """
    rg_raw = resource_group_name if isinstance(resource_group_name, str) else (getattr(resource_group_name, "name", None) or None)
    if not rg_raw:
        raise RuntimeError("resource_group_name required for VNet creation")

    prefix = vnet_prefix or _cfg_attr("vnet_cidr") or "10.1.0.0/16"
    vnet_name = vnet_name or f"vnet-{os.environ.get('RESOURCE_NAME_PREFIX', 'rag')}-{pulumi.get_stack()}"

    vnet = azure_native.network.VirtualNetwork(
        vnet_name,
        resource_group_name=rg_raw,
        virtual_network_name=vnet_name,
        location=_cfg_attr("azure_location") or os.environ.get("AZURE_LOCATION", "eastus"),
        address_space=azure_native.network.AddressSpaceArgs(address_prefixes=[prefix]),
        tags=default_tags()
    )

    aks_subnet = azure_native.network.Subnet(
        f"{vnet_name}-snet-aks",
        resource_group_name=rg_raw,
        virtual_network_name=vnet.name,
        subnet_name="snet-aks",
        address_prefix="10.1.1.0/24",
        delegations=[azure_native.network.DelegationArgs(name="delegation-aks", service_name="Microsoft.ContainerService/managedClusters")]
    )

    appgw_subnet = azure_native.network.Subnet(
        f"{vnet_name}-snet-appgw",
        resource_group_name=rg_raw,
        virtual_network_name=vnet.name,
        subnet_name="snet-appgw",
        address_prefix="10.1.2.0/24"
    )

    return {
        "vnet": vnet,
        "vnet_id": vnet.id,
        "aks_subnet": aks_subnet,
        "aks_subnet_id": aks_subnet.id,
        "appgw_subnet": appgw_subnet
    }


def create_nat_gateway(resource_group_name: str,
                       public_ip_name: Optional[str] = None,
                       nat_name: Optional[str] = None,
                       zone: Optional[str] = None) -> Dict[str, Any]:
    """
    Create Standard Public IP + NAT Gateway. Caller must attach nat to subnet(s).
    """
    rg_raw = resource_group_name if isinstance(resource_group_name, str) else (getattr(resource_group_name, "name", None) or None)
    if not rg_raw:
        raise RuntimeError("resource_group_name required for NAT gateway")

    pip_name = public_ip_name or f"pip-{os.environ.get('RESOURCE_NAME_PREFIX','rag')}-{pulumi.get_stack()}"
    nat_name = nat_name or f"nat-{os.environ.get('RESOURCE_NAME_PREFIX','rag')}-{pulumi.get_stack()}"

    pip = azure_native.network.PublicIPAddress(
        pip_name,
        resource_group_name=rg_raw,
        public_ip_address_name=pip_name,
        location=_cfg_attr("azure_location") or os.environ.get("AZURE_LOCATION", "eastus"),
        sku=azure_native.network.PublicIPAddressSkuArgs(name="Standard"),
        public_ip_allocation_method="Static",
        tags=default_tags()
    )

    nat = azure_native.network.NatGateway(
        nat_name,
        resource_group_name=rg_raw,
        nat_gateway_name=nat_name,
        location=_cfg_attr("azure_location") or os.environ.get("AZURE_LOCATION", "eastus"),
        sku=azure_native.network.NatGatewaySkuArgs(name="Standard"),
        public_ip_addresses=[azure_native.network.SubResourceArgs(id=pip.id)],
        idle_timeout_in_minutes=10,
        tags=default_tags()
    )

    return {"public_ip": pip, "nat_gateway": nat}
