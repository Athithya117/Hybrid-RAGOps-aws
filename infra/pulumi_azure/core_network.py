from __future__ import annotations
import os,re,ipaddress
from typing import Any,Dict,Optional
import pulumi
import pulumi_azure_native as azure_native
from pulumi_azure_native import resources,network,storage
def _env(k:str,d:Optional[str]=None)->Optional[str]:
    return os.environ.get(k) or d
def _bool(k:str,d:bool=False)->bool:
    v=_env(k); 
    if v is None: return d
    return v.lower() in ("1","true","yes")
SUPPORTED_REGIONS={r.lower() for r in ("eastus","eastus2","centralus","westus2","westeurope","northeurope","southeastasia","eastasia","centralindia","southindia","australiaeast","canadacentral","uksouth","francecentral","germanywestcentral","japaneast","koreacentral")}
def _validate_region(region:str)->str:
    if region.lower() not in SUPPORTED_REGIONS: raise RuntimeError(f"AZURE_LOCATION '{region}' not supported; choose one of: {sorted(SUPPORTED_REGIONS)}")
    return region
def _sanitize_prefix(p:str)->str:
    s=re.sub(r'[^a-z0-9]','',p.lower())
    if not s: raise RuntimeError("RESOURCE_NAME_PREFIX must contain at least one ascii letter or digit after sanitization")
    return s
def _validate_rg(name:str)->str:
    if not (1<=len(name)<=90): raise RuntimeError("AZURE_RESOURCE_GROUP_NAME length must be 1..90")
    if not re.fullmatch(r'[A-Za-z0-9._\-\(\)]+',name): raise RuntimeError("AZURE_RESOURCE_GROUP_NAME contains invalid characters")
    return name
def _validate_storage_account(name:str)->str:
    if not re.fullmatch(r'[a-z0-9]{3,24}',name): raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME must match ^[a-z0-9]{3,24}$")
    return name
def _validate_container(name:str)->str:
    if not (3<=len(name)<=63): raise RuntimeError("AZURE_CONTAINER must be 3..63 chars")
    if not re.fullmatch(r'[a-z0-9]([a-z0-9-]*[a-z0-9])?',name): raise RuntimeError("AZURE_CONTAINER invalid; lowercase letters, digits or '-', cannot start/end with '-'")
    return name
def _parse_cidr(cidr:str):
    try: return ipaddress.IPv4Network(cidr,strict=False)
    except Exception as e: raise RuntimeError(f"Invalid CIDR '{cidr}': {e}")
def _ensure_subnet_in_vnet(vnet:ipaddress.IPv4Network,sub:ipaddress.IPv4Network,name:str):
    if not (sub.network_address >= vnet.network_address and sub.broadcast_address <= vnet.broadcast_address): raise RuntimeError(f"{name} {sub} not inside VNET {vnet}")
def _ensure_no_overlap(a:ipaddress.IPv4Network,b:ipaddress.IPv4Network,a_name:str,b_name:str):
    if a.overlaps(b): raise RuntimeError(f"{a_name} {a} overlaps with {b_name} {b}")
stack=pulumi.get_stack()
raw_prefix=_env("RESOURCE_NAME_PREFIX","rag")
prefix=_sanitize_prefix(raw_prefix)
location=_env("AZURE_LOCATION") or _env("AZ_REGION") or "eastus"
_validate_region(location)
rg_name=_env("AZURE_RESOURCE_GROUP_NAME") or f"{prefix}-rg-{stack}"
rg_name=_validate_rg(rg_name)
vnet_name=_env("VNET_NAME") or f"{prefix}-vnet-{stack}"
vnet_cidr=_env("VNET_CIDR") or "10.1.0.0/16"
aks_subnet_prefix=_env("AKS_SUBNET_PREFIX") or "10.1.1.0/24"
appgw_subnet_prefix=_env("APPGW_SUBNET_PREFIX") or "10.1.2.0/24"
create_nat=_bool("CREATE_NAT",False)
provided_sa=_env("AZURE_STORAGE_ACCOUNT_NAME")
if provided_sa:
    storage_account_name=_validate_storage_account(provided_sa.lower())
else:
    candidate=f"{prefix}stor{stack}".lower()
    storage_account_name=_validate_storage_account(candidate[:24])
blob_container_name=_validate_container((_env("AZURE_CONTAINER") or f"{prefix}-data-{stack}"))
vnet_net=_parse_cidr(vnet_cidr)
aks_net=_parse_cidr(aks_subnet_prefix)
appgw_net=_parse_cidr(appgw_subnet_prefix)
_ensure_subnet_in_vnet(vnet_net,aks_net,"AKS subnet")
_ensure_subnet_in_vnet(vnet_net,appgw_net,"AppGW subnet")
_ensure_no_overlap(aks_net,appgw_net,"AKS subnet","AppGW subnet")
pulumi.log.info(f"[core_network] prefix={prefix} stack={stack} location={location} vnet={vnet_net} aks={aks_net} appgw={appgw_net} nat={create_nat}")
rg=None
try:
    existing_rg=resources.get_resource_group(resource_group_name=rg_name)
    rg=existing_rg
    pulumi.log.info(f"[core_network] using existing RG {rg_name}")
except Exception:
    rg=resources.ResourceGroup(rg_name,resource_group_name=rg_name,location=location,tags={"managedBy":"pulumi","project":prefix,"stack":stack})
storage_account=None
sa_exists=False
try:
    existing_sa=storage.get_storage_account(resource_group_name=rg_name,account_name=storage_account_name)
    storage_account=existing_sa
    sa_exists=True
    pulumi.log.info(f"[core_network] found existing storage account {storage_account_name}")
except Exception:
    storage_account=storage.StorageAccount(storage_account_name,resource_group_name=rg_name,account_name=storage_account_name,location=location,sku=storage.SkuArgs(name=storage.SkuName.STANDARD_LRS),kind=storage.Kind.STORAGE_V2,minimum_tls_version="TLS1_2",allow_blob_public_access=False,tags={"managedBy":"pulumi","project":prefix,"stack":stack})
container_account_name=(getattr(storage_account,"name",None) or getattr(storage_account,"account_name",None) or storage_account_name)
blob_container=None
container_exists=False
try:
    if sa_exists:
        existing_container=storage.get_blob_container(resource_group_name=rg_name,account_name=storage_account_name,container_name=blob_container_name)
        blob_container=existing_container
        container_exists=True
        pulumi.log.info(f"[core_network] found existing blob container {blob_container_name} in account {storage_account_name}")
except Exception:
    pass
if not container_exists:
    blob_container=storage.BlobContainer(f"blobcontainer-{blob_container_name}",resource_group_name=rg_name,account_name=container_account_name,container_name=blob_container_name,public_access=storage.PublicAccess.NONE)
nat_public_ip:Optional[Any]=None
nat_gw:Optional[Any]=None
if create_nat:
    pip_name=f"{prefix}-nat-pip-{stack}"
    nat_name=f"{prefix}-nat-{stack}"
    nat_public_ip=network.PublicIPAddress(pip_name,resource_group_name=rg_name,public_ip_address_name=pip_name,location=location,sku=network.PublicIPAddressSkuArgs(name="Standard"),public_ip_allocation_method="Static",tags={"managedBy":"pulumi","project":prefix,"stack":stack})
    nat_gw=network.NatGateway(nat_name,resource_group_name=rg_name,nat_gateway_name=nat_name,location=location,sku=network.NatGatewaySkuArgs(name="Standard"),public_ip_addresses=[network.SubResourceArgs(id=nat_public_ip.id)],idle_timeout_in_minutes=10,tags={"managedBy":"pulumi","project":prefix,"stack":stack})
vnet=None
vnet_exists=False
try:
    existing_vnet=network.get_virtual_network(resource_group_name=rg_name,virtual_network_name=vnet_name)
    vnet=existing_vnet
    vnet_exists=True
    pulumi.log.info(f"[core_network] found existing VNet {vnet_name}")
except Exception:
    vnet=network.VirtualNetwork(vnet_name,resource_group_name=rg_name,virtual_network_name=vnet_name,location=location,address_space=network.AddressSpaceArgs(address_prefixes=[vnet_cidr]),tags={"managedBy":"pulumi","project":prefix,"stack":stack})
aks_subnet=None
if vnet_exists:
    try:
        existing_aks_subnet=network.get_subnet(resource_group_name=rg_name,virtual_network_name=vnet_name,subnet_name="snet-aks")
        aks_subnet=existing_aks_subnet
        pulumi.log.info(f"[core_network] found existing subnet snet-aks")
    except Exception:
        aks_subnet=network.Subnet(f"{vnet_name}-snet-aks",resource_group_name=rg_name,virtual_network_name=vnet_name,subnet_name="snet-aks",address_prefix=aks_subnet_prefix,delegations=[network.DelegationArgs(name="delegation-aks",service_name="Microsoft.ContainerService/managedClusters")],nat_gateway=network.SubResourceArgs(id=nat_gw.id) if nat_gw else None)
else:
    aks_subnet=network.Subnet(f"{vnet_name}-snet-aks",resource_group_name=rg_name,virtual_network_name=vnet.name,subnet_name="snet-aks",address_prefix=aks_subnet_prefix,delegations=[network.DelegationArgs(name="delegation-aks",service_name="Microsoft.ContainerService/managedClusters")],nat_gateway=network.SubResourceArgs(id=nat_gw.id) if nat_gw else None)
appgw_subnet=None
if vnet_exists:
    try:
        existing_appgw_subnet=network.get_subnet(resource_group_name=rg_name,virtual_network_name=vnet_name,subnet_name="snet-appgw")
        appgw_subnet=existing_appgw_subnet
        pulumi.log.info(f"[core_network] found existing subnet snet-appgw")
    except Exception:
        appgw_subnet=network.Subnet(f"{vnet_name}-snet-appgw",resource_group_name=rg_name,virtual_network_name=vnet_name,subnet_name="snet-appgw",address_prefix=appgw_subnet_prefix)
else:
    appgw_subnet=network.Subnet(f"{vnet_name}-snet-appgw",resource_group_name=rg_name,virtual_network_name=vnet.name,subnet_name="snet-appgw",address_prefix=appgw_subnet_prefix)
def _id_of(obj:Any)->Optional[Any]:
    return getattr(obj,"id",None) if obj is not None else None
pulumi.export("resource_group_name",rg_name)
pulumi.export("resource_group_id",_id_of(rg))
pulumi.export("storage_account_name",storage_account_name)
pulumi.export("storage_account_id",_id_of(storage_account))
pulumi.export("blob_container_name",blob_container_name)
pulumi.export("blob_container_id",_id_of(blob_container))
pulumi.export("vnet_name",vnet_name)
pulumi.export("vnet_id",_id_of(vnet))
pulumi.export("aks_subnet_id",_id_of(aks_subnet))
pulumi.export("appgw_subnet_id",_id_of(appgw_subnet))
if nat_gw:
    pulumi.export("nat_public_ip_id",_id_of(nat_public_ip))
    pulumi.export("nat_gateway_id",_id_of(nat_gw))
def outputs()->Dict[str,Any]:
    return {"resource_group_name":rg_name,"resource_group_id":_id_of(rg),"storage_account_name":storage_account_name,"storage_account_id":_id_of(storage_account),"blob_container_name":blob_container_name,"blob_container_id":_id_of(blob_container),"vnet_name":vnet_name,"vnet_id":_id_of(vnet),"aks_subnet_id":_id_of(aks_subnet),"appgw_subnet_id":_id_of(appgw_subnet),"nat_public_ip_id":(_id_of(nat_public_ip) if nat_public_ip else None),"nat_gateway_id":(_id_of(nat_gw) if nat_gw else None)}
