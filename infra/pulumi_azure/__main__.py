from __future__ import annotations
import os
import re
import sys
import uuid
import pulumi
import pulumi_azure_native as azure_native
from pulumi_azure_native import resources, network, containerservice, containerregistry, authorization

def getenv_raw(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None and v != "" else default

def as_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        raise SystemExit(f"Environment variable {name} must be an integer; got: {v}")

def as_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "")

def validate_name(name: str):
    if not re.fullmatch(r"[a-z0-9]{5,50}", name):
        raise SystemExit("ACR_NAME must be 5-50 characters, lowercase letters and numbers only.")

STACK = getenv_raw("PULUMI_STACK", "staging")
LOCATION = getenv_raw("AKS_LOCATION", getenv_raw("AZURE_LOCATION", "eastus"))
rg_name = "rg-e2e-rag"
vnet_name = f"rag-vnet-{STACK}"
snet_aks_name = "snet-aks"
snet_appgw_name = "snet-appgw"

AKS_NETWORK_PLUGIN = getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower()
AKS_NETWORK_POLICY = getenv_raw("AKS_NETWORK_POLICY", "calico")
AKS_OUTBOUND_TYPE = getenv_raw("AKS_OUTBOUND_TYPE", "managedNATGateway")
CREATE_NAT = as_bool("CREATE_NAT", False)
AKS_MAX_PODS = as_int("AKS_MAX_PODS", 60)
AKS_CLUSTER_NAME = getenv_raw("AKS_CLUSTER_NAME", "rag-aks")
AKS_SKU = getenv_raw("AKS_SKU", "standard")
SYSTEM_NODE_COUNT = as_int("SYSTEM_NODE_COUNT", 1)
SYSTEM_NODE_VM = getenv_raw("SYSTEM_NODE_VM_SIZE", "Standard_B2s")
SYSTEM_NODE_MAX_PODS = as_int("SYSTEM_NODE_MAX_PODS", AKS_MAX_PODS)

rg = resources.get_resource_group_output(resource_group_name=rg_name)

vnet = network.VirtualNetwork(
    vnet_name,
    resource_group_name=rg.name,
    address_space=network.AddressSpaceArgs(address_prefixes=["10.1.0.0/16"]),
    location=LOCATION,
    tags={"env": STACK, "managedBy": "pulumi", "project": "rag", "stack": STACK},
)

snet_appgw = network.Subnet(
    snet_appgw_name,
    resource_group_name=rg.name,
    virtual_network_name=vnet.name,
    address_prefix="10.1.2.0/24",
    private_endpoint_network_policies="Disabled",
    private_link_service_network_policies="Enabled",
)

if AKS_NETWORK_PLUGIN == "azure":
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

def agent_profile_args(name: str, vm_size: str, count: int, mode: str = "User", enable_autoscaling: bool = False, min_count: int = None, max_count: int = None):
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

SYSTEM_NODE_VM = getenv_raw("SYSTEM_NODE_VM_SIZE", SYSTEM_NODE_VM)
BALANCED_MIN = as_int("BALANCED_NODE_MIN", 0)
BALANCED_MAX = as_int("BALANCED_NODE_MAX", 1)
BALANCED_VM = getenv_raw("BALANCED_NODE_VM_SIZE", "Standard_B2s")
CPU_HEAVY_MIN = as_int("CPU_HEAVY_NODE_MIN", 0)
CPU_HEAVY_MAX = as_int("CPU_HEAVY_NODE_MAX", 0)
CPU_HEAVY_VM = getenv_raw("CPU_HEAVY_NODE_VM_SIZE", "Standard_B2s")
QDRANT_COUNT = as_int("QDRANT_NODE_COUNT", 0)
QDRANT_VM = getenv_raw("QDRANT_NODE_VM_SIZE", "Standard_B2s")

system_pool = agent_profile_args("systempool", SYSTEM_NODE_VM, SYSTEM_NODE_COUNT, mode="System")
apppool_count = BALANCED_MIN if BALANCED_MIN > 0 else 0
app_pool = agent_profile_args("apppool", BALANCED_VM, apppool_count, mode="User", enable_autoscaling=True, min_count=BALANCED_MIN, max_count=BALANCED_MAX)
qdrant_pool = agent_profile_args("qdrant", QDRANT_VM, QDRANT_COUNT, mode="User")
cpu_pool = agent_profile_args("cpuheavy", CPU_HEAVY_VM, CPU_HEAVY_MIN if CPU_HEAVY_MIN>0 else 0, mode="User", enable_autoscaling=True, min_count=CPU_HEAVY_MIN, max_count=CPU_HEAVY_MAX)
agent_pools = [system_pool, app_pool, qdrant_pool, cpu_pool]
for p in agent_pools:
    if AKS_NETWORK_PLUGIN != "azure":
        try:
            setattr(p, "vnet_subnet_id", None)
        except Exception:
            pass

network_profile = containerservice.ContainerServiceNetworkProfileArgs(
    network_plugin=AKS_NETWORK_PLUGIN,
    network_policy=AKS_NETWORK_POLICY,
    service_cidr="10.0.0.0/16",
    dns_service_ip="10.0.0.10",
    pod_cidr="10.244.0.0/16",
    outbound_type=AKS_OUTBOUND_TYPE,
)

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

pulumi.export("aks_cluster_name", cluster.name)
pulumi.export("vnet_name", vnet.name)
pulumi.export("blob_container_name", "rag-data-515")
pulumi.export("system_pool_name", "systempool")
pulumi.export("note", f"Region={LOCATION}. If apply fails, check region quota, VM size availability, RG permissions and outbound type (AKS_OUTBOUND_TYPE).")

SUBSCRIPTION_ID = getenv_raw("AZURE_SUBSCRIPTION_ID", None)
if not SUBSCRIPTION_ID:
    pulumi.log.warn("AZURE_SUBSCRIPTION_ID not set; some automated role assignment logic may be skipped.")

ACR_NAME = getenv_raw("ACR_NAME", None)
if not ACR_NAME:
    raise SystemExit("ACR_NAME is required.")
validate_name(ACR_NAME)

ACR_RESOURCE_RG = getenv_raw("ACR_RESOURCE_RG", rg_name)
explicit_acr_location = getenv_raw("ACR_LOCATION", None)
if explicit_acr_location:
    ACR_LOCATION = explicit_acr_location
else:
    ACR_LOCATION = LOCATION

ACR_SKU_RAW = (getenv_raw("ACR_SKU", "Standard") or "Standard").capitalize()
if ACR_SKU_RAW not in ("Basic", "Standard", "Premium"):
    raise SystemExit("ACR_SKU must be one of: Basic, Standard, Premium")

india_regions = ("centralindia", "southindia")
if ACR_LOCATION.lower() in india_regions and ACR_SKU_RAW == "Standard":
    pulumi.log.info(f"Upgrading requested ACR_SKU Standard -> Premium for region {ACR_LOCATION} to avoid SkuNotSupported.")
    ACR_SKU = "Premium"
else:
    ACR_SKU = ACR_SKU_RAW

ACR_ADMIN_ENABLED = as_bool("ACR_ADMIN_ENABLED", False)
ACR_PUBLIC_ACCESS = as_bool("ACR_PUBLIC_ACCESS", True)
ACR_RETENTION_DAYS = as_int("ACR_RETENTION_DAYS", 30)
if not (1 <= ACR_RETENTION_DAYS <= 365):
    raise SystemExit("ACR_RETENTION_DAYS must be between 1 and 365")

ENABLE_PRIVATE_ENDPOINTS = as_bool("ENABLE_PRIVATE_ENDPOINTS", False)
ACR_PRIVATE_ENDPOINT_ENABLED = as_bool("ACR_PRIVATE_ENDPOINT_ENABLED", False)
PRIVATE_ENDPOINT_ON = ENABLE_PRIVATE_ENDPOINTS or ACR_PRIVATE_ENDPOINT_ENABLED

if PRIVATE_ENDPOINT_ON and ACR_SKU != "Premium":
    raise SystemExit("ACR Private Endpoint requires ACR_SKU=Premium. Set ACR_SKU=Premium or disable private endpoints.")

if ACR_PRIVATE_ENDPOINT_ENABLED and ACR_PUBLIC_ACCESS:
    raise SystemExit("ACR_PUBLIC_ACCESS must be false when ACR_PRIVATE_ENDPOINT_ENABLED=true")

pulumi.export("acr_config", {
    "rg": ACR_RESOURCE_RG,
    "acr": ACR_NAME,
    "sku": ACR_SKU,
    "location": ACR_LOCATION,
    "private_endpoint_enabled": PRIVATE_ENDPOINT_ON,
    "aks_attach_requested": False,
})

existing_registry_id = None
existing_registry_login = None
existing_registry_sku = None
existing_registry_location = None
try:
    resp = containerregistry.get_registry(resource_group_name=ACR_RESOURCE_RG, registry_name=ACR_NAME)
    existing_registry_id = resp.id
    existing_registry_login = getattr(resp, "login_server", None) or (resp.login_server if hasattr(resp, "login_server") else None)
    existing_registry_sku = getattr(resp, "sku", None).name if getattr(resp, "sku", None) else None
    existing_registry_location = getattr(resp, "location", None)
    pulumi.log.info(f"Found existing ACR '{ACR_NAME}' in RG '{ACR_RESOURCE_RG}', location: {existing_registry_location}, sku: {existing_registry_sku}. Reusing it.")
    if existing_registry_location and existing_registry_location.lower() != ACR_LOCATION.lower():
        pulumi.log.warn(f"Existing ACR location {existing_registry_location} != requested {ACR_LOCATION}; continuing to reuse existing registry.")
    if existing_registry_sku and ACR_SKU and existing_registry_sku.capitalize() != ACR_SKU.capitalize():
        pulumi.log.warn(f"Existing ACR SKU {existing_registry_sku} != requested {ACR_SKU}; continuing to reuse existing registry.")
except Exception:
    pulumi.log.info(f"No existing ACR '{ACR_NAME}' found in RG '{ACR_RESOURCE_RG}'. Will create a new registry in {ACR_LOCATION} with SKU={ACR_SKU}.")

registry = None
if existing_registry_id:
    acr_id_output = pulumi.Output.from_input(existing_registry_id)
    acr_login_output = pulumi.Output.from_input(existing_registry_login)
    acr_provisioning_output = pulumi.Output.from_input(None)
else:
    sku_obj = {"name": ACR_SKU}
    public_access_enum = "Enabled" if ACR_PUBLIC_ACCESS else "Disabled"
    policies_obj = {
        "retentionPolicy": {"days": int(ACR_RETENTION_DAYS), "status": "Enabled"},
        "softDeletePolicy": {"status": "Enabled"},
    }
    registry = containerregistry.Registry(
        "acrRegistry",
        resource_group_name=ACR_RESOURCE_RG,
        registry_name=ACR_NAME,
        sku=sku_obj,
        admin_user_enabled=ACR_ADMIN_ENABLED,
        public_network_access=public_access_enum,
        policies=policies_obj,
        location=ACR_LOCATION,
        tags={"managedBy": "pulumi", "project": "rag"},
    )
    acr_id_output = registry.id
    acr_login_output = registry.login_server
    acr_provisioning_output = registry.provisioning_state

pulumi.export("acr_id", acr_id_output)
pulumi.export("acr_login_server", acr_login_output)
pulumi.export("acr_provisioning_state", acr_provisioning_output)

AKS_RESOURCE_NAME = getenv_raw("AKS_RESOURCE_NAME", None)
AKS_RESOURCE_RG = getenv_raw("AKS_RESOURCE_RG", rg_name)
AKS_ACR_ATTACH = as_bool("AKS_ACR_ATTACH", True)

def attach_acr_to_aks():
    if not AKS_ACR_ATTACH:
        pulumi.log.info("AKS_ACR_ATTACH=false; skipping AcrPull role assignment.")
        return
    principal_output = None
    if AKS_RESOURCE_NAME:
        try:
            mc = containerservice.get_managed_cluster(resource_group_name=AKS_RESOURCE_RG, resource_name=AKS_RESOURCE_NAME)
            ip = getattr(mc, "identity_profile", None) or getattr(mc, "identityProfile", None) or {}
            ki = None
            if ip:
                if isinstance(ip, dict):
                    ki = ip.get("kubeletidentity") or ip.get("kubeletIdentity")
                else:
                    ki = ip.get("kubeletidentity") if hasattr(ip, "get") else None
            if ki:
                principal_str = ki.get("principalId") or ki.get("principal_id") or ki.get("principalid")
                principal_output = pulumi.Output.from_input(principal_str)
            else:
                identity = getattr(mc, "identity", None) or {}
                principal_str = getattr(identity, "principal_id", None) or (identity.get("principalId") if isinstance(identity, dict) else None)
                principal_output = pulumi.Output.from_input(principal_str)
        except Exception as e:
            pulumi.log.warn(f"Unable to locate AKS cluster '{AKS_RESOURCE_NAME}' in RG '{AKS_RESOURCE_RG}': {e}. Skipping AcrPull assignment.")
            return
    else:
        try:
            principal_output = cluster.identity.apply(lambda idt: getattr(idt, "principal_id", None) or (idt.get("principalId") if isinstance(idt, dict) else None))
        except Exception:
            pulumi.log.info("AKS kubelet principal not found automatically; skipping AcrPull role assignment.")
            return

    if not principal_output:
        pulumi.log.info("AKS kubelet principal not found; skipping AcrPull role assignment.")
        return

    if not SUBSCRIPTION_ID:
        pulumi.log.warn("AZURE_SUBSCRIPTION_ID not set; cannot construct roleDefinitionId for AcrPull. Skipping automatic assignment.")
        return

    ACRPULL_GUID = "7f951dda-4ed3-4680-a7ca-43fe172d538d"
    role_def_id = f"/subscriptions/{SUBSCRIPTION_ID}/providers/Microsoft.Authorization/roleDefinitions/{ACRPULL_GUID}"

    def create_ra(principal_id, registry_id_arg):
        ra_name = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{principal_id}:{registry_id_arg}:{ACRPULL_GUID}"))
        return authorization.RoleAssignment(
            "acrAcrPullAssignment",
            scope=registry_id_arg,
            role_assignment_name=ra_name,
            principal_id=principal_id,
            role_definition_id=role_def_id,
            principal_type="ServicePrincipal",
        )

    try:
        pulumi.Output.all(principal_output, acr_id_output).apply(lambda args: create_ra(args[0], args[1]))
        pulumi.export("acr_assigned_to_principal", principal_output)
        pulumi.log.info("Attempted AcrPull role assignment (may be created if principal/registry resolved).")
    except Exception as e:
        pulumi.log.warn(f"Role assignment creation failed: {e}. Manual remediation may be required.")

attach_acr_to_aks()

pulumi.export(
    "manual_attach_command_example",
    acr_id_output.apply(lambda rid: f"az role assignment create --assignee <kubeletPrincipalId> --role AcrPull --scope {rid}") if hasattr(acr_id_output, "apply") else pulumi.Output.from_input("Provide registry_id to generate attach command")
)

pulumi.export("final_note", "All resources requested are declared. Use 'pulumi preview' to validate and then 'pulumi up' to apply.")
