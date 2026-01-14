from __future__ import annotations
import os
import re
import sys
import uuid
import pulumi
from pulumi import Output,ResourceOptions
from pulumi_azure_native import resources,managedidentity,authorization
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import network
import aks
import acr as acr_module
def getenv_raw(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None and v != "" else default
def as_int(name: str, default: int) -> int:
    v = getenv_raw(name, None)
    return default if v is None else int(v)
def as_bool(name: str, default: bool) -> bool:
    v = getenv_raw(name, None)
    return default if v is None else (str(v).strip().lower() not in ("0","false","no",""))
AZURE_SUBSCRIPTION_ID = getenv_raw("AZURE_SUBSCRIPTION_ID", None)
AZURE_RESOURCE_GROUP_NAME = getenv_raw("AZURE_RESOURCE_GROUP_NAME", "rg-e2e-rag")
AZURE_LOCATION = getenv_raw("AZURE_LOCATION", "eastus")
PULUMI_STACK = getenv_raw("PULUMI_STACK", "staging")
AKS_LOCATION = getenv_raw("AKS_LOCATION", AZURE_LOCATION)
AKS_CLUSTER_NAME = getenv_raw("AKS_CLUSTER_NAME", "rag-aks")
AKS_SKU = getenv_raw("AKS_SKU", "standard")
AKS_MAX_PODS = as_int("AKS_MAX_PODS", 60)
SYSTEM_NODE_VM_SIZE = getenv_raw("SYSTEM_NODE_VM_SIZE", "Standard_B2s")
SYSTEM_NODE_COUNT = as_int("SYSTEM_NODE_COUNT", 1)
BALANCED_NODE_VM_SIZE = getenv_raw("BALANCED_NODE_VM_SIZE", "Standard_B2s")
BALANCED_NODE_MIN = as_int("BALANCED_NODE_MIN", 0)
BALANCED_NODE_MAX = as_int("BALANCED_NODE_MAX", 1)
CPU_HEAVY_NODE_VM_SIZE = getenv_raw("CPU_HEAVY_NODE_VM_SIZE", "Standard_B2s")
CPU_HEAVY_NODE_MIN = as_int("CPU_HEAVY_NODE_MIN", 0)
CPU_HEAVY_NODE_MAX = as_int("CPU_HEAVY_NODE_MAX", 0)
QDRANT_NODE_VM_SIZE = getenv_raw("QDRANT_NODE_VM_SIZE", "Standard_B2s")
QDRANT_NODE_COUNT = as_int("QDRANT_NODE_COUNT", 0)
AKS_NETWORK_PLUGIN = getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower()
DEFAULT_OUTBOUND = getenv_raw("AKS_OUTBOUND_TYPE", "userAssignedNATGateway")
CREATE_NAT = as_bool("CREATE_NAT", True)
ACR_NAME = getenv_raw("ACR_NAME", None)
if not ACR_NAME:
    raise SystemExit("ACR_NAME is required.")
if not re.fullmatch(r"[a-z0-9]{5,50}", ACR_NAME):
    raise SystemExit("ACR_NAME must be 5-50 lowercase alphanumeric characters.")
ACR_LOCATION = getenv_raw("ACR_LOCATION", AKS_LOCATION)
ACR_SKU = (getenv_raw("ACR_SKU", "Standard") or "Standard").capitalize()
ACR_ADMIN_ENABLED = as_bool("ACR_ADMIN_ENABLED", False)
ACR_PUBLIC_ACCESS = as_bool("ACR_PUBLIC_ACCESS", True)
ACR_RETENTION_DAYS = as_int("ACR_RETENTION_DAYS", 30)
rg = resources.get_resource_group_output(resource_group_name=AZURE_RESOURCE_GROUP_NAME)
pulumi.export("note", f"Region={ACR_LOCATION}. Use make pulumi-up to apply.")
net = network.create_network(rg, AKS_LOCATION, PULUMI_STACK, AKS_NETWORK_PLUGIN, create_nat=CREATE_NAT)
pulumi.export("vnet_name", net["vnet"].name)
acr_obj = acr_module.create_or_get_registry(resource_group_name=AZURE_RESOURCE_GROUP_NAME, registry_name=ACR_NAME, sku=ACR_SKU, admin_user_enabled=ACR_ADMIN_ENABLED, public_network_access=ACR_PUBLIC_ACCESS, retention_days=ACR_RETENTION_DAYS, location=ACR_LOCATION)
def _acrdict(a):
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    try:
        return {"id": a[0], "login_server": a[1], "provisioning_state": a[2]}
    except Exception:
        return {}
acr = Output.from_input(acr_obj).apply(lambda v: _acrdict(v))
pulumi.export("acr_login_server", acr.apply(lambda a: a.get("login_server")))
pulumi.export("acr_provisioning_state", acr.apply(lambda a: a.get("provisioning_state")))
control_name = f"control-uami-{PULUMI_STACK}"
kubelet_name = f"kubelet-uami-{PULUMI_STACK}"
control_uami = managedidentity.UserAssignedIdentity(control_name, resource_group_name=AZURE_RESOURCE_GROUP_NAME, location=AKS_LOCATION, tags={"managedBy":"pulumi","project":"rag","stack":PULUMI_STACK})
kubelet_uami = managedidentity.UserAssignedIdentity(kubelet_name, resource_group_name=AZURE_RESOURCE_GROUP_NAME, location=AKS_LOCATION, tags={"managedBy":"pulumi","project":"rag","stack":PULUMI_STACK})
pulumi.export("control_uami_id", control_uami.id)
pulumi.export("kubelet_uami_id", kubelet_uami.id)
def _sub_from_env_or_ac(a):
    if AZURE_SUBSCRIPTION_ID:
        return AZURE_SUBSCRIPTION_ID
    aid = a.get("id") if isinstance(a, dict) else None
    if aid and aid.startswith("/"):
        parts = aid.split("/")
        try:
            return parts[2]
        except Exception:
            return None
    return None
subscription_output = acr.apply(lambda a: _sub_from_env_or_ac(a))
mio_role_guid = "f1a07417-d97a-45cb-824c-7a7467783830"
def _create_mio(args):
    pid, scope, sub = args
    if not pid or not scope or not sub:
        return None
    role_def = f"/subscriptions/{sub}/providers/Microsoft.Authorization/roleDefinitions/{mio_role_guid}"
    name = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mio:{scope}:{pid}"))
    return authorization.RoleAssignment("mioAssign", role_assignment_name=name, scope=scope, role_definition_id=role_def, principal_id=pid, principal_type="ServicePrincipal")
mio_ra = Output.all(control_uami.principal_id, kubelet_uami.id, subscription_output).apply(_create_mio)
acrpull_role_guid = "7f951dda-4ed3-4680-a7ca-43fe172d538d"
def _create_acrpull(args):
    principal_id, registry_id, sub = args
    if not principal_id or not registry_id or not sub:
        return None
    role_def = f"/subscriptions/{sub}/providers/Microsoft.Authorization/roleDefinitions/{acrpull_role_guid}"
    name = str(uuid.uuid5(uuid.NAMESPACE_URL, f"acrpull:{registry_id}:{principal_id}"))
    return authorization.RoleAssignment("acrPullAssign", role_assignment_name=name, scope=registry_id, role_definition_id=role_def, principal_id=principal_id, principal_type="ServicePrincipal")
acrpull_ra = Output.all(kubelet_uami.principal_id, acr.apply(lambda a: a.get("id")), subscription_output).apply(_create_acrpull)
pulumi.export("mio_role_assignment", mio_ra.apply(lambda r: r.id if r else None))
pulumi.export("acrpull_role_assignment", acrpull_ra.apply(lambda r: r.id if r else None))
deps = Output.all(control_uami.id, kubelet_uami.id, mio_ra.apply(lambda r: r.id if r else None), acrpull_ra.apply(lambda r: r.id if r else None), net["snet_aks"].id, acr.apply(lambda a: a.get("id")), subscription_output, net.get("nat_gateway").id if net.get("nat_gateway") else None)
def _create_cluster(vals):
    control_id, kubelet_id, _, _, snet_aks_id, _, sub, nat_id = vals
    if not sub:
        raise SystemExit("AZURE_SUBSCRIPTION_ID not set and could not be parsed from ACR id; set AZURE_SUBSCRIPTION_ID.")
    outbound = DEFAULT_OUTBOUND
    if CREATE_NAT:
        outbound = "userAssignedNATGateway"
    if snet_aks_id and outbound == "managedNATGateway":
        pulumi.log.warn("AKS_OUTBOUND_TYPE=managedNATGateway is not allowed with a custom VNet/subnet; falling back to loadBalancer.")
        outbound = "loadBalancer"
    return aks.create_aks_cluster_from_ids(resource_group_name=AZURE_RESOURCE_GROUP_NAME, location=AKS_LOCATION, stack=PULUMI_STACK, aks_cluster_name=AKS_CLUSTER_NAME, aks_sku=AKS_SKU, system_node_vm=SYSTEM_NODE_VM_SIZE, system_node_count=SYSTEM_NODE_COUNT, max_pods=AKS_MAX_PODS, balanced_vm=BALANCED_NODE_VM_SIZE, balanced_min=BALANCED_NODE_MIN, balanced_max=BALANCED_NODE_MAX, cpu_vm=CPU_HEAVY_NODE_VM_SIZE, cpu_min=CPU_HEAVY_NODE_MIN, cpu_max=CPU_HEAVY_NODE_MAX, qdrant_vm=QDRANT_NODE_VM_SIZE, qdrant_count=QDRANT_NODE_COUNT, aks_network_plugin=AKS_NETWORK_PLUGIN, outbound_type=outbound, snet_aks_id=snet_aks_id if snet_aks_id else None, control_uami_id=control_id, kubelet_uami_id=kubelet_id, nat_gateway_id=nat_id if nat_id else None)
cluster = deps.apply(_create_cluster)
net["snet_aks"]._opts = ResourceOptions(depends_on=[cluster])
pulumi.export("aks_cluster_name", cluster.apply(lambda c: c.name) if isinstance(cluster, Output) else getattr(cluster, "name", None))
pulumi.export("kubelet_identity_id", kubelet_uami.id)
