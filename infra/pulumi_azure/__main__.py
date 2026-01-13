from __future__ import annotations
import os
import re
import sys
import uuid
import pulumi
from pulumi import Output
from pulumi_azure_native import resources, managedidentity, authorization

# Ensure local modules importable
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
    return default if v is None else (str(v).strip().lower() not in ("0", "false", "no", ""))

# Required / defaulted envs
AZURE_SUBSCRIPTION_ID = getenv_raw("AZURE_SUBSCRIPTION_ID", None)
AZURE_RESOURCE_GROUP_NAME = getenv_raw("AZURE_RESOURCE_GROUP_NAME", "rg-e2e-rag")
AZURE_LOCATION = getenv_raw("AZURE_LOCATION", "eastus")
PULUMI_STACK = getenv_raw("PULUMI_STACK", "staging")

# AKS / args (kept minimal for clarity)
AKS_LOCATION = getenv_raw("AKS_LOCATION", AZURE_LOCATION)
AKS_CLUSTER_NAME = getenv_raw("AKS_CLUSTER_NAME", "rag-aks")
AKS_SKU = getenv_raw("AKS_SKU", "standard")
AKS_MAX_PODS = as_int("AKS_MAX_PODS", 60)

# ACR
ACR_NAME = getenv_raw("ACR_NAME", None)
if not ACR_NAME:
    raise SystemExit("ACR_NAME is required (export ACR_NAME=...).")
if not re.fullmatch(r"[a-z0-9]{5,50}", ACR_NAME):
    raise SystemExit("ACR_NAME must be 5-50 lowercase alphanumeric characters.")
ACR_LOCATION = getenv_raw("ACR_LOCATION", AKS_LOCATION)
ACR_SKU = (getenv_raw("ACR_SKU", "Standard") or "Standard").capitalize()
ACR_ADMIN_ENABLED = as_bool("ACR_ADMIN_ENABLED", False)
ACR_PUBLIC_ACCESS = as_bool("ACR_PUBLIC_ACCESS", True)
ACR_RETENTION_DAYS = as_int("ACR_RETENTION_DAYS", 30)

# Resource group handle
rg = resources.get_resource_group_output(resource_group_name=AZURE_RESOURCE_GROUP_NAME)
pulumi.export("note", f"Region={ACR_LOCATION}. Use make pulumi-up to apply.")

# Network (unchanged)
net = network.create_network(rg, AKS_LOCATION, PULUMI_STACK, getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower())
pulumi.export("vnet_name", net["vnet"].name)

# 1) Create a user-assigned managed identity that will be the kubelet identity.
#    We create it first so its principal_id and id exist for role assignment and cluster wiring.
kubelet_identity_name = f"kubelet-identity-{PULUMI_STACK}"
kubelet_identity = managedidentity.UserAssignedIdentity(
    kubelet_identity_name,
    resource_group_name=AZURE_RESOURCE_GROUP_NAME,
    location=AKS_LOCATION,
    tags={"managedBy": "pulumi", "project": "rag", "stack": PULUMI_STACK},
)
pulumi.export("kubelet_identity_id", kubelet_identity.id)

# 2) Create (or ensure) ACR via your existing helper.
acr_raw = acr_module.create_registry(
    resource_group_name=AZURE_RESOURCE_GROUP_NAME,
    registry_name=ACR_NAME,
    sku=ACR_SKU,
    admin_user_enabled=ACR_ADMIN_ENABLED,
    public_network_access=ACR_PUBLIC_ACCESS,
    retention_days=ACR_RETENTION_DAYS,
    location=ACR_LOCATION,
)

# Normalize acr_out shape (support both tuple or dict return shapes)
def _normalize_acr(o):
    if o is None:
        return {}
    if isinstance(o, dict):
        return o
    try:
        # old helper returned (id, login_server, provisioning_state)
        return {"id": o[0], "login_server": o[1], "provisioning_state": o[2]}
    except Exception as e:
        raise SystemExit("acr.create_registry returned unexpected shape") from e

acr = Output.from_input(acr_raw).apply(lambda v: _normalize_acr(v))
pulumi.export("acr_login_server", acr.apply(lambda a: a.get("login_server")))
pulumi.export("acr_provisioning_state", acr.apply(lambda a: a.get("provisioning_state")))

# 3) Grant AcrPull to the **user-assigned kubelet identity** (immediately).
#    Use principalType = "ServicePrincipal" (user-assigned MI materializes as SP).
def _declare_acrpull(vals):
    acr_obj, principal_id = vals
    acr_id = acr_obj.get("id") if isinstance(acr_obj, dict) else None
    if not acr_id:
        pulumi.log.warn("ACR id not ready; skipping AcrPull role assignment.")
        return None
    if not principal_id:
        pulumi.log.warn("kubelet identity principal_id not ready; skipping AcrPull role assignment.")
        return None

    # derive subscription id from acr id or fallback to env
    subscription = None
    if isinstance(acr_id, str) and acr_id.startswith("/"):
        parts = acr_id.split("/")
        try:
            subscription = parts[2]
        except Exception:
            subscription = None
    if not subscription:
        subscription = AZURE_SUBSCRIPTION_ID
    if not subscription:
        pulumi.log.warn("AZURE_SUBSCRIPTION_ID not set and cannot parse from ACR id; skipping role assignment.")
        return None

    role_guid = "7f951dda-4ed3-4680-a7ca-43fe172d538d"  # AcrPull
    role_id = f"/subscriptions/{subscription}/providers/Microsoft.Authorization/roleDefinitions/{role_guid}"
    ra_name = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{acr_id}:{principal_id}"))

    authorization.RoleAssignment(
        "kubeletAcrPull",
        role_assignment_name=ra_name,
        scope=acr_id,
        role_definition_id=role_id,
        principal_id=principal_id,
        principal_type="ServicePrincipal",  # IMPORTANT: user-assigned identity -> ServicePrincipal
    )
    pulumi.log.info("AcrPull RoleAssignment declared for kubelet identity.")
    return {"role_assignment_name": ra_name, "principal_id": principal_id}

# Run the RBAC wiring when both are available
Output.all(acr, kubelet_identity.principal_id).apply(_declare_acrpull)

# 4) Create AKS cluster.  IMPORTANT: creating `userAssignedIdentities` requires the key
#    to be the literal ARM resource id. To avoid passing an Output as dict key
#    we call the aks helper with the *identity resource object*; the helper will
#    create the ManagedCluster after resolving identity.id (so the dict key is a plain string).
cluster = aks.create_aks_cluster(
    rg=rg,
    location=AKS_LOCATION,
    stack=PULUMI_STACK,
    aks_cluster_name=AKS_CLUSTER_NAME,
    aks_sku=AKS_SKU,
    system_node_vm=getenv_raw("SYSTEM_NODE_VM_SIZE", "Standard_B2s"),
    system_node_count=as_int("SYSTEM_NODE_COUNT", 1),
    max_pods=AKS_MAX_PODS,
    balanced_vm=getenv_raw("BALANCED_NODE_VM_SIZE", "Standard_B2s"),
    balanced_min=as_int("BALANCED_NODE_MIN", 0),
    balanced_max=as_int("BALANCED_NODE_MAX", 1),
    cpu_vm=getenv_raw("CPU_HEAVY_NODE_VM_SIZE", "Standard_B2s"),
    cpu_min=as_int("CPU_HEAVY_NODE_MIN", 0),
    cpu_max=as_int("CPU_HEAVY_NODE_MAX", 0),
    qdrant_vm=getenv_raw("QDRANT_NODE_VM_SIZE", "Standard_B2s"),
    qdrant_count=as_int("QDRANT_NODE_COUNT", 0),
    aks_network_plugin=getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower(),
    outbound_type=getenv_raw("AKS_OUTBOUND_TYPE", "managedNATGateway"),
    snet_aks_id=net["snet_aks"].id if getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower() == "azure" else None,
    kubelet_identity=kubelet_identity,  # pass the managed identity resource (not its Output id)
)

# cluster may be a resource or an Output-wrapped resource; exports can accept both
pulumi.export("aks_cluster_name", cluster.apply(lambda c: c.name) if isinstance(cluster, Output) else cluster.name)
pulumi.export("system_pool_name", "systempool")
