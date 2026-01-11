from __future__ import annotations
import os, re, sys, uuid
import pulumi
from pulumi_azure_native import resources

# --- small helpers (same semantics as before) ---
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

# --- configuration (keep your defaults) ---
STACK = getenv_raw("PULUMI_STACK", "staging")
LOCATION = getenv_raw("AKS_LOCATION", getenv_raw("AZURE_LOCATION", "eastus"))
rg_name = getenv_raw("AZURE_RESOURCE_GROUP_NAME", "rg-e2e-rag")

# these are passed through to modules; change only at env level
AKS_NETWORK_PLUGIN = getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower()
AKS_NETWORK_POLICY = getenv_raw("AKS_NETWORK_POLICY", "calico")
AKS_OUTBOUND_TYPE = getenv_raw("AKS_OUTBOUND_TYPE", "managedNATGateway")

# AKS sizing
AKS_MAX_PODS = as_int("AKS_MAX_PODS", 60)
AKS_CLUSTER_NAME = getenv_raw("AKS_CLUSTER_NAME", "rag-aks")
AKS_SKU = getenv_raw("AKS_SKU", "standard")
SYSTEM_NODE_COUNT = as_int("SYSTEM_NODE_COUNT", 1)
SYSTEM_NODE_VM = getenv_raw("SYSTEM_NODE_VM_SIZE", "Standard_B2s")
SYSTEM_NODE_MAX_PODS = as_int("SYSTEM_NODE_MAX_PODS", AKS_MAX_PODS)
BALANCED_MIN = as_int("BALANCED_NODE_MIN", 0)
BALANCED_MAX = as_int("BALANCED_NODE_MAX", 1)
BALANCED_VM = getenv_raw("BALANCED_NODE_VM_SIZE", "Standard_B2s")
CPU_HEAVY_MIN = as_int("CPU_HEAVY_NODE_MIN", 0)
CPU_HEAVY_MAX = as_int("CPU_HEAVY_NODE_MAX", 0)
CPU_HEAVY_VM = getenv_raw("CPU_HEAVY_NODE_VM_SIZE", "Standard_B2s")
QDRANT_COUNT = as_int("QDRANT_NODE_COUNT", 0)
QDRANT_VM = getenv_raw("QDRANT_NODE_VM_SIZE", "Standard_B2s")

# ACR config
ACR_NAME = getenv_raw("ACR_NAME", None)
if not ACR_NAME:
    raise SystemExit("ACR_NAME is required.")
validate_name(ACR_NAME)
ACR_RESOURCE_RG = getenv_raw("ACR_RESOURCE_RG", rg_name)
ACR_LOCATION = getenv_raw("ACR_LOCATION", LOCATION)
ACR_SKU_RAW = (getenv_raw("ACR_SKU", "Standard") or "Standard").capitalize()
if ACR_SKU_RAW not in ("Basic", "Standard", "Premium"):
    raise SystemExit("ACR_SKU must be one of: Basic, Standard, Premium")
ACR_SKU = ACR_SKU_RAW
ACR_ADMIN_ENABLED = as_bool("ACR_ADMIN_ENABLED", False)
ACR_PUBLIC_ACCESS = as_bool("ACR_PUBLIC_ACCESS", True)
ACR_RETENTION_DAYS = as_int("ACR_RETENTION_DAYS", 30)
if not (1 <= ACR_RETENTION_DAYS <= 365):
    raise SystemExit("ACR_RETENTION_DAYS must be between 1 and 365")
ENABLE_PRIVATE_ENDPOINTS = as_bool("ENABLE_PRIVATE_ENDPOINTS", False)
ACR_PRIVATE_ENDPOINT_ENABLED = as_bool("ACR_PRIVATE_ENDPOINT_ENABLED", False)
if (ENABLE_PRIVATE_ENDPOINTS or ACR_PRIVATE_ENDPOINT_ENABLED) and ACR_SKU != "Premium":
    raise SystemExit("ACR Private Endpoint requires ACR_SKU=Premium. Set ACR_SKU=Premium or disable private endpoints.")
if ACR_PRIVATE_ENDPOINT_ENABLED and ACR_PUBLIC_ACCESS:
    raise SystemExit("ACR_PUBLIC_ACCESS must be false when ACR_PRIVATE_ENDPOINT_ENABLED=true")

# export some base outputs similar to previous behavior
rg = resources.get_resource_group_output(resource_group_name=rg_name)
pulumi.export("note", f"Region={ACR_LOCATION}. Use make pulumi-up to apply.")

import acr,aks,network
# create network (returns dict with subnet ids / names)
net = network.create_network(rg, STACK, LOCATION, AKS_NETWORK_PLUGIN)

# create aks cluster (uses subnet id if plugin 'azure')
cluster_info = aks.create_aks(
    rg,
    AKS_CLUSTER_NAME,
    LOCATION,
    AKS_SKU,
    AKS_MAX_PODS,
    SYSTEM_NODE_VM,
    SYSTEM_NODE_COUNT,
    SYSTEM_NODE_MAX_PODS,
    BALANCED_VM,
    BALANCED_MIN,
    BALANCED_MAX,
    CPU_HEAVY_VM,
    CPU_HEAVY_MIN,
    CPU_HEAVY_MAX,
    QDRANT_VM,
    QDRANT_COUNT,
    AKS_NETWORK_PLUGIN,
    net.get("snet_aks_id"),
)

pulumi.export("aks_cluster_name", cluster_info["cluster"].name)
pulumi.export("vnet_name", net["vnet"].name)
pulumi.export("system_pool_name", "systempool")

# create ACR (minimal, safe for Basic)
acr.create_acr(
    rg_name=ACR_RESOURCE_RG,
    acr_name=ACR_NAME,
    location=ACR_LOCATION,
    sku=ACR_SKU,
    public_access=ACR_PUBLIC_ACCESS,
    admin_enabled=ACR_ADMIN_ENABLED,
    retention_days=ACR_RETENTION_DAYS,
    attach_to_aks=True,
    aks_cluster=cluster_info.get("cluster"),
)
