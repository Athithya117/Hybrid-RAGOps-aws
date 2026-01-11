"""
Top-level Pulumi program (platform-grade, deterministic).

This program:
 - Validates the exact environment variables you provided (no extras required).
 - Creates network & AKS as before (calls network.create_network and aks.create_aks_cluster
   from your repo's modules; those files are unchanged).
 - Deterministically creates ACR using acr.create_registry(...) (managed lifecycle).
 - Fails fast and loudly on invalid env values or unsupported combinations (e.g., private endpoint without Premium).
"""

from __future__ import annotations
import os
import sys
import re
import pulumi
from pulumi_azure_native import resources

# Ensure local modules importable when Pulumi executes this file as a script
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# network.py and aks.py are existing modules in this directory; import them.
import network
import aks
import acr as acr_module

# ---------------- helpers ----------------
def getenv_raw(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None and v != "" else default

def as_int(name: str, default: int) -> int:
    v = getenv_raw(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        raise SystemExit(f"Environment variable {name} must be an integer; got: {v!r}")

def as_bool(name: str, default: bool) -> bool:
    v = getenv_raw(name, None)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "")

# ---------------- required envs (exact set you provided) ----------------
# STEP 1 envs (assumed already exported by user per your doc)
AZURE_SUBSCRIPTION_ID = getenv_raw("AZURE_SUBSCRIPTION_ID", None)  # optional for Pulumi role assignment; not required to run
AZURE_RESOURCE_GROUP_NAME = getenv_raw("AZURE_RESOURCE_GROUP_NAME", "rg-e2e-rag")
AZURE_ENDPOINT_SUFFIX = getenv_raw("AZURE_ENDPOINT_SUFFIX", "core.windows.net")
AZURE_LOCATION = getenv_raw("AZURE_LOCATION", "eastus")
AZURE_STORAGE_ACCOUNT_NAME = getenv_raw("AZURE_STORAGE_ACCOUNT_NAME", None)
STORAGE_TIER = getenv_raw("STORAGE_TIER", "LRS")
AZURE_CONTAINER = getenv_raw("AZURE_CONTAINER", "rag-data-515")
PULUMI_AZ_CONTAINER = getenv_raw("PULUMI_AZ_CONTAINER", "pulumi-state-515")
BACKUP_AZ_CONTAINER = getenv_raw("BACKUP_AZ_CONTAINER", "backups-515")
BACKUP_PREFIX = getenv_raw("BACKUP_PREFIX", "qdrant/backup")
BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS = as_int("BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS", 7)
BACKUP_AZ_CONTAINER_RETENTION_DAYS = as_int("BACKUP_AZ_CONTAINER_RETENTION_DAYS", 30)
AZURE_STORAGE_CONNECTION_STRING = getenv_raw("AZURE_STORAGE_CONNECTION_STRING", "")

# STEP 2 envs (pulumi stack + AKS + ACR)
PULUMI_STACK = getenv_raw("PULUMI_STACK", "staging")
PULUMI_CONFIG_PASSPHRASE = getenv_raw("PULUMI_CONFIG_PASSPHRASE", None)

AKS_MAX_PODS = as_int("AKS_MAX_PODS", 60)
AKS_CLUSTER_NAME = getenv_raw("AKS_CLUSTER_NAME", "rag-aks")
AKS_SKU = getenv_raw("AKS_SKU", "standard")

SYSTEM_NODE_COUNT = as_int("SYSTEM_NODE_COUNT", 1)
SYSTEM_NODE_VM_SIZE = getenv_raw("SYSTEM_NODE_VM_SIZE", "Standard_B2s")
SYSTEM_NODE_MAX_PODS = as_int("SYSTEM_NODE_MAX_PODS", AKS_MAX_PODS)

BALANCED_NODE_MIN = as_int("BALANCED_NODE_MIN", 0)
BALANCED_NODE_MAX = as_int("BALANCED_NODE_MAX", 1)
BALANCED_NODE_VM_SIZE = getenv_raw("BALANCED_NODE_VM_SIZE", "Standard_B2s")

CPU_HEAVY_NODE_MIN = as_int("CPU_HEAVY_NODE_MIN", 0)
CPU_HEAVY_NODE_MAX = as_int("CPU_HEAVY_NODE_MAX", 0)
CPU_HEAVY_NODE_VM_SIZE = getenv_raw("CPU_HEAVY_NODE_VM_SIZE", "Standard_B2s")

QDRANT_NODE_COUNT = as_int("QDRANT_NODE_COUNT", 0)
QDRANT_NODE_VM_SIZE = getenv_raw("QDRANT_NODE_VM_SIZE", "Standard_B2s")

AKS_LOCATION = getenv_raw("AKS_LOCATION", AZURE_LOCATION)

ACR_NAME = getenv_raw("ACR_NAME", None)
if not ACR_NAME:
    raise SystemExit("ACR_NAME is required. Example: export ACR_NAME=acr49251")
if not re.fullmatch(r"[a-z0-9]{5,50}", ACR_NAME):
    raise SystemExit("ACR_NAME must be 5-50 lowercase alphanumeric characters (regex: [a-z0-9]{5,50}).")

ACR_REPO_PREFIX = getenv_raw("ACR_REPO_PREFIX", "rag")
ACR_LOCATION = getenv_raw("ACR_LOCATION", AKS_LOCATION)
ACR_SKU = getenv_raw("ACR_SKU", "Basic")
if ACR_SKU.strip().lower() not in ("basic", "standard", "premium"):
    raise SystemExit("ACR_SKU must be one of: Basic, Standard, Premium.")
ACR_SKU = ACR_SKU.strip().capitalize()

PULUMI_FORCE_DESTROY = as_bool("PULUMI_FORCE_DESTROY", False)

# Validate private endpoint rules statically
ENABLE_PRIVATE_ENDPOINTS = as_bool("ENABLE_PRIVATE_ENDPOINTS", False)
ACR_PRIVATE_ENDPOINT_ENABLED = as_bool("ACR_PRIVATE_ENDPOINT_ENABLED", False)
PRIVATE_ENDPOINT_ON = ENABLE_PRIVATE_ENDPOINTS or ACR_PRIVATE_ENDPOINT_ENABLED
if PRIVATE_ENDPOINT_ON and ACR_SKU != "Premium":
    raise SystemExit("ACR Private Endpoint requires ACR_SKU=Premium. Set ACR_SKU=Premium or disable private endpoints.")
if ACR_PRIVATE_ENDPOINT_ENABLED and as_bool("ACR_PUBLIC_ACCESS", True):
    raise SystemExit("ACR_PUBLIC_ACCESS must be false when ACR_PRIVATE_ENDPOINT_ENABLED=true")

# ---------------- resource group handle ----------------
rg = resources.get_resource_group_output(resource_group_name=AZURE_RESOURCE_GROUP_NAME)

# ---------------- network (unchanged semantics) ----------------
AKS_NETWORK_PLUGIN = getenv_raw("AKS_NETWORK_PLUGIN", "kubenet").lower()
net = network.create_network(rg, AKS_LOCATION, PULUMI_STACK, AKS_NETWORK_PLUGIN)
vnet = net["vnet"]
snet_appgw = net["snet_appgw"]
snet_aks = net["snet_aks"]

# ---------------- AKS creation (preserve behavior) ----------------
cluster = aks.create_aks_cluster(
    rg=rg,
    location=AKS_LOCATION,
    stack=PULUMI_STACK,
    aks_cluster_name=AKS_CLUSTER_NAME,
    aks_sku=AKS_SKU,
    system_node_vm=SYSTEM_NODE_VM_SIZE,
    system_node_count=SYSTEM_NODE_COUNT,
    max_pods=AKS_MAX_PODS,
    balanced_vm=BALANCED_NODE_VM_SIZE,
    balanced_min=BALANCED_NODE_MIN,
    balanced_max=BALANCED_NODE_MAX,
    cpu_vm=CPU_HEAVY_NODE_VM_SIZE,
    cpu_min=CPU_HEAVY_NODE_MIN,
    cpu_max=CPU_HEAVY_NODE_MAX,
    qdrant_vm=QDRANT_NODE_VM_SIZE,
    qdrant_count=QDRANT_NODE_COUNT,
    aks_network_plugin=AKS_NETWORK_PLUGIN,
    outbound_type=getenv_raw("AKS_OUTBOUND_TYPE", "managedNATGateway"),
    snet_aks_id=snet_aks.id if AKS_NETWORK_PLUGIN == "azure" else None,
)

pulumi.export("aks_cluster_name", cluster.name)
pulumi.export("vnet_name", vnet.name)
pulumi.export("blob_container_name", AZURE_CONTAINER)
pulumi.export("system_pool_name", "systempool")
pulumi.export("note", f"Region={AKS_LOCATION}. If apply fails, check region quota, VM size availability, RG permissions and outbound type (AKS_OUTBOUND_TYPE).")

# ---------------- ACR (deterministic managed create) ----------------
# This code intentionally chooses a deterministic 'managed' path (Pulumi creates the registry).
# It validates inputs and omits preview-only properties for Basic SKU.
ACR_ADMIN_ENABLED = as_bool("ACR_ADMIN_ENABLED", False)
ACR_PUBLIC_ACCESS = as_bool("ACR_PUBLIC_ACCESS", True)
ACR_RETENTION_DAYS = as_int("ACR_RETENTION_DAYS", 30)

pulumi.log.info("Proceeding with deterministic ACR create (Pulumi-managed).")

acr_id, acr_login, acr_state = acr_module.create_registry(
    resource_group_name=rg.name,
    registry_name=ACR_NAME,
    sku=ACR_SKU,
    admin_user_enabled=ACR_ADMIN_ENABLED,
    public_network_access=ACR_PUBLIC_ACCESS,
    retention_days=ACR_RETENTION_DAYS,
    location=ACR_LOCATION,
)

pulumi.export("acr_id", acr_id)
pulumi.export("acr_login_server", acr_login)
pulumi.export("acr_provisioning_state", acr_state)
pulumi.log.info("ACR creation request declared to Pulumi (create will be performed on apply).")
