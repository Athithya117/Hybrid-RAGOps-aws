from __future__ import annotations
import re
import pulumi
from pulumi_azure_native import containerregistry
from pulumi import Output

def _validate_name(name: str):
    if not name:
        raise SystemExit("ACR_NAME is required.")
    if not re.fullmatch(r"[a-z0-9]{5,50}", name):
        raise SystemExit("ACR_NAME must be 5-50 characters, lowercase letters and numbers only (regex: [a-z0-9]{5,50}).")

def _normalize_sku(sku: str) -> str:
    if sku is None:
        raise SystemExit("ACR_SKU is required.")
    s = sku.strip().lower()
    if s not in ("basic", "standard", "premium"):
        raise SystemExit("ACR_SKU must be one of: Basic, Standard, Premium.")
    return s.capitalize()

def _validate_retention(days: int):
    if not isinstance(days, int) or not (1 <= days <= 365):
        raise SystemExit("ACR_RETENTION_DAYS must be an integer between 1 and 365.")

def create_registry(
    resource_group_name: str | Output[str],
    registry_name: str,
    sku: str,
    admin_user_enabled: bool,
    public_network_access: bool,
    retention_days: int,
    location: str,
):
    """
    Creates a Registry resource and returns a dict:
      {"registry": registry_resource, "id": registry.id, "login_server": registry.login_server, "provisioning_state": registry.provisioning_state}
    Deterministic creation; minimal fields for Basic SKU; includes policies for non-Basic SKUs.
    """
    _validate_name(registry_name)
    sku_cap = _normalize_sku(sku)
    _validate_retention(int(retention_days))

    sku_obj = containerregistry.SkuArgs(name=sku_cap)
    public_net = "Enabled" if public_network_access else "Disabled"

    registry_args = {
        "resource_group_name": resource_group_name,
        "registry_name": registry_name,
        "sku": sku_obj,
        "admin_user_enabled": admin_user_enabled,
        "public_network_access": public_net,
        "location": location,
        "tags": {"managedBy": "pulumi", "project": "rag"},
    }

    if sku_cap != "Basic":
        registry_args["policies"] = containerregistry.PoliciesArgs(
            retention_policy=containerregistry.RetentionPolicyArgs(days=int(retention_days), status="Enabled"),
            soft_delete_policy=containerregistry.SoftDeletePolicyArgs(status="Enabled"),
        )

    registry = containerregistry.Registry("acrRegistry", **registry_args)

    return {
        "registry": registry,
        "id": registry.id,
        "login_server": registry.login_server,
        "provisioning_state": registry.provisioning_state,
    }
