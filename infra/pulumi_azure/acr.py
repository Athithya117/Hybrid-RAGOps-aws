"""
ACR helper for Pulumi (pulumi-azure-native v3.11.0 compatibility).

This module exposes a single, deterministic create_registry(...) function.
It does NOT attempt to probe or import existing registries (no mixed intent).
If you need import behavior, create the registry out-of-band and adapt later.

Behavior:
 - Validates inputs strictly.
 - For SKU == "Basic" it omits policies (preview-only combinations can break Basic).
 - Uses only documented fields for Registry resource.
"""

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
    Create and return (id, login_server, provisioning_state) Pulumi outputs.

    This is a deterministic 'managed' creation path: Pulumi will own the registry lifecycle.
    It validates inputs and avoids preview-only fields for Basic SKU.
    """
    _validate_name(registry_name)
    sku_cap = _normalize_sku(sku)
    _validate_retention(int(retention_days))

    # SkuArgs is supported by pulumi-azure-native; use a simple SkuArgs with name.
    sku_obj = containerregistry.SkuArgs(name=sku_cap)

    # public_network_access uses documented values "Enabled"|"Disabled"
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

    # Add policies only for non-Basic to avoid preview validation conflicts when Basic is requested.
    if sku_cap != "Basic":
        registry_args["policies"] = containerregistry.PoliciesArgs(
            retention_policy=containerregistry.RetentionPolicyArgs(days=int(retention_days), status="Enabled"),
            soft_delete_policy=containerregistry.SoftDeletePolicyArgs(status="Enabled"),
        )

    # Create the Registry resource (Pulumi will translate to the documented REST create).
    registry = containerregistry.Registry("acrRegistry", **registry_args)

    # Return canonical outputs (they are Outputs and safe to export).
    return registry.id, registry.login_server, registry.provisioning_state
