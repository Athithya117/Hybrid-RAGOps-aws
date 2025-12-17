import os
import pulumi
import core_network
import auth
_stack = pulumi.get_stack()
_outs = core_network.outputs()
_required = ("aks_subnet_id", "vnet_id", "resource_group_name", "storage_account_name", "blob_container_name")
_missing = [k for k in _required if not _outs.get(k)]
if _missing:
    pulumi.log.error("core_network missing required outputs: " + ",".join(_missing))
    raise SystemExit(1)
pulumi.log.info("[__main__] core_network validated; exports available")
env = {k: os.getenv(k) for k in ("AUTH_MODE","USE_B2C","SPA_EXISTING_CLIENT_ID","API_EXISTING_CLIENT_ID","CREATE_API_CLIENT_SECRET","SPA_REDIRECT_URIS","B2C_TENANT","B2C_POLICY","RESOURCE_NAME_PREFIX")}
auth_mode = (env.get("AUTH_MODE") or "").strip().lower()
if not auth_mode:
    if os.getenv("USE_B2C","").lower() in ("1","true","yes"):
        auth_mode = "external-id"
    else:
        auth_mode = "azuread"
if auth_mode not in ("azuread","import","external-id"):
    pulumi.log.error("AUTH_MODE must be one of: azuread, import, external-id")
    raise SystemExit(1)
try:
    args = {
        "prefix": (env.get("RESOURCE_NAME_PREFIX") or "rag").strip(),
        "auth_mode": auth_mode,
        "spa_existing_client_id": (env.get("SPA_EXISTING_CLIENT_ID") or "").strip() or None,
        "api_existing_client_id": (env.get("API_EXISTING_CLIENT_ID") or "").strip() or None,
        "create_api_client_secret": (os.getenv("CREATE_API_CLIENT_SECRET","false").lower() in ("1","true","yes")),
        "spa_redirect_uris": [u.strip() for u in ((os.getenv("SPA_REDIRECT_URIS") or "").split(",")) if u.strip()],
        "b2c_tenant": (os.getenv("B2C_TENANT") or "").strip() or None,
        "b2c_policy": (os.getenv("B2C_POLICY") or "").strip() or None,
        "tenant_id": os.getenv("AZURE_TENANT_ID") or None
    }
    auth_component = auth.AuthComponent("auth", args)
    pulumi.log.info("[__main__] auth component instantiated; exports available")
except Exception as e:
    pulumi.log.error("[__main__] auth initialization failed:" + str(e))
    raise
