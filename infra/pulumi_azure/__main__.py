import os
import sys
import pulumi
import core_network
import auth
_required = ("aks_subnet_id", "vnet_id", "resource_group_name", "storage_account_name", "blob_container_name")
_outs = core_network.outputs()
_missing = [k for k in _required if not _outs.get(k)]
if _missing:
    pulumi.log.error("core_network missing required outputs: " + ",".join(_missing))
    raise SystemExit(1)
pulumi.log.info("[__main__] core_network validated; exports available")
env = os.environ
use_b2c = env.get("USE_B2C", "false").lower() in ("1","true","yes")
spa_existing = env.get("SPA_EXISTING_CLIENT_ID") or None
api_existing = env.get("API_EXISTING_CLIENT_ID") or None
create_api_secret = env.get("CREATE_API_CLIENT_SECRET", "false").lower() in ("1","true","yes")
spa_redirects_raw = env.get("SPA_REDIRECT_URIS") or ""
spa_redirects = [u.strip() for u in spa_redirects_raw.split(",") if u.strip()]
args = auth.AuthArgs(prefix=env.get("RESOURCE_NAME_PREFIX","rag"),
                     create_spa=(spa_existing is None),
                     spa_redirect_uris=spa_redirects,
                     spa_existing_client_id=spa_existing,
                     create_api=(api_existing is None),
                     api_existing_client_id=api_existing,
                     create_api_client_secret=create_api_secret,
                     owners=None,
                     tenant_id=env.get("AZURE_TENANT_ID"),
                     use_b2c=use_b2c,
                     b2c_tenant=env.get("B2C_TENANT"),
                     b2c_policy=env.get("B2C_POLICY"))
try:
    auth_component = auth.AuthComponent("auth", args)
    pulumi.log.info("[__main__] auth component instantiated; exports available")
except Exception as e:
    pulumi.log.error("[__main__] auth initialization failed:" + str(e))
    raise
