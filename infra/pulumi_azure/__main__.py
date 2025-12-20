import os
import pulumi
import core_network
from auth import AuthComponent
_stack = pulumi.get_stack()
_outs = core_network.outputs()
_required = ("aks_subnet_id","vnet_id","resource_group_name","storage_account_name","blob_container_name")
_missing = [k for k in _required if not _outs.get(k)]
if _missing:
    pulumi.log.error("core_network missing required outputs: "+",".join(_missing))
    raise pulumi.RunError("core_network missing required outputs: "+",".join(_missing))
pulumi.log.info("[__main__] core_network validated; exports available")
supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
args = {"supabase_url": supabase_url,"supabase_anon_key": supabase_anon_key}
auth_component = AuthComponent("auth",args)
pulumi.log.info("[__main__] supabase auth component instantiated; exports available")
