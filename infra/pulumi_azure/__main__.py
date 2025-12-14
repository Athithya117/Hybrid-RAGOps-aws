import pulumi
import sys
import core_network
from edge import EdgeComponent
_required = ("aks_subnet_id", "vnet_id", "resource_group_name", "storage_account_name", "blob_container_name")
_outs = core_network.outputs()
_missing = [k for k in _required if not _outs.get(k)]
if _missing:
    pulumi.log.error("core_network missing required outputs: " + ",".join(_missing))
    raise SystemExit(1)
pulumi.log.info("[__main__] core_network validated; exports available")
try:
    edge = EdgeComponent()
    pulumi.log.info("[__main__] EdgeComponent instantiated; exports available")
except Exception as e:
    pulumi.log.error("[__main__] failed to instantiate EdgeComponent: " + str(e))
    raise
