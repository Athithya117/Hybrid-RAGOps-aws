from __future__ import annotations
import os, re
from typing import Optional, List, Dict, Any
import pulumi
from pulumi import ResourceOptions, Output
import pulumi_azuread as azuread
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
def _is_uuid(v:Optional[str]) -> bool:
    return bool(v and UUID_RE.match(v.strip()))
def _validate_redirect_uris(list_in:List[str]) -> List[str]:
    out=[]
    for u in list_in:
        u=u.strip()
        if not u:
            continue
        if not (u.startswith("https://") or u.startswith("http://")):
            raise pulumi.ResourceError(f"redirect URI must start with http(s)://: {u}", None)
        out.append(u)
    return out
class AuthComponent(pulumi.ComponentResource):
    def __init__(self,name: str,args: Dict[str,Any],opts: Optional[ResourceOptions]=None):
        super().__init__("custom:infra:AuthComponent", name, None, opts)
        prefix = (args.get("prefix") or "rag").strip()
        stack = pulumi.get_stack()
        auth_mode = (args.get("auth_mode") or "azuread").strip().lower()
        tenant_id = args.get("tenant_id") or os.getenv("AZURE_TENANT_ID")
        spa_existing = args.get("spa_existing_client_id")
        api_existing = args.get("api_existing_client_id")
        create_api_secret = bool(args.get("create_api_client_secret"))
        spa_redirect_uris = _validate_redirect_uris(list(args.get("spa_redirect_uris") or []))
        b2c_tenant = args.get("b2c_tenant")
        b2c_policy = args.get("b2c_policy")
        if auth_mode == "external-id":
            if not b2c_tenant or not b2c_policy:
                raise pulumi.ResourceError("external-id mode requires B2C_TENANT and B2C_POLICY", self)
            if not _is_uuid(spa_existing) or not _is_uuid(api_existing):
                raise pulumi.ResourceError("external-id mode requires SPA_EXISTING_CLIENT_ID and API_EXISTING_CLIENT_ID as UUIDs from the external tenant", self)
        if auth_mode == "import":
            if not _is_uuid(spa_existing) or not _is_uuid(api_existing):
                raise pulumi.ResourceError("import mode requires SPA_EXISTING_CLIENT_ID and API_EXISTING_CLIENT_ID as valid UUIDs", self)
        if auth_mode == "azuread":
            pass
        if not tenant_id:
            raise pulumi.ResourceError("AZURE_TENANT_ID is required", self)
        notes: Dict[str,Any] = {}
        spa_client_id_out: Optional[Output[str]] = None
        api_client_id_out: Optional[Output[str]] = None
        api_client_secret_out: Optional[Output[str]] = None
        oidc_issuer: Optional[str] = None
        if auth_mode in ("import","external-id"):
            try:
                existing_spa = azuread.get_application(client_id=spa_existing)
                spa_client_id_out = Output.from_input(existing_spa.application_id or existing_spa.app_id or existing_spa.appId)
                notes["spa_imported"] = True
            except Exception as e:
                raise pulumi.ResourceError(f"failed to import SPA application by client_id '{spa_existing}': {e}", self)
            try:
                existing_api = azuread.get_application(client_id=api_existing)
                api_client_id_out = Output.from_input(existing_api.application_id or existing_api.app_id or existing_api.appId)
                notes["api_imported"] = True
            except Exception as e:
                raise pulumi.ResourceError(f"failed to import API application by client_id '{api_existing}': {e}", self)
            if auth_mode == "external-id":
                issuer = f"https://{b2c_tenant}.b2clogin.com/{b2c_tenant}.onmicrosoft.com/{b2c_policy}/v2.0"
                oidc_issuer = issuer
                notes["oidc_mode"] = "external-id"
            else:
                oidc_issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
                notes["oidc_mode"] = "azuread-import"
        else:
            api_display = f"{prefix}-api-{stack}"
            spa_display = f"{prefix}-spa-{stack}"
            api_app = azuread.Application(api_display, display_name=api_display, owners=None, opts=ResourceOptions(parent=self))
            api_sp = azuread.ServicePrincipal(f"{prefix}-api-sp-{stack}", application_id=api_app.application_id, opts=ResourceOptions(parent=self, depends_on=[api_app]))
            api_client_id_out = api_app.application_id
            notes["api_created"] = True
            if create_api_secret:
                sp_secret = azuread.ServicePrincipalPassword(f"{prefix}-api-secret-{stack}", service_principal_id=api_sp.id, end_date="2299-12-31T00:00:00Z", opts=ResourceOptions(parent=self))
                api_client_secret_out = Output.secret(sp_secret.value)
            spa_app = azuread.Application(spa_display, display_name=spa_display, owners=None, opts=ResourceOptions(parent=self))
            spa_sp = azuread.ServicePrincipal(f"{prefix}-spa-sp-{stack}", application_id=spa_app.application_id, opts=ResourceOptions(parent=self, depends_on=[spa_app]))
            spa_client_id_out = spa_app.application_id
            notes["spa_created"] = True
            oidc_issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
            notes["oidc_mode"] = "azuread-create"
        if spa_client_id_out:
            pulumi.export(f"{prefix}_spa_client_id", spa_client_id_out)
        if api_client_id_out:
            pulumi.export(f"{prefix}_api_client_id", api_client_id_out)
            if api_client_secret_out:
                pulumi.export(f"{prefix}_api_client_secret", api_client_secret_out)
        if oidc_issuer:
            pulumi.export(f"{prefix}_oidc_issuer", oidc_issuer)
        pulumi.export(f"{prefix}_notes", notes)
        self.register_outputs({
            "spa_client_id": spa_client_id_out,
            "api_client_id": api_client_id_out,
            "api_client_secret": api_client_secret_out,
            "oidc_issuer": oidc_issuer,
            "notes": notes
        })
