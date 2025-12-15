from __future__ import annotations
import os
import uuid
from typing import Optional, List, Dict, Any
import pulumi
from pulumi import ResourceOptions, Output
import pulumi_azuread as azuread
def _env(k: str, d: Optional[str] = None) -> Optional[str]:
    return os.environ.get(k) or d
def _is_uuid(v: str) -> bool:
    try:
        uuid.UUID(str(v))
        return True
    except Exception:
        return False
class AuthArgs:
    def __init__(self, prefix: str = "rag", create_spa: bool = True, spa_redirect_uris: Optional[List[str]] = None, spa_existing_client_id: Optional[str] = None, create_api: bool = True, api_display_name: str = "rag-api",
                  api_existing_client_id: Optional[str] = None, create_api_client_secret: bool = False, owners: Optional[List[str]] = None,
                    tenant_id: Optional[str] = None, use_b2c: bool = False, b2c_tenant: Optional[str] = None, b2c_policy: Optional[str] = None):
        self.prefix = prefix
        self.create_spa = create_spa
        self.spa_redirect_uris = spa_redirect_uris or []
        self.spa_existing_client_id = spa_existing_client_id
        self.create_api = create_api
        self.api_display_name = api_display_name
        self.api_existing_client_id = api_existing_client_id
        self.create_api_client_secret = create_api_client_secret
        self.owners = owners or []
        self.tenant_id = tenant_id or _env("AZURE_TENANT_ID")
        self.use_b2c = bool(use_b2c)
        self.b2c_tenant = b2c_tenant
        self.b2c_policy = b2c_policy
class AuthComponent(pulumi.ComponentResource):
    def __init__(self, name: str, args: AuthArgs, opts: Optional[ResourceOptions] = None):
        super().__init__("custom:infra:AuthComponent", name, None, opts)
        if not args.tenant_id and not args.use_b2c:
            raise pulumi.ResourceError("AZURE_TENANT_ID is required for non-B2C flows", self)
        if args.use_b2c:
            if not args.b2c_tenant or not args.b2c_policy:
                raise pulumi.ResourceError("B2C mode requires B2C_TENANT and B2C_POLICY to be set", self)
            if not args.spa_existing_client_id or not args.api_existing_client_id:
                raise pulumi.ResourceError("B2C mode requires pre-registered SPA and API client IDs in the B2C tenant; set SPA_EXISTING_CLIENT_ID and API_EXISTING_CLIENT_ID", self)
            if not (_is_uuid(args.spa_existing_client_id) and _is_uuid(args.api_existing_client_id)):
                raise pulumi.ResourceError("SPA_EXISTING_CLIENT_ID and API_EXISTING_CLIENT_ID must be valid UUIDs", self)
        self.spa_client_id = None
        self.api_client_id = None
        self.api_app_id_uri = None
        self.api_client_secret = None
        notes: Dict[str, Any] = {}
        if args.use_b2c:
            self.spa_client_id = Output.from_input(args.spa_existing_client_id)
            self.api_client_id = Output.from_input(args.api_existing_client_id)
            self.api_app_id_uri = Output.from_input(f"api://{args.api_existing_client_id}")
            tenant_host = args.b2c_tenant if "." in args.b2c_tenant else f"{args.b2c_tenant}.onmicrosoft.com"
            sub = tenant_host.split(".")[0]
            issuer = f"https://{sub}.b2clogin.com/{tenant_host}/{args.b2c_policy}/v2.0"
            openid = issuer + "/.well-known/openid-configuration"
            pulumi.export(f"{args.prefix}_oidc_issuer", issuer)
            pulumi.export(f"{args.prefix}_openid_configuration", openid)
            notes["oidc_mode"] = "b2c"
            notes["spa_imported"] = True
            notes["api_imported"] = True
        else:
            if args.api_existing_client_id:
                if not _is_uuid(args.api_existing_client_id):
                    raise pulumi.ResourceError("API_EXISTING_CLIENT_ID must be a valid UUID", self)
                existing_api = azuread.get_application(client_id=args.api_existing_client_id)
                self.api_client_id = Output.from_input(existing_api.client_id)
                self.api_app_id_uri = Output.from_input(f"api://{existing_api.client_id}")
                notes["api_imported"] = True
            elif args.create_api:
                api_app = azuread.Application(f"{args.prefix}-api", display_name=args.api_display_name, owners=args.owners or None, opts=ResourceOptions(parent=self))
                sp = azuread.ServicePrincipal(f"{args.prefix}-api-sp", application_id=api_app.application_id, opts=ResourceOptions(parent=self))
                client_id = getattr(api_app, "application_id", None) or getattr(api_app, "app_id", None) or getattr(api_app, "client_id", None)
                self.api_client_id = client_id
                self.api_app_id_uri = Output.all(client_id).apply(lambda args: f"api://{args[0]}")
                notes["api_created"] = True
                if args.create_api_client_secret:
                    sp_secret = azuread.ServicePrincipalPassword(f"{args.prefix}-api-secret", service_principal_id=sp.id, end_date="2299-12-31T00:00:00Z", opts=ResourceOptions(parent=self))
                    self.api_client_secret = Output.secret(sp_secret.value)
            if args.spa_existing_client_id:
                if not _is_uuid(args.spa_existing_client_id):
                    raise pulumi.ResourceError("SPA_EXISTING_CLIENT_ID must be a valid UUID", self)
                existing_spa = azuread.get_application(client_id=args.spa_existing_client_id)
                self.spa_client_id = Output.from_input(existing_spa.client_id)
                notes["spa_imported"] = True
            elif args.create_spa:
                spa_app = azuread.Application(f"{args.prefix}-spa", display_name=f"{args.prefix}-spa", owners=args.owners or None, opts=ResourceOptions(parent=self))
                sp_sp = azuread.ServicePrincipal(f"{args.prefix}-spa-sp", application_id=spa_app.application_id, opts=ResourceOptions(parent=self))
                client_id = getattr(spa_app, "application_id", None) or getattr(spa_app, "app_id", None) or getattr(spa_app, "client_id", None)
                self.spa_client_id = client_id
                notes["spa_created"] = True
                if args.spa_redirect_uris:
                    notes["spa_redirects_provided"] = args.spa_redirect_uris
        if self.spa_client_id:
            pulumi.export(f"{args.prefix}_spa_client_id", self.spa_client_id)
        if self.api_client_id:
            pulumi.export(f"{args.prefix}_api_client_id", self.api_client_id)
        if self.api_app_id_uri:
            pulumi.export(f"{args.prefix}_api_app_id_uri", self.api_app_id_uri)
        if self.api_client_secret:
            pulumi.export(f"{args.prefix}_api_client_secret", self.api_client_secret)
        pulumi.export(f"{args.prefix}_notes", notes)
        issuer = None
        if not args.use_b2c:
            issuer = f"https://login.microsoftonline.com/{args.tenant_id}/v2.0"
            pulumi.export(f"{args.prefix}_oidc_issuer", issuer)
            pulumi.export(f"{args.prefix}_openid_configuration", issuer + "/.well-known/openid-configuration")
        self.register_outputs({"spa_client_id": self.spa_client_id,"api_client_id": self.api_client_id,
                               "api_app_id_uri": self.api_app_id_uri,"api_client_secret": self.api_client_secret,"oidc_issuer": issuer,"notes": notes})
