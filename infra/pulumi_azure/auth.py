from __future__ import annotations
import os
import uuid
from typing import Optional, List, Dict, Any
import pulumi
from pulumi import ResourceOptions, Output
import pulumi_azuread as azuread

class AuthArgs:
    def __init__(self,
                 prefix: str = "rag",
                 create_spa: bool = True,
                 spa_redirect_uris: Optional[List[str]] = None,
                 spa_existing_client_id: Optional[str] = None,
                 create_api: bool = True,
                 api_existing_client_id: Optional[str] = None,
                 create_api_client_secret: bool = False,
                 owners: Optional[List[str]] = None,
                 tenant_id: Optional[str] = None,
                 use_b2c: bool = False,
                 b2c_tenant: Optional[str] = None,
                 b2c_policy: Optional[str] = None):
        self.prefix = prefix
        self.create_spa = bool(create_spa)
        self.spa_redirect_uris = spa_redirect_uris or []
        self.spa_existing_client_id = spa_existing_client_id
        self.create_api = bool(create_api)
        self.api_existing_client_id = api_existing_client_id
        self.create_api_client_secret = bool(create_api_client_secret)
        self.owners = owners or []
        self.tenant_id = tenant_id or os.environ.get("AZURE_TENANT_ID")
        self.use_b2c = bool(use_b2c)
        self.b2c_tenant = b2c_tenant
        self.b2c_policy = b2c_policy
        if not self.tenant_id:
            raise RuntimeError("AZURE_TENANT_ID (tenant_id) is required")
        if self.use_b2c and not (self.b2c_tenant and self.b2c_policy):
            pulumi.log.info("[auth] B2C mode enabled but B2C_TENANT or B2C_POLICY missing; expecting existing user flow/policy")

class AuthComponent(pulumi.ComponentResource):
    def __init__(self, name: str, args: AuthArgs, opts: Optional[ResourceOptions] = None):
        super().__init__("custom:infra:AuthComponent", name, None, opts)
        notes: Dict[str, Any] = {}
        self.spa_client_id: Optional[Output[str]] = None
        self.api_client_id: Optional[Output[str]] = None
        self.api_client_secret: Optional[Output[str]] = None
        self.oidc_issuer: Optional[str] = None
        if args.spa_existing_client_id:
            existing_spa = self._lookup_application(args.spa_existing_client_id)
            self.spa_client_id = Output.from_input(existing_spa)
            notes["spa_imported"] = True
        elif args.create_spa:
            spa_app = azuread.Application(f"{args.prefix}-spa",display_name=f"{args.prefix}-spa",owners=args.owners or None,opts=ResourceOptions(parent=self))
            sp_sp = azuread.ServicePrincipal(f"{args.prefix}-spa-sp",application_id=spa_app.application_id,opts=ResourceOptions(parent=self))
            if args.spa_redirect_uris:
                azuread.ApplicationRedirectUris(f"{args.prefix}-spa-redirects",application_id=spa_app.application_id,redirect_uris=args.spa_redirect_uris,type="Spa",opts=ResourceOptions(parent=self,depends_on=[spa_app,sp_sp]))
            self.spa_client_id = spa_app.application_id
            notes["spa_created"] = True
        if args.api_existing_client_id:
            existing_api = self._lookup_application(args.api_existing_client_id)
            self.api_client_id = Output.from_input(existing_api)
            notes["api_imported"] = True
        elif args.create_api:
            api_app = azuread.Application(f"{args.prefix}-api",display_name=args.api_display_name if hasattr(args,'api_display_name') else f"{args.prefix}-api",owners=args.owners or None,opts=ResourceOptions(parent=self))
            api_sp = azuread.ServicePrincipal(f"{args.prefix}-api-sp",application_id=api_app.application_id,opts=ResourceOptions(parent=self))
            self.api_client_id = api_app.application_id
            notes["api_created"] = True
            if args.create_api_client_secret:
                sp_secret = azuread.ServicePrincipalPassword(f"{args.prefix}-api-secret",service_principal_id=api_sp.id,end_date="2299-12-31T00:00:00Z",opts=ResourceOptions(parent=self))
                self.api_client_secret = Output.secret(sp_secret.value)
        if args.use_b2c and args.b2c_tenant and args.b2c_policy:
            issuer = f"https://{args.b2c_tenant}.b2clogin.com/{args.b2c_tenant}.onmicrosoft.com/{args.b2c_policy}/v2.0"
            self.oidc_issuer = issuer
            notes["oidc_mode"] = "b2c"
        else:
            issuer = f"https://login.microsoftonline.com/{args.tenant_id}/v2.0"
            self.oidc_issuer = issuer
            notes["oidc_mode"] = "azuread"
        if self.spa_client_id:
            pulumi.export(f"{args.prefix}_spa_client_id",self.spa_client_id)
        if self.api_client_id:
            pulumi.export(f"{args.prefix}_api_client_id",self.api_client_id)
        if self.api_client_secret:
            pulumi.export(f"{args.prefix}_api_client_secret",self.api_client_secret)
        if self.oidc_issuer:
            pulumi.export(f"{args.prefix}_oidc_issuer",self.oidc_issuer)
        pulumi.export(f"{args.prefix}_notes",notes)
        self.register_outputs({"spa_client_id":self.spa_client_id,"api_client_id":self.api_client_id,"api_client_secret":self.api_client_secret,"oidc_issuer":self.oidc_issuer,"notes":notes})
    def _lookup_application(self, client_id: str) -> str:
        try:
            uuid.UUID(client_id)
        except Exception:
            raise RuntimeError("existing client id must be a valid UUID")
        try:
            app = azuread.get_application(client_id=client_id)
        except Exception as e:
            raise RuntimeError(f"azuread.get_application failed for client_id {client_id}: {e}")
        val = getattr(app,"application_id",None) or getattr(app,"appId",None) or getattr(app,"client_id",None)
        if not val:
            # as a fallback try common properties returned in different provider versions
            for k in ("application_id","appId","client_id","clientId","app_id"):
                v = getattr(app,k,None)
                if v:
                    val = v
                    break
        if not val:
            raise RuntimeError("Unable to determine client id from get_application result")
        return val
