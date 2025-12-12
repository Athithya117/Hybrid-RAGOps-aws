# auth.py
"""
Auth component for Pulumi (Azure AD / Entra External ID import-only).
 - Creates (or imports) SPA application & API application in Azure AD via pulumi_azuread.
 - Does NOT attempt to create a B2C tenant. If use_b2c=True you MUST provide an existing B2C tenant identifier.
 - Does NOT perform network calls during construction (no JWKS discovery).
 - Exposes: spa_client_id, api_client_id, api_app_id_uri, api_client_secret (secret Output), oidc_issuer (string)
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

import pulumi
from pulumi import ResourceOptions, Output

# azuread provider (must be configured by Pulumi provider block or environment)
import pulumi_azuread as azuread


# -------------------------
# Helpers (fail-fast)
# -------------------------
def _cfg_attr(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        import config as cfg  # type: ignore
        if hasattr(cfg, "cfg") and hasattr(cfg.cfg, name):
            return getattr(cfg.cfg, name)
    except Exception:
        pass
    return os.environ.get(name.upper(), default)


def required_env(name: str) -> str:
    v = _cfg_attr(name.lower())
    if v is None:
        raise RuntimeError(f"Missing required configuration: {name}")
    return v


# -------------------------
# Component
# -------------------------
class AuthArgs:
    def __init__(self,
                 prefix: str = "rag",
                 create_spa: bool = True,
                 spa_redirect_uris: Optional[List[str]] = None,
                 spa_existing_client_id: Optional[str] = None,
                 create_api: bool = True,
                 api_display_name: str = "rag-api",
                 api_existing_client_id: Optional[str] = None,
                 api_app_id_uri: Optional[str] = None,
                 create_api_client_secret: bool = False,
                 owners: Optional[List[str]] = None,
                 tenant_id: Optional[str] = None,
                 use_b2c: bool = False,
                 b2c_tenant: Optional[str] = None,
                 b2c_policy: Optional[str] = None):
        self.prefix = prefix
        self.create_spa = create_spa
        self.spa_redirect_uris = spa_redirect_uris or []
        self.spa_existing_client_id = spa_existing_client_id
        self.create_api = create_api
        self.api_display_name = api_display_name
        self.api_existing_client_id = api_existing_client_id
        self.api_app_id_uri = api_app_id_uri
        self.create_api_client_secret = create_api_client_secret
        self.owners = owners or []
        self.tenant_id = tenant_id or _cfg_attr("azure_tenant_id") or os.environ.get("AZURE_TENANT_ID")
        self.use_b2c = use_b2c
        self.b2c_tenant = b2c_tenant or _cfg_attr("b2c_tenant")
        self.b2c_policy = b2c_policy or _cfg_attr("b2c_policy")


class AuthComponent(pulumi.ComponentResource):
    def __init__(self, name: str, args: AuthArgs, opts: Optional[ResourceOptions] = None):
        super().__init__("custom:infra:AuthComponent", name, None, opts)

        # Validation
        if args.create_spa and not (args.spa_existing_client_id or args.spa_redirect_uris):
            raise pulumi.ResourceError("Either spa_existing_client_id or spa_redirect_uris must be provided when create_spa=True", self)

        if args.use_b2c:
            # We cannot create a B2C tenant here - require pre-existing tenant and policy
            if not args.b2c_tenant:
                raise pulumi.ResourceError("B2C usage requires an existing b2c_tenant; create it outside Pulumi and pass b2c_tenant", self)
            # b2c_policy optional: if you want OIDC issuer for user flows, ensure policy present
            pulumi.log.info("[auth] B2C mode enabled - ensure tenant and user flow/policy exist and are configured manually")

        if not args.tenant_id:
            raise pulumi.ResourceError("tenant_id (AZURE_TENANT_ID) is required", self)

        # Outputs
        self.spa_client_id: Optional[Output[str]] = None
        self.api_client_id: Optional[Output[str]] = None
        self.api_app_id_uri: Optional[Output[str]] = None
        self.api_client_secret: Optional[Output[str]] = None
        self.oidc_issuer: Optional[str] = None

        notes: Dict[str, Any] = {}

        # --- API app ---
        if args.create_api:
            if args.api_existing_client_id:
                pulumi.log.info("[auth] Importing existing API application")
                existing = azuread.get_application(client_id=args.api_existing_client_id)
                self.api_client_id = Output.from_input(existing.application_id)
                # identifier_uris may or may not be present
                if getattr(existing, "identifier_uris", None):
                    self.api_app_id_uri = Output.from_input(existing.identifier_uris[0])
                notes["api_imported"] = True
            else:
                pulumi.log.info("[auth] Creating API application")
                api_app = azuread.Application(
                    f"{args.prefix}-api",
                    display_name=args.api_display_name,
                    owners=args.owners or None,
                    opts=ResourceOptions(parent=self)
                )
                # ServicePrincipal
                sp = azuread.ServicePrincipal(
                    f"{args.prefix}-api-sp",
                    application_id=api_app.application_id,
                    opts=ResourceOptions(parent=self, depends_on=[api_app])
                )

                # Compute default app URI if not provided (api://<appId>)
                app_id_uri = args.api_app_id_uri or api_app.application_id.apply(lambda aid: f"api://{aid}")
                self.api_client_id = api_app.application_id
                self.api_app_id_uri = app_id_uri
                notes["api_created"] = True

                # Optionally create secret (marked as secret Output)
                if args.create_api_client_secret:
                    sp_secret = azuread.ServicePrincipalPassword(
                        f"{args.prefix}-api-secret",
                        service_principal_id=sp.id,
                        end_date="2299-12-31T00:00:00Z",
                        opts=ResourceOptions(parent=self)
                    )
                    # wrap as secret output
                    self.api_client_secret = Output.secret(sp_secret.value)

        else:
            pulumi.log.info("[auth] Skipping API creation (create_api=False)")

        # --- SPA app ---
        if args.create_spa:
            if args.spa_existing_client_id:
                pulumi.log.info("[auth] Importing existing SPA application")
                existing_spa = azuread.get_application(client_id=args.spa_existing_client_id)
                self.spa_client_id = Output.from_input(existing_spa.application_id)
                notes["spa_imported"] = True
            else:
                pulumi.log.info("[auth] Creating SPA application")
                spa_app = azuread.Application(
                    f"{args.prefix}-spa",
                    display_name=f"{args.prefix}-spa",
                    owners=args.owners or None,
                    # Note: provider versions differ on 'public_client' or redirect resources;
                    # we keep minimal to be compatible across provider versions.
                    opts=ResourceOptions(parent=self)
                )
                sp_sp = azuread.ServicePrincipal(
                    f"{args.prefix}-spa-sp",
                    application_id=spa_app.application_id,
                    opts=ResourceOptions(parent=self)
                )
                self.spa_client_id = spa_app.application_id
                notes["spa_created"] = True
                # Setting redirect URIs via provider helper resources is provider-version dependent;
                # instead, document that redirect URIs must be set manually if provider lacks helper resource.

        else:
            pulumi.log.info("[auth] Skipping SPA creation (create_spa=False)")

        # OIDC issuer (do not perform HTTP discovery here)
        if args.use_b2c and args.b2c_tenant and args.b2c_policy:
            # B2C issuer for a particular user flow/policy (the user must ensure policy exists)
            issuer = f"https://{args.b2c_tenant}.b2clogin.com/{args.b2c_tenant}.onmicrosoft.com/{args.b2c_policy}/v2.0"
            self.oidc_issuer = issuer
            notes["oidc_mode"] = "b2c"
        else:
            issuer = f"https://login.microsoftonline.com/{args.tenant_id}/v2.0"
            self.oidc_issuer = issuer
            notes["oidc_mode"] = "azuread"

        # Exports (safe: secrets are exported as secret Outputs)
        if self.spa_client_id:
            pulumi.export(f"{args.prefix}_spa_client_id", self.spa_client_id)
        if self.api_client_id:
            pulumi.export(f"{args.prefix}_api_client_id", self.api_client_id)
        if self.api_app_id_uri:
            pulumi.export(f"{args.prefix}_api_app_id_uri", self.api_app_id_uri)
        if self.api_client_secret:
            pulumi.export(f"{args.prefix}_api_client_secret", self.api_client_secret)
        if self.oidc_issuer:
            pulumi.export(f"{args.prefix}_oidc_issuer", self.oidc_issuer)

        pulumi.export(f"{args.prefix}_notes", notes)

        # Register outputs for programmatic consumption
        self.register_outputs({
            "spa_client_id": self.spa_client_id,
            "api_client_id": self.api_client_id,
            "api_app_id_uri": self.api_app_id_uri,
            "api_client_secret": self.api_client_secret,
            "oidc_issuer": self.oidc_issuer,
            "notes": notes
        })
