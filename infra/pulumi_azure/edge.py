from __future__ import annotations
import os
import re
import json
import base64
from typing import Optional
import pulumi
class EdgeComponent(pulumi.ComponentResource):
    def __init__(self, name: str = "edge", opts: Optional[pulumi.ResourceOptions] = None):
        super().__init__("custom:infra:EdgeComponent", name, None, opts)
        mode = (os.getenv("FRONTEND_HOST_MODE") or "cloudflare_quick").strip().lower()
        if mode not in ("cloudflare_quick", "cloudflare_named", "azure_optional"):
            raise RuntimeError("FRONTEND_HOST_MODE must be one of: cloudflare_quick, cloudflare_named, azure_optional")
        service_name = (os.getenv("FRONTEND_SERVICE_NAME") or "frontend").strip()
        service_port = self._parse_port(os.getenv("FRONTEND_SERVICE_PORT") or "80")
        namespace = (os.getenv("FRONTEND_NAMESPACE") or "default").strip()
        replicas = self._parse_positive_int(os.getenv("CLOUDFLARE_TUNNEL_REPLICAS") or "2")
        tunnel_name = (os.getenv("CLOUDFLARE_TUNNEL_NAME") or "").strip()
        creds_b64 = (os.getenv("CLOUDFLARE_TUNNEL_CREDENTIALS_B64") or "").strip()
        api_token = (os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
        account_id = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
        hostname = (os.getenv("FRONTEND_HOSTNAME") or "").strip()
        frontend_public_url = ""
        if mode == "cloudflare_named":
            if not hostname:
                raise RuntimeError("FRONTEND_HOSTNAME is required for cloudflare_named mode")
            if not (creds_b64 or (api_token and account_id)):
                raise RuntimeError("Provide CLOUDFLARE_TUNNEL_CREDENTIALS_B64 or CLOUDFLARE_API_TOKEN with CLOUDFLARE_ACCOUNT_ID for cloudflare_named")
            if creds_b64:
                self._validate_credentials_b64(creds_b64)
            self._validate_hostname(hostname)
            frontend_public_url = "https://" + hostname
        elif mode == "cloudflare_quick":
            try_url = (os.getenv("TRY_CLOUDFLARE_URL") or "").strip()
            frontend_public_url = try_url if try_url else "REPLACE_WITH_TRYCLOUDFLARE_HOSTNAME"
        else:
            frontend_public_url = (os.getenv("FRONTEND_PUBLIC_URL") or "").strip()
            if not frontend_public_url:
                raise RuntimeError("FRONTEND_PUBLIC_URL required when FRONTEND_HOST_MODE=azure_optional")
        pulumi.export("frontend_public_url", frontend_public_url)
        pulumi.export("edge_mode", mode)
        pulumi.export("frontend_service_name", service_name)
        pulumi.export("frontend_service_port", service_port)
        pulumi.export("frontend_namespace", namespace)
        pulumi.export("cloudflare_tunnel_name", tunnel_name)
        pulumi.export("cloudflare_creds_provided", bool(creds_b64))
        self.register_outputs({"frontend_public_url": frontend_public_url, "edge_mode": mode})
    def _parse_positive_int(self, v: str) -> int:
        try:
            x = int(v)
        except Exception:
            raise RuntimeError("Expected integer for replica/count value")
        if x < 1:
            raise RuntimeError("Integer value must be >= 1")
        return x
    def _parse_port(self, p: str) -> int:
        try:
            x = int(p)
        except Exception:
            raise RuntimeError("FRONTEND_SERVICE_PORT must be an integer")
        if x < 1 or x > 65535:
            raise RuntimeError("FRONTEND_SERVICE_PORT must be 1..65535")
        return x
    def _validate_credentials_b64(self, b64: str) -> None:
        try:
            decoded = base64.b64decode(b64.encode("utf-8"))
            payload = json.loads(decoded.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("CLOUDFLARE_TUNNEL_CREDENTIALS_B64 must decode to a JSON object")
            if "TunnelID" not in payload and "Credentials" not in payload and "AccountTag" not in payload:
                pass
        except Exception as e:
            raise RuntimeError("Invalid CLOUDFLARE_TUNNEL_CREDENTIALS_B64: " + str(e))
    def _validate_hostname(self, h: str) -> None:
        if len(h) > 253:
            raise RuntimeError("FRONTEND_HOSTNAME too long")
        if h.endswith("."):
            h = h[:-1]
        label = r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
        fqdn_re = re.compile(r"^" + label + r"(?:\." + label + r")*$")
        if not fqdn_re.match(h):
            raise RuntimeError("FRONTEND_HOSTNAME is not a valid hostname")
def create_component() -> EdgeComponent:
    return EdgeComponent()
if __name__ == "__main__":
    create_component()
