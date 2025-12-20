# infra/pulumi_azure/edge.py
from __future__ import annotations
import os
import re
import json
import base64
from typing import Optional, Dict, Any

import pulumi
from pulumi import Output

# pulumi_kubernetes is optional at import-time; only required when creating k8s resources
try:
    import pulumi_kubernetes as k8s
except Exception:
    k8s = None  # type: ignore

# ---- Helper utils ----
def _parse_positive_int(v: str, name: str) -> int:
    try:
        x = int(v)
    except Exception:
        raise RuntimeError(f"Expected integer for {name}")
    if x < 1:
        raise RuntimeError(f"{name} must be >= 1")
    return x

def _parse_port(p: str) -> int:
    try:
        x = int(p)
    except Exception:
        raise RuntimeError("FRONTEND_SERVICE_PORT must be an integer")
    if x < 1 or x > 65535:
        raise RuntimeError("FRONTEND_SERVICE_PORT must be 1..65535")
    return x

def _validate_hostname(h: str) -> None:
    if len(h) > 253:
        raise RuntimeError("FRONTEND_HOSTNAME too long")
    if h.endswith("."):
        h = h[:-1]
    label = r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
    fqdn_re = re.compile(r"^" + label + r"(?:\." + label + r")*$")
    if not fqdn_re.match(h):
        raise RuntimeError("FRONTEND_HOSTNAME is not a valid hostname")

def _validate_credentials_b64(b64: str) -> None:
    try:
        decoded = base64.b64decode(b64.encode("utf-8"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception as e:
        raise RuntimeError("Invalid CLOUDFLARE_TUNNEL_CREDENTIALS_B64: failed to decode/parse JSON: " + str(e))
    if not isinstance(payload, dict):
        raise RuntimeError("CLOUDFLARE_TUNNEL_CREDENTIALS_B64 must decode to a JSON object")
    # accept common keys used by different cloudflared versions
    if not any(k in payload for k in ("TunnelID", "Credentials", "AccountTag", "TunnelSecret", "tunnel_id")):
        # do not strictly fail here; warn via exception message instead
        raise RuntimeError("Decoded credentials JSON does not look like a Cloudflare tunnel credential (missing expected keys)")

# ---- Pulumi component ----
class EdgeComponent(pulumi.ComponentResource):
    """
    EdgeComponent validates environment-driven configuration and optionally deploys
    a cloudflared Deployment/ConfigMap/Secret in the cluster (when credentials are provided).

    Exports:
      frontend_public_url
      edge_mode
      frontend_service_name
      frontend_service_port
      frontend_namespace
      cloudflare_tunnel_name
      cloudflare_creds_provided
      cloudflare_create_k8s (boolean)
    """
    def __init__(self, name: str = "edge", opts: Optional[pulumi.ResourceOptions] = None):
        super().__init__("custom:infra:EdgeComponent", name, None, opts)

        # ---- read and validate environment ----
        mode = (os.getenv("FRONTEND_HOST_MODE") or "cloudflare_quick").strip().lower()
        if mode not in ("cloudflare_quick", "cloudflare_named", "azure_optional"):
            raise RuntimeError("FRONTEND_HOST_MODE must be one of: cloudflare_quick, cloudflare_named, azure_optional")

        service_name = (os.getenv("FRONTEND_SERVICE_NAME") or "frontend-svc").strip()
        if not service_name:
            raise RuntimeError("FRONTEND_SERVICE_NAME is required")
        service_port = _parse_port(os.getenv("FRONTEND_SERVICE_PORT") or "8000")
        namespace = (os.getenv("FRONTEND_NAMESPACE") or "inference").strip()
        if not namespace:
            raise RuntimeError("FRONTEND_NAMESPACE required")

        tunnel_name = (os.getenv("CLOUDFLARE_TUNNEL_NAME") or "").strip()
        creds_b64 = (os.getenv("CLOUDFLARE_TUNNEL_CREDENTIALS_B64") or "").strip()
        api_token = (os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
        account_id = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
        hostname = (os.getenv("FRONTEND_HOSTNAME") or "").strip()
        try_cloudflare_url = (os.getenv("TRY_CLOUDFLARE_URL") or "").strip()
        frontend_public_url = ""
        create_k8s = (os.getenv("CLOUDFLARE_CREATE_K8S") or "false").lower() in ("1", "true", "yes")
        cloudflared_image = os.getenv("CLOUDFLARE_IMAGE", "cloudflare/cloudflared:2025.11.1")
        replicas = _parse_positive_int(os.getenv("CLOUDFLARE_TUNNEL_REPLICAS") or "2", "CLOUDFLARE_TUNNEL_REPLICAS")

        # mode-specific validation
        if mode == "cloudflare_named":
            # named mode requires a hostname and either credentials or API token + account
            if not hostname:
                raise RuntimeError("FRONTEND_HOSTNAME is required for cloudflare_named mode")
            if not tunnel_name:
                raise RuntimeError("CLOUDFLARE_TUNNEL_NAME is required for cloudflare_named mode")
            if not (creds_b64 or (api_token and account_id)):
                raise RuntimeError("Provide CLOUDFLARE_TUNNEL_CREDENTIALS_B64 or CLOUDFLARE_API_TOKEN with CLOUDFLARE_ACCOUNT_ID for cloudflare_named")
            if creds_b64:
                _validate_credentials_b64(creds_b64)
            _validate_hostname(hostname)
            frontend_public_url = "https://" + hostname

        elif mode == "cloudflare_quick":
            # quick mode needs TRY_CLOUDFLARE_URL available (this is the ephemeral public URL from cloudflared)
            if not try_cloudflare_url:
                # allow fallback to environment-provided FRONTEND_PUBLIC_URL, but prefer TRY_CLOUDFLARE_URL
                frontend_public_url = (os.getenv("FRONTEND_PUBLIC_URL") or "").strip()
                if not frontend_public_url:
                    # don't abort: we can still run in dev, but export placeholder
                    frontend_public_url = "REPLACE_WITH_TRYCLOUDFLARE_HOSTNAME"
            else:
                frontend_public_url = try_cloudflare_url

        else:  # azure_optional
            frontend_public_url = (os.getenv("FRONTEND_PUBLIC_URL") or "").strip()
            if not frontend_public_url:
                raise RuntimeError("FRONTEND_PUBLIC_URL required when FRONTEND_HOST_MODE=azure_optional")

        # export core outputs
        pulumi.export("frontend_public_url", frontend_public_url)
        pulumi.export("edge_mode", mode)
        pulumi.export("frontend_service_name", service_name)
        pulumi.export("frontend_service_port", service_port)
        pulumi.export("frontend_namespace", namespace)
        pulumi.export("cloudflare_tunnel_name", tunnel_name or None)
        pulumi.export("cloudflare_creds_provided", bool(creds_b64))
        pulumi.export("cloudflare_create_k8s", create_k8s)

        # Optionally create Kubernetes resources for cloudflared if credentials are present and user asked to create them
        if create_k8s:
            if not creds_b64:
                raise RuntimeError("CLOUDFLARE_CREATE_K8S=true but CLOUDFLARE_TUNNEL_CREDENTIALS_B64 is not provided (or invalid)")
            if k8s is None:
                raise RuntimeError("pulumi_kubernetes is required to create cloudflared k8s resources; install pulumi_kubernetes")

            # prepare values
            cfg_yaml = {
                "tunnel": tunnel_name,
                "credentials-file": "/etc/cloudflared/credentials.json",
                "ingress": [
                    {"hostname": hostname, "service": f"http://{service_name}.{namespace}.svc.cluster.local:{service_port}"},
                    {"service": "http_status:404"},
                ],
                "no-autoupdate": True,
            }
            # Secret (credentials.json) - put base64-decoded bytes into 'data' (credentials.json)
            secret = k8s.core.v1.Secret(
                resource_name=f"cloudflared-creds-{namespace}",
                metadata=k8s.meta.v1.ObjectMetaArgs(name="cloudflared-tunnel-credentials", namespace=namespace),
                type="Opaque",
                data={
                    "credentials.json": creds_b64  # Kubernetes Secret 'data' expects base64-encoded values
                },
                opts=pulumi.ResourceOptions(parent=self),
            )

            # ConfigMap with config.yml
            cm = k8s.core.v1.ConfigMap(
                resource_name=f"cloudflared-cm-{namespace}",
                metadata=k8s.meta.v1.ObjectMetaArgs(name="cloudflared-config", namespace=namespace),
                data={
                    "config.yml": json.dumps(cfg_yaml) if isinstance(cfg_yaml, dict) else str(cfg_yaml)
                },
                opts=pulumi.ResourceOptions(parent=self),
            )

            # ServiceAccount
            sa = k8s.core.v1.ServiceAccount(
                resource_name=f"cloudflared-sa-{namespace}",
                metadata=k8s.meta.v1.ObjectMetaArgs(name="cloudflared-sa", namespace=namespace),
                opts=pulumi.ResourceOptions(parent=self),
            )

            # Deployment
            deploy = k8s.apps.v1.Deployment(
                resource_name=f"cloudflared-deploy-{namespace}",
                metadata=k8s.meta.v1.ObjectMetaArgs(name="cloudflared", namespace=namespace),
                spec=k8s.apps.v1.DeploymentSpecArgs(
                    replicas=replicas,
                    selector=k8s.meta.v1.LabelSelectorArgs(match_labels={"app": "cloudflared"}),
                    template=k8s.core.v1.PodTemplateSpecArgs(
                        metadata=k8s.meta.v1.ObjectMetaArgs(labels={"app": "cloudflared"}),
                        spec=k8s.core.v1.PodSpecArgs(
                            service_account_name="cloudflared-sa",
                            containers=[
                                k8s.core.v1.ContainerArgs(
                                    name="cloudflared",
                                    image=cloudflared_image,
                                    args=["tunnel", "run", tunnel_name, "--config", "/etc/cloudflared/config.yml"],
                                    volume_mounts=[
                                        k8s.core.v1.VolumeMountArgs(name="config-volume", mount_path="/etc/cloudflared"),
                                        k8s.core.v1.VolumeMountArgs(name="creds-volume", mount_path="/etc/cloudflared", read_only=True),
                                    ],
                                    resources=k8s.core.v1.ResourceRequirementsArgs(
                                        limits={"cpu": "200m", "memory": "256Mi"},
                                        requests={"cpu": "50m", "memory": "64Mi"},
                                    ),
                                )
                            ],
                            volumes=[
                                k8s.core.v1.VolumeArgs(name="config-volume", config_map=k8s.core.v1.ConfigMapVolumeSourceArgs(name="cloudflared-config")),
                                k8s.core.v1.VolumeArgs(name="creds-volume", secret=k8s.core.v1.SecretVolumeSourceArgs(secret_name="cloudflared-tunnel-credentials")),
                            ],
                        ),
                    ),
                ),
                opts=pulumi.ResourceOptions(parent=self),
            )

            # Register created resource names as outputs for visibility
            pulumi.export("cloudflared_k8s_secret", secret.metadata["name"])
            pulumi.export("cloudflared_k8s_configmap", cm.metadata["name"])
            pulumi.export("cloudflared_k8s_deployment", deploy.metadata["name"])

        # finish
        self.register_outputs({
            "frontend_public_url": frontend_public_url,
            "edge_mode": mode,
            "frontend_service_name": service_name,
            "frontend_service_port": service_port,
        })


def create_component() -> EdgeComponent:
    return EdgeComponent()


# when run directly for a quick check (pulumi preview style)
if __name__ == "__main__":
    create_component()
