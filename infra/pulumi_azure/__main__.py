import pulumi

from core_network import (
    ensure_resource_group,
    ensure_storage_account,
    ensure_blob_container,
    create_virtual_network,
)
from aks import create_aks_cluster
from auth import AuthComponent, AuthArgs

# ----------------------------
# 1. Core Network
# ----------------------------

# Create or lookup RG
rg = ensure_resource_group()
resource_group_name = rg.name if hasattr(rg, "name") else rg.resource_group_name

# Storage
sa = ensure_storage_account(resource_group_name)
storage_account_name = sa.name if hasattr(sa, "name") else sa.account_name

container = ensure_blob_container(resource_group_name, storage_account_name)

# VNet + subnets
net = create_virtual_network(resource_group_name)
aks_subnet_id = net["aks_subnet_id"]

# ----------------------------
# 2. AKS Cluster
# ----------------------------
aks_cluster = create_aks_cluster(
    resource_group_name=resource_group_name,
    aks_subnet_id=aks_subnet_id,
)

# ----------------------------
# 3. Auth (Azure AD or B2C)
# ----------------------------
auth = AuthComponent(
    "auth",
    AuthArgs(
        prefix="rag",
        create_spa=True,
        spa_redirect_uris=["http://localhost:3000"],
        create_api=True,
        api_display_name="rag-api",
        create_api_client_secret=True,
    )
)

# Exports
pulumi.export("resource_group", resource_group_name)
pulumi.export("storage_account", storage_account_name)
pulumi.export("aks_cluster_name", aks_cluster["cluster"].name)
pulumi.export("spa_client_id", auth.spa_client_id)
pulumi.export("api_client_id", auth.api_client_id)
