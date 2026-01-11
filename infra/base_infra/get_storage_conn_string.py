import os
import sys

from azure.identity import AzureCliCredential
from azure.mgmt.storage import StorageManagementClient


def require_env(name: str) -> str:
    """
    Fetch required environment variable or fail fast with a clear error.
    """
    value = os.environ.get(name)
    if not value:
        print(f"FATAL: required environment variable '{name}' is not set", file=sys.stderr)
        sys.exit(1)
    return value


# -----------------------------------------------------------------------------
# REQUIRED ENV — strictly aligned with STEP 1 / STEP 2
# -----------------------------------------------------------------------------
SUBSCRIPTION_ID = require_env("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = require_env("AZURE_RESOURCE_GROUP_NAME")
STORAGE_ACCOUNT = require_env("AZURE_STORAGE_ACCOUNT_NAME")

# Optional but supported (sovereign clouds)
ENDPOINT_SUFFIX = os.environ.get("AZURE_ENDPOINT_SUFFIX", "core.windows.net")

# -----------------------------------------------------------------------------
# AUTH — deterministic, no guessing
# -----------------------------------------------------------------------------
# You are already using:
#   az login
#   az account set --subscription ...
#
# Therefore AzureCliCredential is the *correct* and *noise-free* choice.
credential = AzureCliCredential()

client = StorageManagementClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
)

# -----------------------------------------------------------------------------
# FETCH STORAGE ACCOUNT KEYS (fail hard if unavailable)
# -----------------------------------------------------------------------------
try:
    keys = client.storage_accounts.list_keys(
        RESOURCE_GROUP,
        STORAGE_ACCOUNT,
    )
except Exception as e:
    print(
        f"FATAL: failed to list keys for storage account '{STORAGE_ACCOUNT}' "
        f"in resource group '{RESOURCE_GROUP}': {e}",
        file=sys.stderr,
    )
    sys.exit(1)

if not keys.keys or not keys.keys[0].value:
    print(
        f"FATAL: no storage account keys returned for '{STORAGE_ACCOUNT}'",
        file=sys.stderr,
    )
    sys.exit(1)

account_key = keys.keys[0].value

# -----------------------------------------------------------------------------
# BUILD CONNECTION STRING (Azure-documented format)
# -----------------------------------------------------------------------------
connection_string = (
    "DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT};"
    f"AccountKey={account_key};"
    f"EndpointSuffix={ENDPOINT_SUFFIX}"
)

print(connection_string)
