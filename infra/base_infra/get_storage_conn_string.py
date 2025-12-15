import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.storage import StorageManagementClient

# ---- REQUIRED ENV (fail fast) ----
SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP_NAME"]
STORAGE_ACCOUNT = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]

# ---- AUTH ----
credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
client = StorageManagementClient(credential, SUBSCRIPTION_ID)

# ---- FETCH KEYS ----
keys = client.storage_accounts.list_keys(
    RESOURCE_GROUP,
    STORAGE_ACCOUNT
)

key = keys.keys[0].value

# ---- BUILD CONNECTION STRING ----
conn_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT};"
    f"AccountKey={key};"
    f"EndpointSuffix=core.windows.net"
)

print(conn_string)
