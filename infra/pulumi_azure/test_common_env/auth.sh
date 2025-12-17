
# Non-B2C — create SPA + API (create client secret)

```bash
export AZURE_TENANT_ID=$AZURE_TENANT_ID
export USE_B2C="false"
export SPA_REDIRECT_URIS="https://example.com/auth/callback"
export CREATE_API_CLIENT_SECRET="true"
```

# Non-B2C — import existing SPA & API 

az ad app create --display-name test-spa --sign-in-audience AzureADMyOrg
az ad app create --display-name test-api --sign-in-audience AzureADMyOrg

```bash
export USE_B2C="false"
export SPA_EXISTING_CLIENT_ID="b70b7f71-652e-4c2d-8535-59c84b322ee8"
export API_EXISTING_CLIENT_ID="6888bcb6-6c36-4ddc-98b7-901d3e59b83e"

```

# Mode 3 — Microsoft Entra External ID (Customer Identity)
# NOTE: External ID apps MUST already exist in the External ID tenant

export USE_B2C="true"

# External ID (customer identity) tenant name (without .onmicrosoft.com)
export B2C_TENANT="${B2C_TENANT:?B2C_TENANT (External ID tenant name) must already be exported}"

# User Flow / Policy name (e.g. B2C_1_signupsignin)
export B2C_POLICY="${B2C_POLICY:?B2C_POLICY (External ID user flow) must already be exported}"

# Application (client) IDs from the External ID tenant
export SPA_EXISTING_CLIENT_ID="${SPA_EXISTING_CLIENT_ID:?SPA_EXISTING_CLIENT_ID must already be exported}"
export API_EXISTING_CLIENT_ID="${API_EXISTING_CLIENT_ID:?API_EXISTING_CLIENT_ID must already be exported}"

