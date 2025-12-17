
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

