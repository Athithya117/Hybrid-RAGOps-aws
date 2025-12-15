
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


```bash
export USE_B2C="true"
export B2C_TENANT="${B2C_TENANT:?B2C_TENANT must already be exported}"
export B2C_POLICY="${B2C_POLICY:?B2C_POLICY must already be exported}"
export SPA_EXISTING_CLIENT_ID="${SPA_EXISTING_CLIENT_ID:?SPA_EXISTING_CLIENT_ID must already be exported}"
export API_EXISTING_CLIENT_ID="${API_EXISTING_CLIENT_ID:?API_EXISTING_CLIENT_ID must already be exported}"
```

# Microsoft Entra External ID (Customer identity) — Correct, Current Flow
# Azure AD B2C is now Microsoft Entra External ID (customer identity)
# External ID uses a separate tenant (directory) from your workforce tenant
# Applications must be registered inside the External ID tenant
# Infrastructure tools should import existing apps, not create them
# Step 1: Create a Microsoft Entra External ID tenant (customer identity)
# Docs: [https://learn.microsoft.com/entra/external-id/customers/how-to-create-external-tenant](https://learn.microsoft.com/entra/external-id/customers/how-to-create-external-tenant)
# Step 2: Switch directory to the External ID tenant in Azure Portal
# Docs: [https://learn.microsoft.com/entra/external-id/customers/concept-tenant](https://learn.microsoft.com/entra/external-id/customers/concept-tenant)
# Step 3: Create a User Flow (Sign up / Sign in), e.g. B2C_1_signupsignin
# Docs: [https://learn.microsoft.com/entra/external-id/customers/how-to-user-flow-sign-up-sign-in](https://learn.microsoft.com/entra/external-id/customers/how-to-user-flow-sign-up-sign-in)
# Step 4: Register SPA and API applications inside the External ID tenant
# Record both Application (client) IDs
# Docs: [https://learn.microsoft.com/entra/external-id/customers/how-to-register-applications](https://learn.microsoft.com/entra/external-id/customers/how-to-register-applications)
# Step 5: Use the registered app IDs and user-flow details when running Pulumi
# Pulumi imports and wires these apps; it does not create them cross-tenant
# Mode 3 — Microsoft Entra External ID (formerly Azure AD B2C)
# USE_B2C=true enables External ID mode
# B2C_TENANT is the External ID tenant name (without .onmicrosoft.com)
# B2C_POLICY is the User Flow name
# SPA_EXISTING_CLIENT_ID and API_EXISTING_CLIENT_ID must come from that tenant


