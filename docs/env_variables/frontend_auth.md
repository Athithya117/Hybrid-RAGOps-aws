## **Environment Variables for the Frontend + OIDC Auth Layer**

This document describes **all environment variables needed by the SPA (frontend UI) and the frontend gateway service (`frontend_and_auth.py`)** that handles authentication, token validation, and communication with the backend Query API.

It covers **both Azure Entra ID (internal auth)** and **Azure External ID / B2C (external consumer auth)** — both are supported with **no backend database** and **pure stateless OIDC validation**.

---

# 1. Overview

Your frontend layer consists of:

1. **SPA (browser)**

   * Initiates login via OIDC
   * Receives ID Token
   * Calls backend with `Authorization: Bearer <id_token>`

2. **Frontend Auth Gateway (`frontend_and_auth.py`)**

   * Validates ID tokens
   * Uses JWKS (JSON Web Key Set) from Azure for verification
   * Passes safe requests to backend Query Service

Everything is driven by **environment variables**, making the deployment multi-environment and multi-tenant friendly.

---

# 2. Required Environment Variables

These are **mandatory** for any real deployment.

---

## **2.1 OIDC_ISSUER**

```
export OIDC_ISSUER="<OIDC_ISSUER_URL>"
```

### What it is

The URL of your identity provider (IdP).
This tells the frontend backend which authority issued the tokens.

### Two possible modes

#### **Internal Entra ID**

```
https://login.microsoftonline.com/<TENANT_ID>/v2.0
```

#### **External ID / B2C**

```
https://<tenant>.b2clogin.com/<tenant>.onmicrosoft.com/<user-flow-or-policy>/v2.0
```

### When you change it

* Switching environments (dev → staging → prod)
* Switching between internal and external identity
* Changing B2C user flows/policies (e.g., signup vs login)

---

## **2.2 OIDC_AUDIENCE**

```
export OIDC_AUDIENCE="<API_CLIENT_ID_OR_APP_ID_URI>"
```

### What it is

Your backend API’s identifier.
The backend only accepts tokens where:

```
aud == OIDC_AUDIENCE
```

### Where to get it

Azure Portal → App Registrations → API App →

* Application (client) ID **or**
* Expose an API → Application ID URI

### When to change

* New API registration
* Multi-tenant deployments
* Renaming the App ID URI

---

## **2.3 SPA_CLIENT_ID**

```
export SPA_CLIENT_ID="<PUBLIC_CLIENT_ID>"
```

### What it is

The frontend SPA’s application ID (public client).
Used by the browser during login.

### When to change

* New SPA App Registration
* Separate client IDs per environment/customer

---

## **2.4 QUERY_URL**

```
export QUERY_URL="https://<backend-query-service>"
```

### What it is

URL of the backend RAG Query Service that performs `/generate`.

### Examples

* Dev:

  ```
  http://localhost:8080
  ```
* Production behind Front Door:

  ```
  https://api.rag.example.com
  ```

### When to change

* Switching dev/staging/prod backends
* When routing API behind Front Door
* When moving to private AKS internal DNS

---

# 3. Optional Environment Variables

---

## **3.1 ENABLE_CORS**

```
export ENABLE_CORS="true|false"
```

### What it does

Controls CORS headers in backend responses.

### Recommended values

* **false** for production (SPA and API share a domain via Front Door)
* **true** for local dev or staging

---

## **3.2 CORS_ALLOWED_ORIGINS**

```
export CORS_ALLOWED_ORIGINS="https://site.example.com"
```

### What it does

Comma-separated list of allowed origins.

### When to set

Only needed when `ENABLE_CORS="true"`.

---

## **3.3 JWKS_REFRESH_INTERVAL_SECONDS**

```
export JWKS_REFRESH_INTERVAL_SECONDS="900"
```

### What it is

How often the backend refreshes Azure JWKS keys.

### Default

`900` seconds (15 minutes).

### When to tune

* Lower for fast-rotating enterprise keys
* Higher for performance-sensitive production API

---

# 4. Two Stateless Auth Models Supported

Your platform supports **both** Azure-native stateless auth systems with **zero database**:

| Auth Type             | Use Case                 | Needs DB? | Stateless? |
| --------------------- | ------------------------ | --------- | ---------- |
| **Internal Entra ID** | Employees / internal ops | ❌ No      | ✔ Yes      |
| **External ID / B2C** | Public users             | ❌ No      | ✔ Yes      |

Token validation is identical — only the `OIDC_ISSUER` + `OIDC_AUDIENCE` differ.

---

# 5. Environment Examples

## **5.1 Internal Entra ID — Dev**

```bash
export OIDC_ISSUER="https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/v2.0"
export OIDC_AUDIENCE="82c2c1cf-2102-443d-b987-e7dc44fb0d2e"
export SPA_CLIENT_ID="abcd1234-dev-spa"
export QUERY_URL="http://localhost:8080"
export ENABLE_CORS="true"
export CORS_ALLOWED_ORIGINS="http://localhost:5173"
export JWKS_REFRESH_INTERVAL_SECONDS="300"
```

---

## **5.2 Internal Entra ID — Production**

```bash
export OIDC_ISSUER="https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/v2.0"
export OIDC_AUDIENCE="api://rag-prod-api"
export SPA_CLIENT_ID="prod-spa-client-id"
export QUERY_URL="https://query.rag.internal"
export ENABLE_CORS="false"
export JWKS_REFRESH_INTERVAL_SECONDS="900"
```

---

## **5.3 External ID — Dev (B2C Test Flow)**

```bash
export OIDC_ISSUER="https://yourtenant.b2clogin.com/yourtenant.onmicrosoft.com/B2C_1_signupsignin/v2.0"
export OIDC_AUDIENCE="api://b2c-dev-api"
export SPA_CLIENT_ID="b2c-dev-spa"
export QUERY_URL="http://localhost:8080"
export ENABLE_CORS="true"
export CORS_ALLOWED_ORIGINS="http://localhost:5173"
export JWKS_REFRESH_INTERVAL_SECONDS="300"
```

---

## **5.4 External ID — Production**

```bash
export OIDC_ISSUER="https://yourtenant.b2clogin.com/yourtenant.onmicrosoft.com/B2C_1_signin_prod/v2.0"
export OIDC_AUDIENCE="api://rag-prod-api"
export SPA_CLIENT_ID="prod-spa-b2c"
export QUERY_URL="https://api.rag.example.com"
export ENABLE_CORS="false"
export JWKS_REFRESH_INTERVAL_SECONDS="900"
```

---

# 6. Flow Diagram

```
User Browser (SPA)
    ↓ OIDC Login (Entra ID / External ID)
Azure Auth Endpoint
    ↓ ID Token (JWT)
SPA stores token
    ↓ Authorization: Bearer <id_token>
Frontend Auth Service (frontend_and_auth.py)
    ↓ Validates token (issuer, aud, signature from JWKS)
Forwards to Query Service
```

No DB. No sessions. No cookies.
Pure **stateless token verification**.

---
