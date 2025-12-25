# Microsoft Entra ID Authentication (Our Platform)

This platform uses **stateless OpenID Connect (OIDC)** with **Microsoft Entra ID (Azure AD)** as an identity provider.

Authentication issues a **signed JWT directly to the browser** and does **not** maintain server-side sessions, group lookups, or directory access.

---

## 1. Supported Microsoft Auth Features

| Feature                               | Supported |
|--------------------------------------|-----------|
| Microsoft Entra ID (OIDC v2)          | Yes       |
| Single-tenant apps                   | Yes       |
| Multi-tenant apps                    | Yes       |
| Email-based access control            | Yes       |
| Tenant ID restriction                | Yes       |
| Microsoft personal accounts           | Yes (optional) |
| Group claims / Microsoft Graph        | No        |
| Admin SDK / directory queries         | No        |
| Workload Identity / federated tokens  | No        |

**Design principle**

Authentication is **stateless, minimal, and portable**.  
No Microsoft Graph calls, no group overage handling, no privileged credentials.

---

## 2. Required Environment Variables (Microsoft)

To enable Microsoft login:

```bash
export ENABLE_MICROSOFT_AUTH="true"
export MS_CLIENT_ID="..."
export MS_CLIENT_SECRET="..."
````

These values must come from a **Microsoft Entra ID App Registration**. Refer below for setup

---

### Optional (Recommended)

```bash
export MS_TENANT_ID="common"
```

Valid values:

| Value           | Meaning                                        |
| --------------- | ---------------------------------------------- |
| `common`        | Any Entra tenant + personal Microsoft accounts |
| `<tenant-id>`   | Only a single Entra tenant                     |
| `organizations` | Any Entra tenant (no personal accounts)        |
| `consumers`     | Personal Microsoft accounts only               |

Default: `common`

---

### Optional Access Restrictions

```bash
export MICROSOFT_ALLOWED_DOMAINS="company.com"
export MICROSOFT_ALLOWED_TENANT_IDS="<tenant-id-1>,<tenant-id-2>"
```

Restrictions are enforced **after OAuth login, before JWT issuance**.

---

## 3. Redirect URI (Critical)

Redirect URIs are derived automatically from the configured frontend base.

Derived from:

* `FRONTEND_HOSTNAME`, or
* `EXTERNAL_BASE`

---

### Canonical redirect URI

```
https://<FRONTEND_BASE>/auth/callback/microsoft
```

---

### Example

```bash
export FRONTEND_HOSTNAME="ui.athithya.site"
```

Register in Microsoft Entra:

```
https://ui.athithya.site/auth/callback/microsoft
```

> Must match **exactly** (scheme, case, no trailing slash).

---

## 4. Microsoft Entra App Registration (Step-by-Step)

### Step 1: Create App Registration

1. Go to: [https://portal.azure.com](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID → App registrations**
3. Click **New registration**
4. Configure:

   * Name: anything
   * Supported account types:

     * Select `Accounts in any organizational directory (Any Microsoft Entra ID tenant - Multitenant) and personal Microsoft accounts (e.g. Skype, Xbox)`. We can add restriction policies later. 

    * Redirect URI (Web):

     ```
     https://<FRONTEND_HOSTNAME>/auth/callback/microsoft
     ```

---

### Step 2: Create Client Secret

1. Go to **Certificates & secrets**
2. Create **New client secret** by giving any description and set Expires according to the requirements. Click Add
3. Copy:

   * Value → `MS_CLIENT_SECRET`

---

### Step 3: Copy Client ID & Tenant

From **Overview** page:

* Application (client) ID → `MS_CLIENT_ID`
* Directory (tenant) ID → `MS_TENANT_ID` (optional)

---

## 5. Scopes and Claims (What We Use)

The platform requests only:

```
openid email profile
```

What we rely on:

* Verified email
* Issuer
* Subject ID
* Tenant ID (`tid` claim)

What we do **not** use:

* Groups
* Microsoft Graph
* User.Read
* Directory permissions

---

## 6. Access Control Logic

### Tenant Validation

If `MICROSOFT_ALLOWED_TENANT_IDS` is set:

```bash
export MICROSOFT_ALLOWED_TENANT_IDS="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
```

Then:

* Tokens issued by other tenants are rejected

---

### Email Domain Validation

If `MICROSOFT_ALLOWED_DOMAINS` is set:

```bash
export MICROSOFT_ALLOWED_DOMAINS="company.com"
```

Then:

| Email                   | Result  |
| ----------------------- | ------- |
| `user@company.com`      | Allowed |
| `user@corp.company.com` | Denied  |
| `user@gmail.com`        | Denied  |

Domain matching is **exact**.

---

### Failure Behavior

If validation fails:

* **HTTP 403 Forbidden**
* Server-rendered error page explains:

  * Rejected email / tenant
  * Allowed domains / tenants

No JWT is issued.

---

## 7. Token & Session Behavior

| Aspect          | Behavior               |
| --------------- | ---------------------- |
| Session storage | Browser `localStorage` |
| Server sessions | None                   |
| Token type      | JWT (HS256)            |
| Token lifetime  | `JWT_EXP_SECONDS`      |
| Refresh         | Re-login required      |

---

## 8. Debug & Validation

### View computed redirect URIs

```
GET /auth/redirects
```

Example:

```
microsoft → https://ui.athithya.site/auth/callback/microsoft
```

---

### Common Errors

| Error                   | Cause                                  |
| ----------------------- | -------------------------------------- |
| `redirect_uri_mismatch` | URI mismatch in Entra app              |
| Login loops             | HTTP/HTTPS or cookie SameSite mismatch |
| Forbidden (403)         | Tenant or domain not allowed           |

---

## 9. Security Notes

* Always use HTTPS in production
* Rotate client secrets per environment
* Rotate `JWT_SECRET` only with coordinated logout
* Do not expose `/auth/callback/*` except via OAuth redirect

---

## 10. Summary

> Microsoft authentication in this platform uses pure OIDC with Entra ID, optional tenant and email-domain restrictions, and stateless JWT issuance. No Microsoft Graph access, no group claims, and no workload identity are used—keeping authentication minimal, secure, and portable.


