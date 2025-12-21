# Google OAuth Authentication (Our Platform)

This platform uses **stateless OpenID Connect (OIDC)** with Google as an identity provider.
Authentication issues a **signed JWT** to the browser and does **not** maintain server-side user sessions.

---

## 1. Supported Google Auth Features

| Feature                       | Supported |
| ----------------------------- | --------- |
| Google OAuth (OIDC)           | Yes       |
| Email-based access control    | Yes       |
| Domain allowlist              | Yes       |
| Google Groups (Admin SDK)     | No        |
| Service account impersonation | No        |
| Workload Identity / ADC       | No        |
| Organization ID usernames     | No        |

**Design principle:**
We intentionally avoid Admin SDK, service accounts, and group APIs to keep auth **stateless, simple, and portable**.

---

## 2. Required Environment Variables (Google)

To enable Google login:

```bash
export ENABLE_GOOGLE_AUTH="true"          # Enables Google OAuth provider
export GOOGLE_CLIENT_ID="..."             # OAuth client ID from Google Cloud
export GOOGLE_CLIENT_SECRET="..."         # OAuth client secret from Google Cloud
```

### Optional (recommended)

```bash
export GOOGLE_ALLOWED_DOMAINS="company.com,gmail.com"
```

Restricts login to users whose **email domain** matches the list.

---

## 3. Redirect URI (Critical)

Our system **derives redirect URIs automatically** from `FRONTEND_HOSTNAME`.

### Canonical redirect URI format

```
https://<FRONTEND_HOSTNAME>/auth/callback/google
```

### Example

```bash
export FRONTEND_HOSTNAME="ui.athithya.site" # <subdomain>.<second-level-domain>.<top-level-domain>
```

Redirect URI to register in Google Cloud:

```
https://ui.athithya.site/auth/callback/google
```

> This URI must match **exactly** (case-sensitive, no trailing slash).

---

## 4. Google Cloud Setup (Step-by-Step)

### Step 1: Create or select a Google Cloud project

[https://console.cloud.google.com/](https://console.cloud.google.com/)

---

### Step 2: Configure OAuth consent screen

1. Go to **APIs & Services → OAuth consent screen**
2. Choose **External** (or Internal for Workspace-only use)
3. Set:

   * App name
   * Support email
   * Authorized domains (your domain)
4. Save and continue

---

### Step 3: Create OAuth client ID

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Choose **Web application**
4. Set:

   * **Application name**: anything meaningful
   * **Authorized redirect URIs**:

     ```
     https://ui.athithya.site/auth/callback/google
     ```
5. Click **Create**
6. Copy:

   * Client ID → `GOOGLE_CLIENT_ID`
   * Client Secret → `GOOGLE_CLIENT_SECRET`

---

## 5. Access Control Model (Important)

### What we support

✔ Domain-based access control via email
✔ Provider-issued identity verification
✔ Stateless JWT issuance

### What we intentionally do NOT support

✘ Google Groups
✘ Admin SDK impersonation
✘ Service account JSON
✘ Organization-wide directory queries

**Why:**
Those require privileged credentials, long-lived secrets, and stateful validation—opposite of this platform’s design goals.

---

## 6. Token & Session Behavior

| Aspect          | Behavior                         |
| --------------- | -------------------------------- |
| Session storage | Browser localStorage             |
| Server sessions | None                             |
| Token type      | Signed JWT (HS256)               |
| Token lifetime  | Configurable (`JWT_EXP_SECONDS`) |
| Refresh         | Re-login required after expiry   |

---

## 7. Domain Restriction Logic

If configured:

```bash
export GOOGLE_ALLOWED_DOMAINS="company.com"
```

Then:

* `user@company.com` → allowed
* `user@gmail.com` → denied

Validation happens **after OAuth login**, before JWT issuance.

---

## 8. Debug & Validation

### View computed redirect URIs

```
GET /auth/redirects
```

Example output:

```
google → https://ui.athithya.site/auth/callback/google
```

### Common errors

| Error                   | Cause                                         |
| ----------------------- | --------------------------------------------- |
| `redirect_uri_mismatch` | Google console URI does not match derived URI |
| Login loops             | Cookie/SameSite mismatch (HTTP vs HTTPS)      |
| Forbidden               | Email domain not allowed                      |

---

## 9. Security Notes

* Always use **HTTPS** in production
* Do not reuse OAuth secrets across environments
* Rotate `JWT_SECRET` only with coordinated logout
* Do not expose `/auth/callback/*` publicly except via OAuth

---

## 10. Summary (One-paragraph)

> Google OAuth integration in this platform is intentionally minimal: a single redirect URI derived from `FRONTEND_HOSTNAME`, no Admin SDK or service accounts, optional email-domain restriction, and stateless JWT issuance. This keeps authentication secure, debuggable, and portable across Kubernetes, Cloudflare Tunnel, and bare-metal deployments.

---
