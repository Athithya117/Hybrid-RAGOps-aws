# Google OAuth Authentication (Our Platform)

This platform uses **stateless OpenID Connect (OIDC)** with Google as an identity provider.
Authentication issues a **signed JWT** directly to the browser and does **not** maintain any server-side user sessions.

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

**Design principle**

Authentication is kept **stateless, minimal, and portable**.  
We avoid Admin SDKs, group APIs, service accounts, and directory queries to eliminate privileged credentials, long-lived secrets, and stateful dependencies.

---

## 2. Required Environment Variables (Google)

To enable Google login:

```bash
export ENABLE_GOOGLE_AUTH="true"
export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
````

These values must come from a **Google Cloud OAuth Web Application**.

---

### Optional (Recommended to restrict only to your org domain)

```bash
export GOOGLE_ALLOWED_DOMAINS="company.com,gmail.com" 
```

Restricts login to users whose **email domain exactly matches** one of the listed domains.

> Domain matching is **exact** — wildcards and subdomains are not supported.

Examples:

| Email                   | Result  |
| ----------------------- | ------- |
| `user@company.com`      | Allowed |
| `user@corp.company.com` | Denied  |
| `user@gmail.com`        | Allowed |

---

## 3. Redirect URI (Critical)

Redirect URIs are derived automatically from the **configured frontend base URL**.

This is resolved internally from:

* `FRONTEND_HOSTNAME` **or**
* `EXTERNAL_BASE` (if set)

---

### Canonical redirect URI format

```
https://<FRONTEND_BASE>/auth/callback/google
```

---

### Example

```bash
export FRONTEND_HOSTNAME="ui.athithya.site"
```

Redirect URI to register in Google Cloud:

```
https://ui.athithya.site/auth/callback/google
```

> The redirect URI must match **exactly**
> (case-sensitive, no trailing slash, correct scheme).

---

## 4. Google Cloud Setup (Step-by-Step)

### Step 1: Create or select a Google Cloud project

[https://console.cloud.google.com/](https://console.cloud.google.com/)

---

### Step 2: Configure OAuth consent screen

1. Navigate to **APIs & Services → OAuth consent screen**
2. Choose:
   * **External** (default for b2c)
   * **Internal** (Workspace-only, optional)
3. Configure:

   * App name
   * Support email
   * Authorized domains (your frontend domain)
4. Save and continue

---

### Step 3: Create OAuth Client ID

1. Go to **APIs & Services → Credentials**

2. Click **Create Credentials → OAuth client ID**

3. Select **Web application**

4. Configure:

   * Application name (any)
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

✔ Provider-issued identity verification
✔ Email-based domain allowlisting
✔ Stateless JWT issuance

### What we intentionally do NOT support

✘ Google Groups
✘ Admin SDK impersonation
✘ Service account credentials
✘ Organization-wide directory queries

**Reason**

These require privileged credentials, stateful validation, or directory access — all of which violate the platform’s stateless and portable design goals.

---

## 6. Token & Session Behavior

| Aspect          | Behavior                                  |
| --------------- | ----------------------------------------- |
| Session storage | Browser `localStorage`                    |
| Server sessions | None                                      |
| Token type      | Signed JWT (HS256)                        |
| Token lifetime  | Configurable via `JWT_EXP_SECONDS`        |
| Refresh tokens  | Not used (re-login required after expiry) |

JWTs are issued **only after** domain validation succeeds.

---

## 7. Domain Restriction Logic

When `GOOGLE_ALLOWED_DOMAINS` is set:

```bash
export GOOGLE_ALLOWED_DOMAINS="company.com"
```

Flow behavior:

1. User completes Google OAuth successfully
2. Email domain is extracted from the verified email
3. Domain is checked against the allowlist
4. If denied:

   * **HTTP 403** is returned
   * A server-rendered page explains:

     * The rejected email/domain
     * The allowed domain list
5. If allowed:

   * JWT is issued
   * Token is stored in browser `localStorage`

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

---

### Common Errors

| Error                   | Cause                                                             |
| ----------------------- | ----------------------------------------------------------------- |
| `redirect_uri_mismatch` | Redirect URI in Google Console does not exactly match derived URI |
| Login loops             | Cookie / SameSite mismatch (HTTP vs HTTPS, proxy misconfig)       |
| Forbidden (403)         | Email domain not in `GOOGLE_ALLOWED_DOMAINS`                      |

---

## 9. Security Notes

* Always use **HTTPS** in production
* Never reuse OAuth client secrets across environments
* Rotate `JWT_SECRET` only with coordinated logout
* `/auth/callback/*` must only be reachable via OAuth redirect

---

## 10. Summary

> Google OAuth integration in this platform is intentionally minimal: a single redirect URI derived from the configured frontend base, optional exact-match email domain restriction, no Admin SDK or service accounts, and stateless JWT issuance. This design maximizes security, debuggability, and portability across Kubernetes, Cloudflare Tunnel, and bare-metal deployments.

---
