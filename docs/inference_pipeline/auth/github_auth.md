# GitHub OAuth Authentication (Our Platform)

This platform uses **stateless OAuth 2.0** with GitHub as an identity provider.
Authentication issues a **signed JWT** directly to the browser and does **not** maintain server-side user sessions.

---

## 1. Supported GitHub Auth Features

| Feature                        | Supported |
| ------------------------------ | --------- |
| GitHub OAuth (Web Application) | Yes       |
| Organization allowlist         | Yes       |
| User email verification        | Yes       |
| GitHub Teams                   | No        |
| Enterprise Managed Users (EMU) | No        |
| Fine-grained tokens            | No        |
| GitHub App authentication      | No        |

**Design principle**

GitHub authentication is **simple and stateless**.
Only public OAuth APIs are used. No GitHub Apps, no org admin tokens, no team graph traversal.

---

## 2. Required Environment Variables (GitHub)

To enable GitHub login:

```bash
export ENABLE_GITHUB_AUTH="true"
export GITHUB_CLIENT_ID="..."
export GITHUB_CLIENT_SECRET="..."
```

These values must come from a **GitHub OAuth App** (not a GitHub App).

---

## 3. Optional Organization Restriction (Recommended)

```bash
export GITHUB_ALLOWED_ORGS="myorg,anotherorg"
```

Restricts login to users who are members of **at least one** listed GitHub organization.

Notes:

* Org names are **case-insensitive**
* Membership is checked via `GET /user/orgs`
* Requires the OAuth scope `read:org` (already configured in code)

Examples:

| GitHub User Orgs       | Result  |
| ---------------------- | ------- |
| `["myorg"]`            | Allowed |
| `["anotherorg","foo"]` | Allowed |
| `["randomorg"]`        | Denied  |

If unset, **any authenticated GitHub user is allowed**.

---

## 4. Redirect URI (Critical)

Redirect URIs are derived automatically from the **configured frontend base URL**.

Resolved internally from:

* `FRONTEND_HOSTNAME` **or**
* `EXTERNAL_BASE`

---

### Canonical redirect URI format

```
https://<FRONTEND_BASE>/auth/callback/github
```

---

### Example

```bash
export FRONTEND_HOSTNAME="ui.athithya.site"
```

Register this redirect URI in GitHub:

```
https://ui.athithya.site/auth/callback/github
```

> Must match **exactly** (scheme, path, no trailing slash).

---

## 5. GitHub OAuth App Setup

### Step 1: Create OAuth App

1. Go to
   [https://github.com/settings/developers](https://github.com/settings/developers)
2. Select **OAuth Apps**
3. Click **New OAuth App**

---

### Step 2: Configure Application

| Field                  | Value                                           |
| ---------------------- | ----------------------------------------------- |
| Application name       | Any                                             |
| Homepage URL           | `https://ui.athithya.site`                      |
| Authorization callback | `https://ui.athithya.site/auth/callback/github` |

---

### Step 3: Copy Credentials

After creation:

* Client ID → `GITHUB_CLIENT_ID`
* Client Secret → `GITHUB_CLIENT_SECRET`

---

## 6. Access Control Model

### What we support

✔ GitHub-issued identity verification
✔ Organization membership allowlist
✔ Stateless JWT issuance

### What we intentionally do NOT support

✘ GitHub Teams
✘ Enterprise Managed Users
✘ GitHub Apps / App installations
✘ Fine-grained token policies

**Reason**

Those require elevated permissions, org-admin access, or stateful directory traversal — all incompatible with the platform’s stateless design.

---

## 7. Token & Session Behavior

| Aspect          | Behavior                          |
| --------------- | --------------------------------- |
| Session storage | Browser `localStorage`            |
| Server sessions | None                              |
| Token type      | Signed JWT (HS256)                |
| Token lifetime  | `JWT_EXP_SECONDS` (default 1800s) |
| Refresh tokens  | Not used (re-login required)      |

JWTs are issued **only after** org validation succeeds (if configured).

---

## 8. Organization Restriction Logic

When `GITHUB_ALLOWED_ORGS` is set:

```bash
export GITHUB_ALLOWED_ORGS="myorg"
```

Flow behavior:

1. User completes GitHub OAuth
2. Access token is issued
3. Platform queries `/user/orgs`
4. Org list is compared to allowlist
5. If denied:

   * HTTP 403
   * Clear explanation page rendered
6. If allowed:

   * JWT is issued
   * Stored in browser `localStorage`

---

## 9. Debug & Validation

### View computed redirect URIs

```
GET /auth/redirects
```

Example:

```
github → https://ui.athithya.site/auth/callback/github
```

---

### Common Errors

| Error                   | Cause                                                  |
| ----------------------- | ------------------------------------------------------ |
| `redirect_uri_mismatch` | Callback URL mismatch in GitHub OAuth App              |
| 403 Access denied       | User not in `GITHUB_ALLOWED_ORGS`                      |
| Provider not shown      | Client ID/secret missing or `ENABLE_GITHUB_AUTH=false` |

---

## 10. Security Notes

* Always use **HTTPS**
* Use separate OAuth apps per environment
* Rotating `JWT_SECRET` forces re-login
* `/auth/callback/github` must only be used by OAuth

---

## 11. Summary

> GitHub OAuth integration in this platform is deliberately minimal: a single redirect URI derived from the frontend base, optional organization membership restriction via public APIs, no GitHub Apps or admin permissions, and fully stateless JWT-based authentication.

