# Frontend & Auth Gateway — Runtime Documentation (Updated)

## Overview

The Frontend & Auth Gateway is a **FastAPI-based stateless edge service** that:

* Serves the RAG UI (SPA)
* Handles **OAuth-based login** (Google, Microsoft, GitHub)
* Issues **platform-signed JWTs** to the browser
* Proxies authenticated requests to the retrieval backend
* Exposes health endpoints

The service **does not validate third-party JWTs**.
Instead, it performs OAuth login, validates identity once, and issues its **own JWT** for all subsequent requests.

---

## Core Design Principles

* **Stateless**: no server-side sessions
* **Single trust boundary**: only platform-issued JWTs are accepted
* **Minimal OAuth usage**: no admin SDKs, no directory traversal
* **Frontend is the only browser-facing service**
* **Backends are private (cluster-internal)**

---

# Components & Contracts

## HTTP Endpoints

### Public

* `GET /`
  Serves the embedded SPA HTML

* `GET /health`
  Returns basic liveness info

### Authentication

* `GET /auth/login/{provider}`
  Initiates OAuth login (`google | microsoft | github`)

* `GET /auth/callback/{provider}`
  OAuth redirect endpoint; issues JWT on success

  *Note:* the temporary cookies/state used during the OAuth redirect handshake are signed with `SESSION_SECRET` (see Configuration). Rotating `SESSION_SECRET` will break in-flight OAuth logins (CSRF / state mismatches) but does **not** invalidate already-issued platform JWTs.

* `GET /auth/me`
  Returns authenticated user claims
  Requires `Authorization: Bearer <platform-jwt>`

### Protected

* `POST /run`
  Proxies request to retrieval backend
  Requires `Authorization: Bearer <platform-jwt>`

---

## External Dependencies

* OAuth providers:

  * Google
  * Microsoft Entra ID
  * GitHub
* Retrieval backend at `QUERY_URL`
* Browser (SPA)

---

# Security Contract

* All protected endpoints require:

  ```
  Authorization: Bearer <JWT>
  ```
* JWTs are:

  * Issued **only by this service**
  * Signed with `JWT_SECRET`
  * Verified locally (HMAC)
* No external JWKS, no issuer discovery, no token introspection

  *Important:* `JWT_SECRET` is the platform's signing key (HS256). Rotating `JWT_SECRET` invalidates all previously issued JWTs and forces global re-authentication. Keep `JWT_SECRET` stable and identical across replicas.

---

# Configuration (Environment Variables)

## Core Runtime

| Variable            | Description                                                                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `QUERY_URL`         | Retrieval service base URL                                                                                                                                 |
| `FRONTEND_HOSTNAME` | Public frontend hostname (preferred)                                                                                                                       |
| `EXTERNAL_BASE`     | Optional full public base URL                                                                                                                              |
| `JWT_SECRET`        | HMAC secret for signing JWTs — HS256. Rotating this invalidates all existing platform JWTs and forces re-login across the fleet.                           |
| `SESSION_SECRET`    | Cookie signing (OAuth flow only). Used to sign temporary OAuth state/cookies; rotating this breaks in-flight OAuth logins but does not revoke issued JWTs. |

---

## Authentication Providers

### Google

```bash
ENABLE_GOOGLE_AUTH=true
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_ALLOWED_DOMAINS=company.com,gmail.com
```

### Microsoft

```bash
ENABLE_MICROSOFT_AUTH=true
MS_CLIENT_ID=...
MS_CLIENT_SECRET=...
MS_TENANT_ID=b8a65a11-...
MICROSOFT_ALLOWED_DOMAINS=company.com
MICROSOFT_ALLOWED_TENANT_IDS=b8a65a11-...
```

### GitHub

```bash
ENABLE_GITHUB_AUTH=true
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
GITHUB_ALLOWED_ORGS=myorg
```

---

## Optional / Behavioral

| Variable               | Description                    |
| ---------------------- | ------------------------------ |
| `JWT_EXP_SECONDS`      | JWT lifetime (default ~30 min) |
| `COOKIE_SECURE`        | Auto-enabled when HTTPS        |
| `ENABLE_CORS`          | Default `false`                |
| `CORS_ALLOWED_ORIGINS` | `*` or comma-separated         |
| `LOG_LEVEL`            | Default `INFO`                 |
| `HOST`, `PORT`         | Uvicorn bind                   |

---

# OAuth Redirect URI Resolution

Redirect URIs are **derived automatically**.

Priority order:

1. `FRONTEND_HOSTNAME`
2. `EXTERNAL_BASE`
3. Fallback: `http://127.0.0.1:8000`

Canonical format:

```
https://<frontend-base>/auth/callback/<provider>
```

Examples:

```
https://ui.example.com/auth/callback/google
https://ui.example.com/auth/callback/microsoft
https://ui.example.com/auth/callback/github
```

---

# Authentication Flow (Precise)

### 1. Login Initiation

```
Browser → /auth/login/google
```

* Redirects to provider OAuth consent screen

---

### 2. OAuth Callback

```
Provider → /auth/callback/google
```

Steps:

1. Exchange authorization code for access token
2. Fetch user identity info
3. Apply provider-specific allowlist rules
4. On success:

   * Issue platform JWT
   * Return SPA page that stores JWT in browser storage

*Implementation note:* temporary state/cookies used to correlate the login request to the callback are signed with `SESSION_SECRET`. Ensure `SESSION_SECRET` is consistent across replicas during normal operation; rotating it will abort active login flows.

---

### 3. Platform JWT

JWT claims include:

* `sub` — provider user id
* `email`
* `name`
* `provider`
* `iat`, `exp`, `iss`, `aud`

Signed with:

```
HS256(JWT_SECRET)
```

*Important:* rotating `JWT_SECRET` causes all previously issued JWTs to fail verification and forces global re-authentication.

---

### 4. Authenticated Requests

```
POST /run
Authorization: Bearer <platform-jwt>
```

* JWT verified locally
* No external calls
* No JWKS
* No issuer discovery

---

# `/run` Proxy Behavior

* Requires valid JWT
* Accepts JSON body (`query`, `top_k`, etc.)
* Proxies request to:

```
POST {QUERY_URL}/generate
```

* Forwards bearer token
* Returns upstream response verbatim
* Errors:

  * `400` invalid payload
  * `401` invalid/missing JWT
  * `502` upstream failure

---

# `/auth/me`

Returns:

```json
{
  "authenticated": true,
  "user": {
    "email": "...",
    "name": "...",
    "provider": "google"
  }
}
```

* JWT required
* Time-based claims removed

---

# CORS Behavior

* **Disabled by default**
* Only applies to browser-originated cross-origin requests. Not required for RAG8s
* Not required when frontend and backend are same-origin (recommended)

Supported values:

```bash
ENABLE_CORS=true
CORS_ALLOWED_ORIGINS="https://ui.example.com"
```

No regex, no wildcards beyond `*`.

---

# Metrics

* Optional
* Exposed only when enabled
* No auth required

---

# Docker & Runtime

* Base: `python:3.x-slim`
* Entrypoint: `uvicorn frontend_and_auth:app`
* Single process
* No background workers
* Fully async via `httpx`

---

# Error Semantics

| Code | Meaning                        |
| ---- | ------------------------------ |
| 400  | Invalid request payload        |
| 401  | Missing / invalid platform JWT |
| 403  | OAuth allowlist rejection      |
| 502  | Upstream retrieval failure     |

---

# Security Notes

* Platform JWT is the **only trusted credential**
* OAuth tokens are short-lived and discarded after login
* Rotating `JWT_SECRET` forces global logout
* Rotating `SESSION_SECRET` breaks in-flight OAuth logins but does not revoke issued JWTs
* Backends should never be exposed publicly
* `/auth/callback/*` must not be called directly

---

# Summary

> The Frontend & Auth Gateway is a stateless OAuth-to-JWT edge service. It terminates all identity flows, issues platform-owned JWTs, enforces minimal allowlists, and proxies authenticated requests to private RAG backends. No external token validation, no JWKS, and no server-side session state are used.

---
