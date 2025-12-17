# Frontend & Auth Gateway — Runtime Documentation

## Overview

The Frontend & Auth Gateway is a FastAPI application that provides a single-entry point for the RAG UI and enforces OIDC-based authentication for upstream requests. It performs asynchronous JWKS discovery and caching, verifies incoming bearer tokens, proxies authenticated requests to the retrieval service, serves a minimal SPA, and exposes health and Prometheus metrics endpoints.

---

# Components & Contracts

**Processes / endpoints**

* `GET /` — serves the SPA HTML.
* `POST /run` — proxy endpoint that accepts a request body (JSON) containing retrieval parameters and forwards it to the configured `QUERY_URL/generate`. Requires `Authorization: Bearer <token>`.
* `GET /auth/me` — returns authenticated user claims (sanitized).
* `GET /health` — warms JWKS cache and returns OIDC issuer and JWKS URI.
* `GET /metrics` — (optional) Prometheus metrics when `PROMETHEUS_ENABLED=true`.

**External services**

* OIDC issuer (discovery document and JWKS).
* Upstream retrieval service at `QUERY_URL` (proxied for `/run`).
* Browser SPA uses MSAL for sign-in; SPA posts to `/run`.

**Security contract**

* All protected endpoints expect `Authorization: Bearer <JWT>`.
* Tokens are validated against the issuer and accepted audiences; token signatures are verified with JWKs fetched from the JWKS endpoint.

---

# Configuration (Environment Variables)

**Required**

* `OIDC_AUDIENCE` — comma-separated allowed audiences.
* `SPA_CLIENT_ID` — client id used by SPA and accepted `azp`.
* `QUERY_URL` — upstream retrieval service base URL (e.g., `https://retrieval.svc`).
* `FRONTEND_URL` — public front-end origin used to compute `REDIRECT_URI`.
* `OIDC_ISSUER` *or* `AZURE_TENANT_ID` — issuer base URL or Azure tenant id (if `OIDC_ISSUER` omitted, it is computed using `AZURE_TENANT_ID`).

**Behavioral & optional**

* `AUTH_MODE` — `entra` or `external-id` (default `external-id`). Controls SPA redirect path.
* `PROMETHEUS_ENABLED` — `true|false` to enable metrics endpoint.
* `PROMETHEUS_PATH` — metrics path (default `/metrics`).
* `JWKS_REFRESH_INTERVAL_SECONDS` — JWKS cache TTL (default `900`).
* `ENABLE_CORS` — enable CORS middleware (default `false`).
* `CORS_ALLOWED_ORIGINS` — `*` or comma-separated list when CORS enabled.
* `LOG_LEVEL` — logging level (default `INFO`).
* `HOST`, `PORT` — uvicorn host/port when run directly.

---

# Startup & Lifespan Behavior

1. Process starts and configures logging.
2. Required environment variables are validated; missing required variables cause startup failure.
3. CORS middleware is configured if `ENABLE_CORS=true`.
4. Prometheus middleware and `/metrics` endpoint are installed when `PROMETHEUS_ENABLED=true`.
5. JWKS cache is lazily populated: `ensure_jwks_loaded()` is called on `GET /health` and on first token verification. JWKS refreshes after `JWKS_REFRESH_INTERVAL_SECONDS` or on forced refresh when a token `kid` is not found.
6. No persistent background worker threads are required; all network I/O uses `httpx.AsyncClient` and is executed asynchronously.

---

# JWKS & Token Verification — precise flow

1. `verify_token_async(token)` entry:

   * Parse unverified header via `jwt.get_unverified_header(token)` to extract `kid`.
   * `get_jwk_for_kid(kid)`:

     * Ensure JWKS loaded via `ensure_jwks_loaded()`:

       * If `_jwks_uri` undefined, fetch OIDC discovery at `OIDC_ISSUER/.well-known/openid-configuration` and read `jwks_uri`.
       * GET the JWKS `jwks_uri` and cache the JSON (`_jwks_cache`) and timestamp (`_jwks_last_refresh`).
     * Search `_jwks_cache["keys"]` for a key with matching `kid`. If not found, force refresh once and retry.
   * Convert JWK to public key with `RSAAlgorithm.from_jwk(json.dumps(jwk))`.
   * Decode token via `jwt.decode(..., public_key, algorithms=[...], options={"verify_aud": False, "verify_iss": False})`. Signature is verified.
   * Manually validate `iss` equals `OIDC_ISSUER`.
   * Validate audience: token `aud` or `azp` must overlap with configured `OIDC_AUDIENCE` or equal `SPA_CLIENT_ID` (per code logic).
   * On success, return the token payload (claims). On failure raise `HTTPException(status_code=401, ...)`.

---

# Endpoint Details & Behavior

### `GET /`

* Returns the embedded SPA HTML (tailwind + MSAL + minimal JS).
* The template injects: `SPA_CLIENT_ID`, `OIDC_ISSUER`, and configured script CDN.

### `POST /run`

* Authorization: requires `Authorization: Bearer <JWT>`.
* Token verification performed via `verify_token_async`.
* Request body: expects JSON with a `query` field and other RAG parameters.
* Proxies the JSON body to `QUERY_URL/generate` via `httpx.AsyncClient.post(...)` with the same bearer token in the header.
* Returns upstream response body as `text/plain` with upstream HTTP status code.
* Errors:

  * 400 for invalid JSON or missing `query`.
  * 401 when token missing/invalid.
  * 502 when upstream call fails or returns non-200 status.

### `GET /auth/me`

* Authorization: requires `Authorization: Bearer <JWT>`.
* Returns `{ "authenticated": true, "user": <claims minus exp/nbf/iat> }`.

### `GET /health`

* Ensures JWKS cache by calling `ensure_jwks_loaded()` and returns:

  ```json
  { "status": "ok", "issuer": "<OIDC_ISSUER>", "jwks_uri": "<_jwks_uri>" }
  ```

### `GET /metrics` (optional)

* Exposes Prometheus metrics when enabled. Uses `prometheus_client` to generate the scrape payload.

---

# SPA Behavior (client-side summary)

* SPA uses MSAL (browser popup) to perform interactive sign-in.
* On successful sign-in, SPA stores JWT in `sessionStorage` under `app_jwt`.
* SPA issues `POST /run` with JSON payload and `Authorization: Bearer <token>` header.
* SPA uses `GET /auth/me` to verify token and obtain sanitized user info for UI display.

---

# Metrics & Monitoring

When `PROMETHEUS_ENABLED=true`, the gateway exposes low-cardinality metrics:

* `frontend_requests_total{method,endpoint,http_status}` — request count
* `frontend_request_latency_seconds{method,endpoint}` — request latency histogram

Instrumentation is performed in middleware that wraps each HTTP request.

---

# Docker image & runtime

* Base image: `python:3.11-slim`.
* Application file copied: `frontend_and_auth.py`.
* Dependencies installed from `requirements.txt`.
* Non-root user `appuser` created; process runs as that user.
* Runtime command: `uvicorn frontend_and_auth:app --host 0.0.0.0 --port 8000 --workers 1`.
* Docker-level minimal system dependencies: `ca-certificates`.

---

# Error Handling & Response Codes

* **400 Bad Request** — invalid input payload (e.g., malformed JSON, missing `query`).
* **401 Unauthorized** — missing/invalid bearer token, expired token, issuer/audience mismatch.
* **502 Bad Gateway** — failure or non-2xx response from upstream `QUERY_URL`.
* Internal exceptions are logged and surfaced as 502 for upstream failures or 500 for unexpected server errors.

---

# Logging & Observability

* Structured logging via Python `logging` module; log level controlled by `LOG_LEVEL`.
* Critical token verification failures, JWKS fetch errors, and upstream proxy failures are logged with stack traces.
* Prometheus metrics capture request counts and latencies when enabled.
* `/health` includes JWKS URI and issuer for quick debugging (remove or sanitize in restricted environments if needed).

---

# Security & Operational Notes

* JWKS are cached and refreshed periodically to avoid frequent network calls; token `kid` misses force a single refresh.
* Token signature verification uses public keys derived from JWK and supports RSA and ECDSA algorithms listed in the code.
* Audience checks accept tokens where `aud` includes any configured `OIDC_AUDIENCE`; SPA `azp` plus `SPA_CLIENT_ID` is also accepted when present.
* SPA redirect URI is computed from `FRONTEND_URL` and `AUTH_MODE` and must match the OIDC client registration.

---

# Shutdown Behavior

* Normal shutdown closes underlying `httpx` client contexts created per request and stops the uvicorn server. No background tasks persist beyond request scope.

---

# Examples

**Proxy a query (client-side)**

```http
POST /run
Authorization: Bearer <token>
Content-Type: application/json

{ "query": "What is RAG?", "top_k": 5, "enable_tracing": false }
```

**Get authenticated user**

```http
GET /auth/me
Authorization: Bearer <token>
```

**Health check**

```http
GET /health
```

---
