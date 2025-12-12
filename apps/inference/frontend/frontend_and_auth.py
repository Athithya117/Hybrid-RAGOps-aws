#!/usr/bin/env python3
"""
frontend_and_auth.py — async, scalable version (httpx only, no requests)

Features:
- Async JWKS discovery + caching (uses .well-known/openid-configuration -> jwks_uri).
- Token verification async: parse JWT header kid -> get JWK -> convert to public key -> jwt.decode.
- Uses httpx.AsyncClient for all outbound HTTP calls (JWKS discovery and upstream query).
- Prometheus metrics (optional via PROMETHEUS_ENABLED).
- FastAPI app exposing: /, /run, /auth/me, /health, optional /metrics.
- Minimal, low-cardinality metrics: request counter + histogram.

Env vars (required/used):
- OIDC_AUDIENCE, SPA_CLIENT_ID, QUERY_URL, FRONTEND_URL, plus OIDC_ISSUER or AZURE_TENANT_ID
- PROMETHEUS_ENABLED (optional), PROMETHEUS_PATH (optional), JWKS_REFRESH_INTERVAL_SECONDS (optional)

Note: ensure requirements contain:
  fastapi, uvicorn[standard], httpx, PyJWT, prometheus_client, python-multipart (if needed)

"""
from __future__ import annotations

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List

from urllib.parse import urljoin
from string import Template

import httpx
from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import jwt  # PyJWT
from jwt import InvalidTokenError
from jwt.algorithms import RSAAlgorithm

# --- Prometheus ---
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() in ("1", "true", "yes")
PROMETHEUS_PATH = os.getenv("PROMETHEUS_PATH", "/metrics")

if PROMETHEUS_ENABLED:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    REQUEST_COUNT = Counter(
        "frontend_requests_total",
        "Total HTTP requests processed by frontend",
        ["method", "endpoint", "http_status"],
    )
    REQUEST_LATENCY = Histogram(
        "frontend_request_latency_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
    )

# -------------------------
# Logging & pre-checks
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("frontend_auth")

def fatal(msg: str):
    log.critical(msg)
    raise RuntimeError(msg)

# -------------------------
# Env parsing and validation
# -------------------------
AUTH_MODE = os.getenv("AUTH_MODE", "external-id").lower()
if AUTH_MODE not in ("entra", "external-id"):
    fatal('AUTH_MODE must be "entra" or "external-id"')

OIDC_ISSUER = os.getenv("OIDC_ISSUER", "") or ""
OIDC_JWKS_URI_OVERRIDE = os.getenv("OIDC_JWKS_URI", "") or ""
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE", "") or ""
SPA_CLIENT_ID = os.getenv("SPA_CLIENT_ID", "") or ""
QUERY_URL = os.getenv("QUERY_URL", "") or ""
FRONTEND_URL = os.getenv("FRONTEND_URL", "") or ""
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "") or ""

ENABLE_CORS = os.getenv("ENABLE_CORS", "false").lower() in ("1", "true", "yes")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
JWKS_REFRESH_INTERVAL_SECONDS = int(os.getenv("JWKS_REFRESH_INTERVAL_SECONDS", "900"))

missing = []
if not OIDC_AUDIENCE:
    missing.append("OIDC_AUDIENCE")
if not SPA_CLIENT_ID:
    missing.append("SPA_CLIENT_ID")
if not QUERY_URL:
    missing.append("QUERY_URL")
if not FRONTEND_URL:
    missing.append("FRONTEND_URL")
if not OIDC_ISSUER and not AZURE_TENANT_ID:
    missing.append("OIDC_ISSUER or AZURE_TENANT_ID")

if missing:
    fatal("Missing required env vars: " + ", ".join(missing))

ALLOWED_AUDIENCES: List[str] = [s.strip() for s in OIDC_AUDIENCE.split(",") if s.strip()]

if not OIDC_ISSUER:
    OIDC_ISSUER = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"

# compute redirect URI used for registration
REDIRECT_PATH = "/auth/callback/entra" if AUTH_MODE == "entra" else "/auth/callback/external-id"
REDIRECT_URI = FRONTEND_URL.rstrip("/") + REDIRECT_PATH

# -------------------------
# Async JWKS discovery + cache
# -------------------------
_jwks_cache: Optional[Dict[str, Any]] = None  # raw jwks JSON
_jwks_last_refresh: int = 0
_jwks_uri: Optional[str] = OIDC_JWKS_URI_OVERRIDE or None
_jwks_lock = None  # we'll create an AsyncClient-level semaphore if needed

async def fetch_openid_config(client: httpx.AsyncClient, issuer: str) -> Dict[str, Any]:
    well_known = issuer.rstrip("/") + "/.well-known/openid-configuration"
    r = await client.get(well_known, timeout=8.0)
    r.raise_for_status()
    return r.json()

async def fetch_jwks(client: httpx.AsyncClient, jwks_uri: str) -> Dict[str, Any]:
    r = await client.get(jwks_uri, timeout=8.0)
    r.raise_for_status()
    return r.json()

async def ensure_jwks_loaded(force_refresh: bool = False) -> None:
    """
    Ensure _jwks_cache is loaded and not older than JWKS_REFRESH_INTERVAL_SECONDS.
    Performs async HTTP calls with httpx only.
    """
    global _jwks_cache, _jwks_last_refresh, _jwks_uri
    now = int(time.time())
    if _jwks_cache and not force_refresh and (now - _jwks_last_refresh) < JWKS_REFRESH_INTERVAL_SECONDS:
        return

    # Acquire simple lock by creating a client and re-check quickly (race benign)
    async with httpx.AsyncClient() as client:
        # discover jwks_uri if not set
        if not _jwks_uri:
            try:
                cfg = await fetch_openid_config(client, OIDC_ISSUER)
                uri = cfg.get("jwks_uri")
                if not uri:
                    raise RuntimeError("jwks_uri not found in OIDC discovery document")
                _jwks_uri = uri
            except Exception as e:
                log.exception("Failed to fetch OIDC discovery document")
                raise

        # fetch jwks JSON
        try:
            jwks = await fetch_jwks(client, _jwks_uri)
            if not isinstance(jwks, dict) or "keys" not in jwks:
                raise RuntimeError("Invalid JWKS document")
            _jwks_cache = jwks
            _jwks_last_refresh = int(time.time())
            log.info("JWKS loaded; keys=%d", len(_jwks_cache.get("keys", [])))
        except Exception:
            log.exception("Failed to fetch JWKS")
            raise

async def get_jwk_for_kid(kid: str) -> Optional[Dict[str, Any]]:
    """
    Return the JWK dict for a given kid. Will attempt one refresh if kid not found.
    """
    global _jwks_cache
    await ensure_jwks_loaded()
    keys = _jwks_cache.get("keys", []) if _jwks_cache else []
    for k in keys:
        if k.get("kid") == kid:
            return k
    # not found — force a refresh and try once
    log.info("kid %s not found in cached JWKS; forcing refresh", kid)
    await ensure_jwks_loaded(force_refresh=True)
    keys = _jwks_cache.get("keys", []) if _jwks_cache else []
    for k in keys:
        if k.get("kid") == kid:
            return k
    return None

# -------------------------
# Token verification (async)
# -------------------------
async def verify_token_async(token: str) -> Dict[str, Any]:
    """
    Async token verification:
    - parse header to find kid
    - fetch jwk for kid
    - convert jwk -> public key via jwt.algorithms.RSAAlgorithm.from_jwk
    - decode token using public key
    - perform issuer & audience checks manually
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
    except Exception as e:
        log.debug("get_unverified_header failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token header")

    kid = unverified_header.get("kid")
    if not kid:
        log.debug("Token header missing kid")
        raise HTTPException(status_code=401, detail="Missing token kid")

    jwk = await get_jwk_for_kid(kid)
    if not jwk:
        log.debug("JWK not found for kid=%s", kid)
        raise HTTPException(status_code=401, detail="Token public key not found")

    # Convert JWK to PEM-like key object for PyJWT using RSAAlgorithm.from_jwk
    try:
        jwk_json = json.dumps(jwk)
        public_key = RSAAlgorithm.from_jwk(jwk_json)
    except Exception as e:
        log.exception("Failed to construct public key from JWK")
        raise HTTPException(status_code=401, detail="Invalid JWK")

    # decode token (do not auto-verify aud/iss here; we'll check after decode)
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256", "RS384", "ES256", "ES384"],
            options={"verify_signature": True, "verify_aud": False, "verify_iss": False}
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError as e:
        log.debug("JWT decode failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token signature")

    # verify issuer
    token_iss = payload.get("iss")
    if token_iss != OIDC_ISSUER:
        log.debug("issuer mismatch token.iss=%s expected=%s", token_iss, OIDC_ISSUER)
        raise HTTPException(status_code=401, detail="Invalid token issuer")

    # verify audience
    aud = payload.get("aud")
    token_auds = [aud] if isinstance(aud, str) else (list(aud) if isinstance(aud, (list, tuple)) else [])
    azp = payload.get("azp")

    if any(a in ALLOWED_AUDIENCES for a in token_auds):
        return payload

    if azp == SPA_CLIENT_ID and SPA_CLIENT_ID in token_auds and ALLOWED_AUDIENCES:
        return payload

    log.debug("aud mismatch token_auds=%s allowed=%s azp=%s", token_auds, ALLOWED_AUDIENCES, azp)
    raise HTTPException(status_code=401, detail="Invalid token audience")

def extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization must be Bearer token")
    return parts[1]

# -------------------------
# FastAPI app + Prometheus middleware
# -------------------------
app = FastAPI(title="frontend-auth-gateway", docs_url=None, redoc_url=None)

if ENABLE_CORS:
    origins = ["*"] if CORS_ALLOWED_ORIGINS == "*" else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    log.info("CORS enabled for origins: %s", origins)

if PROMETHEUS_ENABLED:
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        start = time.time()
        endpoint = request.url.path or "root"
        method = request.method
        status = 500
        try:
            response = await call_next(request)
            status = getattr(response, "status_code", 500)
            return response
        except Exception:
            status = 500
            raise
        finally:
            elapsed = time.time() - start
            try:
                REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=str(status)).inc()
            except Exception:
                log.debug("Prometheus metric update failed", exc_info=True)

    @app.get(PROMETHEUS_PATH)
    def metrics():
        try:
            payload = generate_latest()
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to generate metrics")

# -------------------------
# SPA HTML via string.Template
# -------------------------
MSAL_CDN = "https://alcdn.msauth.net/browser/2.37.0/js/msal-browser.min.js"

INDEX_TEMPLATE = Template("""<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG UI</title>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50 min-h-screen p-6">
<div class="max-w-4xl mx-auto">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-semibold">RAG UI</h1><div id="auth-top"></div>
  </div>
  <div id="login-cta" class="mb-4"><button id="btn-login" class="bg-blue-600 text-white px-4 py-2 rounded">Sign in</button></div>
  <form id="qry" class="space-y-4 bg-white p-4 rounded shadow">
    <label class="block text-sm font-medium">Query</label>
    <textarea id="query" rows="3" class="mt-1 block w-full border rounded p-2"></textarea>
    <div class="flex items-center space-x-4">
      <label class="flex items-center space-x-2"><input id="enable_tracing" type="checkbox"/><span class="text-sm">Enable tracing</span></label>
      <label class="text-sm">Top K <input id="top_k" type="number" value="5" min="1" max="50" class="ml-2 w-20 border rounded p-1 text-sm"/></label>
    </div>
    <div><button id="ask" type="button" class="bg-blue-600 text-white px-4 py-2 rounded">Ask</button></div>
  </form>
  <div id="result" class="mt-6"></div>
</div>
<script src="$msal_cdn"></script>
<script>
const SPA_CLIENT_ID = "$spa_client_id";
const OIDC_AUTHORITY = "$oidc_issuer";
const QUERY_RUN = "/run";
const msalConfig = {
  auth: { clientId: SPA_CLIENT_ID, authority: OIDC_AUTHORITY, redirectUri: window.location.origin },
  cache: { cacheLocation: "sessionStorage", storeAuthStateInCookie: false }
};
const msalInstance = new msal.PublicClientApplication(msalConfig);

async function signInInteractive(){
  try {
    const loginReq = { scopes: ["openid","profile","email"] };
    await msalInstance.loginPopup(loginReq);
    const accounts = msalInstance.getAllAccounts();
    if(accounts.length>0){
      const acc = accounts[0];
      const tokenReq = { scopes: ["openid","profile","email"], account: acc };
      let tokenResp;
      try { tokenResp = await msalInstance.acquireTokenSilent(tokenReq); } catch(e) { tokenResp = await msalInstance.acquireTokenPopup(tokenReq); }
      const token = tokenResp.accessToken || tokenResp.idToken;
      sessionStorage.setItem('app_jwt', token);
      refreshUi();
    }
  } catch(e) { console.error("login failed", e); alert("Login failed - see console"); }
}

function signOut(){ try{ sessionStorage.removeItem('app_jwt'); }catch(e){} window.location.reload(); }

function parseJwt(t){ try{ const p=t.split('.')[1]; const b=p.replace(/-/g,'+').replace(/_/g,'/'); const j=decodeURIComponent(Array.prototype.map.call(atob(b),c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join('')); return JSON.parse(j);}catch(e){return null;}}

function showTop(user){ const el=document.getElementById('auth-top'); if(!el) return; if(user){ el.innerHTML = '<div class="flex items-center space-x-3"><div class="text-sm">Signed in as <strong>'+user+'</strong></div><button id="top-logout" class="ml-2 bg-gray-100 px-2 py-1 rounded">Sign out</button></div>'; document.getElementById('top-logout').addEventListener('click', signOut); document.getElementById('login-cta').style.display='none'; } else { el.innerHTML = '<button id="top-signin" class="bg-blue-600 text-white px-3 py-1 rounded text-sm">Sign in</button>'; document.getElementById('top-signin').addEventListener('click', signInInteractive); document.getElementById('login-cta').style.display='block'; } }

async function validateToken(token){
  if(!token) return false;
  try{
    const r = await fetch('/auth/me', { headers: { 'Authorization': 'Bearer ' + token } });
    if(r.ok){
      const j = await r.json();
      const u = j.user || {};
      const display = u.email || u.name || u.sub || 'User';
      showTop(display);
      return true;
    }
  }catch(e){ console.debug('validate failed', e); }
  try{ sessionStorage.removeItem('app_jwt'); }catch(e){}
  showTop(null);
  return false;
}

function refreshUi(){ const t=sessionStorage.getItem('app_jwt'); if(!t){ showTop(null); return; } const p=parseJwt(t)||{}; showTop(p.email||p.name||p.sub||'User'); validateToken(t).then(ok=>{ if(!ok) showTop(null); }); }

async function submit(){
  const q=document.getElementById('query').value.trim();
  if(!q){ document.getElementById('result').innerHTML='<div class="bg-red-100 p-3 rounded">Query required</div>'; return; }
  const token=sessionStorage.getItem('app_jwt');
  if(!token){ signInInteractive(); return; }
  const top_k=parseInt(document.getElementById('top_k').value||'5',10);
  const enable_tracing=document.getElementById('enable_tracing').checked===true;
  const payload={ query: q, top_k, enable_tracing, return_chunks: enable_tracing };
  document.getElementById('ask').disabled=true; document.getElementById('ask').innerText='Asking...';
  try{
    const resp = await fetch(QUERY_RUN, { method:'POST', headers: { 'Content-Type':'application/json', 'Authorization': 'Bearer ' + token }, body: JSON.stringify(payload) });
    if(resp.status===401){ sessionStorage.removeItem('app_jwt'); signInInteractive(); return; }
    const text=await resp.text();
    document.getElementById('result').innerHTML = text;
  } catch(e){
    document.getElementById('result').innerHTML='<div class="bg-red-100 p-3 rounded">Request failed: '+String(e)+'</div>';
  } finally{
    document.getElementById('ask').disabled=false; document.getElementById('ask').innerText='Ask';
  }
}

document.addEventListener('DOMContentLoaded', function(){ refreshUi(); document.getElementById('ask').addEventListener('click', submit); document.getElementById('btn-login').addEventListener('click', signInInteractive); });
</script>
</body>
</html>
""")

INDEX_HTML = INDEX_TEMPLATE.substitute(
    msal_cdn=MSAL_CDN,
    spa_client_id=SPA_CLIENT_ID,
    oidc_issuer=OIDC_ISSUER
)

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.post("/run")
async def run(request: Request, authorization: Optional[str] = Header(None)):
    token = extract_bearer(authorization)
    # verify token async
    _ = await verify_token_async(token)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict) or not body.get("query"):
        raise HTTPException(status_code=400, detail="Missing 'query'")

    target = urljoin(QUERY_URL.rstrip("/") + "/", "generate")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(target, json=body, headers=headers)
            text = resp.text
            return PlainTextResponse(content=text, status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        log.error("Upstream returned non-200: %s", str(e))
        raise HTTPException(status_code=502, detail=f"Upstream error: {e.response.status_code}")
    except Exception:
        log.exception("Upstream call failed")
        raise HTTPException(status_code=502, detail="Upstream call failed")

@app.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    token = extract_bearer(authorization)
    claims = await verify_token_async(token)
    safe = {k: v for k, v in claims.items() if k not in ("exp", "nbf", "iat")}
    return {"authenticated": True, "user": safe}

@app.get("/health")
async def health():
    # expose issuer/jwks location for debug; remove jwks in prod if sensitive
    await ensure_jwks_loaded()  # warm cache
    return {"status": "ok", "issuer": OIDC_ISSUER, "jwks_uri": _jwks_uri}

# -------------------------
# Run with uvicorn when executed directly
# -------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("frontend_and_auth:app", host=host, port=port, reload=True)
