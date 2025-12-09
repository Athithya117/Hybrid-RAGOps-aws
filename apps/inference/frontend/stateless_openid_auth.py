# apps/inference/frontend/stateless_openid_auth.py
import os
import time
import logging
import json
from typing import List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
import jwt

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("stateless_openid_auth")

# Required secrets
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("Missing required environment variable: JWT_SECRET")

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("Missing required environment variable: SESSION_SECRET")

# Canonical local base to avoid host mismatch; override with env if needed
CANONICAL_BASE = os.getenv("CANONICAL_BASE", "http://127.0.0.1:8000")
OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", CANONICAL_BASE)
FRONTEND_BASE = os.getenv("FRONTEND_BASE", OAUTH_REDIRECT_BASE)

JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "1800"))
JWT_ISS = os.getenv("JWT_ISS", "stateless-openid-auth")
JWT_AUD = os.getenv("JWT_AUD", "rag-ui")

# Provider feature flags / credentials
ENABLE_GOOGLE = os.getenv("ENABLE_GOOGLE_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_MICROSOFT = os.getenv("ENABLE_MICROSOFT_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_GITHUB = os.getenv("ENABLE_GITHUB_AUTH", "false").lower() in ("1", "true", "yes")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")  # optional override

MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "common")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI")

# Validate provider credential presence when enabled
if ENABLE_GOOGLE and not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
    raise RuntimeError("ENABLE_GOOGLE_AUTH=true but GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET missing")
if ENABLE_MICROSOFT and not (MS_CLIENT_ID and MS_CLIENT_SECRET):
    raise RuntimeError("ENABLE_MICROSOFT_AUTH=true but MS_CLIENT_ID/MS_CLIENT_SECRET missing")
if ENABLE_GITHUB and not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
    raise RuntimeError("ENABLE_GITHUB_AUTH=true but GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET missing")

oauth = OAuth()
if ENABLE_GOOGLE:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
if ENABLE_MICROSOFT:
    oauth.register(
        name="microsoft",
        client_id=MS_CLIENT_ID,
        client_secret=MS_CLIENT_SECRET,
        server_metadata_url=f"https://login.microsoftonline.com/{MS_TENANT_ID}/v2.0/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile offline_access"},
    )
if ENABLE_GITHUB:
    oauth.register(
        name="github",
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "user:email"},
    )

app = FastAPI(title="stateless-openid-auth")


def _enabled_providers() -> List[str]:
    out = []
    if ENABLE_GOOGLE:
        out.append("google")
    if ENABLE_MICROSOFT:
        out.append("microsoft")
    if ENABLE_GITHUB:
        out.append("github")
    return out


def _redirect_uri(provider: str) -> str:
    provider = provider.lower()
    if provider == "google" and GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    if provider == "microsoft" and MS_REDIRECT_URI:
        return MS_REDIRECT_URI
    if provider == "github" and GITHUB_REDIRECT_URI:
        return GITHUB_REDIRECT_URI
    # default canonical base for local dev
    return f"{OAUTH_REDIRECT_BASE.rstrip('/')}/auth/callback/{provider}"


@app.get("/redirects", response_class=HTMLResponse)
async def redirects_page():
    provs = _enabled_providers()
    if not provs:
        body = "<div>No providers enabled. Set ENABLE_*_AUTH and provide client credentials.</div>"
    else:
        rows = []
        for p in provs:
            rows.append(f"<li><strong>{p}</strong>: <code>{_redirect_uri(p)}</code></li>")
        body = "<ul>" + "\n".join(rows) + "</ul>"
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Redirect URIs</title></head><body><h2>Redirect URIs to register</h2>{body}</body></html>"
    return HTMLResponse(html)


@app.get("/fragment/providers", response_class=HTMLResponse)
async def fragment_providers():
    providers = _enabled_providers()
    if not providers:
        return HTMLResponse("<div class='text-sm text-gray-600'>No auth providers configured.</div>")
    btns = []
    for p in providers:
        icon = ""
        if p == "google":
            icon = "<img src='https://www.svgrepo.com/show/355037/google.svg' width='16' style='vertical-align:middle;margin-right:8px'/>"
        if p == "microsoft":
            icon = "<img src='https://www.svgrepo.com/show/354054/microsoft.svg' width='16' style='vertical-align:middle;margin-right:8px'/>"
        if p == "github":
            icon = "<img src='https://www.svgrepo.com/show/303145/github-logo.svg' width='16' style='vertical-align:middle;margin-right:8px'/>"
        btns.append(f"<a class='inline-block bg-white border rounded px-3 py-1 mr-2' href='/auth/login/start/{p}'>{icon}{p.capitalize()}</a>")
    return HTMLResponse("<div class='space-x-2'>" + " ".join(btns) + "</div>")


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    providers = _enabled_providers()
    btns = []
    if "google" in providers:
        btns.append(f"<a href='/auth/login/start/google' class='w-full inline-flex items-center justify-center border rounded py-2 px-3 mb-3'><img src='https://www.svgrepo.com/show/355037/google.svg' width='18' style='margin-right:8px'/>Continue with Google</a>")
    if "microsoft" in providers:
        btns.append(f"<a href='/auth/login/start/microsoft' class='w-full inline-flex items-center justify-center border rounded py-2 px-3 mb-3'><img src='https://www.svgrepo.com/show/354054/microsoft.svg' width='18' style='margin-right:8px'/>Continue with Microsoft</a>")
    if "github" in providers:
        btns.append(f"<a href='/auth/login/start/github' class='w-full inline-flex items-center justify-center border rounded py-2 px-3 mb-3'><img src='https://www.svgrepo.com/show/303145/github-logo.svg' width='18' style='margin-right:8px'/>Continue with GitHub</a>")
    btns_html = "\n".join(btns) if btns else "<div class='text-sm text-gray-500'>No providers enabled.</div>"
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<link href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css' rel='stylesheet'>"
        "<title>Sign in</title></head><body class='bg-gray-50 min-h-screen flex items-center justify-center'>"
        "<div class='max-w-md w-full p-6'><div class='bg-white p-6 rounded shadow'>"
        "<h1 class='text-xl font-semibold mb-3'>Sign in</h1>"
        "<p class='text-sm text-gray-600 mb-4'>Please sign in to continue. You will be redirected to the provider for authentication.</p>"
        f"{btns_html}"
        f"<div class='mt-6 text-xs text-gray-400'>Expected redirect URI (register exactly): <code>{OAUTH_REDIRECT_BASE.rstrip('/')}/auth/callback/&lt;provider&gt;</code></div>"
        "</div></div></body></html>"
    )
    return HTMLResponse(html)


@app.get("/login/start/{provider}")
async def login_start(request: Request, provider: str):
    provider = provider.lower()
    if provider not in _enabled_providers():
        raise HTTPException(status_code=404, detail="Provider not enabled")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=500, detail="OAuth client not available")
    redirect_uri = _redirect_uri(provider)
    log.info("Starting OAuth flow for provider=%s redirect_uri=%s", provider, redirect_uri)
    try:
        # Authlib uses request.session to store state; SessionMiddleware must be configured in app.py
        return await client.authorize_redirect(request, redirect_uri)
    except Exception as e:
        log.exception("authorize_redirect failed")
        return HTMLResponse(f"<!doctype html><html><body><h2>OAuth redirect initiation failed</h2><pre>{str(e)}</pre><p>Use /auth/redirects to see exact redirect URIs.</p></body></html>", status_code=500)


@app.get("/callback/{provider}")
async def callback(request: Request, provider: str):
    provider = provider.lower()
    if provider not in _enabled_providers():
        raise HTTPException(status_code=404, detail="Provider not enabled")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=500, detail="OAuth client not available")
    try:
        token = await client.authorize_access_token(request)
    except OAuthError as err:
        err_msg = getattr(err, "error", str(err))
        log.warning("OAuthError for provider %s: %s", provider, err_msg)
        if "redirect_uri_mismatch" in err_msg or "redirect_uri_mismatch" in str(err):
            expected = _redirect_uri(provider)
            return HTMLResponse(
                "<!doctype html><html><body>"
                "<h2>OAuth Redirect URI MISMATCH</h2>"
                "<p>The provider rejected the redirect because it does not match the redirect URI registered in the provider console.</p>"
                f"<p>App will send: <code>{expected}</code></p>"
                "<p>Fix: Register that exact URI in the provider console or set the corresponding *_REDIRECT_URI env var.</p>"
                "</body></html>", status_code=400
            )
        if "mismatching_state" in err_msg or "mismatching_state" in str(err):
            return HTMLResponse(
                "<!doctype html><html><body>"
                "<h2>OAuth State Validation Failed (mismatching_state)</h2>"
                "<p>Possible causes: cookie not sent on callback (SameSite/host mismatch), different host used for login vs callback, or stale browser session.</p>"
                "<p>Fixes: Use the same host (127.0.0.1), set COOKIE_SAMESITE=lax and COOKIE_SECURE=false for local dev, test in fresh Incognito window.</p>"
                "</body></html>", status_code=400
            )
        return RedirectResponse(url=f"{FRONTEND_BASE.rstrip('/')}/auth/success?error=oauth", status_code=302)
    except Exception:
        log.exception("authorize_access_token failed")
        return RedirectResponse(url=f"{FRONTEND_BASE.rstrip('/')}/auth/success?error=oauth", status_code=302)

    # obtain userinfo
    userinfo = {}
    try:
        userinfo = await client.parse_id_token(request, token)
    except Exception:
        userinfo = token.get("userinfo") or {}
        try:
            if provider == "github":
                resp = await client.get("user", token=token)
                u = resp.json()
                userinfo.update(u)
                if "email" not in userinfo:
                    try:
                        emails = await client.get("user/emails", token=token)
                        e = emails.json()
                        if isinstance(e, list) and e:
                            primary = next((x for x in e if x.get("primary")), e[0])
                            userinfo["email"] = primary.get("email")
                    except Exception:
                        log.debug("github: failed to fetch emails")
        except Exception:
            log.debug("failed to enrich github userinfo")

    sub = userinfo.get("sub") or userinfo.get("id") or userinfo.get("node_id")
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("login") or userinfo.get("preferred_username")
    if not sub:
        log.error("Provider returned no subject identifier: %s", userinfo)
        raise HTTPException(status_code=400, detail="Unable to obtain subject from provider response")

    payload = {
        "iss": JWT_ISS,
        "aud": JWT_AUD,
        "sub": str(sub),
        "provider": provider,
        "email": email,
        "name": name,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXP_SECONDS,
    }
    jwt_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    log.info("Issued JWT for sub=%s provider=%s", sub, provider)

    token_js = json.dumps(jwt_token)
    safe_front = FRONTEND_BASE.rstrip("/")

    html = (
        "<!doctype html><html><head><meta charset='utf-8'></head><body>"
        "<div style='font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:#f3f4f6; display:flex;align-items:center;justify-content:center;min-height:100vh'>"
        "<div style='background:white;padding:24px;border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,0.06);max-width:560px;width:90%;text-align:center'>"
        "<h2>Signing in…</h2><p id='msg' style='color:#6b7280;margin-bottom:0'>Completing sign in and redirecting you back.</p>"
        "</div></div>"
        "<script>"
        "try{ console.info('callback: saving token');"
        f" var tok = {token_js};"
        " try{ localStorage.setItem('app_jwt', tok); console.info('token saved to localStorage'); } catch(e){ console.error('localStorage write failed', e); document.getElementById('msg').textContent='Sign-in failed to save token: '+String(e); }"
        f" setTimeout(function(){{ window.location.replace('{safe_front}'); }}, 50);"
        "}catch(e){ console.error('callback client error', e); var m=document.getElementById('msg'); if(m) m.textContent='Sign-in failed: '+String(e); }"
        "</script></body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/success", response_class=HTMLResponse)
async def success_page():
    safe_front = FRONTEND_BASE.rstrip("/")
    return HTMLResponse("<!doctype html><html><head><meta charset='utf-8'></head><body><script>try{window.location.replace('" + safe_front + "');}catch(e){document.body.innerText='Return to app: " + safe_front + "'}</script></body></html>")


@app.get("/me")
async def me(request: Request):
    # Expects Authorization: Bearer <token>
    auth = request.headers.get("authorization", "")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth.split(" ", 1)[1].strip()
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUD, issuer=JWT_ISS)
        return {"authenticated": True, "user": data}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/logout", response_class=HTMLResponse)
async def logout():
    # stateless logout: client should clear localStorage and return to app root
    safe_front = FRONTEND_BASE.rstrip("/")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'></head><body>"
        "<script>try{localStorage.removeItem('app_jwt')}catch(e){}window.location.replace('" + safe_front + "');</script>"
        "</body></html>"
    )
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "stateless-openid-auth", "providers": _enabled_providers()})
