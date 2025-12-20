import os
import time
import logging
import json
from typing import List, Any, Set
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from authlib.jose.errors import InvalidClaimError
import jwt

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("stateless_openid_auth")

# Non-fatal secret handling for CI/dev: generate ephemeral if missing, but warn loudly.
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    log.warning("JWT_SECRET not set; generating ephemeral secret (NOT for production).")
    JWT_SECRET = os.urandom(32).hex()

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    log.warning("SESSION_SECRET not set; generating ephemeral secret (NOT for production).")
    SESSION_SECRET = os.urandom(32).hex()

OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", os.getenv("CANONICAL_BASE", "http://127.0.0.1:8000")).rstrip("/")
_fb = os.getenv("FRONTEND_BASE")
if _fb:
    FRONTEND_BASE = _fb.rstrip("/")
else:
    FRONTEND_BASE = OAUTH_REDIRECT_BASE.split("/auth/callback")[0].rstrip("/")

JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "1800"))
JWT_ISS = os.getenv("JWT_ISS", "stateless-openid-auth")
JWT_AUD = os.getenv("JWT_AUD", "rag-ui")

ENABLE_GOOGLE = os.getenv("ENABLE_GOOGLE_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_MICROSOFT = os.getenv("ENABLE_MICROSOFT_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_GITHUB = os.getenv("ENABLE_GITHUB_AUTH", "false").lower() in ("1", "true", "yes")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "common")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI")


def _parse_allowed_env(name: str) -> Set[str]:
    v = os.getenv(name, "")
    out = set()
    if not v:
        return out
    for x in v.split(","):
        s = x.strip().lower()
        if s:
            out.add(s)
    return out


GOOGLE_ALLOWED_DOMAINS = _parse_allowed_env("GOOGLE_ALLOWED_DOMAINS")
MICROSOFT_ALLOWED_DOMAINS = _parse_allowed_env("MICROSOFT_ALLOWED_DOMAINS")
MICROSOFT_ALLOWED_TENANT_IDS = _parse_allowed_env("MICROSOFT_ALLOWED_TENANT_IDS")
GITHUB_ALLOWED_DOMAINS = _parse_allowed_env("GITHUB_ALLOWED_DOMAINS")
GITHUB_ALLOWED_ORGS = _parse_allowed_env("GITHUB_ALLOWED_ORGS")

_GOOGLE_SVG = ('<svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg"'
' aria-hidden="true" role="img" style="vertical-align:middle">'
'<path fill="#EA4335" d="M12 10.2v3.6h5.2c-.2 1.2-1.4 3.6-5.2 '
'3.6-3.1 0-5.6-2.6-5.6-5.8S8.9 6.8 12 6.8c1.8 0 2.9.8 3.6 1.5l2.4-2.3C17.2 4 14.8 3 12 '
'3 7.6 3 4 6.6 4 11s3.6 8 8 8c4.6 0 7-3.2 7-7.7 0-.5 0-.9-.1-1.1H12z"/></svg>')
_MICROSOFT_SVG = ('<svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" role="img" style="vertical-align:middle">'
'<rect x="2" y="2" width="9" height="9" fill="#F35325"/><rect x="13" y="2" width="9" height="9" fill="#81BC06"/>'
'<rect x="2" y="13" width="9" height="9" fill="#05A6F0"/><rect x="13" y="13" width="9" height="9" fill="#FFBA08"/>'
"</svg>")
_GITHUB_SVG = ('<svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" role="img" style="vertical-align:middle">'
'<path fill="#111" d="M12 .5C5.6.5.5 5.6.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.3.8-.6v-2.2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.1-1.2-1.4-1.2-1.4-1-.7.1-.7.1-.7 1.1.1 1.7 1.1 1.7 1.1 1 .1 1.6.8 2 .6.1-.8.4-1.4.7-1.8-2.6-.3-5.4-1.3-5.4-5.8 0-1.3.5-2.4 1.3-3.2-.1-.3-.6-1.6.1-3.3 0 0 1-.3 3.3 1.3.9-.3 1.9-.5 2.9-.5s2 .2 2.9.5c2.3-1.6 3.3-1.3 3.3-1.3.7 1.7.2 3 .1 3.3.8.8 1.3 1.9 1.3 3.2 0 4.5-2.8 5.5-5.5 5.8.5.4.8 1.1.8 2.3v3.4c0 .3.2.8.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.6 18.4.5 12 .5z"/>'
"</svg>")

oauth = OAuth()

# Register providers only when both id and secret are present; otherwise leave disabled and warn
if ENABLE_GOOGLE and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(name="google", client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET, server_metadata_url="https://accounts.google.com/.well-known/openid-configuration", client_kwargs={"scope": "openid email profile"})
else:
    if ENABLE_GOOGLE:
        log.warning("ENABLE_GOOGLE_AUTH=true but GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET missing; Google login disabled for this run.")
        ENABLE_GOOGLE = False

if ENABLE_MICROSOFT and MS_CLIENT_ID and MS_CLIENT_SECRET:
    oauth.register(name="microsoft", client_id=MS_CLIENT_ID, client_secret=MS_CLIENT_SECRET, server_metadata_url=f"https://login.microsoftonline.com/{MS_TENANT_ID}/v2.0/.well-known/openid-configuration", client_kwargs={"scope": "openid email profile offline_access"})
else:
    if ENABLE_MICROSOFT:
        log.warning("ENABLE_MICROSOFT_AUTH=true but MS_CLIENT_ID/MS_CLIENT_SECRET missing; Microsoft login disabled for this run.")
        ENABLE_MICROSOFT = False

if ENABLE_GITHUB and GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(name="github", client_id=GITHUB_CLIENT_ID, client_secret=GITHUB_CLIENT_SECRET, access_token_url="https://github.com/login/oauth/access_token", authorize_url="https://github.com/login/oauth/authorize", api_base_url="https://api.github.com/", client_kwargs={"scope": "user:email read:org"})
else:
    if ENABLE_GITHUB:
        log.warning("ENABLE_GITHUB_AUTH=true but GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET missing; GitHub login disabled for this run.")
        ENABLE_GITHUB = False

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
    p = provider.lower()
    if p == "google" and GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    if p == "microsoft" and MS_REDIRECT_URI:
        return MS_REDIRECT_URI
    if p == "github" and GITHUB_REDIRECT_URI:
        return GITHUB_REDIRECT_URI
    base = OAUTH_REDIRECT_BASE
    if base.endswith("/auth/callback"):
        return f"{base}/{p}"
    return f"{base}/auth/callback/{p}"


@app.get("/redirects", response_class=HTMLResponse)
async def redirects_page():
    provs = _enabled_providers()
    if not provs:
        body = "<div>No providers enabled. Set ENABLE_*_AUTH and provide client credentials.</div>"
    else:
        rows = [f"<li><strong>{p}</strong>: <code>{_redirect_uri(p)}</code></li>" for p in provs]
        body = "<ul>" + "\n".join(rows) + "</ul>"
    html = "<!doctype html><html><head><meta charset='utf-8'><title>Redirect URIs</title></head><body><h2>Redirect URIs to register</h2>" + body + "</body></html>"
    return HTMLResponse(html)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    providers = _enabled_providers()
    icons = {"google": _GOOGLE_SVG, "microsoft": _MICROSOFT_SVG, "github": _GITHUB_SVG}
    if not providers:
        btns_html = "<div class='text-sm text-gray-500'>No providers enabled.</div>"
    else:
        btns = []
        for p in providers:
            btns.append("<a href='/auth/login/start/{p}' class='w-full inline-flex items-center justify-center border rounded py-2 px-3 mb-3' aria-label='Continue with {cap}'>{svg}<span style='margin-left:8px'>Continue with {cap}</span></a>".format(p=p, cap=p.capitalize(), svg=icons.get(p, "")))
        btns_html = "\n".join(btns)
    expected_hint = f"{OAUTH_REDIRECT_BASE}/auth/callback/<provider>" if not OAUTH_REDIRECT_BASE.endswith("/auth/callback") else f"{OAUTH_REDIRECT_BASE}/<provider>"
    html = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<link href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css' rel='stylesheet'>"
            "<title>Sign in</title></head><body class='bg-gray-50 min-h-screen flex items-center justify-center'>"
            "<div class='max-w-md w-full p-6'><div class='bg-white p-6 rounded shadow'>"
            "<h1 class='text-xl font-semibold mb-3'>Sign in</h1>"
            f"{btns_html}"
            f"<div class='mt-6 text-xs text-gray-400'>Expected redirect URIs: <code>{expected_hint}</code></div>"
            "</div></div></body></html>")
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
        return await client.authorize_redirect(request, redirect_uri)
    except AssertionError:
        log.exception("SessionMiddleware required but not installed")
        return HTMLResponse("<h2>Server misconfiguration: SessionMiddleware required</h2>", status_code=500)
    except Exception:
        log.exception("authorize_redirect failed")
        return HTMLResponse("<h2>OAuth redirect initiation failed — see server logs</h2>", status_code=500)


async def _fetch_userinfo_with_token(client: Any, provider: str, token: dict) -> dict:
    try:
        resp = await client.get("userinfo", token=token)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        log.debug("userinfo endpoint failed for %s", provider)
    try:
        if provider == "github":
            resp = await client.get("user", token=token)
            if resp.status_code == 200:
                data = resp.json()
                if "email" not in data:
                    try:
                        emails = await client.get("user/emails", token=token)
                        e = emails.json()
                        if isinstance(e, list) and e:
                            primary = next((x for x in e if x.get("primary")), e[0])
                            data["email"] = primary.get("email")
                    except Exception:
                        log.debug("github: failed to fetch emails")
                return data
    except Exception:
        log.debug("provider-specific userinfo fallback failed for %s", provider)
    return {}


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
        if "redirect_uri_mismatch" in err_msg:
            expected = _redirect_uri(provider)
            return HTMLResponse(f"<h2>OAuth Redirect URI MISMATCH</h2><p>App will send: <code>{expected}</code></p>", status_code=400)
        if "mismatching_state" in err_msg:
            return HTMLResponse("<h2>OAuth State Validation Failed</h2><p>Use same host and set COOKIE_SAMESITE=lax & COOKIE_SECURE=false for local dev.</p>", status_code=400)
        if "invalid_client" in err_msg:
            return HTMLResponse("<h2>OAuth client invalid</h2><p>Check client id/secret and app registration.</p>", status_code=400)
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
    except Exception:
        log.exception("authorize_access_token failed")
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
    userinfo = {}
    id_token = token.get("id_token") if isinstance(token, dict) else None
    if id_token:
        try:
            userinfo = await client.parse_id_token(request, token)
        except InvalidClaimError as ice:
            log.warning("parse_id_token InvalidClaimError for %s: %s", provider, str(ice))
        except Exception:
            log.exception("parse_id_token failed with unexpected error")
    if not userinfo:
        userinfo = await _fetch_userinfo_with_token(client, provider, token)
    if not userinfo and id_token:
        try:
            expected_aud = {"google": GOOGLE_CLIENT_ID, "microsoft": MS_CLIENT_ID, "github": GITHUB_CLIENT_ID}.get(provider)
            claims = jwt.decode(id_token, options={"verify_signature": False, "verify_iss": False}, audience=expected_aud)
            if claims.get("exp") and int(claims["exp"]) < int(time.time()):
                raise HTTPException(status_code=401, detail="id_token expired")
            userinfo = claims
            log.warning("Used id_token fallback decode for %s (verify_signature=False); claims keys: %s", provider, list(claims.keys()))
        except jwt.InvalidAudienceError:
            log.exception("id_token audience mismatch in fallback decode")
            return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
        except Exception:
            log.exception("id_token fallback decode failed")
            return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
    if not userinfo:
        log.error("Unable to obtain userinfo for provider %s", provider)
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
    sub = userinfo.get("sub") or userinfo.get("id") or userinfo.get("node_id")
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("login") or userinfo.get("preferred_username")
    if not sub:
        log.error("Provider returned no subject identifier: %s", userinfo)
        raise HTTPException(status_code=400, detail="Unable to obtain subject from provider response")
    if not email:
        log.error("No email from provider %s: %s", provider, userinfo)
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
    dom = email.split("@", 1)[1].lower() if "@" in email else ""
    if provider == "google" and GOOGLE_ALLOWED_DOMAINS and dom not in GOOGLE_ALLOWED_DOMAINS:
        log.warning("google login from disallowed domain %s", dom)
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=forbidden", status_code=302)
    if provider == "microsoft":
        ok = False
        if MICROSOFT_ALLOWED_DOMAINS and dom in MICROSOFT_ALLOWED_DOMAINS:
            ok = True
        tid = None
        if isinstance(userinfo, dict):
            tid = userinfo.get("tid") or userinfo.get("tenant")
        if not ok and tid and MICROSOFT_ALLOWED_TENANT_IDS and tid.lower() in MICROSOFT_ALLOWED_TENANT_IDS:
            ok = True
        if not ok and MICROSOFT_ALLOWED_DOMAINS:
            ok = dom in MICROSOFT_ALLOWED_DOMAINS
        if not ok:
            log.warning("microsoft login rejected domain=%s tid=%s", dom, tid)
            return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=forbidden", status_code=302)
    if provider == "github":
        ok = False
        if GITHUB_ALLOWED_DOMAINS and dom in GITHUB_ALLOWED_DOMAINS:
            ok = True
        if not ok and GITHUB_ALLOWED_ORGS and token.get("access_token"):
            try:
                resp = await client.get("user/orgs", token=token)
                if resp.status_code == 200:
                    orgs = resp.json()
                    org_names = [o.get("login", "").lower() for o in orgs if isinstance(o, dict)]
                    if any(o in GITHUB_ALLOWED_ORGS for o in org_names):
                        ok = True
            except Exception:
                log.exception("github org check failed")
        if not ok:
            log.warning("github login rejected domain=%s", dom)
            return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=forbidden", status_code=302)
    payload = {"iss": JWT_ISS, "aud": JWT_AUD, "sub": str(sub), "provider": provider, "email": email, "name": name, "iat": int(time.time()), "exp": int(time.time()) + JWT_EXP_SECONDS}
    jwt_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    log.info("Issued JWT for sub=%s provider=%s", sub, provider)
    token_js = json.dumps(jwt_token)
    safe_front = FRONTEND_BASE
    html = ("<!doctype html><html><head><meta charset='utf-8'></head><body>"
            "<div style='font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:#f3f4f6; display:flex;align-items:center;justify-content:center;min-height:100vh'>"
            "<div style='background:white;padding:24px;border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,0.06);max-width:560px;width:90%;text-align:center'>"
            "<h2>Signing in…</h2><p id='msg' style='color:#6b7280;margin-bottom:0'>Completing sign in and redirecting you back.</p>"
            "</div></div>"
            "<script>"
            "try{ var tok = " + token_js + "; try{ localStorage.setItem('app_jwt', tok); } catch(e){} setTimeout(function(){ window.location.replace('" + safe_front + "'); }, 50); }catch(e){document.body.innerText='Sign-in failed: '+String(e); }"
            "</script></body></html>")
    return HTMLResponse(content=html)


@app.get("/success", response_class=HTMLResponse)
async def success_page():
    return HTMLResponse("<!doctype html><html><head><meta charset='utf-8'></head><body><script>try{window.location.replace('" + FRONTEND_BASE + "');}catch(e){document.body.innerText='Return to app: " + FRONTEND_BASE + "'}</script></body></html>")


@app.get("/me")
async def me(request: Request):
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
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body><script>try{localStorage.removeItem('app_jwt')}catch(e){}window.location.replace('" + FRONTEND_BASE + "');</script></body></html>"
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "stateless-openid-auth", "providers": _enabled_providers()})
