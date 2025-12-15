# apps/inference/forward_auth/stateless_openid_auth.py
import os
import time
import logging
import json
from typing import Any

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from authlib.jose.errors import InvalidClaimError
import jwt

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("forward_auth")

# --- Config & validation (fail-fast unless explicitly allowed) ---
JWT_SECRET = os.getenv("JWT_SECRET")
SESSION_SECRET = os.getenv("SESSION_SECRET")
ALLOW_MISSING = os.getenv("AUTH_ALLOW_MISSING_SECRETS", "false").lower() in ("1", "true", "yes")
if (not JWT_SECRET or not SESSION_SECRET) and not ALLOW_MISSING:
    raise RuntimeError("Missing required env: JWT_SECRET and SESSION_SECRET are required (or set AUTH_ALLOW_MISSING_SECRETS=1)")

DEV_FALLBACK = os.getenv("DEV_FALLBACK", "false").lower() in ("1", "true", "yes")
COOKIE_MODE = os.getenv("COOKIE_MODE", "both").lower()
if COOKIE_MODE not in ("cookie", "localstorage", "both"):
    raise RuntimeError("COOKIE_MODE must be one of cookie|localstorage|both")

OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", os.getenv("FRONTEND_BASE", "http://127.0.0.1:8080"))
FRONTEND_BASE = os.getenv("FRONTEND_BASE", OAUTH_REDIRECT_BASE)
JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "1800"))
JWT_ISS = os.getenv("JWT_ISS", "stateless-openid-auth")
JWT_AUD = os.getenv("JWT_AUD", "rag-ui")

# Providers toggles and credentials
ENABLE_GOOGLE = os.getenv("ENABLE_GOOGLE_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_MICROSOFT = os.getenv("ENABLE_MICROSOFT_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_GITHUB = os.getenv("ENABLE_GITHUB_AUTH", "false").lower() in ("1", "true", "yes")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "common")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# App init
app = FastAPI(title="forward-auth")

# Session middleware is used only during OAuth redirect flows.
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET or "fallback_session_secret",
    session_cookie=os.getenv("COOKIE_NAME", "app_session"),
    same_site=os.getenv("COOKIE_SAMESITE", "lax"),
    https_only=os.getenv("COOKIE_SECURE", "false").lower() in ("1", "true", "yes"),
)

# OAuth clients
oauth = OAuth()
if ENABLE_GOOGLE and (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if ENABLE_MICROSOFT and (MS_CLIENT_ID and MS_CLIENT_SECRET):
    oauth.register(
        name="microsoft",
        client_id=MS_CLIENT_ID,
        client_secret=MS_CLIENT_SECRET,
        server_metadata_url=f"https://login.microsoftonline.com/{MS_TENANT_ID}/v2.0/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile offline_access"},
    )

if ENABLE_GITHUB and (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
    oauth.register(
        name="github",
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "user:email"},
    )


def _enabled_providers() -> list[str]:
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
    base = OAUTH_REDIRECT_BASE.rstrip("/")
    if base.endswith("/auth/callback"):
        return base
    return f"{base}/auth/callback/{p}"


@app.get("/redirects", response_class=HTMLResponse)
async def redirects_page():
    provs = _enabled_providers()
    if not provs:
        return HTMLResponse("<div>No providers enabled. Set ENABLE_*_AUTH and provide client credentials.</div>")
    rows = [f"<li>{p}: <code>{_redirect_uri(p)}</code></li>" for p in provs]
    return HTMLResponse("<ul>" + "\n".join(rows) + "</ul>")


@app.get("/login")
async def login_redirect():
    return RedirectResponse(url="/auth/login")


@app.get("/login/start/{provider}")
async def login_start(request: Request, provider: str):
    provider = provider.lower()
    if provider not in _enabled_providers():
        raise HTTPException(status_code=404, detail="Provider not enabled")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=500, detail="OAuth client not available")
    try:
        return await client.authorize_redirect(request, _redirect_uri(provider))
    except AssertionError:
        log.exception("SessionMiddleware required")
        return HTMLResponse("<h2>Server misconfiguration: SessionMiddleware required</h2>", status_code=500)
    except Exception:
        log.exception("authorize_redirect failed")
        return HTMLResponse("<h2>OAuth redirect initiation failed</h2>", status_code=500)


async def _fetch_userinfo_with_token(client: Any, provider: str, token: dict) -> dict:
    # Prefer standard userinfo; fall back to provider-specific endpoints.
    try:
        resp = await client.get("userinfo", token=token)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
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
                        pass
                return data
    except Exception:
        pass
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
        msg = getattr(err, "error", str(err))
        log.warning("OAuthError: %s", msg)
        if "redirect_uri_mismatch" in msg:
            return HTMLResponse(f"<h2>OAuth Redirect URI MISMATCH</h2><p>Expected: {_redirect_uri(provider)}</p>", status_code=400)
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)
    except Exception:
        log.exception("authorize_access_token failed")
        return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)

    userinfo = {}
    id_token = token.get("id_token") if isinstance(token, dict) else None

    if id_token:
        try:
            userinfo = await client.parse_id_token(request, token)
        except InvalidClaimError:
            pass
        except Exception:
            log.exception("parse_id_token unexpected error")

    if not userinfo:
        userinfo = await _fetch_userinfo_with_token(client, provider, token)

    if not userinfo and id_token:
        if DEV_FALLBACK:
            try:
                claims = jwt.decode(id_token, options={"verify_signature": False, "verify_iss": False})
                userinfo = claims
                log.warning("Used id_token fallback decode for provider=%s (verify_signature=False)", provider)
            except Exception:
                log.exception("fallback id_token decode failed")
        else:
            log.error("Unable to obtain userinfo and DEV_FALLBACK disabled")
            return RedirectResponse(url=f"{FRONTEND_BASE}/auth/success?error=oauth", status_code=302)

    sub = userinfo.get("sub") or userinfo.get("id") or userinfo.get("node_id")
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("login") or userinfo.get("preferred_username")

    if not sub:
        log.error("No subject id from provider: %s", userinfo)
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

    jwt_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256") if JWT_SECRET else ""

    # HTML that stores JWT to localStorage (used in local/dev flows)
    resp_html = (
        "<!doctype html><html><head><meta charset='utf-8'></head><body>"
        "<div>Signing in…</div>"
        "<script>"
        "try{ var tok = " + json.dumps(jwt_token) + ";"
        " try{ localStorage.setItem('app_jwt', tok); }catch(e){}"
        " setTimeout(function(){ window.location.replace('" + FRONTEND_BASE + "'); }, 50); }catch(e){document.body.innerText='Sign-in failed';}"
        "</script></body></html>"
    )

    # If cookie mode enabled, store a minimal server-side session and set an HttpOnly cookie.
    if COOKIE_MODE in ("cookie", "both"):
        request.session["oauth_sub"] = str(sub)
        request.session["email"] = email or ""
        request.session["name"] = name or ""
        r = HTMLResponse(resp_html)
        secure_flag = os.getenv("COOKIE_SECURE", "false").lower() in ("1", "true", "yes") and os.getenv("ENV", "DEV").upper() == "PROD"
        r.set_cookie(
            os.getenv("COOKIE_NAME", "app_session"),
            "1",
            httponly=True,
            secure=secure_flag,
            samesite=os.getenv("COOKIE_SAMESITE", "lax"),
        )
        return r

    return HTMLResponse(resp_html)


@app.get("/auth/check")
async def forward_auth_check(request: Request):
    """
    Traefik ForwardAuth endpoint contract:
      - Return 200 OK when request is allowed. Include authResponseHeaders like X-Auth-User and X-Auth-Email.
      - Return 401 when unauthorized.
      - Optionally return 302 to redirect to /login (Traefik will forward redirect to client).
    """
    # 1) Bearer JWT path
    auth = request.headers.get("authorization", "")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUD, issuer=JWT_ISS)
            headers = {"X-Auth-User": data.get("email", data.get("sub") or ""), "X-Auth-Email": data.get("email", "")}
            return Response(status_code=200, headers=headers)
        except jwt.ExpiredSignatureError:
            return Response(status_code=401)
        except Exception:
            return Response(status_code=401)

    # 2) Cookie / server-session path
    if COOKIE_MODE in ("cookie", "both"):
        sess = request.session if hasattr(request, "session") else {}
        if sess and sess.get("oauth_sub"):
            headers = {"X-Auth-User": sess.get("email", ""), "X-Auth-Email": sess.get("email", "")}
            return Response(status_code=200, headers=headers)

    # 3) Not authenticated — signal Traefik to block
    return Response(status_code=401)


@app.get("/me")
async def me(request: Request):
    # Prefer Authorization header; fall back to cookie/session
    auth = request.headers.get("authorization", "")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUD, issuer=JWT_ISS)
            return {"authenticated": True, "user": data}
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    if COOKIE_MODE in ("cookie", "both"):
        sess = request.session if hasattr(request, "session") else {}
        if sess and sess.get("oauth_sub"):
            return {"authenticated": True, "user": {"sub": sess.get("oauth_sub"), "email": sess.get("email"), "name": sess.get("name")}}

    raise HTTPException(status_code=401, detail="Missing Authorization header or session")


@app.get("/logout")
async def logout(request: Request):
    try:
        request.session.clear()
    except Exception:
        pass
    r = HTMLResponse(
        "<script>try{localStorage.removeItem('app_jwt')}catch(e){}window.location.replace('" + FRONTEND_BASE + "');</script>"
    )
    r.delete_cookie(os.getenv("COOKIE_NAME", "app_session"))
    return r


@app.get("/health")
async def health():
    return {"status": "ok", "service": "forward-auth", "providers": _enabled_providers()}


# Convenience: allow running the file directly for local testing
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("UVICORN_HOST", "127.0.0.1")
    port = int(os.getenv("UVICORN_PORT", "9000"))
    reload_mode = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run(app, host=host, port=port, reload=reload_mode)
