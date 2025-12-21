# apps/inference/frontend/app.py
import importlib, logging, os, secrets, sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import EXTERNAL_BASE as FRONTEND_BASE, QUERY_URL, COOKIE_NAME, SESSION_SECRET, COOKIE_SAMESITE, COOKIE_SECURE, JWT_SECRET, enabled_providers_effective, get_redirect, REQUIRE_AUTH

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("orchestrator")

if not JWT_SECRET:
    log.warning("JWT_SECRET not set; generating ephemeral secret (NOT for production).")
    JWT_SECRET = secrets.token_hex(32)
if not SESSION_SECRET:
    log.warning("SESSION_SECRET not set; generating ephemeral secret (NOT for production).")
    SESSION_SECRET = secrets.token_hex(32)

OAUTH_REDIRECT_BASE = FRONTEND_BASE

app = FastAPI(title="orchestrator")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie=COOKIE_NAME, same_site=COOKIE_SAMESITE, https_only=COOKIE_SECURE)

# safe include_router helper: accepts modules that export either an APIRouter, or a FastAPI instance (we use its .router)
from fastapi import FastAPI as _FastAPI
from fastapi.routing import APIRouter as _APIRouter

def _get_router(obj):
    # obj may be the module, or module.app may be APIRouter or FastAPI or router-like
    if hasattr(obj, "app"):
        candidate = getattr(obj, "app")
    else:
        candidate = obj
    if isinstance(candidate, _FastAPI):
        return candidate.router
    if isinstance(candidate, _APIRouter):
        return candidate
    # last resort: try attribute 'router'
    if hasattr(candidate, "router"):
        return getattr(candidate, "router")
    # unknown type, return None
    return None

# load auth module (fall back stub)
try:
    auth_mod = importlib.import_module("stateless_openid_auth")
except Exception:
    log.exception("Failed to import stateless_openid_auth; falling back to stub auth router.")
    from fastapi import APIRouter
    _auth_router = APIRouter()
    @_auth_router.get("/login")
    async def _auth_login():
        return JSONResponse({"error": "auth module unavailable"}, status_code=503)
    class _AuthMod:
        app = _auth_router
    auth_mod = _AuthMod()

# load frontend UI module (fall back stub)
try:
    frontend_mod = importlib.import_module("frontend_ui")
except Exception:
    log.exception("Failed to import frontend_ui; falling back to stub frontend router.")
    from fastapi import APIRouter
    _fe_router = APIRouter()
    @_fe_router.get("/")
    async def _fe_index():
        return JSONResponse({"error": "frontend module unavailable"}, status_code=503)
    class _FeMod:
        app = _fe_router
    frontend_mod = _FeMod()

# include routers using the helper
auth_router = _get_router(auth_mod)
if auth_router is not None:
    app.include_router(auth_router, prefix="/auth")
else:
    # last-ditch: if auth_mod itself is a router-like object
    try:
        app.include_router(auth_mod.app, prefix="/auth")
    except Exception:
        log.error("Unable to include auth router cleanly; auth endpoints may be unavailable.")

fe_router = _get_router(frontend_mod)
if fe_router is not None:
    app.include_router(fe_router)
else:
    try:
        app.include_router(frontend_mod.app)
    except Exception:
        log.error("Unable to include frontend router cleanly; frontend endpoints may be unavailable.")

@app.get("/login")
async def login_redirect():
    return RedirectResponse(url="/auth/login")

@app.get("/orchestrator/health")
async def orchestrator_health():
    provs = enabled_providers_effective()
    redirects = {p: get_redirect(p) for p in provs}
    cfg = {"frontend_base": FRONTEND_BASE, "oauth_redirect_base": OAUTH_REDIRECT_BASE, "require_auth": REQUIRE_AUTH, "enabled_providers": provs, "redirects": redirects}
    secrets_ok = bool(JWT_SECRET and SESSION_SECRET)
    masked = lambda s: ("<set>" if s else "<unset>")
    return JSONResponse({"status": "ok", "secrets_ok": secrets_ok, "config": cfg, "masked": {"jwt_secret": masked(JWT_SECRET), "session_secret": masked(SESSION_SECRET)}})

@app.post("/run")
async def run(request: Request):
    if REQUIRE_AUTH:
        auth = request.headers.get("authorization", "")
        if not auth or not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict) or not body.get("query"):
        raise HTTPException(status_code=400, detail="Missing 'query'")
    target = QUERY_URL.rstrip("/") + "/generate"
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    client_host = request.client.host if request.client else None
    if client_host:
        headers["X-Forwarded-For"] = client_host
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(target, json=body, headers=headers)
            content_type = resp.headers.get("content-type", "")
            if "application/json" in (content_type or ""):
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            else:
                return JSONResponse(content={"text": resp.text}, status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        log.error("Upstream returned non-200: %s", str(e))
        raise HTTPException(status_code=502, detail=f"Upstream error: {getattr(e.response, 'status_code', 'unknown')}")
    except Exception:
        log.exception("Upstream call failed")
        raise HTTPException(status_code=502, detail="Upstream call failed")

@app.post("/presign")
async def presign(request: Request):
    if REQUIRE_AUTH:
        auth = request.headers.get("authorization", "")
        if not auth or not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    target = QUERY_URL.rstrip("/") + "/presign"
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(target, json=body, headers=headers)
            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                return JSONResponse(content={"text": resp.text}, status_code=resp.status_code)
    except Exception:
        log.exception("Presign proxy failed")
        raise HTTPException(status_code=502, detail="Presign proxy failed")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port)
