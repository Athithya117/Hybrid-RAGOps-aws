import importlib
import os
import secrets
import sys
import json
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, Response
from starlette.middleware.sessions import SessionMiddleware

# Prometheus metrics
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# use our structured JSON logger
from logger import log

from config import (
    EXTERNAL_BASE as FRONTEND_BASE,
    QUERY_URL,
    COOKIE_NAME,
    SESSION_SECRET,
    COOKIE_SAMESITE,
    COOKIE_SECURE,
    JWT_SECRET,
    enabled_providers_effective,
    get_redirect,
    REQUIRE_AUTH,
)

SERVICE_NAME = os.getenv("SERVICE_NAME", "frontend").strip()
ENV = os.getenv("ENV", "STAGING").upper()

# Prometheus readiness gauge
SERVICE_READY = Gauge("service_ready", "Service readiness (1=ready, 0=not ready)", ["service", "env"])
SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(0)

def _iso_ts():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _json_log(level: str, message: str, **fields):
    lvl = (level or "info").strip().lower()
    try:
        if lvl == "debug":
            log.debug(message, **fields)
        elif lvl in ("warn", "warning"):
            log.warn(message, **fields)
        elif lvl == "info":
            log.info(message, **fields)
        else:
            log.error(message, **fields)
    except Exception:
        try:
            sys.stderr.write(f"logging failed level={lvl} message={message}\n")
        except Exception:
            pass

# Warn about missing secrets but using structured logs
if not JWT_SECRET:
    _json_log("warning", "JWT_SECRET not set; generating ephemeral secret (NOT for production).")
    JWT_SECRET = secrets.token_hex(32)
if not SESSION_SECRET:
    _json_log("warning", "SESSION_SECRET not set; generating ephemeral secret (NOT for production).")
    SESSION_SECRET = secrets.token_hex(32)

OAUTH_REDIRECT_BASE = FRONTEND_BASE

app = FastAPI(title="orchestrator")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie=COOKIE_NAME,
    same_site=COOKIE_SAMESITE,
    https_only=COOKIE_SECURE,
)

try:
    auth_mod = importlib.import_module("stateless_openid_auth")
except Exception as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    _json_log("error", "Failed to import stateless_openid_auth; falling back to stub auth router.", stack=tb)
    from fastapi import APIRouter
    _auth_router = APIRouter()

    @_auth_router.get("/login")
    async def _auth_login():
        return JSONResponse({"error": "auth module unavailable"}, status_code=503)

    class _AuthMod:
        app = _auth_router

    auth_mod = _AuthMod()

try:
    frontend_mod = importlib.import_module("frontend_ui")
except Exception as e:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    _json_log("error", "Failed to import frontend_ui; falling back to stub frontend router.", stack=tb)
    from fastapi import APIRouter
    _fe_router = APIRouter()

    @_fe_router.get("/")
    async def _fe_index():
        return JSONResponse({"error": "frontend module unavailable"}, status_code=503)

    class _FeMod:
        app = _fe_router

    frontend_mod = _FeMod()

from fastapi import FastAPI as _FastAPI
from fastapi.routing import APIRouter as _APIRouter

def _get_router(obj):
    if hasattr(obj, "app"):
        candidate = getattr(obj, "app")
    else:
        candidate = obj
    if isinstance(candidate, _FastAPI):
        return candidate.router
    if isinstance(candidate, _APIRouter):
        return candidate
    if hasattr(candidate, "router"):
        return getattr(candidate, "router")
    return None

auth_router = _get_router(auth_mod)
if auth_router is not None:
    app.include_router(auth_router, prefix="/auth")
else:
    try:
        app.include_router(auth_mod.app, prefix="/auth")
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _json_log("error", "Unable to include auth router cleanly; auth endpoints may be unavailable.", stack=tb)

fe_router = _get_router(frontend_mod)
if fe_router is not None:
    app.include_router(fe_router)
else:
    try:
        app.include_router(frontend_mod.app)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _json_log("error", "Unable to include frontend router cleanly; frontend endpoints may be unavailable.", stack=tb)

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
    _json_log("info", "orchestrator.health", config=cfg, secrets_ok=secrets_ok)
    SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1 if secrets_ok else 0)
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
                _json_log("info", "proxy.run.response", status=resp.status_code, path="/run")
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            else:
                _json_log("info", "proxy.run.response.text", status=resp.status_code, path="/run")
                return JSONResponse(content={"text": resp.text}, status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        status = getattr(e.response, "status_code", None)
        _json_log("error", "Upstream returned non-200", status=status, error=str(e))
        raise HTTPException(status_code=502, detail=f"Upstream error: {status or 'unknown'}")
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _json_log("error", "Upstream call failed", stack=tb)
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
                _json_log("info", "proxy.presign.response", status=resp.status_code, path="/presign")
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                _json_log("info", "proxy.presign.response.text", status=resp.status_code, path="/presign")
                return JSONResponse(content={"text": resp.text}, status_code=resp.status_code)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _json_log("error", "Presign proxy failed", stack=tb)
        raise HTTPException(status_code=502, detail="Presign proxy failed")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# set basic ready state based on secrets presence
SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1 if (JWT_SECRET and SESSION_SECRET) else 0)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port)
