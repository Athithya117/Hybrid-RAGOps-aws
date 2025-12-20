import importlib
import logging
import os
import secrets
import sys
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("orchestrator")


def parse_bool(v, default=False):
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes")


def norm_url(u, default):
    if not u:
        return default
    s = str(u).strip()
    if not s:
        return default
    if s.endswith("/"):
        s = s[:-1]
    if "://" not in s:
        if s.startswith("localhost") or s.startswith("127.") or (":" in s and s.split(":")[0].isdigit()):
            s = "http://" + s
        else:
            s = "https://" + s
    return s


def is_trycloudflare(u):
    if not u:
        return False
    return ".trycloudflare.com" in u.lower()


# secret handling: non-fatal in CI/dev — generate ephemeral secrets with clear warnings
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    log.warning("JWT_SECRET not set; generating ephemeral secret (NOT for production).")
    JWT_SECRET = secrets.token_hex(32)

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    log.warning("SESSION_SECRET not set; generating ephemeral secret (NOT for production).")
    SESSION_SECRET = secrets.token_hex(32)

DEFAULT_BASE = os.getenv("DEFAULT_BASE", "http://127.0.0.1:8000")
FRONTEND_BASE = norm_url(os.getenv("FRONTEND_BASE") or os.getenv("FRONTEND_URL") or DEFAULT_BASE, DEFAULT_BASE)
OAUTH_REDIRECT_BASE_RAW = os.getenv("OAUTH_REDIRECT_BASE")
OAUTH_REDIRECT_BASE = norm_url(OAUTH_REDIRECT_BASE_RAW or FRONTEND_BASE, FRONTEND_BASE)

TEMP_DOMAIN = parse_bool(os.getenv("TEMP_DOMAIN"), False)
CLOUDFLARE_HOSTNAME = os.getenv("CLOUDFLARE_HOSTNAME") or ""
EXPLICIT_COOKIE_SECURE = os.getenv("COOKIE_SECURE")
if EXPLICIT_COOKIE_SECURE is not None:
    COOKIE_SECURE = parse_bool(EXPLICIT_COOKIE_SECURE, False)
else:
    COOKIE_SECURE = FRONTEND_BASE.lower().startswith("https://")
COOKIE_NAME = os.getenv("COOKIE_NAME", "app_session")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

if TEMP_DOMAIN:
    if not is_trycloudflare(FRONTEND_BASE):
        log.warning("TEMP_DOMAIN=true but FRONTEND_BASE is not a trycloudflare URL; continuing (non-fatal).")
    if CLOUDFLARE_HOSTNAME:
        log.warning("TEMP_DOMAIN=true and CLOUDFLARE_HOSTNAME set; ignoring CLOUDFLARE_HOSTNAME for this run.")
else:
    if is_trycloudflare(FRONTEND_BASE) and not CLOUDFLARE_HOSTNAME:
        log.warning("Running with trycloudflare URL while TEMP_DOMAIN=false and CLOUDFLARE_HOSTNAME not set.")

app = FastAPI(title="orchestrator")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie=COOKIE_NAME,
    same_site=COOKIE_SAMESITE,
    https_only=COOKIE_SECURE,
)

# Import auth router defensively; log exception and fall back to stub router
try:
    auth_mod = importlib.import_module("stateless_openid_auth")
except Exception:
    log.exception("Failed to import stateless_openid_auth; falling back to stub auth router.")
    from fastapi import APIRouter
    _auth_router = APIRouter()

    @_auth_router.get("/login")
    async def _auth_login():
        return JSONResponse({"error": "auth module unavailable"}, status_code=503)

    @_auth_router.get("/health")
    async def _auth_health():
        return {"status": "stub", "module": "stateless_openid_auth", "available": False}

    class _AuthMod:
        app = APIRouter()

    _AuthMod.app = _auth_router
    auth_mod = _AuthMod

# Import frontend router defensively; log exception and fall back to stub router
try:
    frontend_mod = importlib.import_module("frontend_ui")
except Exception:
    log.exception("Failed to import frontend_ui; falling back to stub frontend router.")
    from fastapi import APIRouter
    _fe_router = APIRouter()

    @_fe_router.get("/")
    async def _fe_index():
        return JSONResponse({"error": "frontend module unavailable"}, status_code=503)

    @_fe_router.get("/health")
    async def _fe_health():
        return {"status": "stub", "module": "frontend_ui", "available": False}

    class _FeMod:
        app = APIRouter()

    _FeMod.app = _fe_router
    frontend_mod = _FeMod

app.include_router(
    auth_mod.app.router if hasattr(auth_mod, "app") and hasattr(auth_mod.app, "router") else auth_mod.app,
    prefix="/auth",
)
app.include_router(
    frontend_mod.app.router if hasattr(frontend_mod, "app") and hasattr(frontend_mod.app, "router") else frontend_mod.app
)


@app.get("/login")
async def login_redirect():
    return RedirectResponse(url="/auth/login")


@app.get("/orchestrator/health")
async def orchestrator_health():
    cfg = {
        "frontend_base": FRONTEND_BASE,
        "oauth_redirect_base": OAUTH_REDIRECT_BASE,
        "temp_domain": TEMP_DOMAIN,
        "cloudflare_hostname": bool(CLOUDFLARE_HOSTNAME),
        "cookie_secure": COOKIE_SECURE,
    }
    secrets_ok = bool(JWT_SECRET and SESSION_SECRET)
    masked = lambda s: ("<set>" if s else "<unset>")
    return JSONResponse(
        {
            "status": "ok",
            "secrets_ok": secrets_ok,
            "config": cfg,
            "masked": {"jwt_secret": masked(JWT_SECRET), "session_secret": masked(SESSION_SECRET)},
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port)
