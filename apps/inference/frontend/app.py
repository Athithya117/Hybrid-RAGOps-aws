# apps/inference/frontend/app.py
import importlib
import logging
import os
import sys
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("orchestrator")

required_envs = ["JWT_SECRET", "SESSION_SECRET"]
missing = [v for v in required_envs if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

DEFAULT_BASE = os.getenv("DEFAULT_BASE", "http://127.0.0.1:8000")
OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", DEFAULT_BASE)
FRONTEND_BASE = os.getenv("FRONTEND_BASE", DEFAULT_BASE)

SESSION_SECRET = os.getenv("SESSION_SECRET")
COOKIE_NAME = os.getenv("COOKIE_NAME", "app_session")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in ("1", "true", "yes")

app = FastAPI(title="orchestrator")

# Install SessionMiddleware BEFORE importing/mounting auth app
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie=COOKIE_NAME,
    same_site=COOKIE_SAMESITE,
    https_only=COOKIE_SECURE,
)

try:
    auth_mod = importlib.import_module("stateless_openid_auth")
    frontend_mod = importlib.import_module("frontend_ui")
except Exception:
    log.exception("Failed to import submodules. Confirm stateless_openid_auth.py and frontend_ui.py exist and are valid.")
    raise

app.include_router(auth_mod.app.router, prefix="/auth")
app.include_router(frontend_mod.app.router)

@app.get("/login")
async def login_redirect():
    return RedirectResponse(url="/auth/login")

@app.get("/orchestrator/health")
async def orchestrator_health():
    return JSONResponse({"status": "ok", "auth_mounted": True, "frontend_mounted": True})
