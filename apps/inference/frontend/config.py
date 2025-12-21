# apps/inference/frontend/config.py
import os
from urllib.parse import urlparse

def parse_bool_env(v, default=False):
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

def parse_list_env(v):
    if not v:
        return []
    parts = []
    for p in str(v).split(","):
        s = p.strip()
        if s:
            parts.append(s)
    return parts

FRONTEND_HOSTNAME = (os.getenv("FRONTEND_HOSTNAME") or "").strip()
DEFAULT_LOCAL = "http://127.0.0.1:8000"
if FRONTEND_HOSTNAME:
    EXTERNAL_BASE = norm_url(f"https://{FRONTEND_HOSTNAME}", DEFAULT_LOCAL)
else:
    EXTERNAL_BASE = norm_url(os.getenv("FRONTEND_BASE") or os.getenv("FRONTEND_URL") or DEFAULT_LOCAL, DEFAULT_LOCAL)

QUERY_URL = norm_url(os.getenv("QUERY_URL") or "http://retrieval-svc.inference.svc.cluster.local:8001", "http://retrieval-svc.inference.svc.cluster.local:8001")

REQUIRE_AUTH = parse_bool_env(os.getenv("REQUIRE_AUTH"), False)
DISPLAY_SOURCES_IN_UI = parse_bool_env(os.getenv("DISPLAY_SOURCES_IN_UI"), True)
DISPLAY_TOPK_IN_UI = parse_bool_env(os.getenv("DISPLAY_TOPK_IN_UI"), True)

COOKIE_NAME = os.getenv("COOKIE_NAME", "app_session")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
_COOKIE_SEC = os.getenv("COOKIE_SECURE")
if _COOKIE_SEC is not None:
    COOKIE_SECURE = parse_bool_env(_COOKIE_SEC, False)
else:
    COOKIE_SECURE = EXTERNAL_BASE.lower().startswith("https://")

ENABLE_GOOGLE = parse_bool_env(os.getenv("ENABLE_GOOGLE_AUTH"), False)
ENABLE_MICROSOFT = parse_bool_env(os.getenv("ENABLE_MICROSOFT_AUTH"), False)
ENABLE_GITHUB = parse_bool_env(os.getenv("ENABLE_GITHUB_AUTH"), False)

GOOGLE_CLIENT_ID = (os.getenv("GOOGLE_CLIENT_ID") or "").strip()
GOOGLE_CLIENT_SECRET = (os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()
GOOGLE_ALLOWED_DOMAINS = set([s.strip().lower() for s in parse_list_env(os.getenv("GOOGLE_ALLOWED_DOMAINS"))])

MS_CLIENT_ID = (os.getenv("MS_CLIENT_ID") or "").strip()
MS_CLIENT_SECRET = (os.getenv("MS_CLIENT_SECRET") or "").strip()
MS_TENANT_ID = (os.getenv("MS_TENANT_ID") or os.getenv("AZURE_TENANT_ID") or "common").strip()
MICROSOFT_ALLOWED_DOMAINS = set([s.strip().lower() for s in parse_list_env(os.getenv("MICROSOFT_ALLOWED_DOMAINS"))])
MICROSOFT_ALLOWED_TENANT_IDS = set([s.strip().lower() for s in parse_list_env(os.getenv("MICROSOFT_ALLOWED_TENANT_IDS"))])

GITHUB_CLIENT_ID = (os.getenv("GITHUB_CLIENT_ID") or "").strip()
GITHUB_CLIENT_SECRET = (os.getenv("GITHUB_CLIENT_SECRET") or "").strip()
GITHUB_ALLOWED_ORGS = set([s.strip().lower() for s in parse_list_env(os.getenv("GITHUB_ALLOWED_ORGS"))])

GOOGLE_REDIRECT_URI = (os.getenv("GOOGLE_REDIRECT_URI") or "").strip()
MS_REDIRECT_URI = (os.getenv("MS_REDIRECT_URI") or "").strip()
GITHUB_REDIRECT_URI = (os.getenv("GITHUB_REDIRECT_URI") or "").strip()

JWT_SECRET = os.getenv("JWT_SECRET") or ""
SESSION_SECRET = os.getenv("SESSION_SECRET") or ""
JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "1800"))
JWT_ISS = os.getenv("JWT_ISS", "stateless-openid-auth")
JWT_AUD = os.getenv("JWT_AUD", "rag-ui")

def get_redirect(provider: str) -> str:
    p = provider.lower()
    base = EXTERNAL_BASE.rstrip("/")
    # allow explicit override envs first
    if p == "google" and GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    if p == "microsoft" and MS_REDIRECT_URI:
        return MS_REDIRECT_URI
    if p == "github" and GITHUB_REDIRECT_URI:
        return GITHUB_REDIRECT_URI
    if base.endswith("/auth/callback"):
        return f"{base}/{p}"
    return f"{base}/auth/callback/{p}"

def enabled_flags():
    return {"google": ENABLE_GOOGLE, "microsoft": ENABLE_MICROSOFT, "github": ENABLE_GITHUB}

def enabled_providers_effective():
    out = []
    flags = enabled_flags()
    if flags.get("google") and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        out.append("google")
    if flags.get("microsoft") and MS_CLIENT_ID and (MS_CLIENT_SECRET or os.getenv("MS_CLIENT_SECRET") or os.getenv("MS_CLIENT_CERT")):
        out.append("microsoft")
    if flags.get("github") and GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
        out.append("github")
    return out
