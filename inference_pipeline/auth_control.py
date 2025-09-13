# auth_control.py
import os
from datetime import timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
import aioredis
import jwt  # PyJWT

# -----------------------------
# Config / Environment Variables
# -----------------------------
VALKEY_URL = os.getenv("VALKEY_URL", "redis://valkey:6379/0")
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "1000"))  # per user

OAUTH2_CLIENT_ID = os.getenv("OAUTH2_CLIENT_ID")
OAUTH2_CLIENT_SECRET = os.getenv("OAUTH2_CLIENT_SECRET")
ALLOWED_DOMAINS = os.getenv("ALLOWED_DOMAINS", "company.com").split(",")

# -----------------------------
# Redis/Valkey Connection
# -----------------------------
redis = aioredis.from_url(VALKEY_URL, decode_responses=True)

# -----------------------------
# OAuth2 (SSO) Setup
# -----------------------------
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/auth",
    tokenUrl="https://oauth2.googleapis.com/token"
)

# -----------------------------
# Auth + Rate Limit Functions
# -----------------------------
async def verify_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Verify JWT token from OAuth2 provider and ensure allowed domain.
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False})  # verify externally in prod
        email = payload.get("email")
        if not email or not any(email.endswith(f"@{d}") for d in ALLOWED_DOMAINS):
            raise HTTPException(status_code=403, detail="Unauthorized domain")
        return {"user_id": payload.get("sub"), "email": email}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def check_rate_limit(user_id: str) -> bool:
    """
    Increment user's daily counter in Valkey. Returns True if within limit, False if exceeded.
    """
    key = f"user:{user_id}:daily"
    # Use atomic INCR with TTL set to 24h if new
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, timedelta(days=1))
    if current > RATE_LIMIT_PER_DAY:
        return False
    return True

async def auth_and_limit(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Combines auth and rate limiting.
    Returns dict: {"user": user_info, "llm_allowed": bool}
    """
    user = await verify_user(token)
    llm_allowed = await check_rate_limit(user["user_id"])
    return {"user": user, "llm_allowed": llm_allowed}

# -----------------------------
# Example FastAPI usage
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="Auth + RateLimit")

    @app.get("/query")
    async def query(info: dict = Depends(auth_and_limit)):
        if not info["llm_allowed"]:
            return {"mode": "NoLLM", "message": "Daily LLM limit exceeded"}
        return {"mode": "LLM", "message": "LLM request allowed"}

    uvicorn.run(app, host="0.0.0.0", port=8000)
