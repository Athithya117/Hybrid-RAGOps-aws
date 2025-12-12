# file: gateway_auth.py
import os, time, json, asyncio, asyncpg
from fastapi import FastAPI, Request, HTTPException, Depends
from jose import jwt
import httpx
from starlette.middleware.base import BaseHTTPMiddleware

DB_CONN=os.getenv("DB_CONN")
CACHE_TTL=int(os.getenv("CACHE_TTL","60"))
DEFAULT_PROVIDER=os.getenv("DEFAULT_PROVIDER")
API_KEY_HEADER=os.getenv("API_KEY_HEADER","x-api-key")
if not DB_CONN or not DEFAULT_PROVIDER:
    raise RuntimeError("Missing required env vars")

app=FastAPI()

class ProviderCache:
    def __init__(self):
        self.data={}
        self.exp={}
    async def load(self,tenant_id,conn):
        now=time.time()
        if tenant_id in self.data and self.exp[tenant_id]>now:
            return self.data[tenant_id]
        row=await conn.fetchrow("select issuer,jwks_url,audience from auth_providers where tenant_id=$1",tenant_id)
        if not row:
            raise HTTPException(401,"Unknown tenant")
        cfg=dict(row)
        self.data[tenant_id]=cfg
        self.exp[tenant_id]=now+CACHE_TTL
        return cfg

provider_cache=ProviderCache()

async def fetch_jwks(url):
    async with httpx.AsyncClient(timeout=5) as c:
        r=await c.get(url)
        if r.status_code!=200:
            raise HTTPException(401,"JWKS fetch failed")
        return r.json()["keys"]

async def validate_oidc(request,token):
    payload=jwt.get_unverified_claims(token)
    tenant_id=payload.get("tid") or DEFAULT_PROVIDER
    conn=request.app.state.db
    cfg=await provider_cache.load(tenant_id,conn)
    jwks=await fetch_jwks(cfg["jwks_url"])
    try:
        return jwt.decode(token,jwks,audience=cfg["audience"],issuer=cfg["issuer"])
    except:
        raise HTTPException(401,"Invalid token")

async def validate_api_key(request,key):
    row=await request.app.state.db.fetchrow("select tenant_id,scopes from api_keys where api_key_hash=crypt($1,api_key_hash)",key)
    if not row:
        raise HTTPException(401,"Invalid API key")
    return {"sub":"api-client","tenant_id":row["tenant_id"],"scopes":row["scopes"]}

class UnifiedAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        hdr=request.headers.get("authorization")
        api_key=request.headers.get(API_KEY_HEADER)
        if hdr and hdr.lower().startswith("bearer "):
            token=hdr.split(" ",1)[1]
            claims=await validate_oidc(request,token)
            request.state.auth=claims
        elif api_key:
            claims=await validate_api_key(request,api_key)
            request.state.auth=claims
        else:
            raise HTTPException(401,"Missing credentials")
        return await call_next(request)

app.add_middleware(UnifiedAuthMiddleware)

@app.on_event("startup")
async def startup():
    app.state.db=await asyncpg.connect(DB_CONN)

@app.get("/secure")
async def secure(request:Request):
    return {"auth":request.state.auth}
