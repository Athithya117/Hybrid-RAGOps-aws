import os,time,uuid,json,logging
from typing import Optional,Dict,Any
from fastapi import Request,HTTPException,Depends
import redis.asyncio as aioredis
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("auth_rl")
VALKEY_URL=os.getenv("VALKEY_URL","redis://localhost:6379/0")
VALKEY_IS_CLUSTER=os.getenv("VALKEY_IS_CLUSTER","false").lower() in ("1","true","t","yes","y")
SESSION_TTL=int(os.getenv("SESSION_TTL_SECONDS","3600"))
RATE_WINDOW=int(os.getenv("RATE_WINDOW_SECONDS","60"))
RATE_LIMIT=int(os.getenv("RATE_LIMIT","60"))
ANON_RATE_LIMIT=int(os.getenv("ANON_RATE_LIMIT","20"))
VALKEY_MAX_CONN=int(os.getenv("VALKEY_MAX_CONN","50"))
RATE_ALG=os.getenv("RATE_ALG","sliding").lower()
_pool=aioredis.ConnectionPool.from_url(VALKEY_URL,max_connections=VALKEY_MAX_CONN,decode_responses=True)
redis_client=aioredis.Redis(connection_pool=_pool,decode_responses=True)
_sliding_lua="""local key=KEYS[1]
local now=tonumber(ARGV[1])
local window_ms=tonumber(ARGV[2])
local limit=tonumber(ARGV[3])
local member=ARGV[4]
redis.call('ZREMRANGEBYSCORE',key,0,now-window_ms)
redis.call('ZADD',key,now,member)
redis.call('PEXPIRE',key,window_ms)
local cnt=redis.call('ZCARD',key)
return cnt
"""
_token_bucket_lua="""local key=KEYS[1]
local now=tonumber(ARGV[1])
local capacity=tonumber(ARGV[2])
local refill_per_ms=tonumber(ARGV[3])
local requested=tonumber(ARGV[4])
local data=redis.call('HMGET',key,'tokens','last')
local tokens=tonumber(data[1]) or capacity
local last=tonumber(data[2]) or now
local delta=now-last
if delta>0 then
  local add=delta*refill_per_ms
  tokens=tokens+add
  if tokens>capacity then tokens=capacity end
end
local allowed=0
if tokens>=requested then
  tokens=tokens-requested
  allowed=1
end
redis.call('HSET',key,'tokens',tokens,'last',now)
local ttl_ms=math.floor((capacity/refill_per_ms)*2)
if ttl_ms<1000 then ttl_ms=1000 end
redis.call('PEXPIRE',key,ttl_ms)
return allowed
"""
async def _rate_check_sliding(key_prefix:str,identifier:str,window:int,limit:int)->int:
    key=f"rl:{key_prefix}:{identifier}"
    now=int(time.time()*1000)
    member=f"{uuid.uuid4()}:{now}"
    try:
        result=await redis_client.eval(_sliding_lua,1,key,now,window*1000,limit,member)
        return int(result or 0)
    except Exception:
        logger.exception("rate_check_sliding error")
        raise
async def _rate_check_token_bucket(key_prefix:str,identifier:str,window:int,limit:int)->int:
    key=f"rl_tb:{key_prefix}:{identifier}"
    now=int(time.time()*1000)
    capacity=limit
    window_ms=window*1000
    refill_per_ms=float(limit)/float(window_ms) if window_ms>0 else float(limit)
    requested=1
    try:
        result=await redis_client.eval(_token_bucket_lua,1,key,now,capacity,refill_per_ms,requested)
        return int(result or 0)
    except Exception:
        logger.exception("rate_check_token_bucket error")
        raise
async def _rate_check(key_prefix:str,identifier:str,window:int,limit:int)->int:
    if RATE_ALG=="token_bucket":
        return await _rate_check_token_bucket(key_prefix,identifier,window,limit)
    return await _rate_check_sliding(key_prefix,identifier,window,limit)
async def get_session_from_cookie(request:Request)->Dict[str,Any]:
    headers=request.headers
    user=headers.get("x-auth-request-user") or headers.get("x-forwarded-user") or headers.get("x-remote-user")
    email=headers.get("x-auth-request-email") or headers.get("x-forwarded-email") or headers.get("x-remote-email")
    preferred=headers.get("x-auth-request-preferred-username") or headers.get("x-forwarded-preferred-username")
    if user or email or preferred:
        sub=user or preferred or email
        if not sub:
            raise HTTPException(status_code=401,detail="no user in auth headers")
        session={"sub":sub}
        if email: session["email"]=email
        return session
    session_id=request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401,detail="no session")
    try:
        raw=await redis_client.get(f"sess:{session_id}")
    except Exception:
        logger.exception("redis get failed in get_session_from_cookie")
        raise HTTPException(status_code=503,detail="session store unavailable")
    if not raw:
        raise HTTPException(status_code=401,detail="session expired")
    try:
        await redis_client.pexpire(f"sess:{session_id}",SESSION_TTL*1000)
    except Exception:
        logger.debug("failed to refresh session TTL (non-fatal)")
    try:
        data=json.loads(raw)
        if not isinstance(data,dict): raise ValueError("session payload not a dict")
        return data
    except Exception:
        logger.exception("invalid session payload")
        raise HTTPException(status_code=401,detail="invalid session")
def rate_limiter_dep(route_name:str,window:int=RATE_WINDOW,limit:int=RATE_LIMIT):
    async def _dep(request:Request,session:Dict[str,Any]=Depends(get_session_from_cookie)):
        user_id=session.get("sub")
        if not user_id: raise HTTPException(status_code=401,detail="no user id in session")
        try:
            cnt=await _rate_check(route_name,user_id,window,limit)
            if RATE_ALG=="token_bucket":
                if int(cnt)==0: raise HTTPException(status_code=429,detail="rate limit exceeded")
            else:
                if int(cnt)>int(limit): raise HTTPException(status_code=429,detail="rate limit exceeded")
        except HTTPException:
            raise
        except Exception:
            logger.exception("rate_limiter_dep error")
            raise HTTPException(status_code=503,detail="rate limiter unavailable")
        return True
    return _dep
async def anon_rate_limiter(request:Request):
    try:
        ip=request.client.host or "anon"
    except Exception:
        ip="anon"
    try:
        cnt=await _rate_check("anon",ip,RATE_WINDOW,ANON_RATE_LIMIT)
        if RATE_ALG=="token_bucket":
            if int(cnt)==0: raise HTTPException(status_code=429,detail="anon rate limit exceeded")
        else:
            if int(cnt)>int(ANON_RATE_LIMIT): raise HTTPException(status_code=429,detail="anon rate limit exceeded")
    except HTTPException:
        raise
    except Exception:
        logger.exception("anon_rate_limiter failed")
        raise HTTPException(status_code=503,detail="rate limiter unavailable")
    return True
async def close_redis():
    try: await redis_client.close()
    except Exception: logger.debug("redis close failed")
    try: await redis_client.connection_pool.disconnect()
    except Exception: logger.debug("redis pool disconnect failed")
