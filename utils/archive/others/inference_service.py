import os,time,logging,asyncio
from typing import Optional
from urllib.parse import urlparse
import weaviate
from weaviate.classes.init import Auth
from fastapi import FastAPI,Request,HTTPException,Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter,Histogram,Gauge,make_asgi_app
from llm_retriever import retrieve_answer,settings as retriever_settings,init_http_client,close_global_clients
from presign import presign_docs_batch
from auth_and_rate_limits import rate_limiter_dep,redis_client
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("inference_service")
ENV=os.getenv("ENV","dev")
ROUTE="retrieve"
rag_requests_total=Counter("rag_requests_total","Total RAG requests",["env","route"])
rag_request_duration_seconds=Histogram("rag_request_duration_seconds","End-to-end RAG request latency seconds",["env","route","outcome"],buckets=(0.05,0.1,0.25,0.5,1,2.5,5,10,30,60))
rag_errors_total=Counter("rag_errors_total","Failures in pipeline",["env","stage","error_type"])
post_dedupe_count=Gauge("post_dedupe_count","Number of chunks after content dedupe",["env"])
app=FastAPI(title="RAG Inference Service")
metrics_app=make_asgi_app();app.mount("/metrics",metrics_app)
class QueryIn(BaseModel):
    query:str
    do_presign:bool=True
startup_ok = False
@app.on_event("startup")
async def on_start():
    global startup_ok
    try:
        await init_http_client()
    except Exception:
        logger.exception("init_http_client failed");raise
    try:
        await redis_client.ping()
    except Exception:
        logger.exception("redis ping failed at startup");raise
    try:
        url=retriever_settings.WEAVIATE_URL
        parsed=urlparse(url)
        host=parsed.hostname or "localhost"
        port=parsed.port or (443 if parsed.scheme=="https" else 80)
        secure=parsed.scheme=="https"
        auth_cred=None
        if retriever_settings.WEAVIATE_API_KEY:
            auth_cred=Auth.api_key(retriever_settings.WEAVIATE_API_KEY)
        client=weaviate.use_async_with_custom if hasattr(weaviate,"use_async_with_custom") else weaviate.use_async_with_local
        weav=client(host,port,http_secure=secure,auth_credentials=auth_cred) if client is weaviate.use_async_with_custom else client(host=host,port=port,auth_credentials=auth_cred)
        await weav.connect()
        ready=await weav.is_ready()
        if not ready:
            raise RuntimeError("weaviate not ready")
        app.state.weav=weav
    except Exception:
        logger.exception("weaviate client init failed");raise
    startup_ok = True
    logger.info("startup complete")
@app.on_event("shutdown")
async def on_shutdown():
    try:
        await redis_client.close()
    except Exception:
        logger.debug("redis close failed")
    try:
        if getattr(app.state,"weav",None):
            try:
                await app.state.weav.close()
            except Exception:
                logger.debug("weav close failed")
    except Exception:
        logger.debug("weav close wrapper failed")
    try:
        await close_global_clients()
    except Exception:
        logger.debug("close_global_clients failed")
@app.get("/health")
async def health():
    if not startup_ok:
        return JSONResponse({"ok":False,"detail":"starting"},status_code=503)
    try:
        if getattr(app.state,"weav",None):
            ok=await app.state.weav.is_ready()
            if not ok: return JSONResponse({"ok":False,"detail":"weaviate not ready"},status_code=503)
    except Exception:
        return JSONResponse({"ok":False,"detail":"weaviate health check failed"},status_code=503)
    return JSONResponse({"ok":True})
@app.post("/retrieve")
async def retrieve_endpoint(body:QueryIn,request:Request,_rl=Depends(rate_limiter_dep("retrieve"))):
    rag_requests_total.labels(ENV,ROUTE).inc()
    start=time.monotonic()
    try:
        weav=getattr(app.state,"weav",None)
        out=await retrieve_answer(body.query,weav=weav,do_presign=False)
        docs=out.get("docs",[])
        try:
            if body.do_presign and docs:
                await presign_docs_batch(docs,concurrency=8)
        except Exception:
            logger.exception("presign batch failed")
        post_dedupe_count.labels(ENV).set(out.get("retrieval",{}).get("final_chunks",0))
        rag_request_duration_seconds.labels(ENV,ROUTE,"ok").observe(time.monotonic()-start)
        return JSONResponse(out)
    except HTTPException as he:
        rag_request_duration_seconds.labels(ENV,ROUTE,"error").observe(time.monotonic()-start)
        raise he
    except Exception:
        logger.exception("retrieve failed")
        rag_errors_total.labels(ENV,"retrieve","exception").inc()
        rag_request_duration_seconds.labels(ENV,ROUTE,"error").observe(time.monotonic()-start)
        raise HTTPException(status_code=500,detail="internal error")
