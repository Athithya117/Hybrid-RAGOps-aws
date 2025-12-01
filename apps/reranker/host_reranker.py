import os,logging,asyncio
from typing import List
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
logging.basicConfig(level=os.getenv("RERANKER_LOGLEVEL","INFO"))
log=logging.getLogger("host_reranker")
RERANKER_MODEL_NAME=os.getenv("RERANKER_MODEL_NAME","Xenova/ms-marco-MiniLM-L-6-v2")
RERANKER_HOST=os.getenv("RERANKER_HOST","0.0.0.0")
RERANKER_PORT=int(os.getenv("RERANKER_PORT","8202"))
RERANKER_MAX_DOCS=int(os.getenv("RERANKER_MAX_DOCS","50"))
RERANKER_CUDA=os.getenv("RERANKER_CUDA","0").upper() in ("1","TRUE","YES")
app=FastAPI(title="reranker")
class RerankRequest(BaseModel):
    query:str
    documents:List[str]
class RerankResponse(BaseModel):
    scores:List[float]
MODEL=None
MODEL_READY=False
MODEL_ERROR=None
async def load_model():
    global MODEL,MODEL_READY,MODEL_ERROR
    try:
        log.info("background loading reranker %s cuda=%s",RERANKER_MODEL_NAME,RERANKER_CUDA)
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        if RERANKER_CUDA:
            try:
                MODEL=TextCrossEncoder(model_name=RERANKER_MODEL_NAME,providers=["CUDAExecutionProvider"])
            except TypeError:
                MODEL=TextCrossEncoder(model_name=RERANKER_MODEL_NAME)
        else:
            MODEL=TextCrossEncoder(model_name=RERANKER_MODEL_NAME)
        _=MODEL.rerank("init",["a","b"])
        MODEL_READY=True
        log.info("reranker ready")
    except Exception as e:
        MODEL_ERROR=str(e)
        MODEL_READY=False
        log.exception("reranker load failed: %s",e)
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(load_model())
@app.post("/rerank",response_model=RerankResponse)
def rerank(req:RerankRequest):
    if not req.query:
        raise HTTPException(status_code=400,detail="query must be provided")
    if not req.documents or len(req.documents)==0:
        raise HTTPException(status_code=400,detail="documents must be a non-empty list")
    if len(req.documents)>RERANKER_MAX_DOCS:
        raise HTTPException(status_code=400,detail=f"too many documents max={RERANKER_MAX_DOCS}")
    if not MODEL_READY:
        raise HTTPException(status_code=503,detail=f"model not ready error={MODEL_ERROR}")
    try:
        raw=MODEL.rerank(req.query,req.documents)
        scores=[float(x) for x in raw]
        return {"scores":scores}
    except Exception as e:
        log.exception("rerank failed: %s",e)
        raise HTTPException(status_code=500,detail=str(e))
@app.get("/health")
def health():
    return {"status":"ok","model":RERANKER_MODEL_NAME,"ready":bool(MODEL_READY),"error":MODEL_ERROR}
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host=RERANKER_HOST,port=RERANKER_PORT)
