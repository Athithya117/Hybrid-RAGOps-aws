import os,sys,logging,asyncio
from typing import List,Dict,Any
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
logging.basicConfig(level=os.getenv("SPARSE_LOGLEVEL","INFO"))
log=logging.getLogger("host_sparse")
SPARSE_MODEL_NAME=os.getenv("SPARSE_MODEL_NAME","Qdrant/minicoil-v1")
SPARSE_HOST=os.getenv("SPARSE_HOST","0.0.0.0")
SPARSE_PORT=int(os.getenv("SPARSE_PORT","8201"))
SPARSE_BATCH_SIZE=int(os.getenv("SPARSE_BATCH_SIZE","8"))
SPARSE_CUDA=os.getenv("SPARSE_CUDA","0").upper() in ("1","TRUE","YES")
app=FastAPI(title="sparse-embedder")
class SparseOut(BaseModel):
    indices:List[int]
    values:List[float]
class SparseRequest(BaseModel):
    texts:List[str]
class SparseResponse(BaseModel):
    vectors:List[SparseOut]
MODEL=None
MODEL_READY=False
MODEL_ERROR=None
def to_sparse(obj:Any)->Dict[str,Any]:
    if obj is None:
        return {"indices":[],"values":[]}
    if isinstance(obj,dict):
        inds=list(map(int,obj.get("indices",[]))); vals=list(map(float,obj.get("values",[]))); return {"indices":inds,"values":vals}
    if hasattr(obj,"indices") and hasattr(obj,"values"):
        inds=obj.indices.tolist() if hasattr(obj.indices,"tolist") else list(obj.indices)
        vals=obj.values.tolist() if hasattr(obj.values,"tolist") else list(obj.values)
        return {"indices":[int(x) for x in inds],"values":[float(x) for x in vals]}
    if isinstance(obj,(list,tuple)) and len(obj)==2:
        inds,vals=obj; return {"indices":[int(x) for x in inds],"values":[float(x) for x in vals]}
    raise RuntimeError("unsupported sparse object")
async def load_model():
    global MODEL,MODEL_READY,MODEL_ERROR
    try:
        log.info("background loading sparse model %s cuda=%s",SPARSE_MODEL_NAME,SPARSE_CUDA)
        from fastembed import SparseTextEmbedding
        if SPARSE_CUDA:
            try:
                MODEL=SparseTextEmbedding(model_name=SPARSE_MODEL_NAME,providers=["CUDAExecutionProvider"])
            except TypeError:
                MODEL=SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        else:
            MODEL=SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        tmp=list(MODEL.embed(["_init_"],batch_size=1))
        if tmp is None:
            MODEL_ERROR="no output from init embed"; MODEL_READY=False; log.error("sparse init produced no output"); return
        MODEL_READY=True; log.info("sparse model ready")
    except Exception as e:
        MODEL_ERROR=str(e); MODEL_READY=False; log.exception("sparse model load failed: %s",e)
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(load_model())
@app.post("/embed",response_model=SparseResponse)
def embed(req:SparseRequest):
    if not req.texts or len(req.texts)==0:
        raise HTTPException(status_code=400,detail="texts must be a non-empty list")
    if len(req.texts)>SPARSE_BATCH_SIZE:
        raise HTTPException(status_code=400,detail=f"batch too large max={SPARSE_BATCH_SIZE}")
    if not MODEL_READY:
        raise HTTPException(status_code=503,detail=f"model not ready error={MODEL_ERROR}")
    try:
        gen=MODEL.embed(req.texts,batch_size=len(req.texts))
        out=[]
        for s in gen:
            sv=to_sparse(s); out.append(sv)
        return {"vectors":out}
    except Exception as e:
        log.exception("sparse embed failed: %s",e); raise HTTPException(status_code=500,detail=str(e))
@app.get("/health")
def health():
    return {"status":"ok","model":SPARSE_MODEL_NAME,"ready":bool(MODEL_READY),"error":MODEL_ERROR}
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host=SPARSE_HOST,port=SPARSE_PORT)
