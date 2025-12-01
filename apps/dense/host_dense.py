import os,sys,logging
from typing import List
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import numpy as np
from fastembed import TextEmbedding
logging.basicConfig(level=os.getenv("DENSE_LOGLEVEL","INFO"))
log=logging.getLogger("host_dense")
DENSE_MODEL_NAME=os.getenv("DENSE_MODEL_NAME","BAAI/bge-small-en-v1.5")
DENSE_DIM=int(os.getenv("DENSE_DIM","384"))
DENSE_HOST=os.getenv("DENSE_HOST","0.0.0.0")
DENSE_PORT=int(os.getenv("DENSE_PORT","8205"))
DENSE_BATCH_SIZE=int(os.getenv("DENSE_BATCH_SIZE","16"))
DENSE_NORMALIZE=os.getenv("DENSE_NORMALIZE","TRUE").upper() in ("1","TRUE","YES")
DENSE_CUDA=os.getenv("DENSE_CUDA","0").upper() in ("1","TRUE","YES")
app=FastAPI(title="dense-embedder")
class EmbedRequest(BaseModel):
    texts:List[str]
class EmbedResponse(BaseModel):
    vectors:List[List[float]]
def l2_normalize(v):
    a=np.asarray(v,dtype=np.float32)
    n=np.linalg.norm(a)
    if n>0:
        a=a/n
    return a.astype(float).tolist()
try:
    log.info("loading dense model %s cuda=%s",DENSE_MODEL_NAME,DENSE_CUDA)
    if DENSE_CUDA:
        try:
            MODEL=TextEmbedding(model_name=DENSE_MODEL_NAME,providers=["CUDAExecutionProvider"])
        except TypeError:
            MODEL=TextEmbedding(model_name=DENSE_MODEL_NAME)
            log.warning("provider arg not supported, falling back to default provider")
    else:
        MODEL=TextEmbedding(model_name=DENSE_MODEL_NAME)
    tmp=list(MODEL.embed(["_init_"]))
    if not tmp:
        log.error("dense init produced no output")
        raise RuntimeError("dense init failed")
    got_dim=len(tmp[0]) if hasattr(tmp[0],"__len__") else None
    if got_dim is None or got_dim!=DENSE_DIM:
        log.error("embedding dimension mismatch expected=%d got=%s",DENSE_DIM,str(got_dim))
        sys.exit(1)
except Exception as e:
    log.exception("dense model init failed: %s",e)
    sys.exit(1)
@app.post("/embed",response_model=EmbedResponse)
def embed(req:EmbedRequest):
    texts=req.texts
    if not texts or len(texts)==0:
        raise HTTPException(status_code=400,detail="texts must be a non-empty list")
    if len(texts)>DENSE_BATCH_SIZE:
        raise HTTPException(status_code=400,detail=f"batch too large max={DENSE_BATCH_SIZE}")
    try:
        gens=MODEL.embed(texts)
        vecs=[]
        for a in gens:
            if hasattr(a,"astype"):
                v=a.astype(float).tolist()
            else:
                v=[float(x) for x in a]
            if DENSE_NORMALIZE:
                v=l2_normalize(v)
            if len(v)!=DENSE_DIM:
                raise HTTPException(status_code=500,detail="embedding dimension mismatch")
            vecs.append([float(x) for x in v])
        return {"vectors":vecs}
    except Exception as e:
        log.exception("embed failed: %s",e)
        raise HTTPException(status_code=500,detail=str(e))
@app.get("/health")
def health():
    return {"status":"ok","model":DENSE_MODEL_NAME,"dim":DENSE_DIM,"normalize":DENSE_NORMALIZE,"cuda":DENSE_CUDA}
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host=DENSE_HOST,port=DENSE_PORT)
