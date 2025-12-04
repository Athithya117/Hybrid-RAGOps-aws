import os,asyncio,logging
from typing import List,Optional,Tuple
from urllib.parse import urlparse,unquote
import boto3
from botocore.client import Config
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("presign")
S3_PRESIGN_EXPIRES=int(os.getenv("S3_PRESIGN_EXPIRES","300"))
S3_ENDPOINT_URL=os.getenv("S3_ENDPOINT_URL",None)
_s3_client=None
def _get_s3_client(endpoint_url:Optional[str]=None):
    global _s3_client
    if _s3_client is not None: return _s3_client
    sess=boto3.session.Session()
    client_kwargs={}
    if endpoint_url: client_kwargs["endpoint_url"]=endpoint_url
    _s3_client=sess.client("s3",config=Config(signature_version="s3v4"),**client_kwargs) if client_kwargs else sess.client("s3",config=Config(signature_version="s3v4"))
    return _s3_client
def _parse_s3_url(url:str)->Optional[Tuple[str,str]]:
    if not url: return None
    u=urlparse(url)
    if u.scheme=="s3":
        bucket=u.netloc;key=u.path.lstrip("/")
        return (bucket,unquote(key)) if bucket and key else None
    host=u.netloc.lower();path=u.path.lstrip("/")
    if host.endswith(".s3.amazonaws.com") or ".s3." in host:
        parts=host.split(".");bucket=parts[0];key=unquote(path)
        return (bucket,key) if bucket and key else None
    if host=="s3.amazonaws.com":
        segs=path.split("/",1)
        if len(segs)==2:
            bucket,key=segs[0],unquote(segs[1]);return (bucket,key)
    return None
def _generate_presigned_sync(bucket:str,key:str,expires:int=300,endpoint_url:Optional[str]=None)->str:
    client=_get_s3_client(endpoint_url)
    return client.generate_presigned_url("get_object",Params={"Bucket":bucket,"Key":key},ExpiresIn=int(expires))
async def _generate_presigned(bucket:str,key:str,expires:int=300,endpoint_url:Optional[str]=None)->str:
    return await asyncio.to_thread(_generate_presigned_sync,bucket,key,expires,endpoint_url)
async def presign_doc_dict(doc:dict,expires:Optional[int]=None,replace:bool=True)->Optional[str]:
    expires=expires if expires is not None else S3_PRESIGN_EXPIRES
    url=doc.get("metadata",{}).get("source_url") or doc.get("metadata",{}).get("s3_url") or ""
    parsed=_parse_s3_url(url)
    if not parsed: return None
    bucket,key=parsed
    try:
        signed=await _generate_presigned(bucket,key,expires,S3_ENDPOINT_URL)
        doc.setdefault("metadata",{})["signed_url"]=signed
        if replace: doc["metadata"]["source_url"]=signed
        return signed
    except Exception:
        logger.exception("presign failed for %s/%s",bucket,key);return None
async def presign_docs_batch(docs:List[dict],expires:Optional[int]=None,concurrency:int=16)->List[dict]:
    sem=asyncio.Semaphore(concurrency)
    async def _worker(d:dict):
        async with sem:
            try:
                await presign_doc_dict(d,expires=expires)
            except Exception:
                logger.debug("presign failed for doc id %s",d.get("id"))
        return d
    tasks=[asyncio.create_task(_worker(d)) for d in docs]
    out=[]
    for t in asyncio.as_completed(tasks):
        out.append(await t)
    return out
