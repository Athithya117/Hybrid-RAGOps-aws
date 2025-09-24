import os,sys,json,asyncio,hashlib,time
from typing import List,Dict,Any,Optional
import boto3,httpx,weaviate
from weaviate.util import generate_uuid5
from weaviate.classes.config import Configure,Property,DataType
S3_BUCKET=os.environ.get("S3_BUCKET") or ""
S3_PREFIX=os.environ.get("S3_PREFIX","data/chunked/")
AWS_REGION=os.environ.get("AWS_REGION")
EMBEDDING_URL=os.environ.get("EMBEDDING_URL") or "http://localhost:8001/embed"
EMBED_BATCH=int(os.environ.get("EMBED_BATCH","32"))
EMBED_CONCURRENCY=int(os.environ.get("EMBED_CONCURRENCY","8"))
EMBED_TIMEOUT=float(os.environ.get("EMBED_TIMEOUT","60"))
INDEX_BATCH=int(os.environ.get("INDEX_BATCH","128"))
CONCURRENCY=int(os.environ.get("CONCURRENCY","4"))
WEAVIATE_URL=os.environ.get("WEAVIATE_URL","http://localhost:8080")
WEAVIATE_API_KEY=os.environ.get("WEAVIATE_API_KEY","")
META_DATA_FILTERING_FIELDS=[f.strip() for f in os.environ.get("META_DATA_FILTERING_FIELDS","file_name,chunk_type,file_type,tags,timestamp").split(",") if f.strip()]
COLLECTION_NAME=os.environ.get("WEAVIATE_COLLECTION","Chunks")
if not S3_BUCKET:
    print("S3_BUCKET not set",file=sys.stderr);sys.exit(2)
s3=boto3.client("s3",region_name=AWS_REGION) if AWS_REGION else boto3.client("s3")
async def http_post_with_retries(client:httpx.AsyncClient,url:str,payload:Any,timeout:float,tries:int=4)->Any:
    backoff=0.25
    last=None
    for i in range(tries):
        try:
            r=await client.post(url,json=payload,timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last=e
            await asyncio.sleep(backoff*(2**i))
    raise last
async def embedder_call(texts:List[str],client:httpx.AsyncClient)->List[List[float]]:
    payload={"texts":texts}
    resp=await http_post_with_retries(client,EMBEDDING_URL,payload,EMBED_TIMEOUT)
    if isinstance(resp,dict):
        if "embeddings" in resp and isinstance(resp["embeddings"],list):
            return resp["embeddings"]
        if "data" in resp and isinstance(resp["data"],list):
            out=[]
            for it in resp["data"]:
                if isinstance(it,dict) and "embedding" in it:
                    out.append(it["embedding"])
                elif isinstance(it,list):
                    out.append(it)
            if out:
                return out
    if isinstance(resp,list) and all(isinstance(x,list) for x in resp):
        return resp
    raise RuntimeError("Unsupported embedder response shape")
class EmbedClient:
    def __init__(self,concurrency:int, batch_size:int):
        self.semaphore=asyncio.Semaphore(concurrency)
        self.batch_size=batch_size
        limits=httpx.Limits(max_keepalive_connections=concurrency,max_connections=concurrency*2)
        self._client=httpx.AsyncClient(limits=limits)
    async def embed_texts(self,texts:List[str]) -> List[List[float]]:
        if not texts:
            return []
        results=[]
        async def do_batch(batch):
            async with self.semaphore:
                return await embedder_call(batch,self._client)
        tasks=[asyncio.create_task(do_batch(texts[i:i+self.batch_size])) for i in range(0,len(texts),self.batch_size)]
        for fut in asyncio.as_completed(tasks):
            res=await fut
            results.extend(res)
        return results
    async def close(self):
        await self._client.aclose()
def parse_json_objects_from_text(s:str)->List[Dict[str,Any]]:
    s=s.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            j=json.loads(s)
            if isinstance(j,list):
                return j
        except Exception:
            pass
    if "\n" in s:
        lines=[l.strip() for l in s.splitlines() if l.strip()]
        maybe_jsonl=True
        parsed=[]
        for ln in lines:
            try:
                parsed.append(json.loads(ln))
            except Exception:
                maybe_jsonl=False
                break
        if maybe_jsonl:
            return parsed
    objs=[]
    depth=0;start=None;in_str=False;esc=False
    for i,ch in enumerate(s):
        if ch=='"' and not esc:
            in_str=not in_str
        if in_str and ch=='\\' and not esc:
            esc=True
            continue
        if not in_str:
            if ch=='{':
                if depth==0:
                    start=i
                depth+=1
            elif ch=='}':
                depth-=1
                if depth==0 and start is not None:
                    raw=s[start:i+1]
                    try:
                        objs.append(json.loads(raw))
                    except Exception:
                        pass
                    start=None
        if esc:
            esc=False
    if objs:
        return objs
    try:
        j=json.loads(s)
        if isinstance(j,list):
            return j
        if isinstance(j,dict):
            return [j]
    except Exception:
        pass
    raise RuntimeError("Could not parse JSON/JSONL content")
def connect_weaviate():
    headers={}
    if WEAVIATE_API_KEY:
        headers["Authorization"]=f"Bearer {WEAVIATE_API_KEY}"
    parsed=weaviate.connect_to_local if WEAVIATE_URL in ("http://localhost:8080","http://127.0.0.1:8080","localhost:8080") else None
    if parsed and not WEAVIATE_URL.startswith("http"):
        client=weaviate.connect_to_local(headers=headers)
    else:
        host="localhost";port=8080;secure=False
        try:
            u=WEAVIATE_URL
            if u.startswith("http://") or u.startswith("https://"):
                from urllib.parse import urlparse
                p=urlparse(u)
                host=p.hostname or host
                port=p.port or port
                secure=(p.scheme=="https")
        except Exception:
            pass
        client=weaviate.connect_to_custom(http_host=host,http_port=port,http_secure=secure,auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None,headers=None,skip_init_checks=True)
    return client
def ensure_collection(client):
    try:
        client.collections.get(COLLECTION_NAME)
        return
    except Exception:
        pass
    props=[Property(name="text",data_type=DataType.TEXT),Property(name="document_id",data_type=DataType.TEXT),Property(name="chunk_id",data_type=DataType.TEXT),Property(name="chunk_type",data_type=DataType.TEXT),Property(name="token_count",data_type=DataType.INT),Property(name="file_type",data_type=DataType.TEXT),Property(name="file_name",data_type=DataType.TEXT),Property(name="source_url",data_type=DataType.TEXT),Property(name="page_number",data_type=DataType.INT),Property(name="timestamp",data_type=DataType.DATE),Property(name="tags",data_type=DataType.TEXT_ARRAY),Property(name="vector_id",data_type=DataType.NUMBER),Property(name="deleted",data_type=DataType.BOOL)]
    client.collections.create(name=COLLECTION_NAME,vector_config=Configure.Vectors.self_provided(),properties=props)
def weaviate_batch_upsert(client,items:List[Dict[str,Any]]):
    coll=client.collections.use(COLLECTION_NAME)
    with coll.batch.fixed_size(batch_size=INDEX_BATCH) as batch:
        for it in items:
            batch.add_object(properties=it["properties"],vector=it["vector"],uuid=it["uuid"])
async def process_s3_object(key:str,embed_client:EmbedClient,client_weaviate):
    obj=s3.get_object(Bucket=S3_BUCKET,Key=key)
    body=obj["Body"].read().decode("utf-8")
    try:
        chunks=parse_json_objects_from_text(body)
    except Exception as e:
        print("parse error",key,e,file=sys.stderr)
        return
    if not chunks:
        return
    texts=[(c.get("text") or "") for c in chunks]
    embeddings=await embed_client.embed_texts(texts)
    if len(embeddings)!=len(texts):
        raise RuntimeError("embedding count mismatch")
    dim=len(embeddings[0])
    for v in embeddings:
        if len(v)!=dim:
            raise RuntimeError("embedding dimension mismatch")
    batch_items=[]
    for c,vec in zip(chunks,embeddings):
        doc_id=c.get("document_id") or c.get("source_url") or key
        chunk_id=c.get("chunk_id") or hashlib.sha256((json.dumps(c,sort_keys=True)).encode()).hexdigest()
        uuid=generate_uuid5(f"{doc_id}:{chunk_id}")
        v_id=int.from_bytes(hashlib.sha256(f"{doc_id}:{chunk_id}".encode()).digest()[:8],"big",signed=False)
        props={k:v for k,v in c.items() if k!="embedding"}
        props["text"]=c.get("text","")
        props["document_id"]=doc_id
        props["chunk_id"]=chunk_id
        props["vector_id"]=v_id
        props["deleted"]=bool(props.get("deleted",False))
        for m in META_DATA_FILTERING_FIELDS:
            if m in c and m not in props:
                props[m]=c[m]
        batch_items.append({"properties":props,"vector":[float(x) for x in vec],"uuid":str(uuid)})
        if len(batch_items)>=INDEX_BATCH:
            await asyncio.get_running_loop().run_in_executor(None,weaviate_batch_upsert,client_weaviate,batch_items)
            batch_items=[]
    if batch_items:
        await asyncio.get_running_loop().run_in_executor(None,weaviate_batch_upsert,client_weaviate,batch_items)
async def main():
    client_weaviate=connect_weaviate()
    ensure_collection(client_weaviate)
    embed_client=EmbedClient(concurrency=EMBED_CONCURRENCY,batch_size=EMBED_BATCH)
    paginator=s3.get_paginator("list_objects_v2")
    prefix=S3_PREFIX.rstrip("/")+"/"
    keys=[]
    for page in paginator.paginate(Bucket=S3_BUCKET,Prefix=prefix):
        for obj in page.get("Contents",[]):
            k=obj["Key"]
            if k.lower().endswith(".json") or k.lower().endswith(".jsonl"):
                keys.append(k)
    if not keys:
        print("no files to process");await embed_client.close();return
    sem=asyncio.Semaphore(CONCURRENCY)
    async def worker(k):
        async with sem:
            await process_s3_object(k,embed_client,client_weaviate)
    tasks=[asyncio.create_task(worker(k)) for k in keys]
    for t in asyncio.as_completed(tasks):
        try:
            await t
        except Exception as e:
            print("task error",e,file=sys.stderr)
    await embed_client.close()
    try:
        client_weaviate.close()
    except Exception:
        pass
if __name__=="__main__":
    asyncio.run(main())
