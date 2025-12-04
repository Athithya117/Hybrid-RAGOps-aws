import os,time,json,logging,asyncio,random
from dataclasses import dataclass
from typing import Optional,List,Dict,Any,Tuple
import httpx
import numpy as np
import weaviate
from urllib.parse import urlparse
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("llm_retriever")
def _env(name:str,default:Optional[str]=None)->Optional[str]:
    v=os.environ.get(name);return v if v is not None else default
def _bool(name:str,default:bool=False)->bool:
    v=_env(name)
    if v is None: return default
    return v.lower() in ("1","true","t","yes","y")
@dataclass
class Settings:
    WEAVIATE_URL:str=_env("WEAVIATE_URL","http://localhost:8080")
    WEAVIATE_CLASS:str=_env("WEAVIATE_CLASS","Document")
    WEAVIATE_API_KEY:Optional[str]=_env("WEAVIATE_API_KEY",None)
    EMBEDDING_URL:str=_env("EMBEDDING_URL","")
    CROSS_ENCODER_URL:str=_env("CROSS_ENCODER_URL","")
    EMBEDDING_SERVICE_URL:str=_env("EMBEDDING_SERVICE_URL","")
    EMBED_MODEL:str=_env("EMBED_MODEL","")
    RERANK_MODEL:str=_env("RERANK_MODEL","")
    LLM_SERVICE_URL:str=_env("LLM_SERVICE_URL","")
    LLM_MODEL:str=_env("LLM_MODEL","")
    BM25_TOP:int=int(_env("BM25_TOP","100"))
    DENSE_TOP:int=int(_env("DENSE_TOP","256"))
    UNION_CAP:int=int(_env("UNION_CAP","800"))
    STAGE_A_TOP_M:int=int(_env("STAGE_A_TOP_M","50"))
    FUSED_TOP_N:int=int(_env("FUSED_TOP_N","200"))
    RRF_K:int=int(_env("RRF_K","60"))
    DEDUPE_SIM_THRESHOLD:float=float(_env("DEDUPE_SIM_THRESHOLD","0.88"))
    USE_HEAVY_RERANKER:bool=_bool("ENABLE_RERANKER",True)
    HEAVY_TOP_K:int=int(_env("HEAVY_TOP_K","16"))
    HEAVY_BATCH_SIZE:int=int(_env("HEAVY_BATCH_SIZE","8"))
    MAX_CHUNKS_TO_LLM:int=int(_env("MAX_CHUNKS_TO_LLM","6"))
    GLOBAL_REQUEST_TIMEOUT:int=int(_env("GLOBAL_REQUEST_TIMEOUT","30"))
    EMBEDDING_BATCH_SIZE:int=int(_env("EMBEDDING_BATCH_SIZE","32"))
    BIOENCODER_BATCH_SIZE:int=int(_env("BIOENCODER_BATCH_SIZE","16"))
    CROSSENCODER_BATCH_SIZE:int=int(_env("CROSSENCODER_BATCH_SIZE","8"))
    HTTP_RETRY_ATTEMPTS:int=int(_env("HTTP_RETRY_ATTEMPTS","3"))
    HTTP_BACKOFF_BASE:float=float(_env("HTTP_BACKOFF_BASE","0.4"))
    ASYNC_CONCURRENCY:int=int(_env("ASYNC_CONCURRENCY","16"))
    HTTPX_HTTP2:bool=_bool("HTTP2_FOR_HTTPX",False)
    BATCH_FETCH_SIZE:int=int(_env("BATCH_FETCH_SIZE","64"))
    ALPHA_WEIGHT:float=float(max(0.0,min(1.0,float(_env("ALPHA_WEIGHT","0.5")))))
settings=Settings()
@dataclass
class WeaviateDoc:
    id:str
    text:str
    metadata:Dict[str,Any]
    embedding:Optional[np.ndarray]=None
class AsyncHTTPClient:
    def __init__(self,timeout:int=10):
        max_keepalive=int(os.getenv("HTTPX_MAX_KEEPALIVE","20"))
        max_conn=int(os.getenv("HTTPX_MAX_CONNECTIONS","100"))
        limits=httpx.Limits(max_keepalive_connections=max_keepalive,max_connections=max_conn)
        self.client=httpx.AsyncClient(timeout=httpx.Timeout(timeout,read=timeout,write=timeout),limits=limits,http2=settings.HTTPX_HTTP2)
        self.attempts=settings.HTTP_RETRY_ATTEMPTS
        self.backoff_base=settings.HTTP_BACKOFF_BASE
    async def _sleep_backoff(self,attempt:int):
        await asyncio.sleep(self.backoff_base*(2**attempt)+random.random()*0.1)
    async def post_json(self,url:str,json_body:dict,timeout:Optional[int]=None,headers:Optional[dict]=None)->dict:
        attempt=0;last_err=None
        while attempt<self.attempts:
            try:
                r=await self.client.post(url,json=json_body,timeout=timeout or self.client.timeout,headers=headers)
                r.raise_for_status();return r.json()
            except Exception as e:
                last_err=e;logger.warning("POST %s failed (attempt %d/%d): %s",url,attempt+1,self.attempts,e)
                await self._sleep_backoff(attempt);attempt+=1
        logger.exception("POST %s failed after %d attempts",url,self.attempts);raise last_err
    async def get_json(self,url:str,params:Optional[dict]=None,timeout:Optional[int]=None,headers:Optional[dict]=None)->dict:
        attempt=0;last_err=None
        while attempt<self.attempts:
            try:
                r=await self.client.get(url,params=params or {},timeout=timeout or self.client.timeout,headers=headers)
                r.raise_for_status();return r.json()
            except Exception as e:
                last_err=e;logger.warning("GET %s failed (attempt %d/%d): %s",url,attempt+1,self.attempts,e)
                await self._sleep_backoff(attempt);attempt+=1
        logger.exception("GET %s failed after %d attempts",url,self.attempts);raise last_err
    async def close(self):
        try:
            await self.client.aclose()
        except Exception:
            logger.debug("httpx aclose failed")
http: Optional[AsyncHTTPClient] = None
async def init_http_client():
    global http
    if http is None:
        http = AsyncHTTPClient(timeout=settings.GLOBAL_REQUEST_TIMEOUT)
        logger.info("http client initialized")
async def close_global_clients():
    global http
    try:
        if http is not None:
            await http.close()
            http = None
    except Exception:
        logger.exception("close_global_clients failed")
class EmbedderError(Exception): pass
class RemoteEmbedderClient:
    def __init__(self,embed_url:Optional[str]=None,cross_url:Optional[str]=None):
        self.embed_url=embed_url or settings.EMBEDDING_URL or settings.EMBEDDING_SERVICE_URL
        self.cross_url=cross_url or settings.CROSS_ENCODER_URL or settings.EMBEDDING_SERVICE_URL
        if not self.embed_url:
            raise ValueError("EMBEDDING_URL or EMBEDDING_SERVICE_URL must be set")
    def _make_endpoint(self,base:str,default_path:str)->str:
        if base.endswith("/") : base=base[:-1]
        if default_path in base or base.endswith(default_path): return base
        return base.rstrip("/") + "/" + default_path.lstrip("/")
    async def embed_query(self,query:str)->np.ndarray:
        if http is None:
            raise RuntimeError("http client not initialized")
        url=self.embed_url
        if not any(p in url for p in ("/embed","/embed_with_ids")):
            url=self._make_endpoint(url,"/embed")
        payload={"text":query,"model":settings.EMBED_MODEL}
        resp=await http.post_json(url,payload,timeout=settings.GLOBAL_REQUEST_TIMEOUT)
        emb=None
        if isinstance(resp,dict):
            if resp.get("embedding") is not None:
                emb=resp.get("embedding")
            elif resp.get("embeddings") is not None:
                e=resp.get("embeddings")
                if isinstance(e,list) and len(e)>0:
                    first=e[0]
                    if isinstance(first,dict) and first.get("embedding") is not None:
                        emb=first.get("embedding")
                    elif isinstance(first,(list,tuple)):
                        emb=first
                    elif isinstance(first,(int,float)):
                        emb=e
            elif resp.get("onnx_output") is not None:
                maybe=resp.get("onnx_output")
                if isinstance(maybe,list) and len(maybe) and isinstance(maybe[0],(list,float,int)):
                    emb=maybe[0] if isinstance(maybe[0],(list,tuple)) else maybe
            elif "data" in resp and isinstance(resp.get("data"),list) and len(resp["data"])>0:
                a0=resp["data"][0]
                if isinstance(a0,dict) and a0.get("embedding") is not None:
                    emb=a0.get("embedding")
        if emb is None:
            raise EmbedderError("embed response missing 'embedding'")
        vec=np.array(emb,dtype=np.float32);return self._normalize(vec)
    async def embed_texts(self,texts:List[str],batch_size:Optional[int]=None)->np.ndarray:
        if http is None:
            raise RuntimeError("http client not initialized")
        if batch_size is None: batch_size=settings.EMBEDDING_BATCH_SIZE
        out:List[List[float]]=[]
        url=self.embed_url
        if not any(p in url for p in ("/embed","/embed_with_ids")):
            url=self._make_endpoint(url,"/embed")
        for i in range(0,len(texts),batch_size):
            batch=texts[i:i+batch_size];payload={"texts":batch,"model":settings.EMBED_MODEL}
            resp=await http.post_json(url,payload,timeout=settings.GLOBAL_REQUEST_TIMEOUT)
            embs=None
            if isinstance(resp,dict):
                if resp.get("embeddings") is not None:
                    embs=resp.get("embeddings")
                elif resp.get("embedding") is not None:
                    emb=resp.get("embedding");embs=[emb]*len(batch)
                elif resp.get("onnx_output") is not None:
                    maybe=resp.get("onnx_output")
                    if isinstance(maybe,list) and len(maybe)==len(batch):
                        embs=maybe
            if embs is None:
                raise EmbedderError("embed_batch missing 'embeddings'")
            for e in embs:
                out.append(e)
        if len(out)==0:
            return np.zeros((0,0),dtype=np.float32)
        arr=np.array(out,dtype=np.float32);return self._normalize_rows(arr)
    async def embed_with_token_arrays(self,input_ids:List[List[int]],attention_mask:Optional[List[List[int]]]=None,batch_size:Optional[int]=None)->List[List[float]]:
        if http is None:
            raise RuntimeError("http client not initialized")
        url=self._make_endpoint(self.embed_url,"/embed_with_ids")
        payload={"input_ids":input_ids}
        if attention_mask is not None: payload["attention_mask"]=attention_mask
        resp=await http.post_json(url,payload,timeout=settings.GLOBAL_REQUEST_TIMEOUT)
        if not isinstance(resp,dict): raise EmbedderError("embed_with_ids unexpected response")
        embs=resp.get("embeddings")
        if embs is None: raise EmbedderError("embed_with_ids missing 'embeddings'")
        return embs
    async def bi_encoder_score_pairs_from_vectors(self,query_vec:np.ndarray,stored_vecs:List[np.ndarray])->List[float]:
        qv=query_vec
        if len(stored_vecs)==0: return []
        tv=np.stack(stored_vecs,axis=0)
        if tv.ndim==1: tv=tv.reshape(1,-1)
        scores=np.dot(tv,qv);return scores.tolist()
    async def cross_encoder_rerank(self,query:str,candidate_texts:List[str],batch_size:Optional[int]=None)->List[float]:
        if http is None:
            raise RuntimeError("http client not initialized")
        if batch_size is None: batch_size=settings.CROSSENCODER_BATCH_SIZE
        url=self.cross_url or self.embed_url
        if not any(p in url for p in ("/rerank","/cross_rerank")):
            url=self._make_endpoint(url,"/rerank")
        scores:List[float]=[]
        for i in range(0,len(candidate_texts),batch_size):
            batch=candidate_texts[i:i+batch_size];payload={"query":query,"passages":batch,"model":settings.RERANK_MODEL}
            resp=await http.post_json(url,payload,timeout=settings.GLOBAL_REQUEST_TIMEOUT)
            s=None
            if isinstance(resp,dict):
                if resp.get("scores") is not None: s=resp.get("scores")
            if s is None: raise EmbedderError("cross_rerank missing 'scores'")
            scores.extend(s)
        return scores
    def _normalize(self,vec:np.ndarray)->np.ndarray:
        n=np.linalg.norm(vec)
        if n==0: return vec.astype(np.float32)
        return (vec/n).astype(np.float32)
    def _normalize_rows(self,mat:np.ndarray)->np.ndarray:
        if mat.ndim==1: return self._normalize(mat)
        norms=np.linalg.norm(mat,axis=1,keepdims=True);norms[norms==0]=1.0
        return (mat/norms).astype(np.float32)
async def search_bm25(client:weaviate.WeaviateAsyncClient,query:str,top_k:int)->List[Dict]:
    props=["text","title"]
    try:
        q=client.query.get(settings.WEAVIATE_CLASS,props).with_bm25(query=query).with_limit(top_k).with_additional(['id'])
        resp=await q.do()
        docs=resp.get("data",{}).get("Get",{}).get(settings.WEAVIATE_CLASS,[]) or []
        out=[]
        for d in docs:
            aid=d.get("_additional",{}).get("id") or d.get("id") or None
            out.append({"id":aid,"score":d.get("bm25Score",1.0) if isinstance(d,dict) else 1.0,"metadata":{"title":d.get("title")},"text":d.get("text")})
        return out
    except Exception:
        logger.exception("search_bm25 failed");return []
async def search_ann(client:weaviate.WeaviateAsyncClient,vector:np.ndarray,top_k:int)->List[Dict]:
    try:
        vec_json=vector.tolist()
        q=client.query.get(settings.WEAVIATE_CLASS,[]).with_near_vector({"vector":vec_json}).with_limit(top_k).with_additional(['id','distance'])
        resp=await q.do()
        docs=resp.get("data",{}).get("Get",{}).get(settings.WEAVIATE_CLASS,[]) or []
        out=[]
        for d in docs:
            aid=d.get("_additional",{}).get("id")
            dist=d.get("_additional",{}).get("distance",None)
            score=-float(dist) if dist is not None else 0.0
            out.append({"id":aid,"distance":dist,"score":score})
        return out
    except Exception:
        logger.exception("search_ann failed");return []
async def batch_fetch_full(client:weaviate.WeaviateAsyncClient,ids:List[str],batch_size:int=64)->List[WeaviateDoc]:
    out:List[WeaviateDoc]=[]
    sem=asyncio.Semaphore(settings.ASYNC_CONCURRENCY)
    async def _get_one(i:str):
        async with sem:
            try:
                obj=await client.data_object.get(i, with_vector=True)
                if not obj: return None
                props=obj.get("properties",{}) or {}
                text=props.get("text","") or props.get("content","") or ""
                metadata=props.copy();metadata.pop("text",None)
                vec=None
                v=obj.get("vector") or obj.get("_additional",{}).get("vector")
                if v is not None:
                    try:
                        arr=np.array(v,dtype=np.float32);n=np.linalg.norm(arr)
                        if n!=0.0: arr=arr/n
                        vec=arr
                    except Exception:
                        vec=None
                return WeaviateDoc(id=i,text=text,metadata=metadata,embedding=vec)
            except Exception:
                logger.exception("data_object.get failed for %s",i);return None
    tasks=[]
    for i in range(0,len(ids),batch_size):
        chunk=ids[i:i+batch_size]
        tasks=[asyncio.create_task(_get_one(cid)) for cid in chunk]
        for fut in asyncio.as_completed(tasks):
            r=await fut
            if r: out.append(r)
    return out
def intra_list_dedupe(hits:List[Dict])->List[Dict]:
    seen=set();out=[]
    for h in hits:
        hid=h.get("id")
        if hid and hid not in seen:
            seen.add(hid);out.append(h)
    return out
def ordered_unique_union(first:List[str],second:List[str],cap:int)->List[str]:
    seen=set();out=[]
    for id_ in (first+second):
        if id_ and id_ not in seen:
            seen.add(id_);out.append(id_)
            if len(out)>=cap: break
    return out
def rrf_scores_for_list(lst:List[str],k:int)->Dict[str,float]:
    m={}
    for pos,id_ in enumerate(lst):
        if not id_: continue
        rank=pos+1
        m.setdefault(id_,0.0)
        m[id_]+=1.0/(k+rank)
    return m
def weighted_rrf_fuse(sparse_lst:List[str],dense_lst:List[str],k:int,top_n:int,alpha:float)->List[Tuple[str,float]]:
    sparse_map=rrf_scores_for_list(sparse_lst,k)
    dense_map=rrf_scores_for_list(dense_lst,k)
    keys=set(list(sparse_map.keys())+list(dense_map.keys()))
    combined={}
    for key in keys:
        s=sparse_map.get(key,0.0);d=dense_map.get(key,0.0)
        combined[key]=alpha*d + (1.0-alpha)*s
    items=sorted(combined.items(),key=lambda x:(-x[1],x[0]))[:top_n]
    return items
def greedy_embedding_dedupe(candidates:List[WeaviateDoc],embeddings:List[Optional[np.ndarray]],threshold:float)->List[WeaviateDoc]:
    if len(candidates)<=1: return candidates
    keep:List[WeaviateDoc]=[];kept_embs:List[np.ndarray]=[]
    for idx,doc in enumerate(candidates):
        emb=embeddings[idx]
        if emb is None:
            keep.append(doc);continue
        if len(kept_embs)==0:
            keep.append(doc);kept_embs.append(emb);continue
        sims=np.dot(np.vstack(kept_embs),emb)
        if np.any(sims>=threshold): continue
        keep.append(doc);kept_embs.append(emb)
    return keep
def prepare_prompt(system_instructions:str,passages:List[WeaviateDoc],user_query:str)->str:
    parts=[system_instructions.strip()]
    for p in passages:
        prov=p.metadata.get("file_name") or p.metadata.get("title") or p.metadata.get("source_url") or "unknown"
        header_parts=[]
        if prov: header_parts.append(f"source:{prov}")
        if p.metadata.get("source_url"): header_parts.append(f"url:{p.metadata.get('source_url')}")
        if p.metadata.get("file_type"): header_parts.append(f"type:{p.metadata.get('file_type')}")
        if p.metadata.get("page_number") is not None: header_parts.append(f"page:{p.metadata.get('page_number')}")
        parts.append(f"[{' | '.join(header_parts) if header_parts else 'source:unknown'}]\n{p.text.strip()}")
    parts.append("User Query: "+user_query.strip());return "\n\n".join(parts)
def _build_service_url(base:str,default_path:str)->str:
    if not base: return ""
    if base.endswith("/") : base=base[:-1]
    if default_path.lstrip("/") in base: return base
    return base.rstrip("/") + "/" + default_path.lstrip("/")
async def call_llm(prompt:str)->Dict[str,Any]:
    if not settings.LLM_SERVICE_URL:
        logger.warning("LLM_SERVICE_URL not configured; returning prompt excerpt");return {"answer":prompt[:2000],"llm_metadata":{"generated":False,"note":"LLM_SERVICE_URL not configured"}}
    url=settings.LLM_SERVICE_URL
    if not any(p in url for p in ("/generate","/v1/generate")):
        url=_build_service_url(url,"/generate")
    if http is None:
        raise RuntimeError("http client not initialized")
    payload={"model":settings.LLM_MODEL,"prompt":prompt}
    resp=await http.post_json(url,payload,timeout=settings.GLOBAL_REQUEST_TIMEOUT)
    ans=resp.get("text") or resp.get("answer") or ""
    return {"answer":ans,"llm_metadata":{"generated":True}}
async def retrieve_answer(query:str,weav:Optional[weaviate.WeaviateAsyncClient]=None,do_presign:bool=True)->Dict[str,Any]:
    start_all=time.time();timings={}
    embedder=None
    embed_url=settings.EMBEDDING_URL or settings.EMBEDDING_SERVICE_URL
    cross_url=settings.CROSS_ENCODER_URL or settings.EMBEDDING_SERVICE_URL
    if embed_url:
        try:
            embedder=RemoteEmbedderClient(embed_url,cross_url)
        except Exception:
            logger.exception("failed to init embedder client; proceeding BM25-only");embedder=None
    weav=weav
    t0=time.time();q_vec=None
    if embedder is not None:
        try:
            q_vec=await embedder.embed_query(query)
        except Exception:
            logger.exception("query embedding failed");q_vec=None
    timings["query_embed_ms"]=int((time.time()-t0)*1000)
    t0=time.time();bm25_hits=[]
    try:
        if weav is not None:
            bm25_hits=await search_bm25(weav,query,settings.BM25_TOP)
    except Exception:
        logger.exception("bm25 search failed")
    timings["bm25_ms"]=int((time.time()-t0)*1000)
    t0=time.time();ann_hits=[]
    if q_vec is not None and weav is not None:
        try:
            ann_hits=await search_ann(weav,q_vec,settings.DENSE_TOP)
        except Exception:
            logger.exception("ann search failed")
    timings["ann_ms"]=int((time.time()-t0)*1000)
    bm25_hits=intra_list_dedupe(bm25_hits);ann_hits=intra_list_dedupe(ann_hits)
    bm25_ids=[h["id"] for h in bm25_hits if h.get("id")];ann_ids=[h["id"] for h in ann_hits if h.get("id")]
    candidate_ids=ordered_unique_union(bm25_ids,ann_ids,settings.UNION_CAP)
    if not candidate_ids:
        return {"answer":"","docs":[],"retrieval":{},"timings":timings}
    t0=time.time();candidate_docs=[]
    try:
        candidate_docs=await batch_fetch_full(weav,candidate_ids,batch_size=settings.BATCH_FETCH_SIZE)
    except Exception:
        logger.exception("batch_fetch_full failed");candidate_docs=[]
    id_to_doc={d.id:d for d in candidate_docs};ids_with_embs=[];embs=[]
    for cid in candidate_ids:
        doc=id_to_doc.get(cid)
        if doc and doc.embedding is not None:
            ids_with_embs.append(cid);embs.append(doc.embedding)
    bi_ranked_ids=[]
    if embedder is not None and q_vec is not None and len(embs)>0:
        try:
            bi_scores=await embedder.bi_encoder_score_pairs_from_vectors(q_vec,embs)
            paired=list(zip(ids_with_embs,bi_scores));paired_sorted=sorted(paired,key=lambda x:-x[1]);bi_ranked_ids=[pid for pid,score in paired_sorted]
        except Exception:
            logger.exception("bi-encoder scoring failed")
    timings["bi_ms"]=int((time.time()-t0)*1000)
    sparse_list=bm25_ids
    dense_list=bi_ranked_ids if bi_ranked_ids else ann_ids
    fused_ids=[]
    if sparse_list and dense_list:
        fused=weighted_rrf_fuse(sparse_list,dense_list,settings.RRF_K,settings.FUSED_TOP_N,settings.ALPHA_WEIGHT)
        fused_ids=[i for i,_ in fused]
    else:
        fused_ids=sparse_list if sparse_list else dense_list
    fused_top=fused_ids[:settings.STAGE_A_TOP_M]
    t0=time.time();fused_docs=await batch_fetch_full(weav,fused_top,batch_size=settings.BATCH_FETCH_SIZE);id_to_fdoc={d.id:d for d in fused_docs};fused_ordered_docs=[id_to_fdoc[i] for i in fused_top if i in id_to_fdoc]
    embeddings_for_dedupe=[d.embedding for d in fused_ordered_docs];deduped_docs=greedy_embedding_dedupe(fused_ordered_docs,embeddings_for_dedupe,settings.DEDUPE_SIM_THRESHOLD)
    timings["fetch_and_dedupe_ms"]=int((time.time()-t0)*1000)
    final_docs=deduped_docs
    if settings.USE_HEAVY_RERANKER and embedder is not None and len(final_docs)>0:
        texts=[d.text for d in final_docs]
        try:
            cross_scores=await embedder.cross_encoder_rerank(query,texts,batch_size=settings.CROSSENCODER_BATCH_SIZE)
            paired=list(zip(final_docs,cross_scores));paired_sorted=sorted(paired,key=lambda x:-x[1])[:settings.HEAVY_TOP_K];final_docs=[p for p,s in paired_sorted]
        except Exception:
            logger.exception("cross-encoder failed; skipping")
    final_docs=final_docs[:settings.MAX_CHUNKS_TO_LLM]
    system_instructions=os.getenv("LLM_SYSTEM_PROMPT","You are a helpful assistant. Use the sources provided.")
    prompt=prepare_prompt(system_instructions,final_docs,query)
    full_prompt=system_instructions+"\n\n"+prompt
    t0=time.time();llm_out=await call_llm(full_prompt);timings["llm_ms"]=int((time.time()-t0)*1000)
    total_ms=int((time.time()-start_all)*1000)
    docs_out=[]
    for d in final_docs:
        docs_out.append({"id":d.id,"text":d.text,"metadata":d.metadata})
    retrieval_meta={"bm25_count":len(bm25_ids),"ann_count":len(ann_ids),"candidate_count":len(candidate_ids),"fused_count":len(fused_ids),"final_chunks":len(final_docs)}
    return {"answer":llm_out.get("answer",""),"docs":docs_out,"retrieval":retrieval_meta,"timings":timings}
