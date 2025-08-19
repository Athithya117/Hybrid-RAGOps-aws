import os,gc,logging,shutil
from typing import List,Optional
import numpy as np
import onnxruntime
from transformers import PreTrainedTokenizerFast
import grpc_pb2,grpc_pb2_grpc
from ray import serve
from prometheus_client import Counter,Histogram,Gauge,start_http_server
from pathlib import Path
os.environ.setdefault("HF_HOME","/workspace/models/hf")
os.environ.setdefault("MODEL_DIR","/workspace/models/onnx")
os.environ.setdefault("EMBEDDER_OMP_NUM_THREADS","1")
os.environ.setdefault("RERANKER_OMP_NUM_THREADS","1")
os.environ.setdefault("EMBEDDER_BATCH_MAX_SIZE","8")
os.environ.setdefault("RERANKER_BATCH_MAX_SIZE","4")
os.environ.setdefault("EMBEDDER_BATCH_WAIT_TIMEOUT_S","0.05")
os.environ.setdefault("RERANKER_BATCH_WAIT_TIMEOUT_S","0.1")
os.environ.setdefault("EMBEDDER_NUM_CPUS","1")
os.environ.setdefault("RERANKER_NUM_CPUS","1")
os.environ.setdefault("PROMETHEUS_METRICS_PORT","8080")
os.environ.setdefault("PROMETHEUS_DISABLED","0")
os.environ.setdefault("LOG_LEVEL","INFO")
logging.basicConfig(level=os.getenv("LOG_LEVEL"),format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger("rag8s.serve")
def get_env(name:str,default=None,required:bool=False,cast=None):
    val=os.getenv(name,default)
    if required and (val is None or (isinstance(val,str) and val.strip()=="")):
        logger.error("Missing required environment variable: %s",name)
        raise SystemExit(2)
    if cast and val is not None:
        try:
            return cast(val)
        except Exception as e:
            logger.warning("Failed casting env %s=%r with %s: %s",name,val,cast,e)
            return val
    return val
HF_HOME=get_env("HF_HOME","/workspace/models/hf")
MODEL_DIR=get_env("MODEL_DIR","/workspace/models/onnx")
MODEL_EMBEDDER_NAME=get_env("MODEL_EMBEDDER_NAME",None,required=True)
MODEL_RERANKER_NAME=get_env("MODEL_RERANKER_NAME",None,required=True)
EMBEDDER_ONNX_PATH=get_env("EMBEDDER_ONNX_PATH",None)
RERANKER_ONNX_PATH=get_env("RERANKER_ONNX_PATH",None)
# explicit tokenizer/config/env paths (optional)
EMBEDDER_CONFIG_PATH=get_env("EMBEDDER_CONFIG_PATH",None)
EMBEDDER_TOKENIZER_PATH=get_env("EMBEDDER_TOKENIZER_PATH",None)
EMBEDDER_TOKENIZER_CONFIG_PATH=get_env("EMBEDDER_TOKENIZER_CONFIG_PATH",None)
EMBEDDER_SPECIAL_TOKENS_MAP_PATH=get_env("EMBEDDER_SPECIAL_TOKENS_MAP_PATH",None)
RERANKER_CONFIG_PATH=get_env("RERANKER_CONFIG_PATH",None)
RERANKER_TOKENIZER_PATH=get_env("RERANKER_TOKENIZER_PATH",None)
RERANKER_TOKENIZER_CONFIG_PATH=get_env("RERANKER_TOKENIZER_CONFIG_PATH",None)
RERANKER_SPECIAL_TOKENS_MAP_PATH=get_env("RERANKER_SPECIAL_TOKENS_MAP_PATH",None)
EMBEDDER_OMP_NUM_THREADS=int(get_env("EMBEDDER_OMP_NUM_THREADS","1"))
RERANKER_OMP_NUM_THREADS=int(get_env("RERANKER_OMP_NUM_THREADS","1"))
EMBEDDER_BATCH_MAX_SIZE=int(get_env("EMBEDDER_BATCH_MAX_SIZE","8"))
RERANKER_BATCH_MAX_SIZE=int(get_env("RERANKER_BATCH_MAX_SIZE","4"))
EMBEDDER_BATCH_WAIT_TIMEOUT_S=float(get_env("EMBEDDER_BATCH_WAIT_TIMEOUT_S","0.05"))
RERANKER_BATCH_WAIT_TIMEOUT_S=float(get_env("RERANKER_BATCH_WAIT_TIMEOUT_S","0.1"))
PROM_METRICS_PORT=int(get_env("PROMETHEUS_METRICS_PORT","8080"))
PROM_DISABLED=get_env("PROMETHEUS_DISABLED","0")=="1"
INFER_COUNTER=Counter("rag8s_inference_requests_total","Total inference requests",["model"])
INFER_ERRORS=Counter("rag8s_inference_errors_total","Inference errors",["model"])
INFER_LATENCY=Histogram("rag8s_inference_latency_seconds","Latency per inference",["model"])
BATCH_SIZE_HIST=Histogram("rag8s_batch_size","Batch size distribution",["model"])
READY_GAUGE=Gauge("rag8s_initialized","Model initialization state (1=ready,0=not ready)",["model"])
def _basename_from_repo_id(repo_id:str)->str:
    return repo_id.rstrip("/").split("/")[-1]
def _candidate_tokenizer_dirs(repo_id:str):
    name=_basename_from_repo_id(repo_id)
    candidates=[os.path.join(MODEL_DIR,name),os.path.join(HF_HOME,name)]
    hub_prefix=os.path.join(HF_HOME,"hub")
    try:
        if os.path.isdir(hub_prefix):
            for candidate in os.listdir(hub_prefix):
                if name in candidate or repo_id.replace("/","--") in candidate:
                    snap_dir=os.path.join(hub_prefix,candidate,"snapshots")
                    if os.path.isdir(snap_dir):
                        for s in os.listdir(snap_dir):
                            candidates.append(os.path.join(snap_dir,s))
                    candidates.append(os.path.join(hub_prefix,candidate))
    except Exception:
        logger.exception("Error scanning HF hub for tokenizer candidates")
    return [p for p in candidates if p]
def _candidate_onnx_paths(repo_id:str):
    name=_basename_from_repo_id(repo_id)
    candidates=[os.path.join(MODEL_DIR,name,"onnx","model_int8.onnx"),os.path.join(HF_HOME,name,"onnx","model_int8.onnx"),os.path.join(MODEL_DIR,name,"model_int8.onnx")]
    hub_prefix=os.path.join(HF_HOME,"hub")
    try:
        if os.path.isdir(hub_prefix):
            for candidate in os.listdir(hub_prefix):
                if name in candidate or repo_id.replace("/","--") in candidate:
                    snap_dir=os.path.join(hub_prefix,candidate,"snapshots")
                    if os.path.isdir(snap_dir):
                        for s in os.listdir(snap_dir):
                            candidates.append(os.path.join(snap_dir,s,"onnx","model_int8.onnx"))
                            candidates.append(os.path.join(snap_dir,s,"model_int8.onnx"))
                    candidates.append(os.path.join(hub_prefix,candidate,"onnx","model_int8.onnx"))
    except Exception:
        logger.exception("Error scanning HF hub for onnx candidates")
    return [p for p in candidates if p]
def _make_tokenizer_dir_from_files(name:str, tokenizer_path:Optional[str], tokenizer_config_path:Optional[str], special_tokens_path:Optional[str]) -> Optional[str]:
    if not tokenizer_path and not tokenizer_config_path and not special_tokens_path:
        return None
    tmp_dir=Path("/tmp")/f"hf_tokenizer_{name}"
    try:
        tmp_dir.mkdir(parents=True,exist_ok=True)
        copied=False
        mapping=[(tokenizer_path,"tokenizer.json"),(tokenizer_config_path,"tokenizer_config.json"),(special_tokens_path,"special_tokens_map.json")]
        for src, dst in mapping:
            if src:
                s=Path(src)
                if s.is_file():
                    shutil.copy(str(s), str(tmp_dir/dst))
                    os.chmod(str(tmp_dir/dst),0o444)
                    copied=True
                else:
                    logger.warning("Explicit tokenizer file %s for %s not found: %s", dst, name, src)
        if copied:
            return str(tmp_dir)
        # nothing copied -> remove dir if empty
        try:
            if not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except Exception:
            pass
    except Exception:
        logger.exception("Failed preparing tokenizer dir for %s",name)
    return None
def resolve_tokenizer_dir(repo_id:str)->Optional[str]:
    # prefer explicit env-supplied tokenizers for known embedder/reranker
    if repo_id==MODEL_EMBEDDER_NAME:
        explicit=_make_tokenizer_dir_from_files("embedder",EMBEDDER_TOKENIZER_PATH,EMBEDDER_TOKENIZER_CONFIG_PATH,EMBEDDER_SPECIAL_TOKENS_MAP_PATH)
        if explicit:
            logger.info("Using explicit embedder tokenizer dir %s from env paths",explicit)
            return explicit
    if repo_id==MODEL_RERANKER_NAME:
        explicit=_make_tokenizer_dir_from_files("reranker",RERANKER_TOKENIZER_PATH,RERANKER_TOKENIZER_CONFIG_PATH,RERANKER_SPECIAL_TOKENS_MAP_PATH)
        if explicit:
            logger.info("Using explicit reranker tokenizer dir %s from env paths",explicit)
            return explicit
    for p in _candidate_tokenizer_dirs(repo_id):
        try:
            if os.path.isdir(p):
                if os.path.exists(os.path.join(p,"tokenizer.json")) or os.path.exists(os.path.join(p,"tokenizer_config.json")):
                    logger.info("Using tokenizer dir %s for %s",p,repo_id)
                    return p
                for root,_,files in os.walk(p):
                    if "tokenizer.json" in files or "tokenizer_config.json" in files:
                        logger.info("Using tokenizer nested dir %s for %s",root,repo_id)
                        return root
        except Exception:
            logger.exception("Error checking tokenizer candidate %s",p)
    logger.warning("No local tokenizer dir found for %s",repo_id)
    return None
def resolve_onnx_path(repo_id:str)->Optional[str]:
    for p in _candidate_onnx_paths(repo_id):
        try:
            if os.path.isfile(p):
                logger.info("Found ONNX model for %s at %s",repo_id,p)
                return p
        except Exception:
            logger.exception("Error checking onnx candidate %s",p)
    logger.warning("No local ONNX file found for %s",repo_id)
    return None
def check_model(path:str):
    if not os.path.isfile(path):
        logger.error("ONNX model not found: %s",path)
        raise FileNotFoundError(path)
def _set_thread_envs(threads:int):
    try:
        os.environ["OMP_NUM_THREADS"]=str(threads)
        os.environ["MKL_NUM_THREADS"]=str(threads)
    except Exception:
        logger.exception("Failed setting thread envs")
def make_session(path:str,intra_op_threads=1)->onnxruntime.InferenceSession:
    check_model(path)
    _set_thread_envs(intra_op_threads)
    logger.info("Creating ONNX Runtime session for %s with intra_op_threads=%s",path,intra_op_threads)
    opts=onnxruntime.SessionOptions()
    opts.intra_op_num_threads=intra_op_threads
    opts.execution_mode=onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess=onnxruntime.InferenceSession(path,opts,providers=["CPUExecutionProvider"])
    logger.info("ONNX Runtime session created for %s",path)
    return sess
def final_onnx_path_for(repo_env_name:str,explicit_env_path:Optional[str],repo_id:str)->str:
    if explicit_env_path:
        if os.path.isfile(explicit_env_path):
            logger.info("Using explicit ONNX path from env %s=%s",repo_env_name,explicit_env_path)
            return explicit_env_path
        else:
            logger.error("Explicit ONNX path %s set but missing: %s",repo_env_name,explicit_env_path)
            raise FileNotFoundError(explicit_env_path)
    p=resolve_onnx_path(repo_id)
    if p:
        return p
    raise FileNotFoundError(f"ONNX model for {repo_id} not found under MODEL_DIR ({MODEL_DIR}) or HF_HOME ({HF_HOME}).")
def final_tokenizer_dir_for(repo_id:str)->Optional[str]:
    # If explicit tokenizer dir provided via env named TOKENIZER_DIR (not currently used), it could be checked here.
    # Prefer explicit file-based dirs already handled in resolve_tokenizer_dir.
    return resolve_tokenizer_dir(repo_id)
_EMBED_TOKENIZER=None
_EMBED_SESSION=None
_RERANK_TOKENIZER=None
_RERANK_SESSION=None
_initialized=False
def start_prometheus_if_enabled(host:str="0.0.0.0"):
    if PROM_DISABLED:
        logger.info("Prometheus metrics disabled by PROMETHEUS_DISABLED=1")
        return
    try:
        start_http_server(PROM_METRICS_PORT,addr=host)
        logger.info("Prometheus metrics server started on %s:%s",host,PROM_METRICS_PORT)
    except Exception:
        logger.exception("Failed to start Prometheus metrics server")
        raise
def _close_sessions():
    global _EMBED_SESSION,_RERANK_SESSION,_EMBED_TOKENIZER,_RERANK_TOKENIZER,_initialized
    try:
        _EMBED_SESSION=None
        _EMBED_TOKENIZER=None
        _RERANK_SESSION=None
        _RERANK_TOKENIZER=None
        gc.collect()
        _initialized=False
        READY_GAUGE.labels(model="embedder").set(0)
        READY_GAUGE.labels(model="reranker").set(0)
        logger.info("Cleared ONNX sessions and tokenizers")
    except Exception:
        logger.exception("Error closing sessions")
def ensure_initialized():
    global _EMBED_TOKENIZER,_EMBED_SESSION,_RERANK_TOKENIZER,_RERANK_SESSION,_initialized
    if _initialized:
        return
    logger.info("Initializing tokenizers and ONNX sessions for embedder=%s reranker=%s",MODEL_EMBEDDER_NAME,MODEL_RERANKER_NAME)
    try:
        embed_tok_dir=final_tokenizer_dir_for(MODEL_EMBEDDER_NAME)
        if not embed_tok_dir:
            logger.error("Embedder tokenizer dir missing for %s",MODEL_EMBEDDER_NAME)
            raise SystemExit(3)
        _EMBED_TOKENIZER=PreTrainedTokenizerFast.from_pretrained(embed_tok_dir,local_files_only=True,trust_remote_code=True)
        assert getattr(_EMBED_TOKENIZER,"is_fast",True),"Fast tokenizer not loaded"
        logger.info("Embedder tokenizer loaded from %s",embed_tok_dir)
        if EMBEDDER_CONFIG_PATH and not Path(EMBEDDER_CONFIG_PATH).is_file():
            logger.warning("Embedder config path set but missing: %s",EMBEDDER_CONFIG_PATH)
        embed_onnx=final_onnx_path_for("EMBEDDER_ONNX_PATH",EMBEDDER_ONNX_PATH,MODEL_EMBEDDER_NAME)
        _EMBED_SESSION=make_session(embed_onnx,intra_op_threads=EMBEDDER_OMP_NUM_THREADS)
        logger.info("Embedder session ready")
        rerank_tok_dir=final_tokenizer_dir_for(MODEL_RERANKER_NAME)
        if not rerank_tok_dir:
            logger.error("Reranker tokenizer dir missing for %s",MODEL_RERANKER_NAME)
            raise SystemExit(3)
        _RERANK_TOKENIZER=PreTrainedTokenizerFast.from_pretrained(rerank_tok_dir,local_files_only=True,trust_remote_code=True)
        assert getattr(_RERANK_TOKENIZER,"is_fast",True),"Fast tokenizer not loaded"
        logger.info("Reranker tokenizer loaded from %s",rerank_tok_dir)
        if RERANKER_CONFIG_PATH and not Path(RERANKER_CONFIG_PATH).is_file():
            logger.warning("Reranker config path set but missing: %s",RERANKER_CONFIG_PATH)
        rerank_onnx=final_onnx_path_for("RERANKER_ONNX_PATH",RERANKER_ONNX_PATH,MODEL_RERANKER_NAME)
        _RERANK_SESSION=make_session(rerank_onnx,intra_op_threads=RERANKER_OMP_NUM_THREADS)
        logger.info("Reranker session ready")
        _initialized=True
        READY_GAUGE.labels(model="embedder").set(1)
        READY_GAUGE.labels(model="reranker").set(1)
        logger.info("Module-level initialization complete")
    except Exception:
        READY_GAUGE.labels(model="embedder").set(0)
        READY_GAUGE.labels(model="reranker").set(0)
        logger.exception("Initialization failed")
        _close_sessions()
        raise
@serve.batch(max_batch_size=EMBEDDER_BATCH_MAX_SIZE,batch_wait_timeout_s=EMBEDDER_BATCH_WAIT_TIMEOUT_S)
async def _embed_batch(requests:List[grpc_pb2.EmbedRequest])->List[grpc_pb2.EmbedResponse]:
    ensure_initialized()
    model_tag="embedder"
    INFER_COUNTER.labels(model=model_tag).inc(len(requests))
    BATCH_SIZE_HIST.labels(model=model_tag).observe(len(requests))
    try:
        with INFER_LATENCY.labels(model=model_tag).time():
            all_texts=[t for req in requests for t in req.texts]
            if not all_texts:
                return [grpc_pb2.EmbedResponse(embeddings=[]) for _ in requests]
            enc=_EMBED_TOKENIZER(all_texts,padding=True,truncation=True,return_tensors="np")
            outputs=_EMBED_SESSION.run(None,{"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"]})
            embeddings_arr=np.array(outputs[0]).mean(axis=1)
            responses=[]
            idx=0
            for req in requests:
                n=len(req.texts)
                if n==0:
                    responses.append(grpc_pb2.EmbedResponse(embeddings=[]))
                else:
                    slice_arr=embeddings_arr[idx:idx+n]
                    flat=slice_arr.reshape(-1).astype(float).tolist()
                    responses.append(grpc_pb2.EmbedResponse(embeddings=flat))
                    idx+=n
            return responses
    except Exception:
        INFER_ERRORS.labels(model=model_tag).inc(len(requests))
        logger.exception("Embed batch failed")
        return [grpc_pb2.EmbedResponse(embeddings=[]) for _ in requests]
@serve.batch(max_batch_size=RERANKER_BATCH_MAX_SIZE,batch_wait_timeout_s=RERANKER_BATCH_WAIT_TIMEOUT_S)
async def _rerank_batch(requests:List[grpc_pb2.RerankRequest])->List[grpc_pb2.RerankResponse]:
    ensure_initialized()
    model_tag="reranker"
    INFER_COUNTER.labels(model=model_tag).inc(len(requests))
    BATCH_SIZE_HIST.labels(model=model_tag).observe(len(requests))
    try:
        with INFER_LATENCY.labels(model=model_tag).time():
            all_pairs=[pair for req in requests for pair in req.pairs]
            if not all_pairs:
                return [grpc_pb2.RerankResponse(scores=[]) for _ in requests]
            queries=[p.query for p in all_pairs]
            docs=[p.doc for p in all_pairs]
            enc=_RERANK_TOKENIZER(queries,docs,padding=True,truncation=True,return_tensors="np")
            outputs=_RERANK_SESSION.run(None,{"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"]})
            scores_arr=np.array(outputs[0])
            if scores_arr.ndim>1 and scores_arr.shape[1]==1:
                scores_arr=scores_arr.squeeze(axis=1)
            responses=[]
            idx=0
            for req in requests:
                n=len(req.pairs)
                if n==0:
                    responses.append(grpc_pb2.RerankResponse(scores=[]))
                else:
                    slice_scores=scores_arr[idx:idx+n].astype(float).tolist()
                    responses.append(grpc_pb2.RerankResponse(scores=slice_scores))
                    idx+=n
            return responses
    except Exception:
        INFER_ERRORS.labels(model=model_tag).inc(len(requests))
        logger.exception("Rerank batch failed")
        return [grpc_pb2.RerankResponse(scores=[]) for _ in requests]
@serve.deployment(ray_actor_options={"num_cpus":float(get_env("EMBEDDER_NUM_CPUS","1"))})
class EmbedderServicer(grpc_pb2_grpc.EmbedServiceServicer):
    def __init__(self):
        logger.info("Initializing EmbedderServicer for %s",MODEL_EMBEDDER_NAME)
        ensure_initialized()
    async def Embed(self,request:grpc_pb2.EmbedRequest)->grpc_pb2.EmbedResponse:
        return await _embed_batch(request)
@serve.deployment(ray_actor_options={"num_cpus":float(get_env("RERANKER_NUM_CPUS","1"))})
class RerankerServicer(grpc_pb2_grpc.RerankServiceServicer):
    def __init__(self):
        logger.info("Initializing RerankerServicer for %s",MODEL_RERANKER_NAME)
        ensure_initialized()
    async def Rerank(self,request:grpc_pb2.RerankRequest)->grpc_pb2.RerankResponse:
        return await _rerank_batch(request)
def warmup_models():
    try:
        ensure_initialized()
        logger.info("Model warmup completed.")
    except Exception as e:
        logger.exception("Warmup failed: %s",e)
