import os
import gc
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import onnxruntime
from transformers import PreTrainedTokenizerFast
import grpc_pb2, grpc_pb2_grpc
from ray import serve
from fastapi import FastAPI, Request, Response
from starlette.responses import JSONResponse
from traceloop.sdk import Traceloop
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
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
os.environ.setdefault("LOG_LEVEL","INFO")
logging.basicConfig(level=os.getenv("LOG_LEVEL"),format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger("rag8s.serve")
def get_env(name: str, default=None, required: bool=False, cast=None):
    val=os.getenv(name, default)
    if required and (val is None or (isinstance(val, str) and val.strip()=="")):
        logger.error("Missing required environment variable: %s", name)
        raise SystemExit(2)
    if cast and val is not None:
        try:
            return cast(val)
        except Exception as e:
            logger.warning("Failed casting env %s=%r with %s: %s", name, val, cast, e)
            return val
    return val
HF_HOME=get_env("HF_HOME","/workspace/models/hf")
MODEL_DIR=get_env("MODEL_DIR","/workspace/models/onnx")
MODEL_EMBEDDER_NAME=get_env("MODEL_EMBEDDER_NAME",None,required=True)
MODEL_RERANKER_NAME=get_env("MODEL_RERANKER_NAME",None,required=True)
EMBEDDER_ONNX_PATH=get_env("EMBEDDER_ONNX_PATH",None)
RERANKER_ONNX_PATH=get_env("RERANKER_ONNX_PATH",None)
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
PROM_DISABLED=True
PROM_METRICS_PORT=int(get_env("PROMETHEUS_METRICS_PORT","8080"))
PROM_DISABLED=get_env("PROMETHEUS_DISABLED","1")=="1"
OTEL_ENDPOINT=get_env("OTEL_EXPORTER_OTLP_ENDPOINT","http://otel-collector.default.svc.cluster.local:4317")
SERVICE_NAME=get_env("OTEL_SERVICE_NAME","rag8s-onnx-embedder-reranker")
resource=Resource.create({"service.name": SERVICE_NAME})
Traceloop.init(app_name=SERVICE_NAME)
tracer_provider=TracerProvider(resource=resource)
span_exporter=OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
trace.set_tracer_provider(tracer_provider)
tracer=trace.get_tracer(__name__)
metric_exporter=OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True)
metric_reader=PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
meter_provider=MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter=metrics.get_meter("rag8s.embedder", "0.1")
INFER_COUNTER = meter.create_counter("rag8s.inference.requests.total", description="Total inference requests")
INFER_ERRORS = meter.create_counter("rag8s.inference.errors.total", description="Inference errors")
INFER_LATENCY = meter.create_histogram("rag8s.inference.latency.seconds", description="Latency per inference")
BATCH_SIZE_HIST = meter.create_histogram("rag8s.batch.size", description="Batch size distribution")
_EMBED_TOKENIZER=None
_EMBED_SESSION=None
_RERANK_TOKENIZER=None
_RERANK_SESSION=None
_initialized=False
def _basename_from_repo_id(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]
def _candidate_tokenizer_dirs(repo_id: str):
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
def _candidate_onnx_paths(repo_id: str):
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
        logger.exception("Error checking onnx candidate %s", p)
    return [p for p in candidates if p]
def _make_tokenizer_dir_from_files(name: str, tokenizer_path: Optional[str], tokenizer_config_path: Optional[str], special_tokens_path: Optional[str]) -> Optional[str]:
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
        try:
            if not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except Exception:
            pass
    except Exception:
        logger.exception("Failed preparing tokenizer dir for %s", name)
    return None
def resolve_tokenizer_dir(repo_id: str) -> Optional[str]:
    if repo_id==MODEL_EMBEDDER_NAME:
        explicit=_make_tokenizer_dir_from_files("embedder",EMBEDDER_TOKENIZER_PATH,EMBEDDER_TOKENIZER_CONFIG_PATH,EMBEDDER_SPECIAL_TOKENS_MAP_PATH)
        if explicit:
            logger.info("Using explicit embedder tokenizer dir %s from env paths", explicit)
            return explicit
    if repo_id==MODEL_RERANKER_NAME:
        explicit=_make_tokenizer_dir_from_files("reranker",RERANKER_TOKENIZER_PATH,RERANKER_TOKENIZER_CONFIG_PATH,RERANKER_SPECIAL_TOKENS_MAP_PATH)
        if explicit:
            logger.info("Using explicit reranker tokenizer dir %s from env paths", explicit)
            return explicit
    for p in _candidate_tokenizer_dirs(repo_id):
        try:
            if os.path.isdir(p):
                if os.path.exists(os.path.join(p,"tokenizer.json")) or os.path.exists(os.path.join(p,"tokenizer_config.json")):
                    logger.info("Using tokenizer dir %s for %s", p, repo_id)
                    return p
                for root,_,files in os.walk(p):
                    if "tokenizer.json" in files or "tokenizer_config.json" in files:
                        logger.info("Using tokenizer nested dir %s for %s", root, repo_id)
                        return root
        except Exception:
            logger.exception("Error checking tokenizer candidate %s", p)
    logger.warning("No local tokenizer dir found for %s", repo_id)
    return None
def resolve_onnx_path(repo_id: str) -> Optional[str]:
    for p in _candidate_onnx_paths(repo_id):
        try:
            if os.path.isfile(p):
                logger.info("Found ONNX model for %s at %s", repo_id, p)
                return p
        except Exception:
            logger.exception("Error checking onnx candidate %s", p)
    logger.warning("No local ONNX file found for %s", repo_id)
    return None
def check_model(path: str):
    if not os.path.isfile(path):
        logger.error("ONNX model not found: %s", path)
        raise FileNotFoundError(path)
def _set_thread_envs(threads: int):
    try:
        os.environ["OMP_NUM_THREADS"]=str(threads)
        os.environ["MKL_NUM_THREADS"]=str(threads)
    except Exception:
        logger.exception("Failed setting thread envs")
def make_session(path: str, intra_op_threads=1) -> onnxruntime.InferenceSession:
    check_model(path)
    _set_thread_envs(intra_op_threads)
    logger.info("Creating ONNX Runtime session for %s with intra_op_threads=%s", path, intra_op_threads)
    opts=onnxruntime.SessionOptions()
    opts.intra_op_num_threads=intra_op_threads
    opts.execution_mode=onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess=onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    logger.info("ONNX Runtime session created for %s", path)
    return sess
def final_onnx_path_for(repo_env_name: str, explicit_env_path: Optional[str], repo_id: str) -> str:
    if explicit_env_path:
        if os.path.isfile(explicit_env_path):
            logger.info("Using explicit ONNX path from env %s=%s", repo_env_name, explicit_env_path)
            return explicit_env_path
        else:
            logger.error("Explicit ONNX path %s set but missing: %s", repo_env_name, explicit_env_path)
            raise FileNotFoundError(explicit_env_path)
    p=resolve_onnx_path(repo_id)
    if p:
        return p
    raise FileNotFoundError(f"ONNX model for {repo_id} not found under MODEL_DIR ({MODEL_DIR}) or HF_HOME ({HF_HOME}).")
def final_tokenizer_dir_for(repo_id: str) -> Optional[str]:
    return resolve_tokenizer_dir(repo_id)
_initialized_lock=False
def _close_sessions():
    global _EMBED_SESSION, _RERANK_SESSION, _EMBED_TOKENIZER, _RERANK_TOKENIZER, _initialized
    try:
        _EMBED_SESSION=None
        _EMBED_TOKENIZER=None
        _RERANK_SESSION=None
        _RERANK_TOKENIZER=None
        gc.collect()
        _initialized=False
        logger.info("Cleared ONNX sessions and tokenizers")
    except Exception:
        logger.exception("Error closing sessions")
def ensure_initialized():
    global _EMBED_TOKENIZER, _EMBED_SESSION, _RERANK_TOKENIZER, _RERANK_SESSION, _initialized
    if _initialized:
        return
    logger.info("Initializing tokenizers and ONNX sessions for embedder=%s reranker=%s", MODEL_EMBEDDER_NAME, MODEL_RERANKER_NAME)
    try:
        embed_tok_dir=final_tokenizer_dir_for(MODEL_EMBEDDER_NAME)
        if not embed_tok_dir:
            logger.error("Embedder tokenizer dir missing for %s", MODEL_EMBEDDER_NAME)
            raise SystemExit(3)
        _EMBED_TOKENIZER=PreTrainedTokenizerFast.from_pretrained(embed_tok_dir, local_files_only=True, trust_remote_code=True)
        assert getattr(_EMBED_TOKENIZER,"is_fast", True)
        logger.info("Embedder tokenizer loaded from %s", embed_tok_dir)
        if EMBEDDER_CONFIG_PATH and not Path(EMBEDDER_CONFIG_PATH).is_file():
            logger.warning("Embedder config path set but missing: %s", EMBEDDER_CONFIG_PATH)
        embed_onnx=final_onnx_path_for("EMBEDDER_ONNX_PATH", EMBEDDER_ONNX_PATH, MODEL_EMBEDDER_NAME)
        _EMBED_SESSION=make_session(embed_onnx, intra_op_threads=EMBEDDER_OMP_NUM_THREADS)
        logger.info("Embedder session ready")
        rerank_tok_dir=final_tokenizer_dir_for(MODEL_RERANKER_NAME)
        if not rerank_tok_dir:
            logger.error("Reranker tokenizer dir missing for %s", MODEL_RERANKER_NAME)
            raise SystemExit(3)
        _RERANK_TOKENIZER=PreTrainedTokenizerFast.from_pretrained(rerank_tok_dir, local_files_only=True, trust_remote_code=True)
        assert getattr(_RERANK_TOKENIZER,"is_fast", True)
        logger.info("Reranker tokenizer loaded from %s", rerank_tok_dir)
        if RERANKER_CONFIG_PATH and not Path(RERANKER_CONFIG_PATH).is_file():
            logger.warning("Reranker config path set but missing: %s", RERANKER_CONFIG_PATH)
        rerank_onnx=final_onnx_path_for("RERANKER_ONNX_PATH", RERANKER_ONNX_PATH, MODEL_RERANKER_NAME)
        _RERANK_SESSION=make_session(rerank_onnx, intra_op_threads=RERANKER_OMP_NUM_THREADS)
        logger.info("Reranker session ready")
        _initialized=True
        logger.info("Module-level initialization complete")
    except Exception:
        _close_sessions()
        logger.exception("Initialization failed")
        raise
@serve.batch(max_batch_size=EMBEDDER_BATCH_MAX_SIZE, batch_wait_timeout_s=EMBEDDER_BATCH_WAIT_TIMEOUT_S)
async def _embed_batch(requests: List[grpc_pb2.EmbedRequest]) -> List[grpc_pb2.EmbedResponse]:
    ensure_initialized()
    model_tag="embedder"
    INFER_COUNTER.add(len(requests), {"model": model_tag})
    BATCH_SIZE_HIST.record(len(requests), {"model": model_tag})
    start=time.perf_counter()
    try:
        all_texts=[t for req in requests for t in req.texts]
        if not all_texts:
            duration=time.perf_counter()-start
            INFER_LATENCY.record(duration, {"model": model_tag})
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
        duration=time.perf_counter()-start
        INFER_LATENCY.record(duration, {"model": model_tag})
        return responses
    except Exception:
        INFER_ERRORS.add(len(requests), {"model": model_tag})
        logger.exception("Embed batch failed")
        return [grpc_pb2.EmbedResponse(embeddings=[]) for _ in requests]
@serve.batch(max_batch_size=RERANKER_BATCH_MAX_SIZE, batch_wait_timeout_s=RERANKER_BATCH_WAIT_TIMEOUT_S)
async def _rerank_batch(requests: List[grpc_pb2.RerankRequest]) -> List[grpc_pb2.RerankResponse]:
    ensure_initialized()
    model_tag="reranker"
    INFER_COUNTER.add(len(requests), {"model": model_tag})
    BATCH_SIZE_HIST.record(len(requests), {"model": model_tag})
    start=time.perf_counter()
    try:
        all_pairs=[pair for req in requests for pair in req.pairs]
        if not all_pairs:
            duration=time.perf_counter()-start
            INFER_LATENCY.record(duration, {"model": model_tag})
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
        duration=time.perf_counter()-start
        INFER_LATENCY.record(duration, {"model": model_tag})
        return responses
    except Exception:
        INFER_ERRORS.add(len(requests), {"model": model_tag})
        logger.exception("Rerank batch failed")
        return [grpc_pb2.RerankResponse(scores=[]) for _ in requests]
app=FastAPI()
ready_event=None
@app.get("/healthz")
async def healthz():
    status=status_code=503
    try:
        if _initialized:
            status_code=200
            return JSONResponse({"ready": True}, status_code=200)
    except Exception:
        pass
    return JSONResponse({"ready": False}, status_code=503)
@app.post("/embed")
async def http_embed(req: Request):
    try:
        body=await req.json()
        texts=body.get("texts",[])
        handle=serve.get_app_handle("rag8s_embedder_app")
        resp=handle.Embed.remote(grpc_pb2.EmbedRequest(texts=texts)).result()
        return JSONResponse({"embeddings": list(resp.embeddings)})
    except Exception as exc:
        logger.exception("HTTP embed error")
        return JSONResponse({"error": str(exc)}, status_code=500)
@app.post("/rerank")
async def http_rerank(req: Request):
    try:
        body=await req.json()
        pairs=body.get("pairs",[])
        proto_pairs=[grpc_pb2.Pair(query=p.get("query",""), doc=p.get("doc","")) for p in pairs]
        request_proto=grpc_pb2.RerankRequest(pairs=proto_pairs)
        handle=serve.get_app_handle("rag8s_reranker_app")
        resp=handle.Rerank.remote(request_proto).result()
        return JSONResponse({"scores": list(resp.scores)})
    except Exception as exc:
        logger.exception("HTTP rerank error")
        return JSONResponse({"error": str(exc)}, status_code=500)
class GrpcForwarderServicer(grpc_pb2_grpc.EmbedServiceServicer, grpc_pb2_grpc.RerankServiceServicer):
    def __init__(self):
        self._embed_handle=None
        self._rerank_handle=None
    def _ensure_handles(self):
        if self._embed_handle is None:
            for _ in range(40):
                try:
                    self._embed_handle=serve.get_app_handle("rag8s_embedder_app")
                    break
                except Exception:
                    time.sleep(0.25)
        if self._rerank_handle is None:
            for _ in range(40):
                try:
                    self._rerank_handle=serve.get_app_handle("rag8s_reranker_app")
                    break
                except Exception:
                    time.sleep(0.25)
    def Embed(self, request, context):
        try:
            self._ensure_handles()
            return self._embed_handle.Embed.remote(request).result()
        except Exception:
            logger.exception("gRPC Embed forward error")
            context.set_details("internal error")
            context.set_code(5)
            return grpc_pb2.EmbedResponse(embeddings=[])
    def Rerank(self, request, context):
        try:
            self._ensure_handles()
            return self._rerank_handle.Rerank.remote(request).result()
        except Exception:
            logger.exception("gRPC Rerank forward error")
            context.set_details("internal error")
            context.set_code(5)
            return grpc_pb2.RerankResponse(scores=[])
def start_grpc_server(port: int):
    import grpc
    from concurrent import futures
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    forwarder=GrpcForwarderServicer()
    grpc_pb2_grpc.add_EmbedServiceServicer_to_server(forwarder, server)
    grpc_pb2_grpc.add_RerankServiceServicer_to_server(forwarder, server)
    addr=f"[::]:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("gRPC forwarder started on %s", addr)
    return server
def _init_ray(address: str):
    try:
        if address in ("", "local", "none"):
            ray.init()
            return
        if address=="auto":
            try:
                ray.init(address="auto")
                return
            except Exception:
                ray.init()
                return
        ray.init(address=address)
    except Exception:
        logger.exception("Ray init failed")
        raise
def deploy_services():
    attempts=0
    last_exc=None
    while attempts<3:
        try:
            if not ray.is_initialized():
                _init_ray(os.getenv("RAY_ADDRESS","auto"))
            serve.start(detached=True, http_options={"host":"0.0.0.0","port":int(os.getenv("HTTP_PORT","8000"))})
            embed_dep=EmbedderServicer.bind()
            rerank_dep=RerankerServicer.bind()
            serve.run(embed_dep, name="rag8s_embedder_app", route_prefix=os.getenv("EMBED_ROUTE","/rag8s_embedder_app"), blocking=False)
            serve.run(rerank_dep, name="rag8s_reranker_app", route_prefix=os.getenv("RERANK_ROUTE","/rag8s_reranker_app"), blocking=False)
            Ingress=serve.deployment()(serve.ingress(app))
            serve.run(Ingress.bind(), name="rag8s_http_gateway", route_prefix="/", blocking=False)
            try:
                warmup_models()
            except Exception:
                logger.exception("warmup_models failed (continuing)")
            return
        except Exception as exc:
            last_exc=exc
            attempts+=1
            logger.exception("Deploy attempt %s failed", attempts)
            time.sleep(2*attempts)
    logger.error("All deploy attempts failed")
    raise last_exc
def shutdown(signum, frame):
    import signal
    stop_event=None
    try:
        serve.shutdown()
    except Exception:
        logger.exception("Error shutting down Serve")
    try:
        ray.shutdown()
    except Exception:
        logger.exception("Error shutting down Ray")
    os._exit(0)
def main():
    try:
        deploy_services()
    except Exception:
        os._exit(3)
    grpc_server=None
    try:
        grpc_server=start_grpc_server(int(os.getenv("GRPC_PORT","9000")))
    except Exception:
        logger.exception("Failed to start gRPC server")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    if grpc_server:
        try:
            grpc_server.stop(0)
        except Exception:
            logger.exception("Error stopping gRPC server")
    try:
        serve.shutdown()
    except Exception:
        logger.exception("Error shutting down Serve")
    try:
        ray.shutdown()
    except Exception:
        logger.exception("Error shutting down Ray")
if __name__=="__main__":
    main()
