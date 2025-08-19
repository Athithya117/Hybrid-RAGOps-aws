import os,time,logging,signal,threading
from concurrent import futures
import grpc
import ray
from ray import serve
from fastapi import FastAPI,Request,Response
from starlette.responses import JSONResponse
import grpc_pb2,grpc_pb2_grpc
from rayserve_embedder_reranker import EmbedderServicer,RerankerServicer,warmup_models,start_prometheus_if_enabled
from prometheus_client import generate_latest,CONTENT_TYPE_LATEST
os.environ.setdefault("LOG_LEVEL","INFO")
os.environ.setdefault("HTTP_PORT","8000")
os.environ.setdefault("GRPC_PORT","9000")
os.environ.setdefault("RAY_ADDRESS","auto")
os.environ.setdefault("EMBED_ROUTE","/rag8s_embedder_app")
os.environ.setdefault("RERANK_ROUTE","/rag8s_reranker_app")
LOG_LEVEL=os.getenv("LOG_LEVEL","INFO")
logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger("rag8s.entrypoint")
HTTP_PORT=int(os.getenv("HTTP_PORT","8000"))
GRPC_PORT=int(os.getenv("GRPC_PORT","9000"))
RAY_ADDRESS=os.getenv("RAY_ADDRESS","auto")
EMBED_ROUTE=os.getenv("EMBED_ROUTE","/rag8s_embedder_app")
RERANK_ROUTE=os.getenv("RERANK_ROUTE","/rag8s_reranker_app")
ready_event=threading.Event()
stop_event=threading.Event()
app=FastAPI()
@app.get("/healthz")
async def healthz():
    return JSONResponse({"ready":ready_event.is_set()},status_code=(200 if ready_event.is_set() else 503))
@app.get("/metrics")
async def http_metrics():
    try:
        data=generate_latest()
        return Response(content=data,media_type=CONTENT_TYPE_LATEST)
    except Exception:
        logger.exception("HTTP metrics error")
        return JSONResponse({"error":"metrics error"},status_code=500)
@app.post("/embed")
async def http_embed(req:Request):
    try:
        body=await req.json()
        texts=body.get("texts",[])
        handle=serve.get_app_handle("rag8s_embedder_app")
        resp=handle.Embed.remote(grpc_pb2.EmbedRequest(texts=texts)).result()
        return JSONResponse({"embeddings":list(resp.embeddings)})
    except Exception as exc:
        logger.exception("HTTP embed error")
        return JSONResponse({"error":str(exc)},status_code=500)
@app.post("/rerank")
async def http_rerank(req:Request):
    try:
        body=await req.json()
        pairs=body.get("pairs",[])
        proto_pairs=[grpc_pb2.Pair(query=p.get("query",""),doc=p.get("doc","")) for p in pairs]
        request_proto=grpc_pb2.RerankRequest(pairs=proto_pairs)
        handle=serve.get_app_handle("rag8s_reranker_app")
        resp=handle.Rerank.remote(request_proto).result()
        return JSONResponse({"scores":list(resp.scores)})
    except Exception as exc:
        logger.exception("HTTP rerank error")
        return JSONResponse({"error":str(exc)},status_code=500)
class GrpcForwarderServicer(grpc_pb2_grpc.EmbedServiceServicer,grpc_pb2_grpc.RerankServiceServicer):
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
    def Embed(self,request,context):
        try:
            self._ensure_handles()
            return self._embed_handle.Embed.remote(request).result()
        except Exception:
            logger.exception("gRPC Embed forward error")
            context.set_details("internal error")
            context.set_code(grpc.StatusCode.INTERNAL)
            return grpc_pb2.EmbedResponse(embeddings=[])
    def Rerank(self,request,context):
        try:
            self._ensure_handles()
            return self._rerank_handle.Rerank.remote(request).result()
        except Exception:
            logger.exception("gRPC Rerank forward error")
            context.set_details("internal error")
            context.set_code(grpc.StatusCode.INTERNAL)
            return grpc_pb2.RerankResponse(scores=[])
def start_grpc_server(port:int):
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    forwarder=GrpcForwarderServicer()
    grpc_pb2_grpc.add_EmbedServiceServicer_to_server(forwarder,server)
    grpc_pb2_grpc.add_RerankServiceServicer_to_server(forwarder,server)
    addr=f"[::]:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("gRPC forwarder started on %s",addr)
    return server
def _init_ray(address:str):
    try:
        if address in ("","local","none"):
            logger.info("Starting local Ray")
            ray.init()
            return
        if address=="auto":
            try:
                logger.info("Connecting to Ray at address=auto")
                ray.init(address="auto")
                return
            except Exception:
                logger.warning("Ray address=auto not found; starting local Ray")
                ray.init()
                return
        logger.info("Connecting to Ray at address=%s",address)
        ray.init(address=address)
    except Exception:
        logger.exception("Ray init failed")
        raise
def deploy_services():
    attempts=0
    last_exc=None
    while attempts<3 and not stop_event.is_set():
        try:
            if not ray.is_initialized():
                _init_ray(RAY_ADDRESS)
                logger.info("Ray initialized")
            serve.start(detached=True,http_options={"host":"0.0.0.0","port":HTTP_PORT})
            logger.info("Ray Serve HTTP started on 0.0.0.0:%s",HTTP_PORT)
            embed_dep=EmbedderServicer.bind()
            rerank_dep=RerankerServicer.bind()
            serve.run(embed_dep,name="rag8s_embedder_app",route_prefix=EMBED_ROUTE,blocking=False)
            serve.run(rerank_dep,name="rag8s_reranker_app",route_prefix=RERANK_ROUTE,blocking=False)
            Ingress=serve.deployment()(serve.ingress(app))
            serve.run(Ingress.bind(),name="rag8s_http_gateway",route_prefix="/",blocking=False)
            try:
                warmup_models()
            except Exception:
                logger.exception("warmup_models failed (continuing)")
            ready_event.set()
            logger.info("Deployments and HTTP gateway registered and ready")
            return
        except Exception as exc:
            last_exc=exc
            attempts+=1
            logger.exception("Deploy attempt %s failed",attempts)
            time.sleep(2*attempts)
    logger.error("All deploy attempts failed")
    raise last_exc
def shutdown(signum,frame):
    logger.info("Signal %s received, starting shutdown",signum)
    stop_event.set()
    ready_event.clear()
    try:
        serve.shutdown()
    except Exception:
        logger.exception("Error shutting down Serve")
    try:
        ray.shutdown()
    except Exception:
        logger.exception("Error shutting down Ray")
    logger.info("Shutdown finished")
    os._exit(0)
signal.signal(signal.SIGTERM,shutdown)
signal.signal(signal.SIGINT,shutdown)
def main():
    try:
        start_prometheus_if_enabled()
    except Exception:
        logger.exception("Failed to start prometheus server")
        os._exit(3)
    try:
        deploy_services()
    except Exception:
        logger.exception("Initial deploy failed")
        os._exit(3)
    grpc_server=None
    try:
        grpc_server=start_grpc_server(GRPC_PORT)
    except Exception:
        logger.exception("Failed to start gRPC server")
    logger.info("Service entering main loop")
    try:
        while not stop_event.is_set():
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
