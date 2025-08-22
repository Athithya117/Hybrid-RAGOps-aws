import os
import time
import logging
import signal
from concurrent import futures
import grpc
import ray
from ray import serve
from fastapi import FastAPI
from starlette.responses import JSONResponse
from rayserve_embedder_reranker import EmbedderServicer, RerankerServicer, warmup_models, start_grpc_server
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
ready_event=None
stop_event=None
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
    try:
        if not ray.is_initialized():
            _init_ray(RAY_ADDRESS)
        serve.start(detached=True, http_options={"host":"0.0.0.0","port":HTTP_PORT})
        embed_dep=EmbedderServicer.bind()
        rerank_dep=RerankerServicer.bind()
        serve.run(embed_dep, name="rag8s_embedder_app", route_prefix=EMBED_ROUTE, blocking=False)
        serve.run(rerank_dep, name="rag8s_reranker_app", route_prefix=RERANK_ROUTE, blocking=False)
        Ingress=serve.deployment()(serve.ingress(FastAPI()))
        serve.run(Ingress.bind(), name="rag8s_http_gateway", route_prefix="/", blocking=False)
        try:
            warmup_models()
        except Exception:
            logger.exception("warmup_models failed (continuing)")
        return
    except Exception:
        logger.exception("Deploy failed")
        raise
def shutdown(signum, frame):
    try:
        serve.shutdown()
    except Exception:
        logger.exception("Error shutting down Serve")
    try:
        ray.shutdown()
    except Exception:
        logger.exception("Error shutting down Ray")
    os._exit(0)
signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
def main():
    try:
        deploy_services()
    except Exception:
        os._exit(3)
    grpc_server=None
    try:
        grpc_server=start_grpc_server(GRPC_PORT)
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
