import logging,threading,time,signal,os
from http.server import HTTPServer,BaseHTTPRequestHandler
from threading import Event
import ray
from ray import serve
from rayserve_embedder_reranker import EmbedderServicer,RerankerServicer,warmup_models
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger("rag8s.entrypoint")
_app_ready=Event()
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/healthz":
            if _app_ready.is_set():
                self.send_response(200);self.end_headers();self.wfile.write(b"ok")
            else:
                self.send_response(503);self.end_headers();self.wfile.write(b"not ready")
        else:
            self.send_response(404);self.end_headers()
def _start_health_server(port=8000):
    server=HTTPServer(("0.0.0.0",port),_HealthHandler)
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    logger.info("Started health server on port %s",port)
    return server
def _deploy_if_needed():
    attempts=0
    max_attempts=3
    backoff=2
    last_exc=None
    while attempts<max_attempts:
        try:
            if not ray.is_initialized():
                logger.info("ray not initialized; calling ray.init()")
                ray.init()
                logger.info("ray.init() succeeded")
            try:
                serve.start(detached=True,http_options={"location":"NoServer"})
                logger.info("Called serve.start(detached=True, http_options={'location':'NoServer'})")
            except Exception as e:
                logger.warning("serve.start raised (continuing): %s",e)
            logger.info("Binding deployments via .bind() and calling serve.run(..., route_prefix=None, blocking=False)")
            embed_app=EmbedderServicer.bind()
            rerank_app=RerankerServicer.bind()
            try:
                serve.run(embed_app,name="rag8s_embedder_app",route_prefix=None,blocking=False)
                serve.run(rerank_app,name="rag8s_reranker_app",route_prefix=None,blocking=False)
                logger.info("serve.run called for embedder and reranker (route_prefix=None)")
            except Exception as e:
                logger.exception("serve.run failed: %s",e)
                raise
            try:
                warmup_models()
            except Exception as e:
                logger.exception("warmup_models exception: %s",e)
            _app_ready.set()
            return
        except Exception as e:
            last_exc=e
            attempts+=1
            logger.exception("Deploy attempt %s failed: %s",attempts,e)
            time.sleep(backoff*attempts)
    logger.error("All deploy attempts failed, raising")
    raise last_exc
_http_server=_start_health_server(port=int(os.getenv("HTTP_PORT","8000")))
try:
    _deploy_if_needed()
except Exception:
    logger.error("Initial deploy failed, exiting");os._exit(3)
def _graceful_shutdown(signum,frame):
    logger.info("Received signal %s, shutting down Serve and health server",signum)
    try:
        if hasattr(serve,"shutdown"):
            try:
                serve.shutdown()
            except Exception:
                logger.exception("Error shutting down Serve")
    except Exception:
        logger.exception("Error checking serve.shutdown")
    try:
        _http_server.shutdown()
    except Exception:
        logger.exception("Error shutting down health server")
    _app_ready.clear()
    logger.info("Shutdown complete. Exiting.")
    os._exit(0)
signal.signal(signal.SIGTERM,_graceful_shutdown)
signal.signal(signal.SIGINT,_graceful_shutdown)
while True:
    try:
        time.sleep(60)
    except Exception:
        break
