import logging, threading, time, signal, os
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Event
from ray import serve
from rayserve_embedder_reranker import EmbedderServicer, RerankerServicer

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag8s.entrypoint")
_app_ready = Event()

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            if _app_ready.is_set():
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            else:
                self.send_response(503); self.end_headers(); self.wfile.write(b"not ready")
        else:
            self.send_response(404); self.end_headers()

def _start_health_server(port=8000):
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    logger.info("Started health server on port %s", port)
    return server

def _deploy_if_needed():
    try:
        if not serve.is_running():
            serve.start(detached=True); logger.info("Started Ray Serve (detached).")
        EmbedderServicer.deploy(); RerankerServicer.deploy(); logger.info("Deployments applied (Embedder, Reranker).")
        _app_ready.set()
    except Exception as e:
        logger.exception("Failed to start/deploy: %s", e); _app_ready.clear()

app = {"name": "rag8s_onnx_app", "deployments": ["embedder", "reranker"]}
_http_server = _start_health_server(port=int(os.getenv("HTTP_PORT", "8000")))
_deploy_if_needed()

def _graceful_shutdown(signum, frame):
    logger.info("Received signal %s, shutting down Serve and health server", signum)
    try:
        if serve.is_running(): serve.shutdown()
    except Exception:
        logger.exception("Error shutting down Serve")
    try:
        _http_server.shutdown()
    except Exception:
        logger.exception("Error shutting down health server")
    _app_ready.clear()
    logger.info("Shutdown complete. Exiting.")
    os._exit(0)

signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)
