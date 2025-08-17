"""
Entrypoint module for KubeRay serveConfigV2.import_path.

This module is copied into the image and must be importable by the operator.
It tries to be idempotent: importing repeatedly (as KubeRay may do) will not
create duplicates; it also exposes `app` symbol for compatibility.

Behavior:
- If running under a normal Python process, attempt to start Serve and deploy
  the Embedder and Reranker servicers. This path is useful for local dev.
- When used by KubeRay, import triggers are sometimes used by the controller;
  our code is defensive and idempotent.
"""

import logging
import threading
import time
import signal
import os
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Event

from ray import serve

# Import the deployment classes (module must be in the image)
from rayserve_embedder_reranker import EmbedderServicer, RerankerServicer

logger = logging.getLogger("rag8s.entrypoint")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Control flags for health probe
_app_ready = Event()

# Simple HTTP health server (runs in a thread)
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            if _app_ready.is_set():
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"not ready")
        else:
            self.send_response(404)
            self.end_headers()

def _start_health_server(port=8000):
    # bind to localhost only
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Started health server on port %s", port)
    return server

def _deploy_if_needed():
    """
    Start Serve if not running and deploy services idempotently.
    This function is designed to be safe to call multiple times.
    """
    try:
        if not serve.is_running():
            # Start Serve in detached mode for this process
            serve.start(detached=True)
            logger.info("Started Ray Serve (detached).")
        # Deploy the classes (deploy is idempotent)
        EmbedderServicer.deploy()
        RerankerServicer.deploy()
        logger.info("Deployments applied (Embedder, Reranker).")
        _app_ready.set()
    except Exception as e:
        logger.exception("Failed to start/deploy: %s", e)
        _app_ready.clear()
        # Don't re-raise so import does not crash operator; operator can reapply.

# Expose an object that the operator can import; KubeRay expects import_path to resolve
# to something usable. `app` could be an object or dict; we provide informational dict.
app = {"name": "rag8s_onnx_app", "deployments": ["embedder", "reranker"]}

# Start health server (always start — allows probes to talk to container)
_http_server = _start_health_server(port=int(os.getenv("HTTP_PORT", "8000")))

# Attempt to deploy on import
_deploy_if_needed()

# Graceful shutdown handler for SIGTERM
def _graceful_shutdown(signum, frame):
    logger.info("Received signal %s, shutting down Serve and health server", signum)
    try:
        if serve.is_running():
            serve.shutdown()
    except Exception:
        logger.exception("Error shutting down Serve")
    try:
        _http_server.shutdown()
    except Exception:
        logger.exception("Error shutting down health server")
    # set not ready
    _app_ready.clear()
    logger.info("Shutdown complete. Exiting.")
    # allow process to exit
    os._exit(0)

signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)
