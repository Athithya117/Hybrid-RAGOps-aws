import os
os.environ.setdefault("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))
import logging
import coloredlogs
import ray
from ray import serve
import signal
import sys
import time

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
coloredlogs.install(level=LOG_LEVEL)
log = logging.getLogger("rag8s")

def _maybe_init_otel():
    enable = os.getenv("ENABLE_OTEL_COLLECTOR", "false").lower() in ("1", "true", "yes")
    if not enable:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "rag8s-onnx")})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318"))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as e:
        log.warning("otel init failed: %s", e)

def _ray_init_with_fallback():
    addr = os.getenv("RAY_ADDRESS", "auto").strip()
    try:
        if addr in ("", "local", "auto"):
            log.info("ray.init(local)")
            ray.init(ignore_reinit_error=True)
        else:
            log.info("ray.init(address=%s)", addr)
            ray.init(address=addr, ignore_reinit_error=True)
    except Exception as e:
        log.warning("Ray connect failed (%s); falling back to local.", e)
        try:
            ray.shutdown()
        except Exception:
            pass
        ray.init(ignore_reinit_error=True)

def _serve_start(port: int):
    serve.start(http_options={"host": "0.0.0.0", "port": port})

def _graceful_shutdown(sig, frame):
    log.info("shutdown signal received")
    try:
        serve.shutdown()
    except Exception:
        pass
    try:
        ray.shutdown()
    except Exception:
        pass
    sys.exit(0)

if __name__ == "__main__":
    _maybe_init_otel()
    _ray_init_with_fallback()
    from rayserve_embedder_reranker import EmbedderWorker, RerankerWorker, Api
    http_port = int(os.getenv("HTTP_PORT", "8000"))
    _serve_start(http_port)
    embedder = EmbedderWorker.bind()
    reranker = RerankerWorker.bind()
    api = Api.bind(embedder, reranker)
    serve.run(api, route_prefix="/")
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    log.info("Ray Serve app started.")
    while True:
        time.sleep(3600)
