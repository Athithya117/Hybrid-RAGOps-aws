import os
import logging
import gc
from typing import List, Optional
import numpy as np
import onnxruntime
from transformers import PreTrainedTokenizerFast
import grpc_pb2, grpc_pb2_grpc
from ray import serve

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from prometheus_client import Counter, Histogram, Gauge, start_http_server

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag8s.serve")

HF_HOME = os.getenv("HF_HOME", "/workspace/models/hf")
MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models/onnx")
MODEL_EMBEDDER_NAME = os.getenv("MODEL_EMBEDDER_NAME")
MODEL_RERANKER_NAME = os.getenv("MODEL_RERANKER_NAME")
EMBEDDER_ONNX_PATH = os.getenv("EMBEDDER_ONNX_PATH")
RERANKER_ONNX_PATH = os.getenv("RERANKER_ONNX_PATH")

EMBEDDER_OMP_NUM_THREADS = int(os.getenv("EMBEDDER_OMP_NUM_THREADS", "1"))
RERANKER_OMP_NUM_THREADS = int(os.getenv("RERANKER_OMP_NUM_THREADS", "1"))
EMBEDDER_BATCH_MAX_SIZE = int(os.getenv("EMBEDDER_BATCH_MAX_SIZE", "8"))
RERANKER_BATCH_MAX_SIZE = int(os.getenv("RERANKER_BATCH_MAX_SIZE", "4"))
EMBEDDER_BATCH_WAIT_TIMEOUT_S = float(os.getenv("EMBEDDER_BATCH_WAIT_TIMEOUT_S", "0.05"))
RERANKER_BATCH_WAIT_TIMEOUT_S = float(os.getenv("RERANKER_BATCH_WAIT_TIMEOUT_S", "0.1"))

PROM_DISABLED = os.getenv("PROMETHEUS_DISABLED", "1") == "1"

READY_GAUGE = None

INFER_COUNTER = None
INFER_ERRORS = None
INFER_LATENCY = None
BATCH_SIZE_HIST = None

tracer = None
meter = None

_EMBED_TOKENIZER = None
_EMBED_SESSION = None
_RERANK_TOKENIZER = None
_RERANK_SESSION = None
_initialized = False

def init_telemetry():
    global READY_GAUGE, INFER_COUNTER, INFER_ERRORS, INFER_LATENCY, BATCH_SIZE_HIST, tracer, meter
    try:
        resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "rag8s-onnx")})
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)

        metric_exporter = OTLPMetricExporter()
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter("rag8s.embedder", "0.1")

        INFER_COUNTER = meter.create_counter("rag8s.inference.requests.total", description="Total inference requests")
        INFER_ERRORS = meter.create_counter("rag8s.inference.errors.total", description="Inference errors")
        INFER_LATENCY = meter.create_histogram("rag8s.inference.latency.seconds", description="Latency per inference")
        BATCH_SIZE_HIST = meter.create_histogram("rag8s.batch.size", description="Batch size distribution")
        READY_GAUGE = None if PROM_DISABLED else Gauge("rag8s_initialized", "Model initialization state (1=ready,0=not ready)", ["model"])
    except Exception:
        logger.exception("Telemetry init failed")
        tracer = None
        meter = None

def _set_thread_envs(threads: int):
    try:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
    except Exception:
        logger.exception("Failed setting thread envs")

def make_session(path: str, intra_op_threads=1) -> onnxruntime.InferenceSession:
    if not os.path.isfile(path):
        logger.error("ONNX model not found: %s", path)
        raise FileNotFoundError(path)
    _set_thread_envs(intra_op_threads)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return sess

def ensure_initialized():
    global _EMBED_TOKENIZER, _EMBED_SESSION, _RERANK_TOKENIZER, _RERANK_SESSION, _initialized
    if _initialized:
        return
    try:
        embed_tok_dir = None
        if MODEL_EMBEDDER_NAME:
            cand = [os.path.join(MODEL_DIR, os.path.basename(MODEL_EMBEDDER_NAME)), os.path.join(HF_HOME, os.path.basename(MODEL_EMBEDDER_NAME))]
            for p in cand:
                if os.path.isdir(p):
                    embed_tok_dir = p
                    break
        if not embed_tok_dir:
            logger.error("Embedder tokenizer dir missing for %s", MODEL_EMBEDDER_NAME)
            raise SystemExit(3)
        _EMBED_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(embed_tok_dir, local_files_only=True, trust_remote_code=True)
        embed_onnx = EMBEDDER_ONNX_PATH
        _EMBED_SESSION = make_session(embed_onnx, intra_op_threads=EMBEDDER_OMP_NUM_THREADS)
        rerank_tok_dir = None
        if MODEL_RERANKER_NAME:
            cand = [os.path.join(MODEL_DIR, os.path.basename(MODEL_RERANKER_NAME)), os.path.join(HF_HOME, os.path.basename(MODEL_RERANKER_NAME))]
            for p in cand:
                if os.path.isdir(p):
                    rerank_tok_dir = p
                    break
        if not rerank_tok_dir:
            logger.error("Reranker tokenizer dir missing for %s", MODEL_RERANKER_NAME)
            raise SystemExit(3)
        _RERANK_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(rerank_tok_dir, local_files_only=True, trust_remote_code=True)
        rerank_onnx = RERANKER_ONNX_PATH
        _RERANK_SESSION = make_session(rerank_onnx, intra_op_threads=RERANKER_OMP_NUM_THREADS)
        _initialized = True
        if READY_GAUGE:
            READY_GAUGE.labels(model="embedder").set(1)
            READY_GAUGE.labels(model="reranker").set(1)
    except Exception:
        if READY_GAUGE:
            try:
                READY_GAUGE.labels(model="embedder").set(0)
                READY_GAUGE.labels(model="reranker").set(0)
            except Exception:
                pass
        logger.exception("Initialization failed")
        raise

@serve.batch(max_batch_size=EMBEDDER_BATCH_MAX_SIZE, batch_wait_timeout_s=EMBEDDER_BATCH_WAIT_TIMEOUT_S)
async def _embed_batch(requests: List[grpc_pb2.EmbedRequest]) -> List[grpc_pb2.EmbedResponse]:
    ensure_initialized()
    model_tag = "embedder"
    try:
        if INFER_COUNTER is not None:
            INFER_COUNTER.add(len(requests), {"model": model_tag})
        if BATCH_SIZE_HIST is not None:
            BATCH_SIZE_HIST.record(len(requests), {"model": model_tag})
    except Exception:
        pass
    import time
    start = time.perf_counter()
    try:
        all_texts = [t for req in requests for t in req.texts]
        if not all_texts:
            duration = time.perf_counter() - start
            if INFER_LATENCY is not None:
                INFER_LATENCY.record(duration, {"model": model_tag})
            return [grpc_pb2.EmbedResponse(embeddings=[]) for _ in requests]
        enc = _EMBED_TOKENIZER(all_texts, padding=True, truncation=True, return_tensors="np")
        outputs = _EMBED_SESSION.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        embeddings_arr = np.array(outputs[0]).mean(axis=1)
        responses = []
        idx = 0
        for req in requests:
            n = len(req.texts)
            if n == 0:
                responses.append(grpc_pb2.EmbedResponse(embeddings=[]))
            else:
                slice_arr = embeddings_arr[idx:idx + n]
                flat = slice_arr.reshape(-1).astype(float).tolist()
                responses.append(grpc_pb2.EmbedResponse(embeddings=flat))
                idx += n
        duration = time.perf_counter() - start
        if INFER_LATENCY is not None:
            INFER_LATENCY.record(duration, {"model": model_tag})
        return responses
    except Exception:
        if INFER_ERRORS is not None:
            INFER_ERRORS.add(len(requests), {"model": model_tag})
        logger.exception("Embed batch failed")
        return [grpc_pb2.EmbedResponse(embeddings=[]) for _ in requests]

@serve.batch(max_batch_size=RERANKER_BATCH_MAX_SIZE, batch_wait_timeout_s=RERANKER_BATCH_WAIT_TIMEOUT_S)
async def _rerank_batch(requests: List[grpc_pb2.RerankRequest]) -> List[grpc_pb2.RerankResponse]:
    ensure_initialized()
    model_tag = "reranker"
    try:
        if INFER_COUNTER is not None:
            INFER_COUNTER.add(len(requests), {"model": model_tag})
        if BATCH_SIZE_HIST is not None:
            BATCH_SIZE_HIST.record(len(requests), {"model": model_tag})
    except Exception:
        pass
    import time
    start = time.perf_counter()
    try:
        all_pairs = [pair for req in requests for pair in req.pairs]
        if not all_pairs:
            duration = time.perf_counter() - start
            if INFER_LATENCY is not None:
                INFER_LATENCY.record(duration, {"model": model_tag})
            return [grpc_pb2.RerankResponse(scores=[]) for _ in requests]
        queries = [p.query for p in all_pairs]
        docs = [p.doc for p in all_pairs]
        enc = _RERANK_TOKENIZER(queries, docs, padding=True, truncation=True, return_tensors="np")
        outputs = _RERANK_SESSION.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        scores_arr = np.array(outputs[0])
        if scores_arr.ndim > 1 and scores_arr.shape[1] == 1:
            scores_arr = scores_arr.squeeze(axis=1)
        responses = []
        idx = 0
        for req in requests:
            n = len(req.pairs)
            if n == 0:
                responses.append(grpc_pb2.RerankResponse(scores=[]))
            else:
                slice_scores = scores_arr[idx:idx + n].astype(float).tolist()
                responses.append(grpc_pb2.RerankResponse(scores=slice_scores))
                idx += n
        duration = time.perf_counter() - start
        if INFER_LATENCY is not None:
            INFER_LATENCY.record(duration, {"model": model_tag})
        return responses
    except Exception:
        if INFER_ERRORS is not None:
            INFER_ERRORS.add(len(requests), {"model": model_tag})
        logger.exception("Rerank batch failed")
        return [grpc_pb2.RerankResponse(scores=[]) for _ in requests]

@serve.deployment(ray_actor_options={"num_cpus": float(os.getenv("EMBEDDER_NUM_CPUS", "1"))})
class EmbedderServicer(grpc_pb2_grpc.EmbedServiceServicer):
    def __init__(self):
        ensure_initialized()
    async def Embed(self, request: grpc_pb2.EmbedRequest) -> grpc_pb2.EmbedResponse:
        return await _embed_batch(request)

@serve.deployment(ray_actor_options={"num_cpus": float(os.getenv("RERANKER_NUM_CPUS", "1"))})
class RerankerServicer(grpc_pb2_grpc.RerankServiceServicer):
    def __init__(self):
        ensure_initialized()
    async def Rerank(self, request: grpc_pb2.RerankRequest) -> grpc_pb2.RerankResponse:
        return await _rerank_batch(request)

def warmup_models():
    try:
        ensure_initialized()
    except Exception as e:
        logger.exception("Warmup failed: %s", e)
