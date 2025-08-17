"""
Refactored serve module: defines EmbedderServicer and RerankerServicer
as importable classes. No top-level serve.start() or deploy() calls
so this module is import-safe in KubeRay. Use rayserve_entrypoint.py
to handle import-time deployment or returning an app object.
"""

import os
import logging
import sys
import signal
from typing import List

import numpy as np
import onnxruntime
from transformers import PreTrainedTokenizerFast
import grpc_pb2
import grpc_pb2_grpc

from ray import serve

# Metrics
from prometheus_client import Counter, Histogram

# Logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag8s.serve")

# ---- Environment defaults and helpers ----
def get_env(name, default=None, cast=None):
    val = os.getenv(name, default)
    if cast and val is not None:
        return cast(val)
    return val

HF_HOME = get_env("HF_HOME", "/app/models/hf")
MODEL_EMBEDDER_NAME = get_env("MODEL_EMBEDDER_NAME", required=True)
MODEL_RERANKER_NAME = get_env("MODEL_RERANKER_NAME", required=True)

EMBEDDER_ONNX_PATH = get_env("EMBEDDER_ONNX_PATH",
    f"{HF_HOME}/{MODEL_EMBEDDER_NAME.split('/')[-1]}/onnx/model_int8.onnx")
RERANKER_ONNX_PATH = get_env("RERANKER_ONNX_PATH",
    f"{HF_HOME}/{MODEL_RERANKER_NAME.split('/')[-1]}/onnx/model_int8.onnx")

EMBEDDER_OMP_NUM_THREADS = int(get_env("EMBEDDER_OMP_NUM_THREADS", "1"))
RERANKER_OMP_NUM_THREADS = int(get_env("RERANKER_OMP_NUM_THREADS", "1"))

EMBEDDER_BATCH_MAX_SIZE = int(get_env("EMBEDDER_BATCH_MAX_SIZE", "8"))
RERANKER_BATCH_MAX_SIZE = int(get_env("RERANKER_BATCH_MAX_SIZE", "4"))
EMBEDDER_BATCH_WAIT_TIMEOUT_S = float(get_env("EMBEDDER_BATCH_WAIT_TIMEOUT_S", "0.05"))
RERANKER_BATCH_WAIT_TIMEOUT_S = float(get_env("RERANKER_BATCH_WAIT_TIMEOUT_S", "0.1"))

# Metrics
INFER_COUNTER = Counter("rag8s_inference_requests_total", "Total inference requests", ["model"])
INFER_LATENCY = Histogram("rag8s_inference_latency_seconds", "Latency per inference", ["model"])

# ---- Utilities ----
def check_model(path: str):
    if not os.path.isfile(path):
        logger.error("ONNX model not found: %s", path)
        raise FileNotFoundError(path)

def make_session(path: str, intra_op_threads=1) -> onnxruntime.InferenceSession:
    check_model(path)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

# ---- Serve deployments (definitions only) ----
@serve.deployment(ray_actor_options={"num_cpus": float(get_env("EMBEDDER_NUM_CPUS", "1"))})
class EmbedderServicer(grpc_pb2_grpc.EmbedServiceServicer):
    def __init__(self):
        logger.info("Loading embedder tokenizer from %s (cache_dir=%s)", MODEL_EMBEDDER_NAME, HF_HOME)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_EMBEDDER_NAME, cache_dir=HF_HOME, trust_remote_code=True)
        assert self.tokenizer.is_fast, "Fast tokenizer not loaded — install 'tokenizers'"
        logger.info("Loading embedder ONNX from %s", EMBEDDER_ONNX_PATH)
        self.session = make_session(EMBEDDER_ONNX_PATH, intra_op_threads=EMBEDDER_OMP_NUM_THREADS)

    @serve.batch(max_batch_size=EMBEDDER_BATCH_MAX_SIZE, batch_wait_timeout_s=EMBEDDER_BATCH_WAIT_TIMEOUT_S)
    async def Embed(self, requests: List[grpc_pb2.EmbedRequest]) -> grpc_pb2.EmbedResponse:
        model_tag = "embedder"
        INFER_COUNTER.labels(model=model_tag).inc(len(requests))
        with INFER_LATENCY.labels(model=model_tag).time():
            all_texts = [t for req in requests for t in req.texts]
            enc = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors="np")
            outputs = self.session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
            embeddings = outputs[0].mean(axis=1).tolist()
            return grpc_pb2.EmbedResponse(embeddings=embeddings)

@serve.deployment(ray_actor_options={"num_cpus": float(get_env("RERANKER_NUM_CPUS", "1"))})
class RerankerServicer(grpc_pb2_grpc.RerankServiceServicer):
    def __init__(self):
        logger.info("Loading reranker tokenizer from %s (cache_dir=%s)", MODEL_RERANKER_NAME, HF_HOME)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_RERANKER_NAME, cache_dir=HF_HOME, trust_remote_code=True)
        assert self.tokenizer.is_fast, "Fast tokenizer not loaded — install 'tokenizers'"
        logger.info("Loading reranker ONNX from %s", RERANKER_ONNX_PATH)
        self.session = make_session(RERANKER_ONNX_PATH, intra_op_threads=RERANKER_OMP_NUM_THREADS)

    @serve.batch(max_batch_size=RERANKER_BATCH_MAX_SIZE, batch_wait_timeout_s=RERANKER_BATCH_WAIT_TIMEOUT_S)
    async def Rerank(self, requests: List[grpc_pb2.RerankRequest]) -> grpc_pb2.RerankResponse:
        model_tag = "reranker"
        INFER_COUNTER.labels(model=model_tag).inc(len(requests))
        with INFER_LATENCY.labels(model=model_tag).time():
            all_pairs = [pair for req in requests for pair in req.pairs]
            queries = [p.query for p in all_pairs]
            docs = [p.doc for p in all_pairs]
            enc = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors="np")
            outputs = self.session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
            scores = outputs[0].squeeze(axis=1).tolist()
            return grpc_pb2.RerankResponse(scores=scores)

# ---- Helpers for warmup / health ----
def warmup_models():
    """Load ONNX sessions to warm up memory (useful for head warmup)."""
    try:
        _ = make_session(EMBEDDER_ONNX_PATH, intra_op_threads=EMBEDDER_OMP_NUM_THREADS)
        _ = make_session(RERANKER_ONNX_PATH, intra_op_threads=RERANKER_OMP_NUM_THREADS)
        logger.info("Model warmup completed.")
    except Exception as e:
        logger.warning("Warmup failed: %s", e)

def create_health_check(app_ready_flag):
    # The entrypoint will create a simple thread/HTTP server that queries this flag
    def health():
        return app_ready_flag.is_set()
    return health
