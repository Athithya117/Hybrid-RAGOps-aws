# infra/onnx/rayserve-embedder-reranker.py
import os
import sys
import logging
from typing import List

import onnxruntime
from transformers import PreTrainedTokenizerFast
import grpc_pb2
import grpc_pb2_grpc

from ray import serve

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag8s.serve")

# ---------------- Env helpers ----------------
def get_env(name: str, default=None, required: bool = False, cast=None):
    val = os.getenv(name, default)
    if required and not val:
        logger.error(f"Missing required env var: {name}")
        sys.exit(1)
    if cast and val is not None:
        return cast(val)
    return val

# ---------------- Config ----------------
# Set HF_HOME to where Dockerfile places downloaded model files by default
HF_HOME = get_env("HF_HOME", "/app/models/hf")
MODEL_EMBEDDER_NAME = get_env("MODEL_EMBEDDER_NAME", required=True)
MODEL_RERANKER_NAME = get_env("MODEL_RERANKER_NAME", required=True)

# Allow explicit ONNX paths; otherwise compute plausible defaults that match the image layout
EMBEDDER_ONNX_PATH = get_env(
    "EMBEDDER_ONNX_PATH",
    f"{HF_HOME}/{MODEL_EMBEDDER_NAME.split('/')[-1]}/onnx/model_int8.onnx"
)
RERANKER_ONNX_PATH = get_env(
    "RERANKER_ONNX_PATH",
    f"{HF_HOME}/{MODEL_RERANKER_NAME.split('/')[-1]}/onnx/model_int8.onnx"
)

OMP_NUM_THREADS = get_env("OMP_NUM_THREADS", "1", cast=int)
BATCH_MAX_SIZE = get_env("BATCH_MAX_SIZE", "16", cast=int)
BATCH_WAIT_TIMEOUT_S = get_env("BATCH_WAIT_TIMEOUT_S", "0.05", cast=float)

EMBEDDER_NUM_CPUS = get_env("EMBEDDER_NUM_CPUS", "1", cast=float)
RERANKER_NUM_CPUS = get_env("RERANKER_NUM_CPUS", "1", cast=float)

AUTOSCALE_EMBED = {
    "min_replicas": get_env("EMBEDDER_MIN_REPLICAS", "1", cast=int),
    "max_replicas": get_env("EMBEDDER_MAX_REPLICAS", "4", cast=int),
    "target_num_ongoing_requests_per_replica": get_env("EMBEDDER_TARGET_REQUESTS_PER_REPLICA", "10", cast=float),
}
AUTOSCALE_RERANK = {
    "min_replicas": get_env("RERANKER_MIN_REPLICAS", "1", cast=int),
    "max_replicas": get_env("RERANKER_MAX_REPLICAS", "2", cast=int),
    "target_num_ongoing_requests_per_replica": get_env("RERANKER_TARGET_REQUESTS_PER_REPLICA", "5", cast=float),
}

# ---------------- Utilities ----------------
def check_model(path: str):
    if not os.path.isfile(path):
        logger.error("ONNX model not found: %s", path)
        raise FileNotFoundError(path)

def make_session(path: str) -> onnxruntime.InferenceSession:
    check_model(path)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = OMP_NUM_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

# ---------------- Serve deployments (definitions only) ----------------
@serve.deployment(
    ray_actor_options={"num_cpus": EMBEDDER_NUM_CPUS},
    autoscaling_config=AUTOSCALE_EMBED,
)
class EmbedderServicer(grpc_pb2_grpc.EmbedServiceServicer):
    def __init__(self):
        logger.info("Loading embedder tokenizer from %s (cache_dir=%s)", MODEL_EMBEDDER_NAME, HF_HOME)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_EMBEDDER_NAME, cache_dir=HF_HOME, trust_remote_code=True
        )
        assert self.tokenizer.is_fast, "Fast tokenizer not loaded — install 'tokenizers'"
        logger.info("Loading embedder ONNX from %s", EMBEDDER_ONNX_PATH)
        self.session = make_session(EMBEDDER_ONNX_PATH)

    @serve.batch(max_batch_size=BATCH_MAX_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def Embed(self, requests: List[grpc_pb2.EmbedRequest]) -> grpc_pb2.EmbedResponse:
        all_texts = [t for req in requests for t in req.texts]
        enc = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors="np")
        outputs = self.session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        embeddings = outputs[0].mean(axis=1).tolist()
        return grpc_pb2.EmbedResponse(embeddings=embeddings)

@serve.deployment(
    ray_actor_options={"num_cpus": RERANKER_NUM_CPUS},
    autoscaling_config=AUTOSCALE_RERANK,
)
class RerankerServicer(grpc_pb2_grpc.RerankServiceServicer):
    def __init__(self):
        logger.info("Loading reranker tokenizer from %s (cache_dir=%s)", MODEL_RERANKER_NAME, HF_HOME)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_RERANKER_NAME, cache_dir=HF_HOME, trust_remote_code=True
        )
        assert self.tokenizer.is_fast, "Fast tokenizer not loaded — install 'tokenizers'"
        logger.info("Loading reranker ONNX from %s", RERANKER_ONNX_PATH)
        self.session = make_session(RERANKER_ONNX_PATH)

    @serve.batch(max_batch_size=BATCH_MAX_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def Rerank(self, requests: List[grpc_pb2.RerankRequest]) -> grpc_pb2.RerankResponse:
        all_pairs = [pair for req in requests for pair in req.pairs]
        queries = [p.query for p in all_pairs]
        docs = [p.doc for p in all_pairs]
        enc = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors="np")
        outputs = self.session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        scores = outputs[0].squeeze(axis=1).tolist()
        return grpc_pb2.RerankResponse(scores=scores)

# ---------------- Optional helpers for local dev or head warmup ----------------
def warmup_models():
    """Load models into memory (useful if you run warmup on head after Serve app exists)."""
    try:
        _ = make_session(EMBEDDER_ONNX_PATH)
        _ = make_session(RERANKER_ONNX_PATH)
        logger.info("Model warmup completed.")
    except Exception as e:
        logger.warning("Warmup failed: %s", e)

def serve_main_local():
    """Local run for development only (runs serve.start and deploys)."""
    import ray
    from ray import serve as _serve
    ray.init()
    _serve.start(detached=True)
    EmbedderServicer.deploy()
    RerankerServicer.deploy()
    logger.info("Local Serve started and deployments deployed.")

# ---------------- Entrypoint guard ----------------
if __name__ == "__main__":
    # local development; do not rely on this when KubeRay is managing serveConfigV2
    serve_main_local()
