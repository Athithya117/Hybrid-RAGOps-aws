import os
import sys
import logging
from typing import List, Dict

import ray
from ray import serve
import onnxruntime
from transformers import PreTrainedTokenizerFast

import grpc_pb2
import grpc_pb2_grpc

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rayserve_grpc")

# ---------------- Env Loader ----------------
def get_env(name: str, default=None, required: bool = False):
    val = os.getenv(name, default)
    if required and not val:
        logger.error(f"Missing required env var: {name}")
        sys.exit(1)
    return val

# ---------------- Configuration ----------------
HF_HOME                  = get_env("HF_HOME", "/opt/models/hf")
MODEL_EMBEDDER_NAME      = get_env("MODEL_EMBEDDER_NAME", required=True)
MODEL_RERANKER_NAME      = get_env("MODEL_RERANKER_NAME", required=True)
EMBEDDER_ONNX_PATH       = get_env(
    "EMBEDDER_ONNX_PATH",
    f"{HF_HOME}/onnx/{MODEL_EMBEDDER_NAME.split('/')[-1]}/model.onnx"
)
RERANKER_ONNX_PATH       = get_env(
    "RERANKER_ONNX_PATH",
    f"{HF_HOME}/onnx/{MODEL_RERANKER_NAME.split('/')[-1]}/model.onnx"
)
OMP_NUM_THREADS          = int(get_env("OMP_NUM_THREADS", "1"))
BATCH_MAX_SIZE           = int(get_env("BATCH_MAX_SIZE", "16"))
BATCH_WAIT_TIMEOUT_S     = float(get_env("BATCH_WAIT_TIMEOUT_S", "0.05"))

EMBEDDER_NUM_CPUS        = float(get_env("EMBEDDER_NUM_CPUS", "1"))
RERANKER_NUM_CPUS        = float(get_env("RERANKER_NUM_CPUS", "1"))
AUTOSCALE_EMBED          = {
    "min_replicas": int(get_env("RAYSERVE_EMBEDDER_MIN_REPLICAS", "1")),
    "max_replicas": int(get_env("RAYSERVE_EMBEDDER_MAX_REPLICAS", "4")),
    "target_num_ongoing_requests_per_replica":
        float(get_env("RAYSERVE_EMBEDDER_TARGET_ONGOING_REQUESTS_PER_REPLICA", "10")),
}
AUTOSCALE_RERANK         = {
    "min_replicas": int(get_env("RAYSERVE_RERANKER_MIN_REPLICAS", "1")),
    "max_replicas": int(get_env("RAYSERVE_RERANKER_MAX_REPLICAS", "2")),
    "target_num_ongoing_requests_per_replica":
        float(get_env("RAYSERVE_RERANKER_TARGET_ONGOING_REQUESTS_PER_REPLICA", "5")),
}

# ---------------- Utility ----------------
def check_model(path: str):
    if not os.path.isfile(path):
        logger.error(f"ONNX model not found: {path}")
        sys.exit(1)

def make_session(path: str) -> onnxruntime.InferenceSession:
    check_model(path)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = OMP_NUM_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

# ---------------- Deployments ----------------
@serve.deployment(
    ray_actor_options={"num_cpus": EMBEDDER_NUM_CPUS},
    autoscaling_config=AUTOSCALE_EMBED,
)
class EmbedderServicer(grpc_pb2_grpc.EmbedServiceServicer):
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_EMBEDDER_NAME, cache_dir=HF_HOME, trust_remote_code=True
        )
        assert self.tokenizer.is_fast, "Fast tokenizer not loaded—install 'tokenizers'"
        self.session = make_session(EMBEDDER_ONNX_PATH)

    @serve.batch(max_batch_size=BATCH_MAX_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def Embed(self, request: grpc_pb2.EmbedRequest) -> grpc_pb2.EmbedResponse:
        texts = list(request.texts)
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        outputs = self.session.run(None, {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
        embeddings = outputs[0].mean(axis=1).tolist()
        return grpc_pb2.EmbedResponse(embeddings=embeddings)


@serve.deployment(
    ray_actor_options={"num_cpus": RERANKER_NUM_CPUS},
    autoscaling_config=AUTOSCALE_RERANK,
)
class RerankerServicer(grpc_pb2_grpc.RerankServiceServicer):
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_RERANKER_NAME, cache_dir=HF_HOME, trust_remote_code=True
        )
        assert self.tokenizer.is_fast, "Fast tokenizer not loaded—install 'tokenizers'"
        self.session = make_session(RERANKER_ONNX_PATH)

    @serve.batch(max_batch_size=BATCH_MAX_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def Rerank(self, request: grpc_pb2.RerankRequest) -> grpc_pb2.RerankResponse:
        queries = [pair.query for pair in request.pairs]
        docs    = [pair.doc   for pair in request.pairs]
        enc = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors="np")
        outputs = self.session.run(None, {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
        scores = outputs[0].squeeze(axis=1).tolist()
        return grpc_pb2.RerankResponse(scores=scores)
