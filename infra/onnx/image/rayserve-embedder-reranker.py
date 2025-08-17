import os, logging, sys, signal
from typing import List, Optional
import numpy as np
import onnxruntime
from transformers import PreTrainedTokenizerFast
import grpc_pb2, grpc_pb2_grpc
from ray import serve
from prometheus_client import Counter, Histogram

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag8s.serve")

def get_env(name: str, default=None, required: bool = False, cast=None):
    val = os.getenv(name, default)
    if required and (val is None or (isinstance(val, str) and val.strip() == "")):
        logger.error("Missing required environment variable: %s", name)
        raise SystemExit(f"Missing required env var: {name}")
    if cast and val is not None:
        try:
            return cast(val)
        except Exception as e:
            logger.warning("Failed to cast env %s value %r with %s: %s", name, val, cast, e)
            return val
    return val

HF_HOME = get_env("HF_HOME", "/app/models/hf")
MODEL_DIR = get_env("MODEL_DIR", "/app/models/onnx")
MODEL_EMBEDDER_NAME = get_env("MODEL_EMBEDDER_NAME", None, required=True)
MODEL_RERANKER_NAME = get_env("MODEL_RERANKER_NAME", None, required=True)
EMBEDDER_ONNX_PATH = get_env("EMBEDDER_ONNX_PATH", None)
RERANKER_ONNX_PATH = get_env("RERANKER_ONNX_PATH", None)
EMBEDDER_OMP_NUM_THREADS = int(get_env("EMBEDDER_OMP_NUM_THREADS", "1"))
RERANKER_OMP_NUM_THREADS = int(get_env("RERANKER_OMP_NUM_THREADS", "1"))
EMBEDDER_BATCH_MAX_SIZE = int(get_env("EMBEDDER_BATCH_MAX_SIZE", "8"))
RERANKER_BATCH_MAX_SIZE = int(get_env("RERANKER_BATCH_MAX_SIZE", "4"))
EMBEDDER_BATCH_WAIT_TIMEOUT_S = float(get_env("EMBEDDER_BATCH_WAIT_TIMEOUT_S", "0.05"))
RERANKER_BATCH_WAIT_TIMEOUT_S = float(get_env("RERANKER_BATCH_WAIT_TIMEOUT_S", "0.1"))

INFER_COUNTER = Counter("rag8s_inference_requests_total", "Total inference requests", ["model"])
INFER_LATENCY = Histogram("rag8s_inference_latency_seconds", "Latency per inference", ["model"])

def _basename_from_repo_id(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]

def _candidate_tokenizer_dirs(repo_id: str):
    name = _basename_from_repo_id(repo_id)
    candidates = [os.path.join(MODEL_DIR, name), os.path.join(HF_HOME, name)]
    hub_prefix = os.path.join(HF_HOME, "hub")
    try:
        if os.path.isdir(hub_prefix):
            for candidate in os.listdir(hub_prefix):
                if name in candidate or repo_id.replace("/", "--") in candidate:
                    snap_dir = os.path.join(hub_prefix, candidate, "snapshots")
                    if os.path.isdir(snap_dir):
                        for s in os.listdir(snap_dir):
                            candidates.append(os.path.join(snap_dir, s))
                    candidates.append(os.path.join(hub_prefix, candidate))
    except Exception:
        pass
    return [p for p in candidates if p]

def _candidate_onnx_paths(repo_id: str):
    name = _basename_from_repo_id(repo_id)
    candidates = [
        os.path.join(MODEL_DIR, name, "onnx", "model_int8.onnx"),
        os.path.join(HF_HOME, name, "onnx", "model_int8.onnx"),
        os.path.join(MODEL_DIR, name, "model_int8.onnx")
    ]
    hub_prefix = os.path.join(HF_HOME, "hub")
    try:
        if os.path.isdir(hub_prefix):
            for candidate in os.listdir(hub_prefix):
                if name in candidate or repo_id.replace("/", "--") in candidate:
                    snap_dir = os.path.join(hub_prefix, candidate, "snapshots")
                    if os.path.isdir(snap_dir):
                        for s in os.listdir(snap_dir):
                            candidates.append(os.path.join(snap_dir, s, "onnx", "model_int8.onnx"))
                            candidates.append(os.path.join(snap_dir, s, "model_int8.onnx"))
                    candidates.append(os.path.join(hub_prefix, candidate, "onnx", "model_int8.onnx"))
    except Exception:
        pass
    return [p for p in candidates if p]

def resolve_tokenizer_dir(repo_id: str) -> Optional[str]:
    candidates = _candidate_tokenizer_dirs(repo_id)
    for p in candidates:
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "tokenizer.json")) or os.path.exists(os.path.join(p, "tokenizer_config.json")):
                logger.info("Using tokenizer dir %s for %s", p, repo_id)
                return p
            for root, _, files in os.walk(p):
                if "tokenizer.json" in files or "tokenizer_config.json" in files:
                    logger.info("Using tokenizer nested dir %s for %s", root, repo_id)
                    return root
    logger.warning("No local tokenizer dir found for %s (checked candidates: %s)", repo_id, candidates)
    return None

def resolve_onnx_path(repo_id: str) -> Optional[str]:
    candidates = _candidate_onnx_paths(repo_id)
    for p in candidates:
        if os.path.isfile(p):
            logger.info("Found ONNX model for %s at %s", repo_id, p)
            return p
    logger.warning("No local ONNX file found for %s (checked candidates: %s)", repo_id, candidates)
    return None

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

def final_onnx_path_for(repo_env_name: str, explicit_env_path: Optional[str], repo_id: str) -> str:
    if explicit_env_path:
        if os.path.isfile(explicit_env_path):
            logger.info("Using explicit ONNX path from env %s=%s", repo_env_name, explicit_env_path)
            return explicit_env_path
        else:
            logger.error("Explicit ONNX path %s set but file missing: %s", repo_env_name, explicit_env_path)
            raise FileNotFoundError(explicit_env_path)
    p = resolve_onnx_path(repo_id)
    if p:
        return p
    raise FileNotFoundError(f"ONNX model for {repo_id} not found under MODEL_DIR ({MODEL_DIR}) or HF_HOME ({HF_HOME}). Check that models were baked into the image and that MODEL_DIR/HF_HOME are correct.")

def final_tokenizer_dir_for(repo_id: str) -> Optional[str]:
    return resolve_tokenizer_dir(repo_id)

@serve.deployment(ray_actor_options={"num_cpus": float(get_env("EMBEDDER_NUM_CPUS", "1"))})
class EmbedderServicer(grpc_pb2_grpc.EmbedServiceServicer):
    def __init__(self):
        tok_dir = final_tokenizer_dir_for(MODEL_EMBEDDER_NAME)
        if tok_dir:
            logger.info("Loading embedder tokenizer from local dir %s (cache_dir=%s)", tok_dir, HF_HOME)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir, local_files_only=True, trust_remote_code=True)
        else:
            logger.info("Local tokenizer not found for %s. Attempting from_pretrained() using cache_dir=%s", MODEL_EMBEDDER_NAME, HF_HOME)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_EMBEDDER_NAME, cache_dir=HF_HOME, trust_remote_code=True, local_files_only=True)
        assert getattr(self.tokenizer, "is_fast", True), "Fast tokenizer not loaded — install 'tokenizers'"
        onnx_path = final_onnx_path_for("EMBEDDER_ONNX_PATH", EMBEDDER_ONNX_PATH, MODEL_EMBEDDER_NAME)
        logger.info("Loading embedder ONNX from %s", onnx_path)
        self.session = make_session(onnx_path, intra_op_threads=EMBEDDER_OMP_NUM_THREADS)

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
        tok_dir = final_tokenizer_dir_for(MODEL_RERANKER_NAME)
        if tok_dir:
            logger.info("Loading reranker tokenizer from local dir %s (cache_dir=%s)", tok_dir, HF_HOME)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir, local_files_only=True, trust_remote_code=True)
        else:
            logger.info("Local tokenizer not found for %s. Attempting from_pretrained() using cache_dir=%s", MODEL_RERANKER_NAME, HF_HOME)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_RERANKER_NAME, cache_dir=HF_HOME, trust_remote_code=True, local_files_only=True)
        assert getattr(self.tokenizer, "is_fast", True), "Fast tokenizer not loaded — install 'tokenizers'"
        onnx_path = final_onnx_path_for("RERANKER_ONNX_PATH", RERANKER_ONNX_PATH, MODEL_RERANKER_NAME)
        logger.info("Loading reranker ONNX from %s", onnx_path)
        self.session = make_session(onnx_path, intra_op_threads=RERANKER_OMP_NUM_THREADS)

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
            scores_arr = np.array(outputs[0])
            if scores_arr.ndim > 1 and scores_arr.shape[1] == 1:
                scores = scores_arr.squeeze(axis=1).tolist()
            else:
                scores = scores_arr.tolist()
            return grpc_pb2.RerankResponse(scores=scores)

def warmup_models():
    try:
        embedder_onnx = final_onnx_path_for("EMBEDDER_ONNX_PATH", EMBEDDER_ONNX_PATH, MODEL_EMBEDDER_NAME)
        reranker_onnx = final_onnx_path_for("RERANKER_ONNX_PATH", RERANKER_ONNX_PATH, MODEL_RERANKER_NAME)
        _ = make_session(embedder_onnx, intra_op_threads=EMBEDDER_OMP_NUM_THREADS)
        _ = make_session(reranker_onnx, intra_op_threads=RERANKER_OMP_NUM_THREADS)
        logger.info("Model warmup completed.")
    except Exception as e:
        logger.warning("Warmup failed: %s", e)

def create_health_check(app_ready_flag):
    def health():
        return app_ready_flag.is_set()
    return health
