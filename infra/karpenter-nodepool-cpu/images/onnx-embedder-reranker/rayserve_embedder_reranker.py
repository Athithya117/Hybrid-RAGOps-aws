import os
os.environ.setdefault("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", os.environ.get("EMBEDDER_OMP_NUM_THREADS", "1")))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
import logging
import asyncio
import random
from typing import List, Dict, Any

import numpy as np
np.random.seed(0)
random.seed(0)

import coloredlogs
from transformers import AutoTokenizer
import onnxruntime as ort
from ray import serve
from fastapi import FastAPI, Body, HTTPException

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
coloredlogs.install(level=LOG_LEVEL)
logger = logging.getLogger("rag8s")

def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _provider_session_options(num_threads: int) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(num_threads))
    so.inter_op_num_threads = 1
    try:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    except Exception:
        pass
    try:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    so.log_severity_level = 2
    return so

def _select_output(outputs: List[Any], names: List[str], prefer: List[str]) -> Any:
    name2idx = {n: i for i, n in enumerate(names)}
    for p in prefer:
        if p in name2idx:
            return outputs[name2idx[p]]
    return outputs[0]

def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-6, None)
    return (summed / counts).astype(np.float32)

@serve.deployment(ray_actor_options={"num_cpus": _int_env("EMBEDDER_NUM_CPUS", 1)})
class EmbedderWorker:
    def __init__(self):
        model_name = os.getenv("MODEL_EMBEDDER_NAME", "")
        hf_home = os.getenv("HF_HOME", "/workspace/models/hf")
        onnx_path = os.getenv("EMBEDDER_ONNX_PATH")
        num_threads = _int_env("EMBEDDER_OMP_NUM_THREADS", int(os.environ.get("OMP_NUM_THREADS", "1")))
        if not onnx_path or not os.path.isfile(onnx_path):
            raise RuntimeError(f"Missing or invalid EMBEDDER_ONNX_PATH: {onnx_path}")
        if model_name:
            tokenizer_source = os.path.join(hf_home, model_name) if os.path.isdir(os.path.join(hf_home, model_name)) else model_name
        else:
            tokenizer_source = hf_home if os.path.isdir(hf_home) else None
        if not tokenizer_source:
            raise RuntimeError("Tokenizer source not found. Set MODEL_EMBEDDER_NAME or provide valid HF_HOME.")
        logger.info("[Embedder] loading tokenizer from %s", tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        logger.info("[Embedder] loading ONNX %s", onnx_path)
        so = _provider_session_options(num_threads)
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.out_names = [o.name for o in self.sess.get_outputs()]
        warm = int(os.getenv("WARMUP_SAMPLES", "0"))
        if warm > 0:
            try:
                sample = ["warmup input"] * max(1, warm)
                _ = self._embed_sync(sample)
                logger.info("embedder warmup complete")
            except Exception as e:
                logger.warning("embedder warmup failed: %s", e)

    def _embed_sync(self, batch_texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        outs = self.sess.run(None, inputs)
        if "sentence_embedding" in self.out_names:
            emb = _select_output(outs, self.out_names, ["sentence_embedding"])
            return np.asarray(emb, dtype=np.float32).tolist()
        if "last_hidden_state" in self.out_names:
            last = _select_output(outs, self.out_names, ["last_hidden_state"])
            pooled = _mean_pool(last, inputs["attention_mask"])
            return pooled.tolist()
        first = outs[0]
        if getattr(first, "ndim", 0) == 3:
            pooled = _mean_pool(first, inputs["attention_mask"])
            return pooled.tolist()
        return np.asarray(first, dtype=np.float32).tolist()

    @serve.batch(max_batch_size=_int_env("EMBEDDER_BATCH_MAX_SIZE", 8), batch_wait_timeout_s=_float_env("EMBEDDER_BATCH_WAIT_TIMEOUT_S", 0.05))
    async def embed(self, batch_texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._embed_sync, batch_texts)
        return result

@serve.deployment(ray_actor_options={"num_cpus": _int_env("RERANKER_NUM_CPUS", 1)})
class RerankerWorker:
    def __init__(self):
        model_name = os.getenv("MODEL_RERANKER_NAME", "")
        hf_home = os.getenv("HF_HOME", "/workspace/models/hf")
        onnx_path = os.getenv("RERANKER_ONNX_PATH")
        num_threads = _int_env("RERANKER_OMP_NUM_THREADS", int(os.environ.get("OMP_NUM_THREADS", "1")))
        if not onnx_path or not os.path.isfile(onnx_path):
            raise RuntimeError(f"Missing or invalid RERANKER_ONNX_PATH: {onnx_path}")
        if model_name:
            tokenizer_source = os.path.join(hf_home, model_name) if os.path.isdir(os.path.join(hf_home, model_name)) else model_name
        else:
            tokenizer_source = hf_home if os.path.isdir(hf_home) else None
        if not tokenizer_source:
            raise RuntimeError("Tokenizer source not found. Set MODEL_RERANKER_NAME or provide valid HF_HOME.")
        logger.info("[Reranker] loading tokenizer from %s", tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        logger.info("[Reranker] loading ONNX %s", onnx_path)
        so = _provider_session_options(num_threads)
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.out_names = [o.name for o in self.sess.get_outputs()]
        warm = int(os.getenv("WARMUP_SAMPLES", "0"))
        if warm > 0:
            try:
                _ = self._rerank_sync("warmup", ["doc"] * max(1, warm))
                logger.info("reranker warmup complete")
            except Exception as e:
                logger.warning("reranker warmup failed: %s", e)

    def _rerank_sync(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []
        enc = self.tokenizer([query] * len(docs), docs, padding=True, truncation=True, max_length=512, return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        outs = self.sess.run(None, inputs)
        logits = _select_output(outs, self.out_names, ["logits"])
        arr = np.asarray(logits, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.tolist()

    @serve.batch(max_batch_size=_int_env("RERANKER_BATCH_MAX_SIZE", 4), batch_wait_timeout_s=_float_env("RERANKER_BATCH_WAIT_TIMEOUT_S", 0.10))
    async def rerank(self, payloads: List[Dict[str, Any]]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(None, self._rerank_sync, str(p.get("query", "")), list(map(str, p.get("documents", [])))) for p in payloads]
        results = await asyncio.gather(*futures)
        return results

@serve.deployment
class Api:
    def __init__(self, embedder_ref, reranker_ref):
        self.app = FastAPI()
        self.embedder = embedder_ref
        self.reranker = reranker_ref

        @self.app.get("/healthz")
        async def healthz():
            return {"status": "ok"}

        @self.app.get("/ready")
        async def ready():
            return {"status": "ready"}

        @self.app.post("/v1/embed")
        async def embed(body: Dict[str, Any] = Body(...)):
            texts = body.get("texts") or body.get("documents") or []
            if not isinstance(texts, list):
                raise HTTPException(status_code=400, detail="texts must be a list of strings")
            try:
                handle = serve.get_deployment_handle("EmbedderWorker")
                result = await handle.embed.remote(texts)
                return {"embeddings": result}
            except Exception as e:
                logger.exception("embed endpoint error")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/rerank")
        async def rerank(body: Dict[str, Any] = Body(...)):
            query = body.get("query", "")
            docs = body.get("documents", [])
            if not isinstance(docs, list):
                raise HTTPException(status_code=400, detail="documents must be a list of strings")
            try:
                handle = serve.get_deployment_handle("RerankerWorker")
                result = await handle.rerank.remote({"query": query, "documents": docs})
                return {"scores": result}
            except Exception as e:
                logger.exception("rerank endpoint error")
                raise HTTPException(status_code=500, detail=str(e))

    def _configure_app(self):
        return self.app
