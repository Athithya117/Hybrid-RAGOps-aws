from __future__ import annotations
import os
import sys
import json
import logging
import signal
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

EMBEDDER_MODEL_DIR = os.environ.get("EMBEDDER_MODEL_DIR", "/opt/models/gte-modernbert-base")
EMBEDDER_MODEL_PATH = os.environ.get("EMBEDDER_MODEL_PATH", os.path.join(EMBEDDER_MODEL_DIR, "model_fp16.onnx"))
RERANKER_MODEL_DIR = os.environ.get("RERANKER_MODEL_DIR", "/opt/models/gte-reranker-modernbert-base")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", os.path.join(RERANKER_MODEL_DIR, "model_fp16.onnx"))

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8001"))
SERVER_DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH") or "512")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "128"))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", "0.02"))
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "1"))
ORT_INTER_THREADS = int(os.environ.get("ORT_INTER_THREADS", "1"))
FORCE_CPU = os.environ.get("FORCE_CPU", "") in ("1", "true", "True")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
CUDA_DEVICE_ENV = os.environ.get("CUDA_DEVICE", "0")

logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("host_onnx_models")

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))

def _cuda_visible_via_env() -> List[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not v:
        return []
    v = v.strip()
    if v == "":
        return []
    return [p.strip() for p in v.split(",") if p.strip() not in ("", "-1")]

def _nvidia_smi_device_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=3).decode("utf-8", errors="ignore")
        if not out.strip():
            return 0
        return len([l for l in out.splitlines() if l.strip()])
    except Exception:
        return 0

def _cuda_device_count_fallback() -> int:
    env_list = _cuda_visible_via_env()
    if env_list:
        return len(env_list)
    nvidia_count = _nvidia_smi_device_count()
    if nvidia_count > 0:
        return nvidia_count
    return 0

def _get_allocated_cuda_device_index() -> int:
    env_list = _cuda_visible_via_env()
    if env_list:
        try:
            return int(env_list[0])
        except Exception:
            pass
    try:
        return int(CUDA_DEVICE_ENV)
    except Exception:
        return 0

def _read_json_if_exists(p: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _ensure_pad_token(tokenizer: Any) -> Any:
    try:
        if getattr(tokenizer, "pad_token", None) is not None:
            return tokenizer
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("pad_token set to eos_token for tokenizer")
            return tokenizer
        if getattr(tokenizer, "cls_token", None) is not None:
            tokenizer.pad_token = tokenizer.cls_token
            logger.info("pad_token set to cls_token for tokenizer")
            return tokenizer
        try:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Added special token '[PAD]' and set as pad_token")
        except Exception:
            try:
                tokenizer.pad_token = "[PAD]"
                logger.info("Set pad_token to literal '[PAD]'")
            except Exception as e:
                logger.warning("Failed to add pad token: %s", e)
    except Exception as e:
        logger.warning("ensure_pad_token error: %s", e)
    return tokenizer

def load_tokenizer_from_dir(model_dir: str) -> Optional[Any]:
    tok_json = os.path.join(model_dir, "tokenizer.json")
    sp_model = os.path.join(model_dir, "tokenizer.model")
    special_tokens_map = os.path.join(model_dir, "special_tokens_map.json")
    tokenizer_config = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_json):
        try:
            logger.info("Loading fast tokenizer from tokenizer.json at %s", tok_json)
            tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
            stm = _read_json_if_exists(special_tokens_map)
            if isinstance(stm, dict) and stm:
                try:
                    tok.add_special_tokens(stm)
                except Exception:
                    logger.debug("special_tokens_map apply ignored or failed")
            cfg = _read_json_if_exists(tokenizer_config)
            if isinstance(cfg, dict):
                mml = cfg.get("model_max_length")
                if mml:
                    try:
                        tok.model_max_length = int(mml)
                    except Exception:
                        pass
            tok = _ensure_pad_token(tok)
            return tok
        except Exception as e:
            logger.warning("PreTrainedTokenizerFast(tokenizer.json) failed: %s", e)
    if os.path.exists(sp_model):
        try:
            logger.info("Loading SentencePiece tokenizer from %s (use_fast=False)", model_dir)
            tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
            tok = _ensure_pad_token(tok)
            return tok
        except Exception as e:
            logger.warning("SentencePiece AutoTokenizer failed: %s", e)
    try:
        logger.info("Attempting AutoTokenizer.from_pretrained(use_fast=True) for %s", model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        tok = _ensure_pad_token(tok)
        return tok
    except Exception as e:
        logger.warning("AutoTokenizer(use_fast=True) fallback failed: %s", e)
    logger.warning("Could not load tokenizer from %s; continuing without tokenizer (use ID endpoints).", model_dir)
    return None

def create_onnx_session(model_path: str, prefer_cuda: bool = True) -> Tuple[ort.InferenceSession, str]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_INTRA_THREADS
    so.inter_op_num_threads = ORT_INTER_THREADS
    providers = ort.get_available_providers()
    visible_cuda = _cuda_device_count_fallback()
    want_cuda = prefer_cuda and ("CUDAExecutionProvider" in providers) and (visible_cuda > 0) and (not FORCE_CPU)
    provider_used = "CPUExecutionProvider"
    if want_cuda:
        device_idx = _get_allocated_cuda_device_index()
        if device_idx < visible_cuda:
            try:
                sess = ort.InferenceSession(model_path, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], provider_options=[{"device_id": int(device_idx)}, {}])
                provider_used = "CUDAExecutionProvider"
                logger.info("Created ONNX CUDAExecutionProvider with device_id=%s", device_idx)
                return sess, provider_used
            except Exception as e:
                logger.warning("Failed to create CUDAExecutionProvider with device_id=%s: %s", device_idx, e)
        try:
            sess = ort.InferenceSession(model_path, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            provider_used = "CUDAExecutionProvider"
            logger.info("Created ONNX CUDAExecutionProvider without explicit device_id")
            return sess, provider_used
        except Exception as e:
            logger.warning("Failed to create CUDAExecutionProvider (no device_id): %s", e)
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    provider_used = "CPUExecutionProvider"
    logger.info("Created ONNX CPUExecutionProvider")
    return sess, provider_used

def mean_pooling(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[..., :, None]
    masked = hidden_states * mask
    sums = masked.sum(axis=1)
    counts = mask.sum(axis=1)
    counts = np.maximum(counts, 1e-9)
    return sums / counts

def pick_embedding_from_outputs(outputs: List[np.ndarray], sess: ort.InferenceSession, tokenized: Dict[str, np.ndarray]) -> np.ndarray:
    output_names = [o.name for o in sess.get_outputs()]
    chosen = None
    for n, out in zip(output_names, outputs):
        nl = n.lower()
        if "pooled" in nl or "sentence" in nl or "cls" in nl:
            chosen = out
            break
    if chosen is None:
        for n, out in zip(output_names, outputs):
            if getattr(out, "ndim", None) == 2:
                chosen = out
                break
    if chosen is None:
        chosen = outputs[0]
    if getattr(chosen, "ndim", None) == 3:
        att = tokenized.get("attention_mask")
        if att is None:
            embeddings = chosen.mean(axis=1)
        else:
            embeddings = mean_pooling(chosen, att)
    elif getattr(chosen, "ndim", None) == 2:
        embeddings = chosen
    else:
        embeddings = chosen.reshape((chosen.shape[0], -1))
    embeddings = np.asarray(embeddings).astype(float)
    return embeddings

def map_inputs(sess: ort.InferenceSession, tokenized: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    sess_input_names = [i.name for i in sess.get_inputs()]
    lowered = {k.lower(): v for k, v in tokenized.items()}
    feed: Dict[str, np.ndarray] = {}
    for sin in sess_input_names:
        sl = sin.lower()
        if "input" in sl and "id" in sl and "input_ids" in lowered:
            feed[sin] = lowered["input_ids"]
            continue
        if "attention" in sl and "mask" in sl and "attention_mask" in lowered:
            feed[sin] = lowered["attention_mask"]
            continue
        if ("token_type" in sl or "segment" in sl) and "token_type_ids" in lowered:
            feed[sin] = lowered["token_type_ids"]
            continue
        for tk, arr in lowered.items():
            if tk in sl or sl in tk or tk.replace("_", "") in sl.replace("_", ""):
                feed[sin] = arr
                break
    if not feed:
        raise RuntimeError("Could not map tokenizer outputs to ONNX inputs")
    return feed

class AsyncBatcher:
    def __init__(self, sess: ort.InferenceSession, tokenizer: Any, abs_max: int, server_max: int, max_batch: int = MAX_BATCH, batch_wait_s: float = BATCH_WAIT_S):
        self.sess = sess
        self.tokenizer = tokenizer
        self.abs_max = abs_max
        self.server_max = server_max
        self.max_batch = int(max_batch)
        self.batch_wait_s = float(batch_wait_s)
        self._lock = asyncio.Lock()
        self._pending_requests: List[Dict[str, Any]] = []
        self._pending_futures: List[asyncio.Future] = []
        self._flush_task: Optional[asyncio.Task] = None

    async def enqueue(self, item: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._pending_requests.append(item)
            self._pending_futures.append(fut)
            pending_len = len(self._pending_requests)
            if self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_worker())
            if pending_len >= self.max_batch:
                if not self._flush_task.done():
                    asyncio.create_task(self._trigger_flush())
        try:
            result = await fut
            return result
        finally:
            async with self._lock:
                if self._flush_task and self._flush_task.done():
                    self._flush_task = None

    async def _trigger_flush(self):
        await asyncio.sleep(0)

    async def _flush_worker(self):
        try:
            await asyncio.sleep(self.batch_wait_s)
            async with self._lock:
                requests = self._pending_requests
                futures = self._pending_futures
                self._pending_requests = []
                self._pending_futures = []
            if not requests:
                return
            try:
                results = self._process_batch(requests)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
                return
            if not isinstance(results, list) or len(results) != len(futures):
                err = RuntimeError("Batch processing produced unexpected result length")
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(err)
                return
            for fut, res in zip(futures, results):
                if not fut.done():
                    fut.set_result(res)
        finally:
            async with self._lock:
                self._flush_task = None

    def _process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_texts: List[str] = []
        per_text_maxes: List[int] = []
        slices: List[Tuple[int, int]] = []
        for req in requests:
            start = len(all_texts)
            if isinstance(req, dict) and req.get("texts") is not None:
                ts = req["texts"]
                if not isinstance(ts, list):
                    ts = [ts]
                requested = req.get("max_length", None)
                effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                all_texts.extend([str(t) for t in ts])
                per_text_maxes.extend([effective_for_req] * len(ts))
                slices.append((start, len(ts)))
            elif isinstance(req, dict) and req.get("text") is not None:
                requested = req.get("max_length", None)
                effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                all_texts.append(str(req["text"]))
                per_text_maxes.extend([effective_for_req])
                slices.append((start, 1))
            else:
                requested = None
                effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                all_texts.append(str(req))
                per_text_maxes.append(effective_for_req)
                slices.append((start, 1))
        if len(all_texts) == 0:
            return [{"embeddings": [], "onnx_output": None, "used_max_length": None} for _ in requests]
        batch_max = max(per_text_maxes) if per_text_maxes else self.server_max
        tok = self.tokenizer(all_texts, padding=True, truncation=True, max_length=batch_max, return_tensors="np", return_attention_mask=True, return_token_type_ids=True)
        tokenized: Dict[str, np.ndarray] = {}
        for k, v in tok.items():
            arr = v
            if arr.dtype in (np.int32, np.int64):
                arr = arr.astype(np.int64)
            tokenized[k] = arr
        try:
            feed = map_inputs(self.sess, tokenized)
        except Exception as e:
            return [{"error": f"Input mapping failed: {e}"} for _ in requests]
        try:
            outputs = self.sess.run(None, feed)
        except Exception as e:
            return [{"error": f"ONNX runtime failed: {e}"} for _ in requests]
        embeddings = pick_embedding_from_outputs(outputs, self.sess, tokenized)
        results: List[Dict[str, Any]] = []
        for slice_start, length in slices:
            if length == 0:
                results.append({"embeddings": [], "onnx_output": None, "used_max_length": None})
            else:
                slice_emb = embeddings[slice_start:slice_start + length]
                used_for_request = int(max(per_text_maxes[slice_start:slice_start + length]))
                results.append({"embeddings": slice_emb.tolist(), "onnx_output": None, "used_max_length": used_for_request})
        return results

def rerank_pairs(sess: ort.InferenceSession, tokenizer: Any, query: str, passages: List[str], max_length: Optional[int], abs_max: int, server_max: int) -> List[float]:
    if not passages:
        return []
    effective_max = clamp_max_length(max_length, server_max, abs_max)
    pairs_q = [query] * len(passages)
    tok = tokenizer(pairs_q, passages, padding=True, truncation=True, max_length=effective_max, return_tensors="np", return_attention_mask=True, return_token_type_ids=True)
    tokenized: Dict[str, np.ndarray] = {}
    for k, v in tok.items():
        arr = v
        if arr.dtype in (np.int32, np.int64):
            arr = arr.astype(np.int64)
        tokenized[k] = arr
    feed = map_inputs(sess, tokenized)
    outputs = sess.run(None, feed)
    output_names = [o.name for o in sess.get_outputs()]
    chosen = None
    for n, out in zip(output_names, outputs):
        nl = n.lower()
        if "logit" in nl or "score" in nl or "label" in nl:
            chosen = out
            break
    if chosen is None:
        for out in outputs:
            if getattr(out, "ndim", None) == 1:
                chosen = out
                break
        if chosen is None:
            for out in outputs:
                if getattr(out, "ndim", None) == 2 and out.shape[1] in (1, 2):
                    chosen = out
                    break
    if chosen is None:
        chosen = outputs[0]
    chosen = np.asarray(chosen)
    if chosen.ndim == 2 and chosen.shape[1] == 2:
        scores = chosen[:, 1].astype(float).tolist()
    elif chosen.ndim == 2 and chosen.shape[1] == 1:
        scores = chosen[:, 0].astype(float).tolist()
    elif chosen.ndim == 1:
        scores = chosen.astype(float).tolist()
    elif chosen.ndim == 2:
        scores = chosen.mean(axis=1).astype(float).tolist()
    else:
        scores = chosen.reshape((chosen.shape[0], -1)).mean(axis=1).astype(float).tolist()
    return scores

def run_session_with_token_arrays(sess: ort.InferenceSession, arrays: Dict[str, List[List[int]]]) -> List[List[float]]:
    np_inputs: Dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        arr = np.array(v)
        if arr.dtype.kind in ("i", "u"):
            arr = arr.astype(np.int64)
        elif arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        np_inputs[k.lower()] = arr
    feed = map_inputs(sess, np_inputs)
    outputs = sess.run(None, feed)
    emb = pick_embedding_from_outputs(outputs, sess, np_inputs)
    return emb.tolist()

def get_abs_max_from_tokenizer(tok: Any) -> int:
    try:
        m = int(getattr(tok, "model_max_length", 0) or 0)
        return 8192 if (m <= 0 or m > 1_000_000) else m
    except Exception:
        return 8192

def clamp_max_length(requested: Optional[int], server_default: int, abs_max: int) -> int:
    if requested is None:
        v = int(server_default)
    else:
        try:
            v = int(requested)
        except Exception:
            v = int(server_default)
    return max(1, min(v, abs_max))

app = FastAPI()

embedder_sess: Optional[ort.InferenceSession] = None
embedder_tok: Optional[Any] = None
embedder_abs_max: int = 8192
embedder_server_max: int = SERVER_DEFAULT_MAX_LENGTH
embedder_batcher: Optional[AsyncBatcher] = None

reranker_sess: Optional[ort.InferenceSession] = None
reranker_tok: Optional[Any] = None
reranker_abs_max: int = 8192
reranker_server_max: int = SERVER_DEFAULT_MAX_LENGTH

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder_sess, embedder_tok, embedder_abs_max, embedder_server_max, embedder_batcher
    global reranker_sess, reranker_tok, reranker_abs_max, reranker_server_max

    try:
        logger.info("Loading embedder tokenizer (hardcoded path)...")
        embedder_tok = load_tokenizer_from_dir(EMBEDDER_MODEL_DIR)
        embedder_sess, prov_e = create_onnx_session(EMBEDDER_MODEL_PATH, prefer_cuda=not FORCE_CPU)
        if embedder_tok is not None:
            embedder_abs_max = get_abs_max_from_tokenizer(embedder_tok)
            embedder_server_max = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), embedder_abs_max))
            embedder_batcher = AsyncBatcher(embedder_sess, embedder_tok, embedder_abs_max, embedder_server_max)
            logger.info("Embedder tokenizer present.")
        else:
            logger.warning("Embedder tokenizer not available; text-based /embed disabled. Use /embed_with_ids.")
        logger.info("Embedder loaded provider=%s inputs=%s outputs=%s", prov_e, [i.name for i in embedder_sess.get_inputs()], [o.name for o in embedder_sess.get_outputs()])
    except Exception as e:
        logger.exception("Failed to load embedder: %s", e)
        raise

    try:
        logger.info("Loading reranker tokenizer (hardcoded path)...")
        reranker_tok = load_tokenizer_from_dir(RERANKER_MODEL_DIR)
        reranker_sess, prov_r = create_onnx_session(RERANKER_MODEL_PATH, prefer_cuda=not FORCE_CPU)
        if reranker_tok is not None:
            reranker_abs_max = get_abs_max_from_tokenizer(reranker_tok)
            reranker_server_max = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), reranker_abs_max))
            logger.info("Reranker tokenizer present.")
        else:
            logger.warning("Reranker tokenizer not available; text-based /rerank disabled. Use /rerank_with_ids.")
        logger.info("Reranker loaded provider=%s inputs=%s outputs=%s", prov_r, [i.name for i in reranker_sess.get_inputs()], [o.name for o in reranker_sess.get_outputs()])
    except Exception as e:
        logger.exception("Failed to load reranker: %s", e)
        raise

    yield
    logger.info("Shutting down host_onnx_models")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    embedder_present = os.path.exists(EMBEDDER_MODEL_PATH)
    reranker_present = os.path.exists(RERANKER_MODEL_PATH)
    visible_cuda = _cuda_device_count_fallback()
    return JSONResponse({
        "status": "ok",
        "embedder_model": EMBEDDER_MODEL_PATH if embedder_present else "missing",
        "reranker_model": RERANKER_MODEL_PATH if reranker_present else "missing",
        "cuda_visible": visible_cuda,
        "embedder_tokenizer_present": bool(embedder_tok),
        "reranker_tokenizer_present": bool(reranker_tok),
    })

@app.post("/embed")
async def embed(request: Request):
    if embedder_tok is None:
        return JSONResponse({"error": "Tokenizer missing. Use /embed_with_ids or provide tokenizer files."}, status_code=400)
    if embedder_batcher is None:
        return JSONResponse({"error": "Embedder not ready."}, status_code=503)
    body = await request.json()
    try:
        res = await embedder_batcher.enqueue(body)
        return JSONResponse(res)
    except Exception as e:
        logger.exception("embed handler failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/embed_with_ids")
async def embed_with_ids(request: Request):
    if embedder_sess is None:
        return JSONResponse({"error": "Embedder session not ready."}, status_code=503)
    body = await request.json()
    input_ids = body.get("input_ids")
    if input_ids is None:
        return JSONResponse({"error": "Provide 'input_ids' as a list of lists"}, status_code=400)
    arrays = {"input_ids": input_ids}
    if "attention_mask" in body:
        arrays["attention_mask"] = body["attention_mask"]
    if "token_type_ids" in body:
        arrays["token_type_ids"] = body["token_type_ids"]
    try:
        emb = run_session_with_token_arrays(embedder_sess, arrays)
        return JSONResponse({"embeddings": emb})
    except Exception as e:
        logger.exception("embed_with_ids failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/rerank")
async def rerank(request: Request):
    if reranker_tok is None:
        return JSONResponse({"error": "Reranker tokenizer missing. Use /rerank_with_ids or provide tokenizer files."}, status_code=400)
    body = await request.json()
    query = body.get("query") or body.get("text") or body.get("q")
    passages = body.get("passages") or body.get("candidates") or body.get("docs") or body.get("contexts")
    max_length = body.get("max_length", None)
    if query is None or passages is None:
        return JSONResponse({"error": "Provide 'query' and 'passages' (list)"}, status_code=400)
    if not isinstance(passages, list):
        return JSONResponse({"error": "'passages' must be a list"}, status_code=400)
    try:
        scores = rerank_pairs(reranker_sess, reranker_tok, str(query), [str(p) for p in passages], max_length, reranker_abs_max, reranker_server_max)
        return JSONResponse({"scores": scores})
    except Exception as e:
        logger.exception("rerank failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/rerank_with_ids")
async def rerank_with_ids(request: Request):
    if reranker_sess is None:
        return JSONResponse({"error": "Reranker session not ready."}, status_code=503)
    body = await request.json()
    pair_ids = body.get("pair_input_ids")
    pair_attn = body.get("pair_attention_mask")
    if pair_ids is None:
        return JSONResponse({"error": "Provide 'pair_input_ids' as list-of-lists for each (query,passage) pair."}, status_code=400)
    arrays = {"input_ids": pair_ids}
    if pair_attn:
        arrays["attention_mask"] = pair_attn
    try:
        logits_or_emb = run_session_with_token_arrays(reranker_sess, arrays)
        return JSONResponse({"logits_or_embeddings": logits_or_emb})
    except Exception as e:
        logger.exception("rerank_with_ids failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

def _install_signal_handlers():
    def _term(signum, frame):
        logger.info("Received signal %s, exiting", signum)
        try:
            sys.exit(0)
        except Exception:
            pass
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

if __name__ == "__main__":
    _install_signal_handlers()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
