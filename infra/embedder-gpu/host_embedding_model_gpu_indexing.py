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
from transformers import AutoTokenizer

try:
    import ray
    from ray import serve
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse
    import uvicorn
except Exception:
    ray = None
    serve = None
    StarletteRequest = None
    JSONResponse = None
    uvicorn = None

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8001"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models/onnx/gte-modernbert-base-onnx-int8")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
SERVER_DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH") or "512")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "128"))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", "0.02"))
NUM_REPLICAS = int(os.environ.get("NUM_REPLICAS", "1"))
REPLICA_CPUS = float(os.environ.get("REPLICA_CPUS", "1.0"))
REPLICA_GPUS = float(os.environ.get("REPLICA_GPUS", "1.0"))
CUSTOM_RESOURCE = os.environ.get("CUSTOM_RESOURCE", "")
CUSTOM_RESOURCE_UNITS = float(os.environ.get("CUSTOM_RESOURCE_UNITS", "1"))
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "1"))
ORT_INTER_THREADS = int(os.environ.get("ORT_INTER_THREADS", "1"))
CUDA_DEVICE_ENV = os.environ.get("CUDA_DEVICE", "0")
SKIP_SERVE = os.environ.get("SKIP_SERVE", "") in ("1", "true", "True")
FORCE_CPU = os.environ.get("FORCE_CPU", "") in ("1", "true", "True")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("embedder-prod")

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


def create_onnx_session_and_tokenizer(model_path: str = MODEL_PATH, model_dir: str = MODEL_DIR, prefer_cuda: bool = True) -> Tuple[ort.InferenceSession, Any, str]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
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
                return sess, tokenizer, provider_used
            except Exception as e:
                logger.warning("Failed to create CUDAExecutionProvider with device_id=%s: %s", device_idx, e)
        try:
            sess = ort.InferenceSession(model_path, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            provider_used = "CUDAExecutionProvider"
            logger.info("Created ONNX CUDAExecutionProvider without explicit device_id")
            return sess, tokenizer, provider_used
        except Exception as e:
            logger.warning("Failed to create CUDAExecutionProvider (no device_id): %s", e)
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    provider_used = "CPUExecutionProvider"
    logger.info("Created ONNX CPUExecutionProvider")
    return sess, tokenizer, provider_used


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


def mean_pooling(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[..., :, None]
    masked = hidden_states * mask
    sums = masked.sum(axis=1)
    counts = mask.sum(axis=1)
    counts = np.maximum(counts, 1e-9)
    return sums / counts


def embed_texts(sess: ort.InferenceSession, tokenizer: Any, texts: List[str], max_length: Optional[int] = None) -> List[List[float]]:
    if not texts:
        return []
    server_max = int(max_length or SERVER_DEFAULT_MAX_LENGTH)
    abs_max = get_abs_max_from_tokenizer(tokenizer)
    effective_max = clamp_max_length(max_length, server_max, abs_max)
    tok = tokenizer(texts, padding=True, truncation=True, max_length=effective_max, return_tensors="np", return_attention_mask=True, return_token_type_ids=True)
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
    return [emb.tolist() for emb in embeddings]


def run_local_sample(texts: Optional[List[str]] = None, prefer_cuda: bool = True) -> None:
    texts = texts or ["hello"]
    try:
        sess, tokenizer, provider = create_onnx_session_and_tokenizer(prefer_cuda=prefer_cuda)
    except Exception as exc:
        logger.exception("Failed to create session/tokenizer: %s", exc)
        print("Sample embedding failed: see logs", file=sys.stderr)
        return
    logger.info("ORT providers (driver): %s", ort.get_available_providers())
    visible = _cuda_device_count_fallback()
    logger.info("Visible CUDA devices (heuristic, driver): %s", visible)
    print("Selected provider (driver):", provider)
    try:
        embeddings = embed_texts(sess, tokenizer, texts, max_length=None)
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        print("Sample embedding failed: see logs", file=sys.stderr)
        return
    for i, emb in enumerate(embeddings):
        arr = np.array(emb, dtype=float)
        out = {"text": texts[i], "provider": provider, "embedding_len": int(arr.size), "embedding_first8": arr[:8].tolist()}
        print(json.dumps(out))
        print("Full vector length:", arr.size)


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
                per_text_maxes.append(effective_for_req)
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
            output_names = [o.name for o in self.sess.get_outputs()]
        except Exception as e:
            return [{"error": f"ONNX runtime failed: {e}"} for _ in requests]
        chosen = None
        chosen_name = None
        for n, out in zip(output_names, outputs):
            nl = n.lower()
            if "pooled" in nl or "sentence" in nl or "cls" in nl:
                chosen = out
                chosen_name = n
                break
        if chosen is None:
            for n, out in zip(output_names, outputs):
                if getattr(out, "ndim", None) == 2:
                    chosen = out
                    chosen_name = n
                    break
        if chosen is None:
            chosen = outputs[0]
            chosen_name = output_names[0]
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
        results: List[Dict[str, Any]] = []
        for slice_start, length in slices:
            if length == 0:
                results.append({"embeddings": [], "onnx_output": chosen_name, "used_max_length": None})
            else:
                slice_emb = embeddings[slice_start:slice_start + length]
                used_for_request = int(max(per_text_maxes[slice_start:slice_start + length]))
                results.append({"embeddings": slice_emb.tolist(), "onnx_output": chosen_name, "used_max_length": used_for_request})
        return results


class EmbedderImpl:
    def __init__(self):
        self.log = logging.getLogger(f"embedder-replica-{os.getpid()}")
        self.log.setLevel(logging.INFO)
        self.log.info("Initializing embedder replica")
        self.log.info("Replica env: CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=3).decode()
        except Exception as e:
            out = f"nvidia-smi failed: {e}"
        self.log.info("Replica nvidia-smi: %s", out)
        self.log.info("ONNX available providers at replica startup: %s", ort.get_available_providers())
        if not os.path.exists(MODEL_PATH):
            self.log.error("ONNX model missing: %s", MODEL_PATH)
            raise RuntimeError("Model missing")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
        self.abs_max = get_abs_max_from_tokenizer(self.tokenizer)
        self.server_max = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), self.abs_max))
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_INTRA_THREADS
        so.inter_op_num_threads = ORT_INTER_THREADS
        providers = ort.get_available_providers()
        visible_cuda_count = _cuda_device_count_fallback()
        want_cuda = ("CUDAExecutionProvider" in providers) and (visible_cuda_count > 0) and (REPLICA_GPUS > 0) and (not FORCE_CPU)
        if REPLICA_GPUS > 0 and want_cuda is False:
            self.log.error("Replica requested GPUs (REPLICA_GPUS=%s) but no visible CUDA devices in this process (visible=%s). Failing replica to allow reschedule.", REPLICA_GPUS, visible_cuda_count)
            raise RuntimeError("No visible CUDA devices in replica while REPLICA_GPUS>0")
        if want_cuda:
            device_idx = _get_allocated_cuda_device_index()
            if device_idx < visible_cuda_count:
                try:
                    self.sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], provider_options=[{"device_id": int(device_idx)}, {}])
                    self.log.info("Replica created CUDAExecutionProvider device=%s", device_idx)
                except Exception as exc:
                    self.log.warning("Failed to create CUDA session with device id; falling back to CPU: %s", exc)
                    self.sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
            else:
                try:
                    self.sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    self.log.info("Replica created CUDAExecutionProvider (no explicit device id)")
                except Exception as exc:
                    self.log.warning("Failed to create CUDA session; falling back to CPU: %s", exc)
                    self.sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        else:
            self.sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
            self.log.info("Replica created CPUExecutionProvider")
        self.batcher: Optional[AsyncBatcher] = None
        self.log.info("Replica ready. sess_inputs=%s sess_outputs=%s", [i.name for i in self.sess.get_inputs()], [o.name for o in self.sess.get_outputs()])

    async def _handle_health(self) -> "JSONResponse":
        model_exists = os.path.exists(MODEL_PATH)
        return JSONResponse({"status": "ok", "model_path": MODEL_PATH if model_exists else "missing"})

    async def _handle_embed(self, body: Dict[str, Any]) -> "JSONResponse":
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        texts = None
        text = None
        max_length = None
        if "texts" in body:
            t = body.get("texts")
            if not isinstance(t, list) or not all(isinstance(x, (str, bytes)) for x in t):
                return JSONResponse({"error": "'texts' must be list of strings"}, status_code=400)
            texts = [str(x) for x in t]
        elif "text" in body:
            t = body.get("text")
            if not isinstance(t, (str, bytes)):
                return JSONResponse({"error": "'text' must be string"}, status_code=400)
            text = str(t)
        else:
            return JSONResponse({"error": "Provide 'text' or 'texts'"}, status_code=400)
        if "max_length" in body:
            try:
                max_length = int(body.get("max_length"))
            except Exception:
                return JSONResponse({"error": "'max_length' must be integer"}, status_code=400)
        normalized: Dict[str, Any] = {}
        if texts is not None:
            normalized["texts"] = texts
        else:
            normalized["text"] = text
        if max_length is not None:
            normalized["max_length"] = max_length
        try:
            if self.batcher is None:
                self.batcher = AsyncBatcher(self.sess, self.tokenizer, self.abs_max, self.server_max)
            result = await self.batcher.enqueue(normalized)
            return JSONResponse(result)
        except Exception as exc:
            self.log.exception("embed failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def __call__(self, request: "StarletteRequest"):
        path = request.url.path
        method = request.method.upper()
        if path == "/health" and method == "GET":
            return await self._handle_health()
        if path == "/embed" and method == "POST":
            body = await request.json()
            return await self._handle_embed(body)
        return JSONResponse({"error": "not found"}, status_code=404)


if serve is not None:
    Embedder = serve.deployment()(EmbedderImpl)
else:
    Embedder = None


def serve_deploy(ray_address: Optional[str] = None):
    if ray is None or serve is None:
        raise RuntimeError("Ray/Serve not available in this python environment")
    try:
        addr = ray_address if ray_address is not None else "auto"
        if addr == "auto":
            ray.init(address="auto", ignore_reinit_error=True)
        elif addr == "":
            return
        else:
            ray.init(address=addr, ignore_reinit_error=True)
    except Exception as e:
        logger.warning("ray.init() warning/fallback: %s", e)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    try:
        available = ray.available_resources()
    except Exception:
        available = {}
    logger.info("Ray available resources: %s", available)
    ray_actor_options: Dict[str, Any] = {"num_cpus": REPLICA_CPUS}
    try:
        if REPLICA_GPUS > 0:
            ray_actor_options["num_gpus"] = REPLICA_GPUS
    except Exception:
        pass
    if CUSTOM_RESOURCE:
        if CUSTOM_RESOURCE in available and available.get(CUSTOM_RESOURCE, 0) >= CUSTOM_RESOURCE_UNITS:
            ray_actor_options["resources"] = {CUSTOM_RESOURCE: CUSTOM_RESOURCE_UNITS}
    serve.start(detached=True, http_options={"host": HOST, "port": PORT})
    app = Embedder.options(num_replicas=max(1, NUM_REPLICAS), ray_actor_options=ray_actor_options).bind()
    serve.run(app, route_prefix="/")


def _install_signal_handlers():
    def _term(signum, frame):
        logger.info("Received signal %s, exiting", signum)
        try:
            if ray is not None and ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)


if __name__ == "__main__":
    _install_signal_handlers()
    prefer_cuda = (not FORCE_CPU)
    logger.info("MODEL_PATH: %s", MODEL_PATH)
    logger.info("Producing sample embedding for 'hello' (prefer_cuda=%s) ...", prefer_cuda)
    run_local_sample(["hello"], prefer_cuda=prefer_cuda)
    sys.stdout.flush()
    if SKIP_SERVE:
        logger.info("SKIP_SERVE set; exiting after sample.")
        sys.exit(0)
    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    if ray_address == "":
        if StarletteRequest is None or JSONResponse is None or uvicorn is None:
            logger.error("Starlette/uvicorn not available in this environment; cannot run local server.")
            sys.exit(1)
        embedder = EmbedderImpl()
        async def asgi_app(scope, receive, send):
            if scope["type"] != "http":
                await send({"type": "http.response.start", "status": 405, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            req = StarletteRequest(scope, receive=receive)
            res = await embedder(req)
            await res(scope, receive, send)
        logger.info("Starting local uvicorn server on %s:%s", HOST, PORT)
        uvicorn.run(asgi_app, host=HOST, port=PORT, log_level="info")
    else:
        logger.info("Deploying to Ray Serve (ray_address=%s) ...", ray_address)
        serve_deploy(ray_address)
