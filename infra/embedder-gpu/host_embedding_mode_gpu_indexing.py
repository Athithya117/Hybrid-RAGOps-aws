#!/usr/bin/env python3
# host_embedding_model_gpu_inference.py
# GPU-aware variant with improved health, fast-fail when REQUIRE_GPU=true,
# and clearer logging for provisioning.

import os
import sys
import time
import logging
import socket
import signal
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
import onnxruntime as ort
import ray
from ray import serve
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

# ---------- env/config ----------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models/onnx/gte-modernbert-base-onnx-int8")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
SERVER_DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "100"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", "0.03"))

# Ray / Serve sizing
NUM_REPLICAS = int(os.environ.get("NUM_REPLICAS", "1"))
REPLICA_CPUS = float(os.environ.get("REPLICA_CPUS", "1.0"))
REPLICA_GPUS = float(os.environ.get("REPLICA_GPUS", "0"))  # set to >0 to request GPUs
CUSTOM_RESOURCE = os.environ.get("CUSTOM_RESOURCE", "")
CUSTOM_RESOURCE_UNITS = float(os.environ.get("CUSTOM_RESOURCE_UNITS", "1"))

# ONNX runtime threads
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "1"))
ORT_INTER_THREADS = int(os.environ.get("ORT_INTER_THREADS", "1"))

# optional explicit CUDA device id (useful with multi-GPU nodes)
CUDA_DEVICE_ID = int(os.environ.get("CUDA_DEVICE_ID", "0"))

# Require GPU: fail-fast if set
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "false").lower() in ("1", "true", "yes")

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))

# ---------- logging ----------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("embedder-gpu")

class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    max_length: Optional[int] = None

# ---------- helpers ----------
def is_port_open(host: str, port: int, timeout=0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False

def find_free_port(start: int = 8000, host: str = "127.0.0.1") -> int:
    p = start
    while p < 65535:
        if not is_port_open(host, p):
            return p
        p += 1
    raise RuntimeError("No free ports")

def load_local_tokenizer(local_dir: str):
    if not os.path.isdir(local_dir):
        raise RuntimeError(f"Tokenizer folder not found: {local_dir}")
    try:
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True)
        log.info("Loaded tokenizer (fast=%s)", getattr(tok, "is_fast", None))
        return tok
    except Exception as e:
        log.warning("Fast tokenizer failed: %s", e)
    tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
    log.info("Loaded tokenizer (fast=%s)", getattr(tok, "is_fast", None))
    return tok

def get_abs_max_from_tokenizer(tok) -> int:
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
            feed[sin] = lowered["input_ids"]; continue
        if "attention" in sl and "mask" in sl and "attention_mask" in lowered:
            feed[sin] = lowered["attention_mask"]; continue
        if ("token_type" in sl or "segment" in sl) and "token_type_ids" in lowered:
            feed[sin] = lowered["token_type_ids"]; continue
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

# ---------------- Batcher ----------------
class AsyncBatcher:
    def __init__(self, sess: ort.InferenceSession, tokenizer, abs_max: int, server_max: int,
                 max_batch: int = 64, batch_wait_s: float = 0.03):
        self.sess = sess
        self.tokenizer = tokenizer
        self.abs_max = abs_max
        self.server_max = server_max
        self.max_batch = max_batch
        self.batch_wait_s = batch_wait_s

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
            except Exception as e:
                log.exception("Batch processing failed")
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
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
        log.info("Batch tokenization: total_texts=%d batch_max=%d", len(all_texts), batch_max)

        t0 = time.perf_counter()
        tok = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=batch_max,
            return_tensors="np",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        token_time = time.perf_counter() - t0

        tokenized: Dict[str, np.ndarray] = {}
        for k, v in tok.items():
            arr = v
            if arr.dtype in (np.int32, np.int64):
                arr = arr.astype(np.int64)
            tokenized[k] = arr

        try:
            feed = map_inputs(self.sess, tokenized)
        except Exception as e:
            log.exception("Input mapping failed")
            return [{"error": f"Input mapping failed: {e}"} for _ in requests]

        t1 = time.perf_counter()
        try:
            outputs = self.sess.run(None, feed)
            output_names = [o.name for o in self.sess.get_outputs()]
        except Exception as e:
            log.exception("ONNX runtime failed")
            return [{"error": f"ONNX runtime failed: {e}"} for _ in requests]
        inf_time = time.perf_counter() - t1

        # choose output
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
            try:
                embeddings = chosen.reshape((chosen.shape[0], -1))
            except Exception:
                return [{"error": "Unexpected ONNX output shape", "onnx_output": chosen_name} for _ in requests]

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

# ---------------- FastAPI app ----------------
app = FastAPI(title="ONNX Embedding (gpu-serve)")

MODEL_LOADED = False
BATCHER: Optional[AsyncBatcher] = None

def choose_providers_and_options():
    prov = ort.get_available_providers()
    log.info("ONNX Runtime providers available: %s", prov)
    if "CUDAExecutionProvider" in prov:
        # explicit provider options (device id). provider_options list must align with providers list
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": CUDA_DEVICE_ID}, {}]
    return ["CPUExecutionProvider"], [{}]

def initialize_model_and_batcher():
    global MODEL_LOADED, BATCHER
    if MODEL_LOADED:
        return
    log.info("Loading tokenizer + ONNX from %s", MODEL_DIR)
    tokenizer = load_local_tokenizer(MODEL_DIR)
    ABS_MAX = get_abs_max_from_tokenizer(tokenizer)
    SERVER_MAX = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), ABS_MAX))

    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_INTRA_THREADS
    so.inter_op_num_threads = ORT_INTER_THREADS

    providers, provider_options = choose_providers_and_options()
    # if REQUIRE_GPU is set and CUDA provider not present, fail fast
    if REQUIRE_GPU and "CUDAExecutionProvider" not in providers:
        log.error("REQUIRE_GPU=true but CUDAExecutionProvider not available (providers=%s). Exiting.", providers)
        sys.exit(2)

    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers, provider_options=provider_options)
    log.info("ONNX model loaded. inputs=%s outputs=%s", [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()])

    BATCHER = AsyncBatcher(sess=sess, tokenizer=tokenizer, abs_max=ABS_MAX, server_max=SERVER_MAX,
                           max_batch=MAX_BATCH, batch_wait_s=BATCH_WAIT_S)
    MODEL_LOADED = True

@app.get("/health")
async def health():
    prov = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in prov
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "model_path": MODEL_PATH if model_exists else "missing",
        "onnx_providers": prov,
        "cuda_available": has_cuda,
        "require_gpu": REQUIRE_GPU
    }

@app.post("/embed")
async def embed(payload: EmbedRequest, request: Request):
    start = time.perf_counter()
    if not MODEL_LOADED:
        initialize_model_and_batcher()

    if payload.texts:
        normalized = {"texts": payload.texts}
    elif payload.text:
        normalized = {"text": payload.text}
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'")

    if payload.max_length is not None:
        normalized["max_length"] = int(payload.max_length)

    try:
        result = await BATCHER.enqueue(normalized)
        return result
    except Exception as e:
        log.exception("embed failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        total = time.perf_counter() - start
        log.info("request finished path=%s total_s=%.3f", request.url.path, total)

# ---------------- Deployment class for Serve ----------------
@serve.deployment()
@serve.ingress(app)
class Embedder:
    pass

# ---------------- run helpers ----------------
def run_uvicorn(port: int):
    log.info("Starting uvicorn on %s:%d", HOST, port)
    uvicorn.run(app, host=HOST, port=port, log_level="info")

def serve_deploy_with_ray(ray_address: str):
    try:
        if ray_address == "auto":
            try:
                ray.init(address="auto")
                log.info("Connected to Ray cluster (auto)")
            except Exception:
                log.info("No cluster detected; starting local Ray")
                ray.init()
        elif ray_address:
            ray.init(address=ray_address)
            log.info("Connected to Ray at %s", ray_address)
        else:
            log.info("RAY_ADDRESS empty -> not using Ray Serve")
            return False, None
    except Exception as e:
        log.exception("ray.init failed: %s", e)
        return False, None

    try:
        available = ray.available_resources()
    except Exception:
        available = {}
    log.info("Ray available resources: %s", available)

    # build ray_actor_options
    ray_actor_options = {"num_cpus": REPLICA_CPUS}
    if REPLICA_GPUS and REPLICA_GPUS > 0:
        # prefer standard num_gpus field — Ray will only place on GPU nodes
        ray_actor_options["num_gpus"] = float(REPLICA_GPUS)
        log.info("Requesting %s GPUs per replica", REPLICA_GPUS)

    if CUSTOM_RESOURCE:
        if CUSTOM_RESOURCE in available and available.get(CUSTOM_RESOURCE, 0) >= CUSTOM_RESOURCE_UNITS:
            ray_actor_options.setdefault("resources", {})[CUSTOM_RESOURCE] = CUSTOM_RESOURCE_UNITS
            log.info("Using custom resource '%s'", CUSTOM_RESOURCE)
        else:
            log.info("Custom resource '%s' not available; will deploy without it", CUSTOM_RESOURCE)

    deploy_options = {"num_replicas": max(1, NUM_REPLICAS)}

    try:
        serve.start(detached=True, http_options={"host": "0.0.0.0", "port": PORT})
    except Exception as e:
        log.info("serve.start note: %s", e)

    try:
        dep = Embedder.options(**deploy_options, ray_actor_options=ray_actor_options)
        if hasattr(dep, "deploy") and callable(getattr(dep, "deploy")):
            log.info("Using legacy dep.deploy()")
            dep.deploy()
            return True, "deployed"
        else:
            bound = dep.bind()
            serve.run(bound, route_prefix="/")
            return True, "served"
    except Exception as e:
        log.exception("serve.run / deploy failed: %s", e)
        return False, None

# ---------------- main ----------------
def main():
    if not os.path.exists(MODEL_PATH):
        log.error("ONNX model not found at %s; please set MODEL_DIR correctly.", MODEL_PATH)
        sys.exit(1)

    # quick provider check before starting heavy init (helps fail fast in autoscaler)
    providers = ort.get_available_providers()
    log.info("Initial ONNX providers (pre-load): %s", providers)
    if REQUIRE_GPU and "CUDAExecutionProvider" not in providers:
        log.error("REQUIRE_GPU=true but CUDAExecutionProvider not available (providers=%s). Exiting.", providers)
        sys.exit(2)

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    if ray_address == "":
        port = PORT if not is_port_open("127.0.0.1", PORT) else find_free_port(PORT + 1)
        run_uvicorn(port)
        return

    ok, status = serve_deploy_with_ray(ray_address)
    if ok:
        time.sleep(0.5)
        if is_port_open("127.0.0.1", PORT):
            log.info("Serve appears to be listening on port %d. Check http://127.0.0.1:%d/health", PORT, PORT)
            signal.pause()
        else:
            port = PORT if not is_port_open("127.0.0.1", PORT) else find_free_port(PORT + 1)
            log.info("Serve proxy not listening on %d; falling back to uvicorn on %d", PORT, port)
            run_uvicorn(port)
    else:
        if is_port_open("127.0.0.1", PORT):
            log.info("Port %d already bound. If Serve proxy is up, use it. Blocking to keep process alive.", PORT)
            signal.pause()
        else:
            port = PORT if not is_port_open("127.0.0.1", PORT) else find_free_port(PORT + 1)
            log.info("Falling back to uvicorn on %d", port)
            run_uvicorn(port)

if __name__ == "__main__":
    main()
