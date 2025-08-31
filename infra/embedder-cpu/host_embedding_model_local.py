"""
Final single-file server:
 - Default: runs a local FastAPI + Uvicorn server (no Ray).
 - Optional: run with Ray Serve when RUN_MODE=ray (ray must be installed).
Fixed model path: /workspace/models/onnx/gte-modernbert-base-onnx-int8

Usage:
  # Local FastAPI:
  python3 infra/host_embedding_model_final_allinone.py
  # Ray Serve:
  RUN_MODE=ray python3 infra/host_embedding_model_final_allinone.py

"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# configuration / env
RUN_MODE = os.environ.get("RUN_MODE", "").lower()  # "ray" for Ray Serve
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
ROUTE_PREFIX = os.environ.get("ROUTE_PREFIX", "/")
MODEL_DIR = "/workspace/models/onnx/gte-modernbert-base-onnx-int8"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
SERVER_DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "500"))  # operator default
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", "0.030"))
WORKER_REPLICAS = int(os.environ.get("WORKER_REPLICAS", "1"))
WORKER_GPU = float(os.environ.get("WORKER_GPU", "0"))


# transformers tokenizer
from transformers import AutoTokenizer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-server-final")

class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    # per-request override - optional, validated server-side
    max_length: Optional[int] = None

# ---------------- utilities ----------------
def choose_providers() -> List[str]:
    prov = ort.get_available_providers()
    log.info("ONNX Runtime providers available: %s", prov)
    # prefer CUDA if available
    if "CUDAExecutionProvider" in prov:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def load_local_tokenizer(local_dir: str):
    if not os.path.isdir(local_dir):
        raise RuntimeError(f"Tokenizer folder not found: {local_dir}")
    # try fast tokenizer first
    try:
        log.info("Loading fast tokenizer from local dir: %s", local_dir)
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True)
        log.info("Loaded fast tokenizer (local). is_fast=%s", getattr(tok, "is_fast", None))
        return tok
    except Exception as e_fast:
        log.warning("Fast tokenizer load failed: %s", e_fast)
    # try slow tokenizer
    try:
        log.info("Loading slow tokenizer from local dir: %s", local_dir)
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
        log.info("Loaded slow tokenizer (local). is_fast=%s", getattr(tok, "is_fast", None))
        return tok
    except Exception as e_slow:
        log.error("Slow tokenizer load failed: %s", e_slow)

    msg = (
        f"Failed to load tokenizer from local dir: {local_dir}\n"
        "Ensure tokenizer.json and tokenizer_config.json (or tokenizer.model/vocab) exist in that folder.\n"
        "If you see conversion errors mentioning SentencePiece or tiktoken, install them:\n"
        "  pip install sentencepiece tiktoken\n"
    )
    raise RuntimeError(msg)

def get_abs_max_from_tokenizer(tok) -> int:
    """
    Return safe absolute model max length. If tokenizer.model_max_length is missing or nonsensical,
    fall back to 8192 (the previous default used).
    """
    try:
        m = int(getattr(tok, "model_max_length", 0) or 0)
        if m <= 0 or m > 1_000_000:
            return 8192
        return m
    except Exception:
        return 8192

def clamp_max_length(requested: Optional[int], server_default: int, abs_max: int) -> int:
    """
    Validate and clamp requested max length.
    - If requested is None: return server_default (clamped)
    - Force 1 <= value <= abs_max
    """
    if requested is None:
        v = int(server_default)
    else:
        try:
            v = int(requested)
        except Exception:
            v = int(server_default)
    v = max(1, min(v, abs_max))
    return v

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
        raise RuntimeError(f"Could not map tokenizer outputs to ONNX inputs: {sess_input_names} vs {list(tokenized.keys())}")
    return feed

def mean_pooling(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[..., :, None]
    masked = hidden_states * mask
    sums = masked.sum(axis=1)
    counts = mask.sum(axis=1)
    counts = np.maximum(counts, 1e-9)
    return sums / counts

# ---------------- FastAPI local server ----------------
def build_fastapi_app():
    app = FastAPI(title="ONNX Embedding Local")

    if not os.path.exists(MODEL_PATH):
        log.error("ONNX model not found at %s", MODEL_PATH)
        raise SystemExit(f"ONNX model not found at {MODEL_PATH}")

    # load tokenizer first (we need its model_max_length for validation)
    tokenizer = load_local_tokenizer(MODEL_DIR)
    ABS_MAX = get_abs_max_from_tokenizer(tokenizer)
    SERVER_MAX = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), ABS_MAX))
    log.info("Tokenizer ABS_MAX=%s, SERVER_DEFAULT_MAX=%s", ABS_MAX, SERVER_MAX)

    # ONNX session
    providers = choose_providers()
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    log.info("ONNX model loaded. inputs=%s outputs=%s", [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()])

    @app.get("/health")
    async def health():
        return {"status": "ok", "abs_max_length": ABS_MAX, "server_default_max_length": SERVER_MAX}

    @app.post("/embed")
    async def embed(payload: EmbedRequest):
        # normalize input
        if payload.texts:
            texts = payload.texts
        elif payload.text:
            texts = [payload.text]
        else:
            raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'")

        # determine effective max length for this request (clamped)
        requested = getattr(payload, "max_length", None)
        effective_max = clamp_max_length(requested, SERVER_MAX, ABS_MAX)
        log.info("Request received: n_texts=%d requested_max=%s effective_max=%d", len(texts), requested, effective_max)

        flat_texts = [str(t) for t in texts]
        tok = tokenizer(
            flat_texts,
            padding=True,             # pads to longest in batch (here single request)
            truncation=True,
            max_length=effective_max,
            return_tensors="np",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        tokenized = {}
        for k, v in tok.items():
            arr = v
            if arr.dtype in (np.int32, np.int64):
                arr = arr.astype(np.int64)
            tokenized[k] = arr

        try:
            feed = map_inputs(sess, tokenized)
        except Exception as e:
            log.exception("Input mapping failed")
            raise HTTPException(status_code=500, detail=f"Input mapping failed: {e}")

        try:
            outputs = sess.run(None, feed)
            output_names = [o.name for o in sess.get_outputs()]
        except Exception as e:
            log.exception("ONNX runtime failed")
            raise HTTPException(status_code=500, detail=f"ONNX runtime failed: {e}")

        # choose embedding output
        chosen = None; chosen_name = None
        for name, out in zip(output_names, outputs):
            nl = name.lower()
            if "pooled" in nl or "sentence" in nl or "cls" in nl:
                chosen = out; chosen_name = name; break
        if chosen is None:
            for name, out in zip(output_names, outputs):
                if getattr(out, "ndim", None) == 2:
                    chosen = out; chosen_name = name; break
        if chosen is None:
            chosen = outputs[0]; chosen_name = output_names[0]

        if getattr(chosen, "ndim", None) == 3:
            att = tokenized.get("attention_mask")
            if att is None:
                emb = chosen.mean(axis=1)
            else:
                emb = mean_pooling(chosen, att)
        elif getattr(chosen, "ndim", None) == 2:
            emb = chosen
        else:
            emb = chosen.reshape((chosen.shape[0], -1))

        emb = np.asarray(emb).astype(float)
        return {"embeddings": emb.tolist(), "onnx_output": chosen_name, "used_max_length": effective_max}

    return app

# ---------------- Ray Serve mode ----------------
def start_ray_serve():
    import ray
    from ray import serve

    ray.init(ignore_reinit_error=True)
    serve.start(detached=False, http_options={"host": HOST, "port": PORT})

    @serve.deployment(
        name="onnx_worker",
        num_replicas=WORKER_REPLICAS,
        ray_actor_options={"num_gpus": WORKER_GPU} if WORKER_GPU and WORKER_GPU > 0 else None,
    )
    class ONNXWorker:
        def __init__(self, model_path: str, tokenizer_dir: str):
            log.info("ONNXWorker initializing...")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found at {model_path}")

            # load tokenizer first so we know abs max
            self.tokenizer = load_local_tokenizer(tokenizer_dir)
            self.abs_max = get_abs_max_from_tokenizer(self.tokenizer)
            self.server_max = max(1, min(int(SERVER_DEFAULT_MAX_LENGTH), self.abs_max))
            log.info("ONNXWorker tokenizer loaded: abs_max=%s server_default=%s", self.abs_max, self.server_max)

            providers = choose_providers()
            self.sess = ort.InferenceSession(model_path, providers=providers)
            log.info("ONNXWorker loaded model inputs=%s outputs=%s", [i.name for i in self.sess.get_inputs()], [o.name for o in self.sess.get_outputs()])

        @serve.batch(max_batch_size=MAX_BATCH, batch_wait_timeout_s=BATCH_WAIT_S)
        async def process(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Each request may be:
             - {"text": "...", "max_length": 512}
             - {"texts": ["a","b"], "max_length": 2048}
            We'll flatten all texts and keep per-text effective max_length.
            For simplicity we tokenize the whole batch with max_length = max(effective_max for batch).
            (Bucketed tokenization would reduce padding; left as an optimization.)
            """
            # flatten and collect per-text max_length
            all_texts: List[str] = []
            per_text_maxes: List[int] = []
            slices: List[Tuple[int,int]] = []

            for req in requests:
                start = len(all_texts)
                # If request is a dict (from ingress), handle accordingly
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
                    # raw string / other
                    requested = None
                    effective_for_req = clamp_max_length(requested, self.server_max, self.abs_max)
                    all_texts.append(str(req))
                    per_text_maxes.append(effective_for_req)
                    slices.append((start, 1))

            if len(all_texts) == 0:
                return [{"embeddings": [], "onnx_output": None, "used_max_length": None} for _ in requests]

            # For now, pick the batch max to tokenize (simple, deterministic)
            batch_max = max(per_text_maxes) if per_text_maxes else self.server_max
            log.info("Batch tokenization: total_texts=%d batch_max=%d", len(all_texts), batch_max)

            tok = self.tokenizer(
                all_texts,
                padding=True,  # pads to longest (here batch_max due to truncation)
                truncation=True,
                max_length=batch_max,
                return_tensors="np",
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            tokenized = {}
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

            try:
                outputs = self.sess.run(None, feed)
                output_names = [o.name for o in self.sess.get_outputs()]
            except Exception as e:
                log.exception("ONNX runtime failed")
                return [{"error": f"ONNX runtime failed: {e}"} for _ in requests]

            # choose embedding output
            chosen = None; chosen_name = None
            for n,out in zip(output_names, outputs):
                nl = n.lower()
                if "pooled" in nl or "sentence" in nl or "cls" in nl:
                    chosen = out; chosen_name = n; break
            if chosen is None:
                for n,out in zip(output_names, outputs):
                    if getattr(out, "ndim", None) == 2:
                        chosen = out; chosen_name = n; break
            if chosen is None:
                chosen = outputs[0]; chosen_name = output_names[0]

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
                    return [{"error":"Unexpected ONNX output shape", "onnx_output": chosen_name} for _ in requests]

            embeddings = np.asarray(embeddings).astype(float)

            # build per-request responses from slices
            results: List[Dict[str, Any]] = []
            idx = 0
            for slice_start, length in slices:
                if length == 0:
                    results.append({"embeddings": [], "onnx_output": chosen_name, "used_max_length": None})
                else:
                    slice_emb = embeddings[slice_start:slice_start+length]
                    # The used max for those texts is per_text_maxes[slice_start:slice_start+length]
                    # For simplicity, return the max used for that request (could return list if multi)
                    used_for_request = int(max(per_text_maxes[slice_start:slice_start+length]))
                    results.append({"embeddings": slice_emb.tolist(), "onnx_output": chosen_name, "used_max_length": used_for_request})
            return results

    # ingress with FastAPI
    app = FastAPI(title="Ray Serve ONNX Ingress")

    @serve.deployment(name="ingress")
    @serve.ingress(app)
    class Ingress:
        def __init__(self, worker_handle):
            self.worker = worker_handle

        @app.get("/health")
        async def health(self):
            return {"status": "ok"}

        @app.post("/embed")
        async def embed(self, payload: EmbedRequest):
            # normalize payload to dict passed to worker
            if payload.texts:
                normalized = {"texts": payload.texts}
            elif payload.text:
                normalized = {"text": payload.text}
            else:
                raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'")

            # include optional per-request max_length
            if payload.max_length is not None:
                normalized["max_length"] = int(payload.max_length)

            try:
                result = await self.worker.process.remote(normalized)
            except Exception as e:
                log.exception("Worker call failed")
                raise HTTPException(status_code=500, detail=str(e))
            return result

    # bind and run
    worker = ONNXWorker.bind(MODEL_PATH, MODEL_DIR)
    ingress = Ingress.bind(worker)
    serve.run(ingress, route_prefix=ROUTE_PREFIX)
    log.info("Ray Serve application deployed (route_prefix=%s)", ROUTE_PREFIX)

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    if not os.path.isdir(MODEL_DIR):
        log.error("MODEL_DIR not found: %s", MODEL_DIR)
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        log.error("MODEL_PATH not found: %s", MODEL_PATH)
        sys.exit(1)

    if RUN_MODE == "ray":
        log.info("Starting Ray Serve mode (RUN_MODE=ray). Make sure ray is installed.")
        try:
            start_ray_serve()
        except Exception as e:
            log.exception("Ray Serve failed to start: %s", e)
            raise
    else:
        log.info("Starting local FastAPI (uvicorn) server at %s:%d", HOST, PORT)
        app = build_fastapi_app()
        import uvicorn
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")



sudo chown -R user:user .git
git config --global user.name "Athithya-Sakthivel"
git config --global user.email "athithyasakthivel1@gmail.com"
git add .
git commit -m "embedder-cpu"
