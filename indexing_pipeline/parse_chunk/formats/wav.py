import os
import sys
import time
import json
import logging
import hashlib
import unicodedata
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import boto3
from botocore.exceptions import ClientError

try:
    import soundfile as sf
    import numpy as np
    import tiktoken
    import torch
    from transformers import WhisperProcessor
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    import onnxruntime as ort
except Exception as e:
    raise ImportError(f"Missing runtime packages for wav parser: {e}") from e

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wav_parser")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/").lstrip("/").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").lstrip("/").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
PARSER_VERSION = os.getenv("PARSER_VERSION_MP3", "onnx-whisper-base-int8-static-inc-v1")
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "/workspace/models"))
MODEL_DIR = WORKSPACE_MODELS / "onnx" / "whisper-base-int8-static-inc"
FFMPEG = os.getenv("FFMPEG_PATH", "ffmpeg")
AUDIO_SLICE_SECONDS = int(os.getenv("AUDIO_SLICE_SECONDS", "30"))
AUDIO_SLICE_OVERLAP_SEC = float(os.getenv("AUDIO_SLICE_OVERLAP_SEC", "1.0"))
AUDIO_MAX_TOKENS_PER_CHUNK = int(os.getenv("AUDIO_MAX_TOKENS_PER_CHUNK", "800"))
TOKEN_OVERLAP = max(1, int(round(AUDIO_MAX_TOKENS_PER_CHUNK * 0.10)))
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))
ORT_PROVIDER = os.getenv("ORT_PROVIDER", "CPUExecutionProvider")
S3_RETRIES = int(os.getenv("S3_RETRIES", "4"))
S3_RETRY_BASE = float(os.getenv("S3_RETRY_BASE", "0.5"))
GEN_RETRY = int(os.getenv("GEN_RETRY", "1"))

os.environ.setdefault("OMP_NUM_THREADS", str(WHISPER_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(WHISPER_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(WHISPER_CPU_THREADS))
try:
    torch.set_num_threads(WHISPER_CPU_THREADS)
except Exception:
    pass

s3 = boto3.client("s3")
enc = tiktoken.get_encoding(ENC_NAME)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split()).strip()

def retry_s3(fn, retries=S3_RETRIES, base=S3_RETRY_BASE):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 == retries:
                raise
            time.sleep(base * (2 ** i))
    raise last

def run_cmd(cmd: list, timeout: int = 60):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise RuntimeError("command timeout")
    if proc.returncode != 0:
        raise RuntimeError(err.strip())
    return out, err

def ensure_model_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Model dir missing: {path}")
    required = ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx", "tokenizer.json", "vocab.json", "merges.txt"]
    missing = [r for r in required if not (path / r).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files in {path}: {missing}")

def model_sanity_check(model_dir: Path, processor: WhisperProcessor):
    enc_path = str(model_dir / "encoder_model.onnx")
    sess = ort.InferenceSession(enc_path, providers=[ORT_PROVIDER])
    inputs = sess.get_inputs()
    sample = np.zeros((16000,), dtype=np.float32)
    try:
        feats = processor(sample, sampling_rate=16000, return_tensors="np").input_features
    except Exception:
        feats = processor(sample, sampling_rate=16000, return_tensors="pt").input_features.numpy()
    model_input = inputs[0]
    if len(model_input.shape) >= 2 and model_input.shape[-1] not in (None, -1):
        if feats.shape[-1] != model_input.shape[-1]:
            raise RuntimeError("processor feature dim mismatch")

def read_wav(path: str):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr

def slice_audio_array(arr: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start_i = int(round(start_s * sr))
    end_i = int(round(end_s * sr))
    return arr[start_i:end_i]

def format_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"

def write_payload_and_upload(text: str, doc_id: str, chunk_id: str, s3_key: str, token_ct: int, start_s: float, end_s: float, parse_ms: int) -> None:
    payload = {
        "document_id": doc_id,
        "chunk_id": chunk_id,
        "chunk_type": "token_window",
        "text": canonicalize_text(text),
        "embedding": None,
        "source": {
            "file_type": "audio/wav",
            "source_url": f"s3://{S3_BUCKET}/{s3_key}",
            "text_checksum": sha256_hex(text)
        },
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parser_version": PARSER_VERSION,
            "token_count": token_ct,
            "token_encoder": ENC_NAME,
            "model_path": str(MODEL_DIR),
            "model_compute": "int8",
            "cpu_threads": WHISPER_CPU_THREADS,
            "beam_size": WHISPER_BEAM,
            "chunk_length_s": AUDIO_SLICE_SECONDS,
            "parse_chunk_duration_ms": parse_ms
        },
        "audio_range": [format_ts(start_s), format_ts(end_s)]
    }
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{chunk_id}.{ext}"
    if not FORCE_OVERWRITE:
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=out_key)
            logger.info("Skipping existing chunk %s", chunk_id)
            return
        except ClientError:
            pass
    body = (json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" else json.dumps(payload, indent=2, ensure_ascii=False).encode()
    retry_s3(lambda: s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json"))
    logger.info("Wrote chunk %s [%s,%s] → %s", chunk_id, payload["audio_range"][0], payload["audio_range"][1], out_key)

def normalize_pred_ids(pred_ids):
    if torch.is_tensor(pred_ids):
        pred_ids = pred_ids.detach().cpu().numpy()
    if isinstance(pred_ids, np.ndarray):
        if pred_ids.ndim == 2:
            pred_ids = pred_ids[0]
        return pred_ids.tolist()
    if isinstance(pred_ids, list) and len(pred_ids) > 0 and isinstance(pred_ids[0], (list, np.ndarray)):
        first = pred_ids[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        return first
    if isinstance(pred_ids, list):
        return pred_ids
    try:
        return list(map(int, pred_ids))
    except Exception:
        return []

def safe_generate(model, input_features, num_beams: int):
    last = None
    for attempt in range(GEN_RETRY + 1):
        try:
            with torch.no_grad():
                outputs = model.generate(input_features, num_beams=num_beams, max_length=2048)
            return outputs
        except Exception as e:
            last = e
            time.sleep(0.2)
    raise last

def make_token_chunks_from_audio_array(audio_array: np.ndarray, sr: int, model, processor,
                                       slice_sec: int, slice_overlap_sec: float,
                                       audio_max_tokens: int, token_overlap: int):
    duration = len(audio_array) / sr
    step = max(0.001, slice_sec - slice_overlap_sec)
    buffer_tokens: List[int] = []
    buffer_token_slice_starts: List[float] = []
    buffer_token_slice_ends: List[float] = []
    chunks = []
    start = 0.0
    while start < duration:
        end = min(duration, start + slice_sec)
        if end - start < 0.01:
            break
        slice_arr = slice_audio_array(audio_array, sr, start, end)
        if slice_arr.size == 0:
            start += step
            continue
        t0 = time.perf_counter()
        try:
            inputs = processor(slice_arr, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features
            outputs = safe_generate(model, input_features, WHISPER_BEAM)
        except Exception:
            start += step
            continue
        t_ms = int((time.perf_counter() - t0) * 1000)
        pred_ids = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        ids = normalize_pred_ids(pred_ids)
        if not ids:
            start += step
            continue
        try:
            slice_text = processor.batch_decode([ids], skip_special_tokens=True)[0]
        except Exception:
            try:
                slice_text = processor.tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                slice_text = ""
        slice_tokens = enc.encode(slice_text) if slice_text else []
        n = len(slice_tokens)
        if n == 0:
            start += step
            continue
        for idx in range(n):
            buffer_tokens.append(slice_tokens[idx])
            buffer_token_slice_starts.append(start)
            buffer_token_slice_ends.append(end)
        while len(buffer_tokens) >= audio_max_tokens:
            chunk_tokens = buffer_tokens[:audio_max_tokens]
            chunk_slice_starts = buffer_token_slice_starts[:audio_max_tokens]
            chunk_slice_ends = buffer_token_slice_ends[:audio_max_tokens]
            text = enc.decode(chunk_tokens)
            token_ct = len(chunk_tokens)
            chunk_start_s = min(chunk_slice_starts) if chunk_slice_starts else 0.0
            chunk_end_s = max(chunk_slice_ends) if chunk_slice_ends else chunk_start_s
            chunks.append({"text": text, "token_count": token_ct, "audio_range": [chunk_start_s, chunk_end_s], "parse_ms": t_ms})
            after_tokens = buffer_tokens[audio_max_tokens:]
            after_starts = buffer_token_slice_starts[audio_max_tokens:]
            after_ends = buffer_token_slice_ends[audio_max_tokens:]
            tail_tokens = chunk_tokens[-token_overlap:] if token_overlap > 0 else []
            tail_starts = chunk_slice_starts[-token_overlap:] if token_overlap > 0 else []
            tail_ends = chunk_slice_ends[-token_overlap:] if token_overlap > 0 else []
            buffer_tokens = tail_tokens + after_tokens
            buffer_token_slice_starts = tail_starts + after_starts
            buffer_token_slice_ends = tail_ends + after_ends
        start += step
    if buffer_tokens:
        text = enc.decode(buffer_tokens)
        chunk_start_s = min(buffer_token_slice_starts) if buffer_token_slice_starts else 0.0
        chunk_end_s = max(buffer_token_slice_ends) if buffer_token_slice_ends else chunk_start_s
        chunks.append({"text": text, "token_count": len(buffer_tokens), "audio_range": [chunk_start_s, chunk_end_s], "parse_ms": 0})
    return chunks

_processor = None
_model = None

def _ensure_model_loaded():
    global _processor, _model
    if _processor is not None and _model is not None:
        return
    ensure_model_dir(MODEL_DIR)
    _processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))
    _model = ORTModelForSpeechSeq2Seq.from_pretrained(str(MODEL_DIR), provider=ORT_PROVIDER)
    try:
        model_sanity_check(MODEL_DIR, _processor)
    except Exception:
        pass

def parse_file(s3_key: str, manifest: dict) -> dict:
    _ensure_model_loaded()
    start_all = time.perf_counter()
    try:
        obj = retry_s3(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
    except Exception as e:
        logger.exception("Failed to get s3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    body = obj["Body"].read()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wavproc_"))
    tmp_wav = tmp_dir / Path(s3_key).name
    with open(tmp_wav, "wb") as f:
        f.write(body)
    try:
        audio_array, sr = read_wav(str(tmp_wav))
    except Exception:
        try:
            conv_path = Path(str(tmp_wav) + ".conv.wav")
            run_cmd([FFMPEG, "-y", "-i", str(tmp_wav), "-ar", "16000", "-ac", "1", "-f", "wav", str(conv_path)], timeout=60)
            audio_array, sr = read_wav(str(conv_path))
            tmp_wav = conv_path
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.exception("Failed to read or convert audio %s: %s", s3_key, e)
            return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    duration_s = len(audio_array) / sr
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(obj.get("LastModified", "")))
    chunks = make_token_chunks_from_audio_array(
        audio_array,
        sr,
        _model,
        _processor,
        slice_sec=AUDIO_SLICE_SECONDS,
        slice_overlap_sec=AUDIO_SLICE_OVERLAP_SEC,
        audio_max_tokens=AUDIO_MAX_TOKENS_PER_CHUNK,
        token_overlap=TOKEN_OVERLAP,
    )
    saved = 0
    for idx, c in enumerate(chunks):
        chunk_id = f"{doc_id}_{idx+1}"
        t0 = time.perf_counter()
        write_payload_and_upload(c["text"], doc_id, chunk_id, s3_key, c["token_count"], c["audio_range"][0], c["audio_range"][1], int((time.perf_counter() - t0) * 1000))
        saved += 1
    total_ms = int((time.perf_counter() - start_all) * 1000)
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    logger.info("Completed parsing %d chunks for %s in %d ms total", saved, s3_key, total_ms)
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}
