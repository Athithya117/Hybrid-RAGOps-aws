import os
import time
import json
import logging
import hashlib
import unicodedata
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError
import soundfile as sf
import numpy as np
import tiktoken
from faster_whisper import WhisperModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wav_parser_fw")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").lstrip("/").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
AUDIO_MAX_TOKENS_PER_CHUNK = int(os.getenv("AUDIO_MAX_TOKENS_PER_CHUNK", "800"))
TOKEN_OVERLAP = max(1, int(round(AUDIO_MAX_TOKENS_PER_CHUNK * 0.10)))
S3_RETRIES = int(os.getenv("S3_RETRIES", "4"))
S3_RETRY_BASE = float(os.getenv("S3_RETRY_BASE", "0.5"))
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
FFMPEG = os.getenv("FFMPEG_PATH", "ffmpeg")
FW_COMPUTE = os.getenv("FW_COMPUTE", "int8")
FW_CPU_THREADS = int(os.getenv("FW_CPU_THREADS", str(max(1, (os.cpu_count() or 1) // 2))))
WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "/workspace/models"))
FW_MODEL_PATH = WORKSPACE_MODELS / "faster_whisper" / "faster-whisper-base"
FW_MODEL_BIN = FW_MODEL_PATH / "model.bin"

s3 = boto3.client("s3")
enc: Optional[tiktoken.Encoding] = None
_model: Optional[WhisperModel] = None


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


def run_cmd(cmd: List[str], timeout: int = 60):
    import subprocess
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


def get_encoder():
    global enc
    if enc is not None:
        return enc
    try:
        enc = tiktoken.get_encoding(ENC_NAME)
        return enc
    except Exception as e:
        logger.exception("Failed to load tiktoken encoder '%s': %s", ENC_NAME, e)
        raise


def write_payload_and_upload(text: str, doc_id: str, chunk_id: str, s3_key: str, token_ct: int, start_s: float, end_s: float, parse_ms: int) -> None:
    payload = {
        "document_id": doc_id,
        "chunk_id": chunk_id,
        "chunk_type": "token_window",
        "text": canonicalize_text(text),
        "embedding": None,
        "source": {"file_type": "audio/wav", "source_url": f"s3://{S3_BUCKET}/{s3_key}", "text_checksum": sha256_hex(text)},
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parser_version": "faster-whisper-v1",
            "token_count": token_ct,
            "token_encoder": ENC_NAME,
            "model_compute": FW_COMPUTE,
            "cpu_threads": FW_CPU_THREADS,
            "beam_size": WHISPER_BEAM,
            "chunk_duration_ms": parse_ms
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


def format_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"


def ensure_model_loaded():
    global _model
    if _model is not None:
        return
    if not FW_MODEL_PATH.exists() or not FW_MODEL_BIN.exists():
        raise RuntimeError(f"Missing faster-whisper model at {FW_MODEL_PATH}; ensure model.bin is present")
    try:
        _model = WhisperModel(str(FW_MODEL_PATH), device="cpu", compute_type=FW_COMPUTE, cpu_threads=FW_CPU_THREADS)
        logger.info("Loaded faster-whisper model from %s compute=%s cpu_threads=%d", FW_MODEL_PATH, FW_COMPUTE, FW_CPU_THREADS)
    except Exception:
        _model = WhisperModel(str(FW_MODEL_PATH), device="cpu", compute_type="int8", cpu_threads=FW_CPU_THREADS)
        logger.info("Loaded faster-whisper model with fallback int8")


def read_wav(path: str):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def make_token_chunks_from_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    enc_local = get_encoder()
    token_list: List[int] = []
    token_times: List[float] = []
    for seg in segments:
        seg_text = getattr(seg, "text", "") or ""
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        if seg_start is None or seg_end is None:
            seg_start = 0.0
            seg_end = seg_start + 0.0
        duration = max(0.0, float(seg_end) - float(seg_start))
        tokens = enc_local.encode(seg_text) if seg_text else []
        n = len(tokens)
        if n == 0:
            continue
        for i, t in enumerate(tokens):
            frac = (i + 0.5) / n
            ts = seg_start + frac * duration if duration > 0 else seg_start
            token_list.append(t)
            token_times.append(float(ts))
    if not token_list:
        return [{"text": "", "token_count": 0, "audio_range": [0.0, 0.0], "parse_ms": 0}]
    out: List[Dict[str, Any]] = []
    i = 0
    L = len(token_list)
    while i < L:
        j = min(i + AUDIO_MAX_TOKENS_PER_CHUNK, L)
        slice_tokens = token_list[i:j]
        start_time = token_times[i]
        end_time = token_times[j - 1]
        text = enc_local.decode(slice_tokens)
        out.append({"text": text, "token_count": len(slice_tokens), "audio_range": [float(start_time), float(end_time)], "parse_ms": 0})
        if j == L:
            break
        i = j - TOKEN_OVERLAP
    return out


def parse_file_with_fw(s3_key: str, manifest: dict) -> dict:
    ensure_model_loaded()
    start_all = time.perf_counter()
    try:
        obj = retry_s3(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
    except Exception as e:
        logger.exception("Failed to get s3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    body = obj["Body"].read()
    tmp_dir = Path(tempfile.mkdtemp(prefix="wavproc_fw_"))
    tmp_wav = tmp_dir / Path(s3_key).name
    with open(tmp_wav, "wb") as f:
        f.write(body)
    try:
        audio_array, sr = read_wav(str(tmp_wav))
    except Exception:
        try:
            conv_path = Path(str(tmp_wav) + ".conv.wav")
            run_cmd([FFMPEG, "-y", "-i", str(tmp_wav), "-ar", "16000", "-ac", "1", "-f", "wav", str(conv_path)], timeout=120)
            audio_array, sr = read_wav(str(conv_path))
            tmp_wav = conv_path
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.exception("Failed to read or convert audio %s: %s", s3_key, e)
            return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    t0 = time.perf_counter()
    try:
        segments, info = _model.transcribe(str(tmp_wav), beam_size=WHISPER_BEAM, vad_filter=False)
    except Exception as e:
        logger.exception("Transcription failed for %s with compute=%s: %s", s3_key, FW_COMPUTE, e)
        try:
            segments, info = _model.transcribe(str(tmp_wav), beam_size=1, vad_filter=False)
        except Exception as e2:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.exception("Fallback transcription also failed for %s: %s", s3_key, e2)
            return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    parse_ms = int((time.perf_counter() - t0) * 1000)
    chunks = make_token_chunks_from_segments(segments)

    # Distribute parse_ms across chunks: prefer audio-duration weighting, fallback to token-count weighting.
    total_parse_ms = int(parse_ms)
    total_audio = 0.0
    for c in chunks:
        try:
            start_s = float(c["audio_range"][0])
            end_s = float(c["audio_range"][1])
        except Exception:
            start_s = 0.0
            end_s = 0.0
        dur = max(0.0, end_s - start_s)
        c["_dur_for_weight"] = dur
        total_audio += dur

    if total_audio > 0.0:
        assigned_ms = 0
        for c in chunks:
            dur = c.get("_dur_for_weight", 0.0)
            if dur <= 0.0:
                c["parse_ms"] = 0
            else:
                c["parse_ms"] = int(round(total_parse_ms * (dur / total_audio)))
            assigned_ms += c["parse_ms"]
    else:
        total_tokens = sum(max(0, int(c.get("token_count", 0))) for c in chunks) or 1
        assigned_ms = 0
        for c in chunks:
            tc = max(0, int(c.get("token_count", 0)))
            c["parse_ms"] = int(round(total_parse_ms * (tc / total_tokens))) if tc > 0 else 0
            assigned_ms += c["parse_ms"]

    diff = total_parse_ms - assigned_ms
    if chunks:
        chunks[-1]["parse_ms"] = int(max(0, chunks[-1].get("parse_ms", 0) + diff))

    for c in chunks:
        c.pop("_dur_for_weight", None)

    saved = 0
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(obj.get("LastModified", "")))
    for idx, c in enumerate(chunks):
        chunk_id = f"{doc_id}_{idx+1}"
        try:
            write_payload_and_upload(c["text"], doc_id, chunk_id, s3_key, c["token_count"], c["audio_range"][0], c["audio_range"][1], c["parse_ms"])
            saved += 1
        except Exception:
            logger.exception("Failed to upload chunk %s for %s", chunk_id, s3_key)
            continue
    total_ms = int((time.perf_counter() - start_all) * 1000)
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    logger.info("Completed parsing %d chunks for %s in %d ms total", saved, s3_key, total_ms)
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}


def parse_file(s3_key: str, manifest: dict) -> dict:
    if not S3_BUCKET:
        logger.error("S3_BUCKET not configured; cannot parse %s", s3_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    try:
        return parse_file_with_fw(s3_key, manifest)
    except Exception:
        logger.exception("Unhandled exception in parse_file for %s", s3_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}


__all__ = ["parse_file"]
