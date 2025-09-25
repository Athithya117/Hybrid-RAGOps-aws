#!/usr/bin/env python3
from __future__ import annotations
import os
import time
import json
import logging
import hashlib
import unicodedata
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
import soundfile as sf
import numpy as np

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

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
WORKSPACE_MODELS: Path = Path(os.getenv("WORKSPACE_MODELS", "/workspace/models")).resolve()
FW_MODEL_PATH: Path = WORKSPACE_MODELS / "faster_whisper" / "faster-whisper-base"
FW_MODEL_BIN: Path = FW_MODEL_PATH / "model.bin"
PARSER_VERSION = os.getenv("PARSER_VERSION_WAV", "faster-whisper-v1")

s3 = boto3.client("s3")
enc: Optional[Any] = None
_model: Optional[Any] = None

def get_encoder():
    global enc
    if enc is not None:
        return enc
    if tiktoken is None:
        raise RuntimeError("tiktoken not available")
    enc = tiktoken.get_encoding(ENC_NAME)
    return enc

def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def canonicalize_text(s: Any) -> str:
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

def format_ts_ms(seconds: float) -> str:
    ms = int(round(max(0.0, float(seconds)) * 1000.0))
    s_total, msecs = divmod(ms, 1000)
    h, rem = divmod(s_total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{int(h):02d}:{int(mm):02d}:{int(ss):02d}.{int(msecs):03d}"

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
            continue
        duration = max(0.0, float(seg_end) - float(seg_start))
        if not seg_text:
            continue
        tokens = enc_local.encode(seg_text)
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
        text = get_encoder().decode(slice_tokens)
        out.append({"text": text, "token_count": len(slice_tokens), "audio_range": [float(start_time), float(end_time)], "parse_ms": 0})
        if j == L:
            break
        i = j - TOKEN_OVERLAP
    return out

def ensure_model_loaded():
    global _model
    if _model is not None:
        return
    if WhisperModel is None:
        raise RuntimeError("faster_whisper not available")
    if not FW_MODEL_PATH.exists() or not FW_MODEL_BIN.exists():
        raise RuntimeError(f"Missing faster-whisper model at {FW_MODEL_PATH}; ensure model.bin is present")
    _model = WhisperModel(str(FW_MODEL_PATH), device="cpu", compute_type=FW_COMPUTE, cpu_threads=FW_CPU_THREADS)
    logger.info("Loaded faster-whisper model from %s compute=%s cpu_threads=%d", FW_MODEL_PATH, FW_COMPUTE, FW_CPU_THREADS)

def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False

class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f".{ext}", dir="/tmp")
        self.count = 0
        self._first = True
        if self.ext == "json":
            self.temp.write(b"[\n")
            self.temp.flush()
    def write_payload(self, payload: Dict[str, Any]) -> int:
        self.count += 1
        if self.ext == "jsonl":
            line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.temp.write(line)
        else:
            pretty = json.dumps(payload, ensure_ascii=False, indent=2)
            indented = ("\n".join("  " + ln for ln in pretty.splitlines()) + "\n").encode("utf-8")
            if not self._first:
                self.temp.write(b",\n")
            self.temp.write(indented)
            self._first = False
        self.temp.flush()
        return 1
    def finalize_and_upload(self, out_key: str) -> Tuple[int, str]:
        if self.ext == "json":
            self.temp.write(b"]\n")
        self.temp.flush()
        self.temp.close()
        try:
            s3.upload_file(self.temp.name, S3_BUCKET, out_key, ExtraArgs={"ContentType": self.content_type})
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            return self.count, out_key
        except Exception:
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            raise

def _derive_file_name_from_s3_key(s3_key: str) -> str:
    return os.path.basename(s3_key) or s3_key

def sanitize_payload_for_weaviate(payload: Dict[str, Any]) -> None:
    for k in list(payload.keys()):
        v = payload.get(k)
        if k == "tags":
            if v is None:
                payload[k] = []
            elif isinstance(v, (list, tuple)):
                payload[k] = [str(x) for x in v]
            else:
                payload[k] = [str(v)]
            continue
        if v is None:
            payload.pop(k, None)
            continue
        if isinstance(v, (list, tuple, dict)):
            try:
                payload[k] = json.dumps(v)
            except Exception:
                payload[k] = str(v)
            continue
        if not isinstance(v, (str, int, float, bool)):
            payload[k] = str(v)

def parse_file_with_fw(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()

    try:
        head_obj = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        logger.exception("Failed to head s3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}
    last_modified = head_obj.get("LastModified", "")
    doc_id = manifest.get("file_hash") or sha256_hex(s3_key + str(last_modified or ""))
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"

    if not FORCE_OVERWRITE:
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=out_key)
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("Skipping entire file because chunked file exists: %s", out_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
        except ClientError:
            pass
        except Exception:
            pass

    try:
        ensure_model_loaded()
    except Exception as e:
        logger.exception("Model load failed (aborting): %s", e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": "model-load-failed"}

    try:
        obj = retry_s3(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
        body = obj["Body"].read()
    except Exception as e:
        logger.exception("Failed to get s3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0}

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
            c["parse_ms"] = int(round(total_parse_ms * (dur / total_audio))) if dur > 0 else 0
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
    ext_out = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_key, ext=ext_out)
    file_name = manifest.get("file_name") if isinstance(manifest, dict) and manifest.get("file_name") else _derive_file_name_from_s3_key(s3_key)

    try:
        for idx, c in enumerate(chunks):
            chunk_id = f"{doc_id}_{idx+1}"
            start_s, end_s = c.get("audio_range", [0.0, 0.0])
            audio_range = [format_ts_ms(float(start_s)), format_ts_ms(float(end_s))] if start_s is not None and end_s is not None else None

            payload = {
                "document_id": doc_id or "",
                "file_name": file_name,
                "chunk_id": chunk_id or "",
                "chunk_type": "audio",
                "text": canonicalize_text(c.get("text", "") or ""),
                "token_count": int(c.get("token_count", 0) or 0),
                "embedding": None,
                "file_type": "audio/wav",
                "source_url": f"s3://{S3_BUCKET}/{s3_key}" if S3_BUCKET else None,
                "page_number": None,
                "slide_range": None,
                "row_range": None,
                "token_range": None,
                "audio_range": audio_range if audio_range is not None else None,
                "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "parser_version": PARSER_VERSION or "",
                "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                "layout_tags": [],
                "used_ocr": False,
                "heading_path": [],
                "headings": [],
                "line_range": None
            }

            sanitize_payload_for_weaviate(payload)

            writer.write_payload(payload)
            saved += 1
            logger.info("Buffered chunk %s %s", chunk_id, audio_range)
    except Exception as e:
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.exception("Error while buffering chunks for %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}

    try:
        if saved == 0:
            try:
                if writer and writer.temp:
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("No chunks produced for %s", s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Wrote %d chunks for %s → %s (%d ms total)", count, s3_key, uploaded_key, total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error("Failed to upload chunked file for %s error=%s", s3_key, str(e_up))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

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
