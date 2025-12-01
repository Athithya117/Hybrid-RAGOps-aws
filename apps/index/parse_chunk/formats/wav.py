from __future__ import annotations
import os,time,json,logging,hashlib,unicodedata,tempfile,shutil
from datetime import datetime,timezone
from pathlib import Path
from typing import List,Dict,Any,Optional,Tuple,Generator
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
try:
    import spacy
    from spacy.pipeline import Sentencizer
except Exception:
    spacy = None
    Sentencizer = None
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wav_parser_fw")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").lstrip("/").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        logger.warning("Invalid int for env %s: %r; using default %d", name, v, default)
        return default
MAX_TOKENS_PER_CHUNK = _env_int("MAX_TOKENS_PER_CHUNK", 500)
MIN_TOKENS_PER_CHUNK = _env_int("MIN_TOKENS_PER_CHUNK", 100)
NUMBER_OF_OVERLAPPING_SENTENCES = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES", 2)
TOKEN_OVERLAP = max(1, int(round(MAX_TOKENS_PER_CHUNK * 0.10))) if MAX_TOKENS_PER_CHUNK > 1 else 1
S3_RETRIES = int(os.getenv("S3_RETRIES", "4"))
S3_RETRY_BASE = float(os.getenv("S3_RETRY_BASE", "0.5"))
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
FFMPEG = os.getenv("FFMPEG_PATH", "ffmpeg")
FW_COMPUTE = os.getenv("FW_COMPUTE", "int8")
FW_CPU_THREADS = int(os.getenv("FW_CPU_THREADS", str(max(1, (os.cpu_count() or 1) // 2))))

WORKSPACE_MODELS: Path = Path(os.getenv("WORKSPACE_MODELS", "/indexing_pipeline/models")).resolve()
FW_MODEL_PATH: Path = WORKSPACE_MODELS / "faster_whisper" / "faster-whisper-base"
FW_MODEL_BIN: Path = FW_MODEL_PATH / "model.bin"


PARSER_VERSION = os.getenv("PARSER_VERSION_WAV", "faster-whisper-v1")
s3 = boto3.client("s3")
_model: Optional[Any] = None
def get_tiktoken_encoder_safe():
    if tiktoken is None:
        return (lambda txt: txt.split(), lambda toks: " ".join(toks), "whitespace")
    try:
        enc = None
        try:
            enc = tiktoken.get_encoding(ENC_NAME)
        except Exception:
            try:
                if hasattr(tiktoken, "encoding_for_model"):
                    enc = tiktoken.encoding_for_model("gpt2")
                else:
                    enc = None
            except Exception:
                enc = None
        if enc is not None:
            return (lambda txt: enc.encode(txt), lambda toks: enc.decode(toks), "tiktoken")
    except Exception:
        pass
    return (lambda txt: txt.split(), lambda toks: " ".join(toks), "whitespace")
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
    return Path(s3_key).name or s3_key
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
def make_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    text = text or ""
    if spacy is not None:
        try:
            nlp = spacy.blank("en")
            try:
                nlp.add_pipe("sentencizer")
            except Exception:
                if Sentencizer is not None:
                    nlp.add_pipe(Sentencizer())
                else:
                    nlp.add_pipe("sentencizer")
            doc = nlp(text)
            return [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
        except Exception:
            pass
    pattern = re.compile(r'(?s).*?[\.\!\?]["\']?\s+|.+$')
    items = []
    pos = 0
    for m in pattern.finditer(text):
        s = m.group(0)
        if not s or s.strip() == "":
            pos = m.end()
            continue
        start = pos
        end = pos + len(s)
        items.append((s.strip(), start, end))
        pos = m.end()
    if not items and text.strip():
        items = [(text.strip(), 0, len(text))]
    return items
def sentences_from_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        seg_text = getattr(seg, "text", "") or ""
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        if seg_start is None or seg_end is None:
            continue
        seg_text_clean = seg_text.strip()
        if not seg_text_clean:
            continue
        seg_len = len(seg_text_clean)
        seg_duration = max(0.0, float(seg_end) - float(seg_start))
        sents = make_sentences_with_offsets(seg_text_clean)
        if not sents:
            out.append({"text": seg_text_clean, "start": float(seg_start), "end": float(seg_end)})
            continue
        for s_text, s_sc, s_ec in sents:
            rel_start = max(0, min(seg_len, s_sc))
            rel_end = max(0, min(seg_len, s_ec))
            if seg_len > 0:
                frac_start = rel_start / seg_len
                frac_end = rel_end / seg_len
            else:
                frac_start = 0.0
                frac_end = 1.0
            abs_start = float(seg_start) + frac_start * seg_duration
            abs_end = float(seg_start) + frac_end * seg_duration
            out.append({"text": s_text.strip(), "start": abs_start, "end": abs_end})
    return out
def split_long_sentence_by_tokens(sentence_text: str, sent_start: float, sent_end: float, max_tokens: int, encode_fn, decode_fn) -> List[Dict[str, Any]]:
    enc_tokens = encode_fn(sentence_text)
    if not enc_tokens:
        return []
    L = len(enc_tokens)
    duration = max(0.0, float(sent_end) - float(sent_start))
    out: List[Dict[str, Any]] = []
    i = 0
    while i < L:
        j = min(i + max_tokens, L)
        slice_tokens = enc_tokens[i:j]
        start_frac = (i + 0.0) / L
        end_frac = (j - 1 + 0.5) / L
        abs_start = sent_start + start_frac * duration
        abs_end = sent_start + min(1.0, end_frac) * duration
        try:
            text = decode_fn(slice_tokens)
        except Exception:
            text = " ".join(str(x) for x in slice_tokens)
        out.append({"text": text, "start": float(abs_start), "end": float(abs_end), "token_count": len(slice_tokens)})
        if j == L:
            break
        i = j
    return out
def make_sentence_chunks_from_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    encode_fn, decode_fn, backend = get_tiktoken_encoder_safe()
    max_tokens = _env_int("MAX_TOKENS_PER_CHUNK", MAX_TOKENS_PER_CHUNK)
    min_tokens = _env_int("MIN_TOKENS_PER_CHUNK", MIN_TOKENS_PER_CHUNK)
    overlap_sentences = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES", NUMBER_OF_OVERLAPPING_SENTENCES)
    sentences = sentences_from_segments(segments)
    sent_items: List[Dict[str, Any]] = []
    for i, s in enumerate(sentences):
        txt = s.get("text", "") or ""
        st = s.get("start", 0.0)
        ed = s.get("end", 0.0)
        sent_items.append({"text": txt, "start": float(st), "end": float(ed), "orig_idx": i, "is_remainder": False})
    out_chunks: List[Dict[str, Any]] = []
    i = 0
    n = len(sent_items)
    prev_chunk: Optional[Dict[str, Any]] = None
    while i < n:
        cur_token_count = 0
        chunk_sent_texts: List[str] = []
        chunk_start_idx = i
        chunk_start_time = sent_items[i]["start"] if i < n else None
        chunk_end_time = None
        is_truncated_sentence = False
        while i < n:
            sent_text = sent_items[i]["text"]
            sent_start = sent_items[i]["start"]
            sent_end = sent_items[i]["end"]
            tok_ids = encode_fn(sent_text)
            sent_tok_len = len(tok_ids)
            if cur_token_count + sent_tok_len > max_tokens:
                if not chunk_sent_texts:
                    pieces = split_long_sentence_by_tokens(sent_text, sent_start, sent_end, max_tokens, encode_fn, decode_fn)
                    if not pieces:
                        i += 1
                        break
                    first = pieces[0]
                    chunk_sent_texts.append(first["text"])
                    cur_token_count += int(first.get("token_count", 0))
                    chunk_end_time = first["end"]
                    is_truncated_sentence = True
                    if len(pieces) > 1:
                        remainders = []
                        for p in pieces[1:]:
                            remainders.append({"text": p["text"], "start": p["start"], "end": p["end"], "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True})
                        sent_items[i] = {"text": "", "start": None, "end": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True}
                        sent_items[i+1:i+1] = remainders
                    else:
                        i += 1
                    break
                else:
                    break
            else:
                chunk_sent_texts.append(sent_text)
                cur_token_count += sent_tok_len
                chunk_end_time = sent_end
                i += 1
        if not chunk_sent_texts:
            logger.warning("Empty chunk at idx %d; advancing to avoid infinite loop", i)
            i += 1
            continue
        chunk_text = " ".join(chunk_sent_texts).strip()
        chunk_meta = {"text": chunk_text, "token_count": cur_token_count, "start_time": chunk_start_time, "end_time": chunk_end_time, "start_sentence_idx": chunk_start_idx, "end_sentence_idx": i, "is_truncated_sentence": is_truncated_sentence}
        new_start = max(chunk_start_idx + 1, chunk_meta["end_sentence_idx"] - overlap_sentences)
        if prev_chunk is None:
            prev_chunk = chunk_meta
        else:
            if chunk_meta["token_count"] < min_tokens:
                prev_chunk["text"] = (prev_chunk["text"] + " " + chunk_meta["text"]).strip()
                prev_chunk["token_count"] = prev_chunk["token_count"] + chunk_meta["token_count"]
                prev_chunk["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
                prev_chunk["end_time"] = chunk_meta["end_time"]
                prev_chunk["is_truncated_sentence"] = prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
            else:
                out_chunks.append(prev_chunk)
                prev_chunk = chunk_meta
        i = new_start
        n = len(sent_items)
    if prev_chunk is not None:
        out_chunks.append(prev_chunk)
    formatted: List[Dict[str, Any]] = []
    for c in out_chunks:
        st = c.get("start_time", 0.0) or 0.0
        ed = c.get("end_time", 0.0) or 0.0
        formatted.append({"text": c.get("text", "") or "", "token_count": int(c.get("token_count", 0) or 0), "audio_range": [float(st), float(ed)], "parse_ms": 0})
    return formatted
def make_token_chunks_from_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    encode_fn = get_tiktoken_encoder_safe()[0]
    token_list: List[Any] = []
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
        tokens = encode_fn(seg_text)
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
        j = min(i + MAX_TOKENS_PER_CHUNK, L)
        slice_tokens = token_list[i:j]
        start_time = token_times[i]
        end_time = token_times[j - 1]
        text = get_tiktoken_encoder_safe()[1](slice_tokens)
        out.append({"text": text, "token_count": len(slice_tokens), "audio_range": [float(start_time), float(end_time)], "parse_ms": 0})
        if j == L:
            break
        i = j - TOKEN_OVERLAP
    return out
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
    chunks = make_sentence_chunks_from_segments(segments)
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
                "figures": "[]",
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
