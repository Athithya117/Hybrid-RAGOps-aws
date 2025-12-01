from __future__ import annotations
import os, io, sys, json, time, logging, hashlib, boto3, unicodedata, re, tempfile, importlib
from datetime import datetime
from botocore.exceptions import ClientError
from typing import List, Tuple, Dict, Generator, Optional, Any
from contextlib import contextmanager
from PIL import Image
import spacy
try:
    from spacy.pipeline import Sentencizer
except Exception:
    Sentencizer = None
logger = logging.getLogger("pdf_parser")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers[:] = [handler]
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        logger.warning("Invalid env var for %s: %r; falling back to %d", name, v, default)
        return default
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f"):
        return False
    return default
S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
PDF_DISABLE_OCR = _env_bool("PDF_DISABLE_OCR", False)
PDF_FORCE_OCR = _env_bool("PDF_FORCE_OCR", False)
PDF_OCR_ENGINE = os.getenv("PDF_OCR_ENGINE", "auto").lower()
PDF_OCR_STRICT = os.getenv("PDF_OCR_STRICT", "false").lower() == "true"
PDF_TESSERACT_LANG = os.getenv("PDF_TESSERACT_LANG", "eng")
PDF_OCR_RENDER_DPI = _env_int("PDF_OCR_RENDER_DPI", 300)
PDF_MIN_IMG_SIZE_BYTES = _env_int("PDF_MIN_IMG_SIZE_BYTES", 3072)
MAX_TOKENS_PER_CHUNK = _env_int("MAX_TOKENS_PER_CHUNK", 512)
MIN_TOKENS_PER_CHUNK = _env_int("MIN_TOKENS_PER_CHUNK", 100)
NUMBER_OF_OVERLAPPING_SENTENCES = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES", 2)
PARSER_VERSION_PDF = os.getenv("PARSER_VERSION_PDF", "pdf-v1")
S3_PUT_RETRIES = _env_int("S3_PUT_RETRIES", 3)
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", "0.3"))
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
s3 = boto3.client("s3")
try:
    import tiktoken
    enc = tiktoken.get_encoding(ENC_NAME)
except Exception:
    enc = None
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def local_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r'[ \t]+$', '', ln) for ln in s.split("\n")]
    return "\n".join(lines).strip()
def try_decode_bytes(b: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return b.decode(encoding)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")
def token_count_for(text: str) -> int:
    if not text:
        return 0
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())
def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        return False
    except Exception:
        return False
def s3_put_object_with_retries(key: str, body: bytes, content_type: str = "application/json") -> None:
    for attempt in range(1, S3_PUT_RETRIES + 1):
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)
            return
        except Exception:
            if attempt < S3_PUT_RETRIES:
                time.sleep(S3_PUT_BACKOFF * attempt)
    raise Exception("s3 put failed after retries")
def _derive_doc_id_from_head(s3_key: str, head_obj: dict, manifest: dict) -> str:
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        return manifest.get("file_hash")
    etag = head_obj.get("ETag", "") if isinstance(head_obj, dict) else ""
    if isinstance(etag, str):
        etag = etag.strip('"')
    if etag:
        return sha256_hex(s3_key + str(etag))
    lm = head_obj.get("LastModified", "") if isinstance(head_obj, dict) else ""
    if lm:
        return sha256_hex(s3_key + str(lm))
    base = os.path.basename(s3_key)
    if base:
        return base
    return sha256_hex(s3_key)
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
class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        tmpdir = os.getenv("TMPDIR") or None
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f".{ext}", dir=tmpdir)
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
            pretty = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
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
@contextmanager
def without_cwd_on_syspath():
    saved = list(sys.path)
    try:
        cwd = os.getcwd()
        sys.path = [p for p in sys.path if p not in ("", cwd)]
        yield
    finally:
        sys.path[:] = saved
class TokenEncoder:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.backend = "whitespace"
        try:
            import tiktoken
            enc_local = tiktoken.encoding_for_model(model_name) if hasattr(tiktoken, "encoding_for_model") else tiktoken.get_encoding(model_name)
            self.encode = lambda txt: enc_local.encode(txt)
            self.decode = lambda toks: enc_local.decode(toks)
            self.backend = "tiktoken"
        except Exception:
            self.encode = lambda txt: txt.split()
            self.decode = lambda toks: " ".join(toks)
def split_long_sentence_by_words(sent_text: str, max_tokens: int, encoder: TokenEncoder) -> List[str]:
    words = sent_text.split()
    pieces: List[str] = []
    cur_words: List[str] = []
    cur_tok = 0
    for w in words:
        toks = encoder.encode(w)
        l = len(toks)
        if cur_tok + l > max_tokens:
            if cur_words:
                pieces.append(" ".join(cur_words))
                cur_words = []
                cur_tok = 0
            if l > max_tokens:
                tok_ids = encoder.encode(w)
                i = 0
                while i < len(tok_ids):
                    chunk_ids = tok_ids[i:i + max_tokens]
                    pieces.append(encoder.decode(chunk_ids))
                    i += max_tokens
                continue
        cur_words.append(w)
        cur_tok += l
    if cur_words:
        pieces.append(" ".join(cur_words))
    return pieces
class SentenceChunker:
    def __init__(self, max_tokens_per_chunk: Optional[int] = None, overlap_sentences: Optional[int] = None, token_model: str = "gpt2", nlp=None, min_tokens_per_chunk: Optional[int] = None):
        self.max_tokens_per_chunk = _env_int("MAX_TOKENS_PER_CHUNK", MAX_TOKENS_PER_CHUNK) if max_tokens_per_chunk is None else int(max_tokens_per_chunk)
        self.overlap_sentences = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES", NUMBER_OF_OVERLAPPING_SENTENCES) if overlap_sentences is None else int(overlap_sentences)
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.min_tokens_per_chunk = _env_int("MIN_TOKENS_PER_CHUNK", MIN_TOKENS_PER_CHUNK) if min_tokens_per_chunk is None else int(min_tokens_per_chunk)
        if self.min_tokens_per_chunk < 0:
            raise ValueError("min_tokens_per_chunk must be >= 0")
        self.encoder = TokenEncoder(model_name=token_model)
        self.nlp = nlp or self._make_sentencizer()
    @staticmethod
    def _make_sentencizer():
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            try:
                if Sentencizer is not None:
                    nlp.add_pipe("sentencizer")
                else:
                    nlp.add_pipe("sentencizer")
            except Exception:
                try:
                    nlp.add_pipe(Sentencizer())
                except Exception:
                    raise RuntimeError("Failed to add Sentencizer to spaCy pipeline.")
            return nlp
    def _sentences_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        doc = self.nlp(text)
        sents = [(sent.text.strip(), int(sent.start_char), int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
        return sents
    def chunk_document(self, text: str) -> Generator[Dict, None, None]:
        sentences = self._sentences_with_offsets(text)
        sent_items: List[Dict] = [{"text": s, "start_char": sc, "end_char": ec, "orig_idx": i, "is_remainder": False} for i, (s, sc, ec) in enumerate(sentences)]
        i = 0
        n = len(sent_items)
        prev_chunk = None
        while i < n:
            cur_token_count = 0
            chunk_sent_texts: List[str] = []
            chunk_start_idx = i
            chunk_start_char: Optional[int] = sent_items[i]["start_char"] if i < n else None
            chunk_end_char: Optional[int] = None
            is_truncated_sentence = False
            while i < n:
                sent_text = sent_items[i]["text"]
                tok_ids = self.encoder.encode(sent_text)
                sent_tok_len = len(tok_ids)
                if sent_tok_len > self.max_tokens_per_chunk:
                    pieces = split_long_sentence_by_words(sent_text, self.max_tokens_per_chunk, self.encoder)
                    if not pieces:
                        pieces = [sent_text[:1000]]
                    sent_items[i]["text"] = pieces[0]
                    for j, rem in enumerate(pieces[1:], start=1):
                        insert_idx = i + j
                        sent_items.insert(insert_idx, {"text": rem, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True})
                    n = len(sent_items)
                    tok_ids = self.encoder.encode(sent_items[i]["text"])
                    sent_tok_len = len(tok_ids)
                if cur_token_count + sent_tok_len > self.max_tokens_per_chunk:
                    if not chunk_sent_texts:
                        prefix_tok_ids = tok_ids[: self.max_tokens_per_chunk]
                        prefix_text = self.encoder.decode(prefix_tok_ids)
                        chunk_sent_texts.append(prefix_text)
                        cur_token_count = len(prefix_tok_ids)
                        is_truncated_sentence = True
                        remainder_tok_ids = tok_ids[self.max_tokens_per_chunk : ]
                        if remainder_tok_ids:
                            remainder_text = self.encoder.decode(remainder_tok_ids)
                            sent_items[i] = {"text": remainder_text, "start_char": None, "end_char": None, "orig_idx": sent_items[i]["orig_idx"], "is_remainder": True}
                        else:
                            i += 1
                        break
                    else:
                        break
                else:
                    chunk_sent_texts.append(sent_text)
                    cur_token_count += sent_tok_len
                    chunk_end_char = sent_items[i]["end_char"]
                    i += 1
            if not chunk_sent_texts:
                i += 1
                continue
            chunk_text = " ".join(chunk_sent_texts).strip()
            chunk_meta = {"text": chunk_text, "token_count": cur_token_count, "start_sentence_idx": chunk_start_idx, "end_sentence_idx": i, "start_char": chunk_start_char, "end_char": chunk_end_char, "is_truncated_sentence": is_truncated_sentence}
            new_start = max(chunk_start_idx + 1, chunk_meta["end_sentence_idx"] - self.overlap_sentences)
            if prev_chunk is None:
                prev_chunk = chunk_meta
            else:
                if chunk_meta["token_count"] < self.min_tokens_per_chunk:
                    prev_chunk["text"] = prev_chunk["text"] + " " + chunk_meta["text"]
                    prev_chunk["token_count"] = prev_chunk["token_count"] + chunk_meta["token_count"]
                    prev_chunk["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
                    prev_chunk["end_char"] = chunk_meta["end_char"]
                    prev_chunk["is_truncated_sentence"] = prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
                else:
                    yield prev_chunk
                    prev_chunk = chunk_meta
            i = new_start
            n = len(sent_items)
        if prev_chunk is not None:
            yield prev_chunk
    @classmethod
    def from_env(cls, **kwargs):
        max_tokens = _env_int("MAX_TOKENS_PER_CHUNK", MAX_TOKENS_PER_CHUNK)
        overlap = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES", NUMBER_OF_OVERLAPPING_SENTENCES)
        min_tokens = _env_int("MIN_TOKENS_PER_CHUNK", MIN_TOKENS_PER_CHUNK)
        token_model = os.getenv("TOKEN_ENCODER_MODEL", os.getenv("TOKEN_ENCODER", "gpt2"))
        return cls(max_tokens_per_chunk=max_tokens, overlap_sentences=overlap, token_model=token_model, nlp=None, min_tokens_per_chunk=min_tokens)
def import_fitz_local():
    with without_cwd_on_syspath():
        try:
            return importlib.import_module("fitz")
        except Exception:
            return importlib.import_module("pymupdf")
def import_pdfplumber():
    with without_cwd_on_syspath():
        return importlib.import_module("pdfplumber")
def crop_page_to_pil_and_bytes(page, bbox: Tuple[float, float, float, float], dpi: int = PDF_OCR_RENDER_DPI) -> Tuple[Image.Image, bytes]:
    fitz = import_fitz_local()
    rect = fitz.Rect(bbox)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    png_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(png_bytes))
    return img, png_bytes
def run_ocr_on_pil_image(engine_name: str, engine_obj, pil_img: Image.Image) -> str:
    if engine_name == "rapidocr" and engine_obj is not None:
        try:
            import numpy as np
            import cv2
            img_arr = None
            if isinstance(pil_img, Image.Image):
                img_arr = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
            elif isinstance(pil_img, (bytes, bytearray)):
                nparr = np.frombuffer(pil_img, np.uint8)
                img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                try:
                    img_arr = np.asarray(pil_img)
                except Exception:
                    img_arr = None
            if img_arr is None:
                logger.error("RapidOCR input image conversion failed")
                return ""
            res = engine_obj(img_arr)
            if isinstance(res, tuple) and len(res) >= 1:
                ocr_result = res[0]
            else:
                ocr_result = res
            lines: List[str] = []
            if isinstance(ocr_result, list):
                for item in ocr_result:
                    if isinstance(item, dict) and "text" in item:
                        txt = item.get("text") or item.get("rec") or ""
                        if txt:
                            lines.append(str(txt))
                            continue
                    if isinstance(item, (list, tuple)):
                        found = False
                        for element in item:
                            if isinstance(element, str) and element.strip():
                                lines.append(element.strip())
                                found = True
                                break
                            if isinstance(element, (list, tuple)) and element and isinstance(element[0], str):
                                lines.append(element[0].strip())
                                found = True
                                break
                        if found:
                            continue
                        try:
                            joined = " ".join([str(x) for x in item if x is not None])
                            if joined.strip():
                                lines.append(joined.strip())
                                continue
                        except Exception:
                            pass
                    try:
                        s = str(item)
                        if s and s.strip():
                            lines.append(s.strip())
                    except Exception:
                        pass
            else:
                try:
                    s = str(ocr_result)
                    if s and s.strip():
                        lines.append(s.strip())
                except Exception:
                    pass
            return "\n".join([ln for ln in lines if ln])
        except Exception:
            logger.exception("RapidOCR failed to OCR image")
            return ""
    if engine_name == "tesseract" and engine_obj is not None:
        try:
            pytesseract = engine_obj
            return pytesseract.image_to_string(pil_img, lang=PDF_TESSERACT_LANG)
        except Exception:
            logger.exception("Tesseract OCR failed to OCR image")
            return ""
    return ""
def _create_rapidocr_engine(model_dir: str|None = None):
    models_path = model_dir or os.getenv("RAPIDOCR_MODEL_DIR", "/opt/models/rapidocr")
    tried = []
    last_exc = None
    candidates = ("rapidocr_onnxruntime","rapidocr")
    for module_name in candidates:
        try:
            with without_cwd_on_syspath():
                mod = importlib.import_module(module_name)
            RapidOCR = getattr(mod, "RapidOCR", None)
            if RapidOCR is None:
                raise ImportError(f"module {module_name} does not expose RapidOCR")
            try:
                eng = RapidOCR(model_dir=models_path)
            except TypeError:
                eng = RapidOCR(models_path)
            return eng
        except Exception as e:
            tried.append((module_name, repr(e)))
            last_exc = e
    raise ImportError("RapidOCR import failed; tried: " + "; ".join(f"{m}:{err}" for m,err in tried)) from last_exc
def get_pdf_image_ocr_engine():
    if PDF_DISABLE_OCR and not PDF_FORCE_OCR:
        logger.info("PDF_DISABLE_OCR=true and PDF_FORCE_OCR=false -> skipping OCR")
        return "none", None
    choice = (PDF_OCR_ENGINE or "auto").lower()
    if choice == "rapidocr":
        try:
            eng = _create_rapidocr_engine()
            logger.info("Using RapidOCR model_dir=%s", os.getenv("RAPIDOCR_MODEL_DIR", "/opt/models/rapidocr"))
            return "rapidocr", eng
        except Exception as e:
            logger.exception("Requested RapidOCR but import/create failed: %s", e)
            if PDF_OCR_STRICT or PDF_FORCE_OCR:
                raise
            return "none", None
    if choice == "tesseract":
        try:
            with without_cwd_on_syspath():
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
                logger.info("Using Tesseract OCR")
                return "tesseract", pytesseract
        except Exception as e:
            logger.exception("Requested Tesseract but import failed: %s", e)
            if PDF_OCR_STRICT or PDF_FORCE_OCR:
                raise
            return "none", None
    if choice == "auto":
        try:
            eng = _create_rapidocr_engine()
            logger.info("Auto-selected RapidOCR model_dir=%s", os.getenv("RAPIDOCR_MODEL_DIR", "/opt/models/rapidocr"))
            return "rapidocr", eng
        except Exception as e_rapid:
            logger.warning("RapidOCR auto-select failed: %s", repr(e_rapid))
            try:
                with without_cwd_on_syspath():
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
                    logger.info("Auto-selected Tesseract")
                    return "tesseract", pytesseract
            except Exception as e_tess:
                logger.warning("Tesseract auto-select failed: %s", repr(e_tess))
                try:
                    import onnxruntime
                except Exception:
                    logger.error("onnxruntime import failed; if you want RapidOCR install onnxruntime or onnxruntime-gpu (pip).")
                logger.error("No OCR engine available. OCR will be skipped.")
                return "none", None
    try:
        eng = _create_rapidocr_engine()
        logger.info("Fallback RapidOCR")
        return "rapidocr", eng
    except Exception as e:
        logger.exception("Fallback RapidOCR failed: %s", e)
        return "none", None
def rect_area(rect: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)
def intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)
def overlap_fraction(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    a_area = rect_area(a)
    if a_area == 0:
        return 0.0
    return intersection_area(a, b) / a_area
def cluster_blocks_into_columns(blocks: List[dict], gap_multiplier: float = 1.5) -> List[List[dict]]:
    if not blocks:
        return []
    centers = [((b["bbox"][0] + b["bbox"][2]) / 2.0, i) for i, b in enumerate(blocks)]
    centers.sort(key=lambda x: x[0])
    xs = [c for c, _ in centers]
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)] or [0]
    med_gap = sorted(gaps)[len(gaps)//2] if gaps else 0
    if med_gap == 0:
        med_gap = max(gaps) if gaps else 50
    split_indices = []
    for idx, g in enumerate(gaps):
        if g > med_gap * gap_multiplier:
            split_indices.append(idx)
    groups = []
    start = 0
    for si in split_indices:
        group_idxs = [centers[j][1] for j in range(start, si+1)]
        groups.append([blocks[k] for k in group_idxs])
        start = si + 1
    group_idxs = [centers[j][1] for j in range(start, len(centers))]
    groups.append([blocks[k] for k in group_idxs])
    return groups
def assemble_column_text(column_blocks: List[dict]) -> str:
    if not column_blocks:
        return ""
    col_sorted = sorted(column_blocks, key=lambda b: b["bbox"][1])
    pieces = []
    prev_y = None
    for b in col_sorted:
        y0 = b["bbox"][1]
        if prev_y is None or (y0 - prev_y) > 50:
            pieces.append(b["text"].strip())
        else:
            pieces.append(" " + b["text"].strip())
        prev_y = b["bbox"][3]
    return "\n\n".join([p.strip() for p in "".join(pieces).split("\n\n") if p.strip()])
def reflow_and_clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    text = re.sub(r'[\x00-\x1F]+', ' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def extract_page_clean_and_figures(pdf_path: str, pageno: int, overlap_threshold: float = 0.25, image_ocr_engine_name: str = "none", image_ocr_engine_obj=None):
    fitz = import_fitz_local()
    pdfplumber = import_pdfplumber()
    doc = fitz.open(pdf_path)
    plumb = pdfplumber.open(pdf_path)
    try:
        page = doc[pageno]
    except Exception:
        plumb.close(); doc.close(); raise
    p_plumb = plumb.pages[pageno]
    table_bboxes: List[Tuple[float, float, float, float]] = []
    tables = []
    try:
        tables = p_plumb.find_tables()
    except Exception:
        tables = []
    for t in tables:
        try:
            table_bboxes.append(tuple(t.bbox))
        except Exception:
            pass
    blocks = page.get_text("dict").get("blocks", [])
    text_blocks: List[dict] = []
    image_bboxes: List[Tuple[float, float, float, float]] = []
    for b in blocks:
        if b.get("type") == 0:
            bbox = tuple(b.get("bbox"))
            text = ""
            for line in b.get("lines", []):
                spans = [s.get("text", "") for s in line.get("spans", [])]
                text += " ".join(spans) + "\n"
            text_blocks.append({"bbox": bbox, "text": text.strip()})
        elif b.get("type") == 1:
            bbox = tuple(b.get("bbox"))
            try:
                img, png_bytes = crop_page_to_pil_and_bytes(page, bbox, dpi=PDF_OCR_RENDER_DPI)
                if len(png_bytes) >= PDF_MIN_IMG_SIZE_BYTES:
                    image_bboxes.append(bbox)
            except Exception:
                pass
    figure_bboxes = table_bboxes + image_bboxes
    caption_map = {}
    content_blocks = []
    for tb in text_blocks:
        tb_bbox = tb["bbox"]
        overlapped = False
        for fb in figure_bboxes:
            if overlap_fraction(tb_bbox, fb) > overlap_threshold:
                overlapped = True
                if tb_bbox[1] >= fb[3] and (tb_bbox[1] - fb[3]) < 80:
                    caption_map.setdefault(fb, []).append(tb["text"])
                break
        if not overlapped:
            content_blocks.append(tb)
    columns = cluster_blocks_into_columns(content_blocks)
    col_texts = [assemble_column_text(col) for col in columns]
    clean_text = "\n\n".join([ct for ct in col_texts if ct]).strip()
    clean_text = reflow_and_clean_text(clean_text)
    figures_texts: List[str] = []
    for t in tables:
        try:
            rows = t.extract()
            if rows:
                lines: List[str] = []
                for row in rows:
                    lines.append("\t".join([str(c) if c is not None else "" for c in row]))
                figures_texts.append("\n".join(lines))
        except Exception:
            pass
    processed_bboxes = [tuple(t.bbox) for t in tables] if tables else []
    for fb in image_bboxes:
        if fb in processed_bboxes:
            continue
        try:
            pil_img, png_bytes = crop_page_to_pil_and_bytes(page, fb, dpi=PDF_OCR_RENDER_DPI)
            if len(png_bytes) < PDF_MIN_IMG_SIZE_BYTES:
                continue
            ocr_text = run_ocr_on_pil_image(image_ocr_engine_name, image_ocr_engine_obj, pil_img)
            caption_list = caption_map.get(fb, [])
            caption_text = "\n".join(caption_list) if caption_list else ""
            combined = (caption_text + "\n" + ocr_text).strip() if caption_text else ocr_text.strip()
            if combined:
                combined = reflow_and_clean_text(combined)
                figures_texts.append(combined)
        except Exception:
            pass
    plumb.close(); doc.close()
    return clean_text, figures_texts
def download_s3_object_to_temp(s3_key: str) -> str:
    tmpdir = os.getenv("TMPDIR") or None
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tmpdir)
    try:
        s3.download_fileobj(Bucket=S3_BUCKET, Key=s3_key, Fileobj=tf)
    finally:
        tf.flush(); tf.close()
    return tf.name
def _now_iso_z() -> str:
    sd = os.getenv("SOURCE_DATE_EPOCH")
    if sd:
        try:
            return datetime.utcfromtimestamp(int(sd)).isoformat() + "Z"
        except Exception:
            pass
    return datetime.utcnow().isoformat() + "Z"
def process_pdf_s3_object(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    try:
        head_obj = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception:
        head_obj = {}
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    try:
        local_pdf = download_s3_object_to_temp(s3_key)
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.error("Could not download S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if isinstance(manifest, dict) and manifest.get("file_hash"):
            doc_id = manifest.get("file_hash")
        else:
            doc_id = local_file_sha256(local_pdf)
        out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
        if not FORCE_OVERWRITE and s3_object_exists(out_key):
            try:
                os.unlink(local_pdf)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("Skipping because chunked file exists: %s", out_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
        img_ocr_name, img_ocr_obj = get_pdf_image_ocr_engine()
        chunker = SentenceChunker.from_env()
        fitz = import_fitz_local()
        doc = fitz.open(local_pdf)
        writer = S3DocWriter(doc_id=doc_id, s3_path=f"s3://{S3_BUCKET}/{s3_key}", ext=ext)
        saved = 0
        for pageno in range(len(doc)):
            page_start = time.perf_counter()
            clean_text, figures_texts = extract_page_clean_and_figures(local_pdf, pageno, overlap_threshold=0.25, image_ocr_engine_name=img_ocr_name, image_ocr_engine_obj=img_ocr_obj)
            used_ocr = bool(figures_texts)
            if not clean_text:
                chunk_id = f"{doc_id}_p{pageno+1}_0"
                payload = {"document_id": doc_id, "file_name": os.path.basename(s3_key), "chunk_id": chunk_id, "chunk_type": "pdf_page_chunk", "text": "", "token_count": 0, "embedding": None, "figures": figures_texts or [], "file_type": "application/pdf", "source_url": f"s3://{S3_BUCKET}/{s3_key}", "page_number": pageno+1, "timestamp": _now_iso_z(), "parser_version": PARSER_VERSION_PDF, "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [], "layout_tags": [], "used_ocr": used_ocr, "heading_path": [], "headings": [], "line_range": None, "layout_bbox": None}
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved += 1
                logger.info("Buffered empty page chunk %s", chunk_id)
                continue
            for idx, chunk in enumerate(chunker.chunk_document(clean_text)):
                chunk_id = f"{doc_id}_p{pageno+1}_{idx}"
                payload = {"document_id": doc_id, "file_name": os.path.basename(s3_key), "chunk_id": chunk_id, "chunk_type": "pdf_page_chunk", "text": chunk["text"], "token_count": int(chunk["token_count"]), "embedding": None, "figures": figures_texts or [], "file_type": "application/pdf", "source_url": f"s3://{S3_BUCKET}/{s3_key}", "page_number": pageno+1, "timestamp": _now_iso_z(), "parser_version": PARSER_VERSION_PDF, "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [], "layout_tags": [], "used_ocr": used_ocr, "heading_path": [], "headings": [], "line_range": None, "layout_bbox": None}
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved += 1
            page_ms = int((time.perf_counter() - page_start) * 1000)
            logger.info("Processed page %d (%d ms) chunks so far %d", pageno+1, page_ms, saved)
        if saved == 0:
            try:
                if writer and getattr(writer, "temp", None):
                    try:
                        os.unlink(writer.temp.name)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                os.unlink(local_pdf)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("No chunks produced for %s", s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        try:
            os.unlink(local_pdf)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Wrote %d chunks for %s → %s (%d ms)", count, s3_key, uploaded_key, total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e:
        try:
            if 'writer' in locals() and getattr(writer, "temp", None):
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            os.unlink(local_pdf)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.exception("Error while processing %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
def parse_file(s3_key: str, manifest: dict) -> dict:
    start = time.perf_counter()
    if S3_BUCKET is None:
        raise RuntimeError("S3_BUCKET must be set in environment")
    try:
        result = process_pdf_s3_object(s3_key, manifest or {})
        return result
    except Exception as e:
        total_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("parse_file error for %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
if __name__ == "__main__":
    engine_name, engine_obj = get_pdf_image_ocr_engine()
    logger.info("Engine result: %s %s", engine_name, "object_loaded" if engine_obj else "none")
