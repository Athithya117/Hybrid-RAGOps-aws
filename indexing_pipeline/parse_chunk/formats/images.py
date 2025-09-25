#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import logging
import hashlib
import time
import tempfile
from io import BytesIO
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
    from PIL import Image, ImageSequence
    import numpy as np
    import cv2
except Exception as e:
    print(f"[FATAL] Missing dependency: {e}. Install boto3 pillow numpy opencv-python.", file=sys.stderr)
    raise
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None
try:
    import tiktoken
except Exception:
    tiktoken = None
RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[37m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;41m"
}
class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"
log = logging.getLogger("images_parser")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
log.handlers[:] = [handler]
REQUIRED_ENV = ["S3_BUCKET", "S3_RAW_PREFIX", "S3_CHUNKED_PREFIX", "CHUNK_FORMAT"]
missing = [v for v in REQUIRED_ENV if os.getenv(v) is None]
if missing:
    log.critical(f"Missing required env vars: {missing}")
    sys.exit(2)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl")
PARSER_VERSION = os.getenv("PARSER_VERSION_IMAGES", "images-parser-v1")
OCR_ENGINE_DESIRED = os.getenv("IMAGE_OCR_ENGINE", "auto").lower()
MIN_IMG_BYTES = int(os.getenv("IMAGE_MIN_IMG_SIZE_BYTES", "1024"))
MIN_WIDTH = int(os.getenv("IMAGE_MIN_WIDTH", "1600"))
TESSERACT_PSM = int(os.getenv("TESSERACT_PSM", "6"))
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "30.0"))
S3_MAX_RETRIES = int(os.getenv("S3_MAX_RETRIES", "3"))
TOKEN_ENCODER = os.getenv("TOKEN_ENCODER", "cl100k_base")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
boto_config = BotoConfig(retries={"max_attempts": S3_MAX_RETRIES, "mode": "standard"})
s3 = boto3.client("s3", config=boto_config)
EXT_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif"
}
ALLOWED_EXTS = set(EXT_MAP.keys())
def _tesseract_ready() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
def _rapidocr_ready() -> bool:
    if RapidOCR is None:
        return False
    try:
        RapidOCR()
        return True
    except Exception:
        return False
OCR_BACKEND = "none"
if OCR_ENGINE_DESIRED == "tesseract":
    OCR_BACKEND = "tesseract" if _tesseract_ready() else "none"
elif OCR_ENGINE_DESIRED == "rapidocr":
    OCR_BACKEND = "rapidocr" if _rapidocr_ready() else "none"
else:
    if _tesseract_ready():
        OCR_BACKEND = "tesseract"
    elif _rapidocr_ready():
        OCR_BACKEND = "rapidocr"
    else:
        OCR_BACKEND = "none"
if OCR_BACKEND == "rapidocr":
    ocr_rapid = RapidOCR()
log.info(f"OCR backend: {OCR_BACKEND}")
ENCODER = None
if tiktoken is not None:
    try:
        ENCODER = tiktoken.get_encoding(TOKEN_ENCODER)
    except Exception:
        ENCODER = None
def blob_hash(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr[..., ::-1].copy()
def load_image_bytes_to_bgr(img_bytes: bytes) -> Tuple[Optional[np.ndarray], Tuple[int, int, int]]:
    try:
        img = Image.open(BytesIO(img_bytes))
        if getattr(img, "is_animated", False):
            img = ImageSequence.Iterator(img)[0]
        arr = pil_to_bgr(img)
        h, w = arr.shape[:2]
        c = arr.shape[2] if arr.ndim == 3 else 1
        return arr, (w, h, c)
    except Exception as e:
        log.debug(f"load_image_bytes_to_bgr failed: {e}")
        return None, (0, 0, 0)
def upscale_if_needed(img: np.ndarray, min_w: int = MIN_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= min_w:
        return img
    scale = min_w / w
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
def denoise_and_sharpen(img: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(den, -1, kernel)
    return sharp
def adaptive_threshold_gray(img_gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)
def _gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)
def _morph_close(bin_img: np.ndarray, k=3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
def preprocess_variants(img_bgr: np.ndarray) -> List[np.ndarray]:
    pre = upscale_if_needed(img_bgr, MIN_WIDTH)
    pre = denoise_and_sharpen(pre)
    gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    variants = []
    v1 = _apply_clahe(gray)
    v1t = adaptive_threshold_gray(v1)
    variants.append(cv2.cvtColor(v1t, cv2.COLOR_GRAY2BGR))
    g = _gamma_correction(gray, 0.8)
    _, v2t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(v2t, cv2.COLOR_GRAY2BGR))
    mb = cv2.medianBlur(gray, 3)
    v3t = adaptive_threshold_gray(mb)
    variants.append(cv2.cvtColor(v3t, cv2.COLOR_GRAY2BGR))
    _, ots = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = _morph_close(ots, k=3)
    variants.append(cv2.cvtColor(close, cv2.COLOR_GRAY2BGR))
    variants.append(pre)
    return variants
def tesseract_extract_lines_and_conf(img_bgr: np.ndarray, psm: int = TESSERACT_PSM) -> tuple[List[str], float]:
    try:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img_bgr[..., ::-1])
        config = f'--oem 1 --psm {psm}'
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config=config)
        texts = data.get("text", [])
        if not texts:
            return [], 0.0
        tokens = []
        n = len(texts)
        for i in range(n):
            txt = (texts[i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data.get("conf", [0]*n)[i])
            except Exception:
                conf = 0.0
            left = int(data.get("left", [0]*n)[i])
            top = int(data.get("top", [0]*n)[i])
            block = int(data.get("block_num", [0]*n)[i])
            par = int(data.get("par_num", [0]*n)[i])
            line = int(data.get("line_num", [0]*n)[i])
            tokens.append({"text": txt, "conf": conf, "left": left, "top": top, "block": block, "par": par, "line": line})
        groups: Dict[Tuple[int,int,int], List[dict]] = {}
        for t in tokens:
            key = (t["block"], t["par"], t["line"])
            groups.setdefault(key, []).append(t)
        lines = []
        all_confs = []
        for (block, par, line_idx), toks in groups.items():
            toks_sorted = sorted(toks, key=lambda x: x["left"])
            line_text = " ".join(t["text"] for t in toks_sorted)
            left_min = min(t["left"] for t in toks_sorted)
            lines.append({"text": line_text, "block": block, "par": par, "left": left_min})
            all_confs.extend([t["conf"] for t in toks_sorted])
        lines_sorted = sorted(lines, key=lambda x: (x["block"], x["par"], x["left"]))
        out_lines = [l["text"] for l in lines_sorted]
        avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0
        return out_lines, float(avg_conf)
    except Exception as e:
        log.debug(f"tesseract_extract_lines_and_conf failed: {e}")
        return [], 0.0
def rapidocr_extract_lines_with_boxes(img_bgr: np.ndarray) -> List[str]:
    try:
        res = ocr_rapid(img_bgr)
        items = []
        if not res:
            return []
        for it in res[0]:
            text = ""
            left = 0.0
            top = 0.0
            if isinstance(it, (list, tuple)):
                if len(it) >= 3 and isinstance(it[0], (list, tuple)):
                    bbox = it[0]
                    text = (it[1] or "").strip()
                    xs = bbox[::2]
                    ys = bbox[1::2]
                    left = float(min(xs)) if xs else 0.0
                    top = float(min(ys)) if ys else 0.0
                elif len(it) >= 2 and isinstance(it[0], (int, float)):
                    try:
                        left = float(it[0])
                        top = float(it[1]) if len(it) > 1 else 0.0
                        text = (it[2] if len(it) > 2 else it[1] or "").strip()
                    except Exception:
                        text = (it[-2] if len(it) >= 2 else it[-1] or "").strip()
                else:
                    text = (it[-2] if len(it) >= 2 else it[-1] or "").strip()
            else:
                text = str(it).strip()
            if text:
                items.append({"text": text, "left": left, "top": top})
        if not items:
            return []
        items_sorted = sorted(items, key=lambda x: (x["top"], x["left"]))
        rows: List[List[dict]] = []
        tops = [i["top"] for i in items_sorted]
        median_top = float(np.median(tops)) if tops else 0.0
        tol = max(10.0, median_top * 0.02) if median_top > 0 else 12.0
        for it in items_sorted:
            if not rows:
                rows.append([it])
                continue
            last_row = rows[-1]
            last_median = float(np.median([r["top"] for r in last_row]))
            if abs(it["top"] - last_median) <= max(tol, 12.0):
                last_row.append(it)
            else:
                rows.append([it])
        out_lines = []
        for row in rows:
            row_sorted = sorted(row, key=lambda x: x["left"])
            out_lines.append(" ".join(x["text"] for x in row_sorted))
        return out_lines
    except Exception as e:
        log.debug(f"rapidocr_extract_lines_with_boxes failed: {e}")
        return []
def fix_hyphenation(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        if out and out[-1].endswith("-"):
            out[-1] = out[-1][:-1] + ln.lstrip()
        else:
            out.append(ln)
    return out
def dedupe_lines(lines: List[str]) -> List[str]:
    seen, out = set(), []
    for l in lines:
        key = (l or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(l)
    return out
def do_ocr(img_bgr: np.ndarray) -> List[str]:
    variants = preprocess_variants(img_bgr)
    best_lines: List[str] = []
    best_conf = -1.0
    psm_candidates = [TESSERACT_PSM, 3, 4] if pytesseract else []
    if OCR_BACKEND == "tesseract":
        for v in variants:
            for psm in psm_candidates:
                lines, conf = tesseract_extract_lines_and_conf(v, psm=psm)
                if lines and conf > best_conf:
                    best_conf = conf
                    best_lines = lines
                if best_conf >= 95.0:
                    break
            if best_conf >= 95.0:
                break
        if best_conf >= OCR_CONFIDENCE_THRESHOLD and best_lines:
            return fix_hyphenation(dedupe_lines(best_lines))
        if RapidOCR is not None:
            rr = rapidocr_extract_lines_with_boxes(variants[0])
            if rr:
                return fix_hyphenation(dedupe_lines(rr))
        return fix_hyphenation(dedupe_lines(best_lines))
    elif OCR_BACKEND == "rapidocr":
        pre = variants[0]
        rr = rapidocr_extract_lines_with_boxes(pre)
        if rr:
            return fix_hyphenation(dedupe_lines(rr))
        if _tesseract_ready():
            for v in variants:
                lines, conf = tesseract_extract_lines_and_conf(v, psm=TESSERACT_PSM)
                if conf > best_conf and lines:
                    best_conf = conf
                    best_lines = lines
            return fix_hyphenation(dedupe_lines(best_lines))
    else:
        log.error("No OCR backend available")
        return []
def _derive_source_key_from_path(s3_path: str) -> str:
    prefix = f"s3://{S3_BUCKET}/"
    if s3_path.startswith(prefix):
        return s3_path[len(prefix):]
    return ""
def _compute_token_count(text: str) -> int:
    if not text:
        return 0
    if ENCODER is not None:
        try:
            return len(ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split())
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
def parse_image_s3_object(s3_key: str, manifest: Optional[dict] = None) -> dict:
    t_all = time.perf_counter()
    manifest = manifest or {}
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except ClientError as e:
        log.error(f"S3 HEAD error {s3_key}: {e}")
        return {"saved_chunks": 0, "error": str(e)}
    content_len = head.get("ContentLength", 0) or 0
    etag = head.get("ETag")
    if isinstance(etag, str):
        etag = etag.strip('"')
    last_modified = head.get("LastModified")
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash")
    else:
        if etag:
            doc_id = sha256_hex(s3_key + str(etag))
        else:
            doc_id = sha256_hex(s3_key + (str(last_modified.isoformat()) if last_modified is not None else ""))
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    ext_out = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext_out}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - t_all) * 1000)
        log.info("Skipping entire file because chunked file exists: %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        raw = obj["Body"].read()
    except ClientError as e:
        log.error(f"S3 GET error {s3_key}: {e}")
        return {"saved_chunks": 0, "error": str(e)}
    size = len(raw)
    if size < MIN_IMG_BYTES:
        log.warning(f"Skipping {s3_key} after download: {size} bytes < MIN_IMG_BYTES ({MIN_IMG_BYTES})")
        return {"saved_chunks": 0, "skipped_bytes": size}
    _, ext = os.path.splitext(s3_key.lower())
    content_type = EXT_MAP.get(ext, "application/octet-stream")
    chunk_index = 0
    saved = 0
    start = time.perf_counter()
    img_bgr, _ = load_image_bytes_to_bgr(raw)
    used_ocr = False
    final_text = ""
    if img_bgr is not None:
        try:
            lines = do_ocr(img_bgr)
            if lines:
                used_ocr = True
                final_text = "\n".join(lines)
        except Exception as e:
            log.warning(f"OCR failed for {s3_key}: {e}")
    else:
        log.warning(f"Cannot decode image {s3_key}")
    parse_ms = int((time.perf_counter() - start) * 1000)
    token_ct = _compute_token_count(final_text)
    manifest_tags = manifest.get("tags", []) if isinstance(manifest, dict) else []
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_key, ext=ext_out)
    try:
        chunk_index += 1
        chunk_id = f"{doc_id}_{chunk_index}"
        payload = {
            "document_id": doc_id or "",
            "file_name": os.path.basename(s3_key),
            "chunk_id": chunk_id or "",
            "chunk_type": "image",
            "text": final_text or "",
            "token_count": int(token_ct or 0),
            "embedding": None,
            "file_type": content_type,
            "source_url": s3_path,
            "page_number": None,
            "slide_range": None,
            "row_range": None,
            "token_range": None,
            "audio_range": None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parser_version": PARSER_VERSION or "",
            "tags": manifest_tags or [],
            "layout_tags": ["image"],
            "used_ocr": bool(used_ocr),
            "heading_path": [],
            "headings": [],
            "line_range": None
        }
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
        sanitize_payload_for_weaviate(payload)
        writer.write_payload(payload)
        saved += 1
    except Exception as e:
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        total_ms = int((time.perf_counter() - t_all) * 1000)
        log.error(f"Error while buffering chunks for {s3_key}: {e}")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if saved == 0:
            try:
                if writer and writer.temp:
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - t_all) * 1000)
            log.info("No chunks produced for %s", s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - t_all) * 1000)
        log.info("Wrote %d chunks for %s → %s (%d ms) ocr_used=%s parse_ms=%d tokens=%d", count, s3_key, uploaded_key, total_ms, used_ocr, parse_ms, token_ct)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - t_all) * 1000)
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        log.error("Failed to upload chunked file for %s error=%s", s3_key, str(e_up))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}
def list_image_keys(prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            _, ext = os.path.splitext(key.lower())
            if ext in ALLOWED_EXTS:
                keys.append(key)
    return keys
def parse_file(s3_key: str, manifest: Optional[dict] = None) -> dict:
    if manifest is None:
        manifest = {}
    start = time.perf_counter()
    result = parse_image_s3_object(s3_key, manifest)
    if isinstance(manifest, dict):
        manifest.setdefault("parsed_by", []).append({"module": "images", "timestamp": datetime.utcnow().isoformat() + "Z"})
        manifest["parsed_chunks"] = manifest.get("parsed_chunks", 0) + result.get("saved_chunks", 0)
        manifest["parse_completed_at"] = datetime.utcnow().isoformat() + "Z"
        duration_ms = int((time.perf_counter() - start) * 1000)
        manifest["parse_duration_ms"] = manifest.get("parse_duration_ms", 0) + duration_ms
    return result
def main():
    keys = list_image_keys(S3_RAW_PREFIX)
    if not keys:
        log.info("No image keys found.")
        return
    log.info(f"Found {len(keys)} image(s) under {S3_RAW_PREFIX}. Processing sequentially...")
    total_saved = 0
    errors = 0
    processed = 0
    for key in keys:
        manifest = {}
        manifest_key = key + ".manifest.json"
        try:
            mf = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
            manifest = json.load(mf["Body"])
        except Exception:
            manifest = {}
        res = parse_image_s3_object(key, manifest)
        processed += 1
        total_saved += res.get("saved_chunks", 0)
        if res.get("error"):
            errors += 1
            log.warning(f"Error for {key}: {res.get('error')}")
    log.info(f"Completed. Processed: {processed}, Saved chunks: {total_saved}, Errors: {errors}")
if __name__ == "__main__":
    main()
