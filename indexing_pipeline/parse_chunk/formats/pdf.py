#!/usr/bin/env python3
import os
import sys
import json
import logging
import boto3
import fitz
import pdfplumber
import numpy as np
import time
import tempfile
from io import BytesIO
from datetime import datetime
from typing import Tuple, Optional
from botocore.exceptions import ClientError

log = logging.getLogger("pdf_parser_minimal_schema")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.handlers[:] = [handler]

REQUIRED = [
    "S3_BUCKET", "S3_RAW_PREFIX", "S3_CHUNKED_PREFIX",
    "CHUNK_FORMAT", "PDF_DISABLE_OCR", "PDF_OCR_ENGINE", "PDF_FORCE_OCR",
    "PDF_OCR_RENDER_DPI", "PDF_MIN_IMG_SIZE_BYTES"
]
missing = [v for v in REQUIRED if os.getenv(v) is None]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
DISABLE_OCR = os.getenv("PDF_DISABLE_OCR", "false").lower() == "true"
FORCE_OCR = os.getenv("PDF_FORCE_OCR", "false").lower() == "true"
OCR_BACKEND = os.getenv("PDF_OCR_ENGINE", "tesseract").lower()
RENDER_DPI = int(os.getenv("PDF_OCR_RENDER_DPI", "500"))
MIN_IMG_BYTES = int(os.getenv("PDF_MIN_IMG_SIZE_BYTES", "3072"))
assert CHUNK_FORMAT in ("json", "jsonl")

PDF_WINDOW_SIZE = int(os.getenv("PDF_WINDOW_SIZE", "800"))
PDF_PAGE_THRESHOLD = PDF_WINDOW_SIZE
PDF_WINDOW_OVERLAP = float(os.getenv("PDF_WINDOW_OVERLAP", "0.1"))
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"

s3 = boto3.client("s3")

try:
    import tiktoken
    TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL") or os.getenv("EMBEDDING_MODEL")
    if TOKENIZER_MODEL:
        enc = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    else:
        enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))
    def encode_tokens(text: str):
        return enc.encode(text)
    def decode_tokens(token_list) -> str:
        return enc.decode(token_list)
except Exception:
    enc = None
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text.split())
    def encode_tokens(text: str):
        return text.split()
    def decode_tokens(token_list) -> str:
        return " ".join(token_list)

def _tesseract_ready():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def _rapidocr_ready():
    try:
        from rapidocr_onnxruntime import RapidOCR
        RapidOCR()
        return True
    except Exception:
        return False

TESSERACT_OK = False
RAPID_OK = False
if OCR_BACKEND == "tesseract":
    TESSERACT_OK = _tesseract_ready()
    if not TESSERACT_OK:
        RAPID_OK = _rapidocr_ready()
        if RAPID_OK:
            OCR_BACKEND = "rapidocr"
        else:
            OCR_BACKEND = "none"
elif OCR_BACKEND == "rapidocr":
    RAPID_OK = _rapidocr_ready()
    if not RAPID_OK:
        TESSERACT_OK = _tesseract_ready()
        if TESSERACT_OK:
            OCR_BACKEND = "tesseract"
        else:
            OCR_BACKEND = "none"
else:
    OCR_BACKEND = "none"

if OCR_BACKEND == "tesseract":
    try:
        import pytesseract
    except Exception:
        OCR_BACKEND = "none"

if OCR_BACKEND == "rapidocr":
    from rapidocr_onnxruntime import RapidOCR
    ocr_rapid = RapidOCR()

def is_valid_text(text: str) -> bool:
    t = (text or "").strip()
    return len(t) > 20 and any(c.isalpha() for c in t)

def is_ocr_line_valid(text: str, min_ratio: float = 0.6) -> bool:
    t = (text or "").strip()
    if len(t) < 5:
        return False
    alnum = sum(c.isalnum() for c in t)
    return (alnum / len(t)) >= min_ratio

def dedupe_lines(lines: list) -> list:
    seen, out = set(), []
    for l in lines:
        key = (l or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(l)
    return out

def do_ocr(img: "np.ndarray") -> list:
    lines = []
    if OCR_BACKEND == "tesseract":
        from PIL import Image
        import pytesseract
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(Image.fromarray(bin_img))
        for l in raw.splitlines():
            if is_ocr_line_valid(l):
                lines.append(l.strip())
    elif OCR_BACKEND == "rapidocr":
        res = ocr_rapid(img)
        if res and isinstance(res[0], (list, tuple)):
            for item in res[0]:
                if len(item) >= 2:
                    text = item[1].strip()
                    if is_ocr_line_valid(text):
                        lines.append(text)
    return dedupe_lines(lines)

def is_valid_table(table: list) -> bool:
    if len(table) < 2 or len(table[0]) < 2:
        return False
    total_cells = sum(len(r) for r in table)
    alpha_cells = sum(1 for row in table for cell in row if any(c.isalpha() for c in (cell or "")))
    return total_cells > 0 and (alpha_cells / total_cells) >= 0.5

class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        self.count = 0
        self._first = True
        if self.ext == "json":
            self.temp.write(b"[")
    def write_payload(self, payload: dict) -> int:
        self.count += 1
        if self.ext == "jsonl":
            line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.temp.write(line)
        else:
            j = json.dumps(payload, ensure_ascii=False)
            if not self._first:
                self.temp.write(b",")
            self.temp.write(j.encode("utf-8"))
            self._first = False
        return 1
    def finalize_and_upload(self, out_key: str) -> Tuple[int, str]:
        if self.ext == "json":
            self.temp.write(b"]")
        self.temp.flush()
        self.temp.close()
        try:
            s3.upload_file(self.temp.name, S3_BUCKET, out_key, ExtraArgs={"ContentType": "application/json"})
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

def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False

def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    doc_id = manifest.get("file_hash") or os.path.basename(s3_key)
    source = f"s3://{S3_BUCKET}/{s3_key}"
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"

    # If chunked file already exists and we are not forcing overwrite, skip early.
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info(f"Skipping entire file because chunked file exists: {out_key}")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}

    mp = fitz.open(stream=raw, filetype="pdf")
    pp = pdfplumber.open(BytesIO(raw))
    saved = 0
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_key, ext=ext)
    try:
        for idx, page in enumerate(mp):
            page_num = idx + 1
            page_chunk_id = f"{doc_id}_page_{page_num}"
            t0 = time.perf_counter()
            pl = pp.pages[idx]
            raw_words = pl.extract_words(use_text_flow=True) or []
            lines_map = {}
            for w in raw_words:
                top = round(w["top"], 1)
                lines_map.setdefault(top, []).append(w)
            line_items = [(top, " ".join(w["text"] for w in sorted(ws, key=lambda x: x["x0"]))) for top, ws in sorted(lines_map.items())]
            static_txt = "\n".join(t for _, t in line_items)
            needs_ocr = not DISABLE_OCR and (FORCE_OCR or not is_valid_text(static_txt))
            pix = page.get_pixmap(dpi=RENDER_DPI) if needs_ocr else None
            used_ocr = False
            if needs_ocr and pix:
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    import cv2
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                ocr_lines = do_ocr(arr)
                if ocr_lines:
                    line_items = [(0.0, "\n".join(ocr_lines))]
                    used_ocr = True
            img_items = []
            if pl.images:
                if not pix:
                    pix = page.get_pixmap(dpi=RENDER_DPI)
                full = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    import cv2
                    full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)
                for img in pl.images:
                    x0, y0, x1, y1 = img.get("x0"), img.get("top"), img.get("x1"), img.get("bottom")
                    px0 = int(x0 * pix.width / page.rect.width)
                    py0 = int(y0 * pix.height / page.rect.height)
                    px1 = int(x1 * pix.width / page.rect.width)
                    py1 = int(y1 * pix.height / page.rect.height)
                    crop = full[py0:py1, px0:px1]
                    if crop.size >= MIN_IMG_BYTES and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                        pieces = do_ocr(crop)
                        if pieces:
                            img_items.append((y0, "\n".join(pieces)))
                            used_ocr = True
            table_items = []
            try:
                raw_tables = pl.extract_tables() or []
            except Exception as e:
                log.warning(f"Table extraction error on p{page_num}: {e}")
                raw_tables = []
            for tbl in raw_tables:
                norm = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in tbl]
                if is_valid_table(norm):
                    if len(norm) >= 2:
                        header = "| " + " | ".join(norm[0]) + " |"
                        sep = "| " + " | ".join(["---"] * len(norm[0])) + " |"
                        rows = ["| " + " | ".join(r) + " |" for r in norm[1:]]
                        md_table = "\n".join([header, sep] + rows)
                    else:
                        md_table = "\n".join(["\t".join(r) for r in norm])
                    table_items.append((0.0, md_table))
            merged = line_items + img_items + table_items
            merged.sort(key=lambda x: x[0])
            raw_lines = [l for _, l in merged]
            md_lines = [f"## Page {page_num}"]
            for l in raw_lines:
                for ln in l.split("\n"):
                    if ln.strip():
                        md_lines.append(ln.strip())
            clean = [ln for ln in md_lines if is_ocr_line_valid(ln)]
            clean = dedupe_lines(clean)
            final_text = "\n\n".join(clean)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            try:
                token_count = count_tokens(final_text) if final_text and final_text.strip() else 0
            except Exception:
                token_count = len(final_text.split()) if final_text and final_text.strip() else 0
            page_payload = {
                "document_id": doc_id or None,
                "chunk_id": page_chunk_id,
                "chunk_type": "page",
                "text": final_text or None,
                "token_count": token_count if token_count is not None else None,
                "embedding": None,
                "file_type": "application/pdf",
                "source_url": source,
                "page_number": page_num,
                "slide_range": None,
                "row_range": None,
                "token_range": [0, token_count - 1] if token_count and token_count > 0 else None,
                "audio_range": None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "parser_version": manifest.get("parser_version") if isinstance(manifest, dict) else None,
                "tags": manifest.get("tags") if isinstance(manifest, dict) else [],
                "layout_tags": ["page"],
                "used_ocr": bool(used_ocr),
                "heading_path": [],
                "headings": [],
                "line_range": None
            }
            writer.write_payload(page_payload)
            log.info(f"Buffered parsed page {page_num} (tokens={token_count}, ocr_used={used_ocr})")
            saved += 1
            try:
                if token_count and token_count > PDF_PAGE_THRESHOLD:
                    encoded = encode_tokens(final_text)
                    total_tokens = len(encoded)
                    window_size = int(PDF_WINDOW_SIZE)
                    overlap = int(max(1, window_size * PDF_WINDOW_OVERLAP))
                    step = window_size - overlap
                    if step <= 0:
                        step = window_size
                    sub_index = 0
                    start_t = 0
                    while start_t < total_tokens:
                        end_t = min(start_t + window_size, total_tokens)
                        window_tokens = encoded[start_t:end_t]
                        try:
                            window_text = decode_tokens(window_tokens)
                        except Exception:
                            if isinstance(window_tokens, list) and window_tokens and isinstance(window_tokens[0], str):
                                window_text = " ".join(window_tokens)
                            else:
                                window_text = final_text
                        sub_chunk_id = f"{doc_id}_page_{page_num}_sub_{sub_index}"
                        sub_token_count = end_t - start_t
                        sub_payload = {
                            "document_id": doc_id or None,
                            "chunk_id": sub_chunk_id,
                            "chunk_type": "page_subchunk",
                            "text": window_text or None,
                            "token_count": sub_token_count,
                            "embedding": None,
                            "file_type": "application/pdf",
                            "source_url": source,
                            "page_number": page_num,
                            "slide_range": None,
                            "row_range": None,
                            "token_range": [start_t, end_t - 1],
                            "audio_range": None,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "parser_version": manifest.get("parser_version") if isinstance(manifest, dict) else None,
                            "tags": manifest.get("tags") if isinstance(manifest, dict) else [],
                            "layout_tags": ["page", "subchunk"],
                            "used_ocr": bool(used_ocr),
                            "heading_path": [],
                            "headings": [],
                            "line_range": None
                        }
                        writer.write_payload(sub_payload)
                        log.info(f"Buffered subchunk {sub_chunk_id} tokens {start_t}-{end_t-1}")
                        saved += 1
                        sub_index += 1
                        start_t += step
            except Exception as e:
                log.warning(f"Failed to produce subchunks for page {page_num}: {e}")
    except Exception as e:
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        mp.close()
        pp.close()
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.exception("Fatal error while parsing %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if saved == 0:
            try:
                if writer and writer.temp:
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("No chunks produced for %s", s3_key)
            mp.close()
            pp.close()
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        mp.close()
        pp.close()
        log.info(f"Wrote {count} chunks for {s3_key} → {uploaded_key} ({total_ms} ms total) ocr_used=%s", bool(True))
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
        mp.close()
        pp.close()
        log.error(f"Failed to upload chunked file for {s3_key} error={str(e_up)}")
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}

if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key or not key.lower().endswith(".pdf"):
                continue
            log.info(f"Routing parse_file for s3://{S3_BUCKET}/{key}")
            manifest_key = key + ".manifest.json"
            try:
                mf = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf["Body"])
            except Exception:
                manifest = {}
            parse_file(key, manifest)
