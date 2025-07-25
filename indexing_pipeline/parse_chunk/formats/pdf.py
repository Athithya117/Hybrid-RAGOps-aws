# indexing_pipeline/parse_chunk/formats/pdf.py

import os, time, json
from io import BytesIO
from datetime import datetime

import fitz  # PyMuPDF
import pdfplumber
import boto3
from rapidocr_onnxruntime import RapidOCR
from indexing_pipeline.parse_chunk.router import log

# ENVIRONMENT VARIABLES + DEFAULTS
S3_BUCKET = os.getenv("S3_BUCKET")
assert S3_BUCKET, "S3_BUCKET must be set"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "jsonl").lower()
assert CHUNK_FORMAT in ("jsonl", "json")
IS_MULTILINGUAL = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
RAPIDOCR_CACHE = os.getenv("RAPIDOCR_CACHE", "/tmp")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_CONF_THRESHOLD = float(os.getenv("OCR_CONF_THRESHOLD", "0.5"))
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "20"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup S3 and OCR
s3 = boto3.client("s3")
os.makedirs(RAPIDOCR_CACHE, exist_ok=True)
ocr = RapidOCR(
    det_model_dir=os.path.join(RAPIDOCR_CACHE, "det"),
    rec_model_dir=os.path.join(RAPIDOCR_CACHE, "rec"),
    cls_model_dir=os.path.join(RAPIDOCR_CACHE, "cls")
)

def parse_page(doc, raw_bytes, pi):
    info = {"page_number": pi + 1, "errors": [], "decision_trace": []}

    text = ""
    source = "empty"
    conf = 0.0

    # 1. Try PyMuPDF
    try:
        page = doc.load_page(pi)
        text0 = page.get_text("text", flags=0).strip()
        info["decision_trace"].append(f"pymupdf_len={len(text0)}")
        if len(text0) >= MIN_TEXT_LEN:
            return text0, "pymupdf", 1.0, info
    except Exception as e:
        info["errors"].append(f"pymupdf:{e}")

    # 2. Try pdfplumber
    try:
        with pdfplumber.open(BytesIO(raw_bytes)) as pdf2:
            p2 = pdf2.pages[pi]
            text2 = (p2.extract_text() or "").strip()
            info["decision_trace"].append(f"pdfplumber_len={len(text2)}")
            if len(text2) >= MIN_TEXT_LEN:
                return text2, "pdfplumber", 0.9, info
    except Exception as e:
        info["errors"].append(f"pdfplumber:{e}")

    # 3. Finally fallback to OCR
    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        ocr_res, _ = ocr(pix.tobytes("png"))
        filtered = [t for _, t, c in ocr_res if c >= OCR_CONF_THRESHOLD]
        cleaned = " ".join(filtered).strip()
        avg = sum(c for _, _, c in ocr_res) / len(ocr_res) if ocr_res else 0.0
        info["decision_trace"].append(f"ocr_len={len(cleaned)}_conf={avg:.2f}")
        if len(cleaned) >= MIN_TEXT_LEN:
            return cleaned, "rapidocr", avg, info
    except Exception as e:
        info["errors"].append(f"ocr:{e}")

    # fallback to whatever best
    return text, source, conf, info

def parse_file(s3_key, manifest):
    log(f"Parsing PDF {s3_key} (multi={IS_MULTILINGUAL})", level=LOG_LEVEL)
    try:
        raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    except Exception as e:
        log(f"S3 download error: {e}", level="ERROR")
        return []

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as e:
        log(f"Invalid PDF open: {e}", level="ERROR")
        return []

    chunks = []
    total_chars = 0

    for pi in range(doc.page_count):
        start = time.time()
        text, src, conf, info = parse_page(doc, raw, pi)
        elapsed = int((time.time() - start) * 1000)
        total_chars += len(text)

        cid = f"{manifest['sha256']}_{pi}"
        payload = {
            "document_id": manifest["sha256"],
            "chunk_id": cid,
            "chunk_index": pi,
            "text": text,
            "source_path": f"s3://{S3_BUCKET}/{s3_key}",
            "source_hash": manifest["sha256"],
            "file_type": "pdf",
            "page_number": pi + 1,
            "start_time": None,
            "end_time": None,
            "line_range": None,
            "bbox": None,
            "metadata": {
                "is_multilingual": IS_MULTILINGUAL,
                "is_ocr": (src == "rapidocr"),
                "chunk_type": "page",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tags": [],
                "layout_tags": [],
                "confidence": conf,
                "errors": info["errors"],
                "decision_trace": info["decision_trace"],
            },
            "entities": [],
            "embedding": []
        }
        chunks.append({"id": f"chunk_{cid}", "payload": payload, "tables": None})
        log(f"Page {pi}: src={src} conf={conf:.2f} len={len(text)} time={elapsed}ms",
            level="DEBUG")

    doc.close()
    log(f"Extracted {len(chunks)} pages, {total_chars} chars total", level=LOG_LEVEL)

    return chunks if CHUNK_FORMAT == "jsonl" else [json.loads(json.dumps(c)) for c in chunks]
