# indexing_pipeline/parse_chunk/formats/pdf.py

import os
import time
import logging
from datetime import datetime

import fitz  # PyMuPDF
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
import boto3

# ─── Config & Environment ─────────────────────────────────────────────────────

DEVICE = os.getenv("DEVICE", "cpu").lower()
USE_GPU = (DEVICE == "gpu")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8" if USE_GPU else "1"))

S3_BUCKET = os.getenv("S3_BUCKET") or RuntimeError("Missing S3_BUCKET env")
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "50"))
LINE_COUNT_THRESH = int(os.getenv("LINE_COUNT_THRESH", "10"))
RECT_COUNT_THRESH = int(os.getenv("RECT_COUNT_THRESH", "5"))
SPARSE_BLOCK_WIDTH = int(os.getenv("SPARSE_BLOCK_WIDTH", "200"))

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── OCR Initialization ──────────────────────────────────────────────────────

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=USE_GPU)

s3 = boto3.client("s3")

# ─── Utility Functions ───────────────────────────────────────────────────────

def render_page(page, dpi=150):
    pix = page.get_pixmap(dpi=dpi)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

def needs_ocr(page) -> bool:
    txt = page.get_text("text").strip()
    if len(txt) < MIN_TEXT_CHARS:
        return True
    if page.get_images():
        return True
    dr = page.get_drawings()
    if sum(d["type"] == "line" for d in dr) >= LINE_COUNT_THRESH:
        return True
    if sum(d["type"] == "rect" for d in dr) >= RECT_COUNT_THRESH:
        return True
    blocks = page.get_text("blocks")
    if len(blocks) < 3 or any((b[2] - b[0]) < SPARSE_BLOCK_WIDTH for b in blocks):
        return True
    return False

def ocr_image(img_np):
    res = ocr.ocr(img_np, cls=True, batch_size=BATCH_SIZE)
    return res[0] if isinstance(res, list) else []

# ─── Main parse_file Function ───────────────────────────────────────────────

def parse_file(s3_key: str, manifest: dict) -> list:
    ts = datetime.utcnow().isoformat()
    body = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    size = len(body)
    doc = fitz.open(stream=body, filetype="pdf")

    chunks = []
    for idx, page in enumerate(doc):
        t0 = time.time()
        text = page.get_text("text").strip()
        used_ocr = False

        if needs_ocr(page):
            img = preprocess(render_page(page, dpi=150))
            try:
                ocr_results = ocr_image(img)
                used_ocr = True
                lines = [row[1][0] for row in ocr_results]
                if lines:
                    text = "\n".join(lines)
            except Exception as e:
                logger.error(f"OCR failed on {s3_key} page {idx+1}: {e}")

        duration = int((time.time() - t0) * 1000)
        payload = {
            "document_id": manifest["sha256"],
            "chunk_id": f"{manifest['sha256'][:8]}_p{idx+1}",
            "chunk_index": idx,
            "text": text,
            "tables": [],
            "source_path": f"s3://{S3_BUCKET}/{s3_key}",
            "file_type": "pdf",
            "page_number": idx + 1,
            "bbox": [0, 0, page.rect.width, page.rect.height],
            "metadata": {"is_ocr": used_ocr, "timestamp": ts},
            "entities": [],
            "embedding": [],
            "parse_duration_ms": duration,
            "source_size_bytes": size
        }
        chunks.append({"id": payload["chunk_id"], "payload": payload})

    doc.close()
    return chunks
