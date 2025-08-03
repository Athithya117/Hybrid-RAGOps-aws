import os
import sys
import re
import json
import logging
import hashlib
import boto3
import fitz               # PyMuPDF
import pdfplumber
import cv2
import numpy as np
import time
from io import BytesIO
from datetime import datetime
from PIL import Image
import pytesseract

# --- Validate env vars ---
REQUIRED = [
    "S3_BUCKET", "S3_RAW_PREFIX", "S3_CHUNKED_PREFIX",
    "CHUNK_FORMAT", "DISABLE_OCR", "OCR_ENGINE", "FORCE_OCR",
    "OCR_RENDER_DPI", "MIN_IMG_SIZE_BYTES" 
]
missing = [v for v in REQUIRED if os.getenv(v) is None]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}")

# --- Configs ---
S3_BUCKET         = os.getenv("S3_BUCKET")
S3_RAW_PREFIX     = os.getenv("S3_RAW_PREFIX").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX").rstrip("/") + "/"
CHUNK_FORMAT      = os.getenv("CHUNK_FORMAT", "json").lower()
DISABLE_OCR       = os.getenv("DISABLE_OCR", "false").lower() == "true"
FORCE_OCR         = os.getenv("FORCE_OCR", "false").lower() == "true"
OCR_BACKEND       = os.getenv("OCR_ENGINE", "tesseract").lower()
RENDER_DPI        = int(os.getenv("OCR_RENDER_DPI", "500"))
MIN_IMG_BYTES     = int(os.getenv("MIN_IMG_SIZE_BYTES", "3072"))
IS_MULTILINGUAL   = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
TESSERACT_LANG    = os.getenv("TESSERACT_LANG", "eng")
TESSERACT_CMD     = os.getenv("TESSERACT_CMD", "tesseract")
assert CHUNK_FORMAT in ("json", "jsonl"), "CHUNK_FORMAT must be 'json' or 'jsonl'"

# --- Logging & AWS ---
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("pdf_parser")
s3 = boto3.client("s3")

# --- OCR setup ---
if OCR_BACKEND in ("tesseract", "indicocr"):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    TESS_CONFIG = f"-l {TESSERACT_LANG} --oem 1 --psm 6"
elif OCR_BACKEND == "rapidocr":
    from rapidocr_onnxruntime import RapidOCR
    ocr_rapid = RapidOCR()

# --- Helpers ---
def is_valid_text(text: str) -> bool:
    t = text.strip()
    return len(t) > 20 and any(c.isalpha() for c in t)

def is_ocr_line_valid(text: str, min_ratio: float = 0.6) -> bool:
    t = text.strip()
    if len(t) < 5:
        return False
    alnum = sum(c.isalnum() for c in t)
    return (alnum / len(t)) >= min_ratio

def dedupe_lines(lines: list[str]) -> list[str]:
    seen, out = set(), []
    for l in lines:
        key = l.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(l)
    return out

def do_ocr(img: np.ndarray) -> list[str]:
    lines = []
    if OCR_BACKEND in ("tesseract", "indicocr"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(Image.fromarray(bin_img), config=TESS_CONFIG)
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

def is_valid_table(table: list[list[str]]) -> bool:
    if len(table) < 2 or len(table[0]) < 2:
        return False
    total_cells = sum(len(r) for r in table)
    alpha_cells = sum(1 for row in table for cell in row if any(c.isalpha() for c in (cell or "")))
    return total_cells > 0 and (alpha_cells / total_cells) >= 0.5

# --- Core parsing function ---
def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()

    doc_id = manifest.get("file_hash", "")
    source = f"s3://{S3_BUCKET}/{s3_key}"

    mp = fitz.open(stream=raw, filetype="pdf")
    pp = pdfplumber.open(BytesIO(raw))
    saved = 0

    for idx, page in enumerate(mp):
        page_num = idx + 1
        chunk_id = f"{doc_id}_{page_num}"
        t0 = time.perf_counter()
        pl = pp.pages[idx]

        figures = []
        raw_words = pl.extract_words(use_text_flow=True) or []
        lines_map = {}
        for w in raw_words:
            top = round(w["top"], 1)
            lines_map.setdefault(top, []).append(w)
        line_items = [
            (top, " ".join(w["text"] for w in sorted(ws, key=lambda x: x["x0"])))
            for top, ws in sorted(lines_map.items())
        ]

        static_txt = "\n".join(t for _, t in line_items)
        needs_ocr = not DISABLE_OCR and (FORCE_OCR or not is_valid_text(static_txt))
        pix = page.get_pixmap(dpi=RENDER_DPI) if needs_ocr else None
        used_ocr = False
        if needs_ocr and pix:
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
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
                full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)
            for img in pl.images:
                x0, y0, x1, y1 = img["x0"], img["top"], img["x1"], img["bottom"]
                px0 = int(x0 * pix.width / page.rect.width)
                py0 = int(y0 * pix.height / page.rect.height)
                px1 = int(x1 * pix.width / page.rect.width)
                py1 = int(y1 * pix.height / page.rect.height)
                crop = full[py0:py1, px0:px1]
                if crop.size >= MIN_IMG_BYTES and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                    pieces = do_ocr(crop)
                    if pieces:
                        img_items.append((y0, "\n".join(pieces)))
                        figures.extend(pieces)
                        used_ocr = True

        table_items = []
        tables_data = []
        try:
            raw_tables = pl.extract_tables() or []
        except Exception as e:
            log.warning(f"Table extraction error on p{page_num}: {e}")
            raw_tables = []
        for i, tbl in enumerate(raw_tables, 1):
            norm = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in tbl]
            if is_valid_table(norm):
                tables_data.append(norm)
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
        payload = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_type": "page",
            "text": final_text,
            "embedding": None,
            "source": {
                "file_type": "application/pdf",
                "source_path": source,
                "page_number": page_num,
                "start_time": None,
                "end_time": None,
                "line_range": None,
                "bbox": None
            },
            "graph": {
                "graph_node_id": chunk_id,
                "parent_id": f"{doc_id}_page{page_num}",
                "previous_id": f"{doc_id}_{page_num - 1}" if page_num > 1 else None,
                "next_id": f"{doc_id}_{page_num + 1}"
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tags": [],
                "layout_tags": ["page"],
                "entities": [],
                "used_ocr": used_ocr,
                "is_multilingual": IS_MULTILINGUAL,
                "num_images": len(img_items),
                "num_tables": len(tables_data),
                "parse_chunk_duration": duration_ms,
                "custom": {}
            }
        }

        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        out_key = f"{S3_CHUNKED_PREFIX}{chunk_id}.{ext}"
        body = (
            (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            if ext == "jsonl"
            else json.dumps(payload, indent=2, ensure_ascii=False).encode()
        )
        s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json")
        log.info(f"Parsed page {page_num} in {duration_ms} ms → {out_key}")
        saved += 1

    mp.close()
    pp.close()
    total_ms = int((time.perf_counter() - start_all) * 1000)
    log.info(f"Completed parsing {saved} pages in {total_ms} ms total")
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}

if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".pdf"):
                continue
            log.info(f"Routing parse_file for s3://{S3_BUCKET}/{key}")
            manifest_key = key + ".manifest.json"
            try:
                mf = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf["Body"])
            except Exception:
                manifest = {}
            parse_file(key, manifest)
