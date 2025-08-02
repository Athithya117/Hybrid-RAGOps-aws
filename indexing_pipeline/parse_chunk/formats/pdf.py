import os
import sys
import json
import logging
import boto3
import fitz  # PyMuPDF
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
    "S3_BUCKET","S3_RAW_PREFIX","S3_CHUNKED_PREFIX",
    "CHUNK_FORMAT","DISABLE_OCR","OCR_ENGINE","FORCE_OCR",
    "OCR_RENDER_DPI","MIN_IMG_SIZE_BYTES","IS_MULTILINGUAL"
]
missing = [v for v in REQUIRED if v not in os.environ or os.environ[v]==""]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}")

# --- Configs ---
S3_BUCKET         = os.getenv("S3_BUCKET")
S3_RAW_PREFIX     = os.getenv("S3_RAW_PREFIX")
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX")
CHUNK_FORMAT      = os.getenv("CHUNK_FORMAT","json").lower()
DISABLE_OCR       = os.getenv("DISABLE_OCR","false").lower()=="true"
FORCE_OCR         = os.getenv("FORCE_OCR","false").lower()=="true"
OCR_BACKEND       = os.getenv("OCR_ENGINE","tesseract").lower()
RENDER_DPI        = int(os.getenv("OCR_RENDER_DPI","500"))
MIN_IMG_BYTES     = int(os.getenv("MIN_IMG_SIZE_BYTES","3072"))
IS_MULTILINGUAL   = os.getenv("IS_MULTILINGUAL","false").lower()=="true"
assert CHUNK_FORMAT in ("json","jsonl"), "CHUNK_FORMAT must be 'json' or 'jsonl'"

# --- Logging & AWS ---
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("pdf_parser")
s3 = boto3.client("s3")

# --- OCR setup ---
if OCR_BACKEND in ("tesseract","indicocr"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD","tesseract")
    langs = "+".join(os.getenv("TESSERACT_LANG","eng").split())
    TESS_CONFIG = f"-l {langs} --oem 1 --psm 6"
elif OCR_BACKEND=="rapidocr":
    from rapidocr_onnxruntime import RapidOCR
    ocr_rapid = RapidOCR()

def is_valid_text(text: str) -> bool:
    t = text.strip()
    return len(t) > 20 and any(c.isalpha() for c in t)

def do_ocr(img: np.ndarray) -> list[str]:
    if OCR_BACKEND in ("tesseract","indicocr"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(Image.fromarray(bin_), config=TESS_CONFIG)
        return [l.strip() for l in raw.splitlines() if len(l.strip())>=3 and any(c.isalpha() for c in l)]
    else:
        res = ocr_rapid(img)
        return [item[1].strip() for item in (res[0] if res else []) if len(item)>=2 and item[1].strip()]

def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    doc_id = manifest.get("file_hash")
    source = f"s3://{S3_BUCKET}/{s3_key}"

    mp = fitz.open(stream=raw, filetype="pdf")
    pp = pdfplumber.open(BytesIO(raw))
    saved = 0

    for idx, page in enumerate(mp):
        page_num = idx + 1
        chunk_id = f"{doc_id}_{idx}"
        t0 = time.perf_counter()

        pl_page = pp.pages[idx]
        words = pl_page.extract_words(use_text_flow=True)
        lines_map = {}
        for w in words:
            top = round(w["top"], 1)
            lines_map.setdefault(top, []).append(w)
        line_items = []
        for top in sorted(lines_map):
            row = sorted(lines_map[top], key=lambda x: x["x0"])
            text_line = " ".join(w["text"] for w in row)
            line_items.append((top, text_line))

        # OCR fallback
        combined_txt = "\n".join(t for _, t in line_items)
        needs_ocr = not DISABLE_OCR and (FORCE_OCR or not is_valid_text(combined_txt))
        pix = page.get_pixmap(dpi=RENDER_DPI) if needs_ocr else None
        if needs_ocr and pix:
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            ocr_lines = do_ocr(arr)
            if ocr_lines:
                line_items = [(0, "\n".join(ocr_lines))]

        # Image OCR
        images = pl_page.images
        arr_full = None
        if images:
            if pix is None:
                pix = page.get_pixmap(dpi=RENDER_DPI)
            arr_full = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr_full = cv2.cvtColor(arr_full, cv2.COLOR_BGRA2BGR)

        img_items = []
        if arr_full is not None:
            scale_x = pix.width / page.rect.width
            scale_y = pix.height / page.rect.height
            for i, img in enumerate(images, start=1):
                x0, y0, x1, y1 = img["x0"], img["top"], img["x1"], img["bottom"]
                px0, py0 = int(x0 * scale_x), int(y0 * scale_y)
                px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
                crop = arr_full[py0:py1, px0:px1]
                if crop.size < MIN_IMG_BYTES:
                    continue
                h, w = crop.shape[:2]
                if h < 50 or w < 50:
                    continue
                ocr_lines = do_ocr(crop)
                if not ocr_lines:
                    continue
                img_items.append((y0, f"[IMG_OCR:{i}]\n" + "\n".join(ocr_lines)))

        # Table extraction
        table_items = []
        tables_data = []
        try:
            tables = pl_page.extract_tables()
            for i, table in enumerate(tables, 1):
                try:
                    norm_table = []
                    for row in table:
                        clean_row = [
                            (cell if isinstance(cell, str) else "")
                            .replace("\n", " ").strip()
                            for cell in row
                        ]
                        norm_table.append(clean_row)
                    tables_data.append(norm_table)
                    table_text = "\n".join(["\t".join(r) for r in norm_table if any(r)])
                    table_items.append((0, f"[TABLE_{i}]\n{table_text}"))
                except Exception as e:
                    log.warning(f"Failed to process table p{page_num}#{i}: {e}")
        except Exception as e:
            log.warning(f"Table extraction error on page {page_num}: {e}")

        # Merge blocks
        merged = [(y, t) for y, t in line_items] + img_items + table_items
        merged.sort(key=lambda x: x[0])
        final_text = "\n".join(t for _, t in merged)

        payload = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "page_number": page_num,
            "source_type": "pdf",
            "source_path": source,
            "parser": OCR_BACKEND,
            "ocr_token_range": [0, len(final_text)],
            "text": final_text,
            "tables": tables_data,
            "metadata": {
                "used_ocr": bool(needs_ocr and pix),
                "is_multilingual": IS_MULTILINGUAL,
                "num_images": len(img_items),
                "num_tables": len(tables_data),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "parse_chunk_duration": None,
                "custom": {}
            }
        }

        duration_ms = int((time.perf_counter() - t0) * 1000)
        payload["metadata"]["parse_chunk_duration"] = duration_ms
        log.info(f"Parsed page {page_num} in {duration_ms} ms")

        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        out_key = f"{S3_CHUNKED_PREFIX}{chunk_id}.{ext}"
        body = (
            (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            if ext == "jsonl"
            else json.dumps(payload, indent=2, ensure_ascii=False).encode()
        )
        s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=body, ContentType="application/json")
        saved += 1

    mp.close()
    pp.close()
    total_ms = int((time.perf_counter() - start_all) * 1000)
    log.info(f"Completed parsing {saved} pages in {total_ms} ms total")
    return {"saved_chunks": saved, "total_parse_duration_ms": total_ms}
