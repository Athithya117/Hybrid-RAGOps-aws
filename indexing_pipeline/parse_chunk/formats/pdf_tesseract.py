import os
import json
import logging
import boto3
import fitz
import pdfplumber
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from rapidocr_onnxruntime import RapidOCR

# --- ENV ---
S3_BUCKET       = os.environ["S3_BUCKET"]
CHUNK_FORMAT    = os.getenv("CHUNK_FORMAT", "json").lower()
IMAGE_PREFIX    = os.getenv("S3_IMAGE_PREFIX", "data/images/")
CHUNK_PREFIX    = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")
MIN_IMG_SIZE    = int(os.getenv("MIN_IMG_SIZE_BYTES", "2048"))
RENDER_DPI      = int(os.getenv("RENDER_DPI", "300"))
FORCE_OCR       = os.getenv("FORCE_OCR", "false").lower() == "true"
DEBUG_SAVE_IMG  = os.getenv("DEBUG_SAVE_IMG", "false").lower() == "true"

assert CHUNK_FORMAT in ("json", "jsonl"), "CHUNK_FORMAT must be 'json' or 'jsonl'"

log = logging.getLogger("pdf_parser")
log.setLevel(logging.INFO)
s3 = boto3.client("s3")
ocr = RapidOCR()

def is_valid_text(text: str) -> bool:
    t = text.strip()
    return len(t) > 20 and any(c.isalpha() for c in t)

def parse_file(s3_key: str, manifest: dict):
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    source = f"s3://{S3_BUCKET}/{s3_key}"
    doc_id = manifest["sha256"]

    mp_doc = fitz.open(stream=raw, filetype="pdf")
    pl_doc = pdfplumber.open(BytesIO(raw))
    saved_chunks = 0

    for idx in range(len(mp_doc)):
        page_start = datetime.utcnow()

        page_num = idx + 1
        chunk_id = f"{doc_id}_{idx}"
        payload = {
            "document_id":     doc_id,
            "chunk_id":        chunk_id,
            "page_number":     page_num,
            "line_range":      None,
            "source_type":     "pdf",
            "source_path":     source,
            "start_time":      None,
            "end_time":        None,
            "text":            "",
            "tables":          [],
            "images":          [],
            "image_ocr":       [],
            "html_blocks":     [],
            "markdown_blocks": [],
            "metadata": {
                "used_ocr":     False,
                "num_tables":   0,
                "num_images":   0,
                "timestamp":    datetime.utcnow().isoformat() + "Z",
                "custom":       {}
            }
        }

        fitz_page       = mp_doc[idx]
        raw_text        = fitz_page.get_text("text") or ""
        embedded_images = fitz_page.get_images(full=True) or []

        # Decide whether to render & OCR
        needs_render = FORCE_OCR or not is_valid_text(raw_text) or bool(embedded_images)
        pix, arr = None, None

        if needs_render:
            pix = fitz_page.get_pixmap(dpi=RENDER_DPI)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if DEBUG_SAVE_IMG:
                cv2.imwrite(f"/tmp/page_{page_num}.png", arr)

        # Full-page OCR
        ocr_lines = []
        if needs_render and arr is not None:
            try:
                ocr_out = ocr(arr)
                if ocr_out:
                    for r in (ocr_out[0] or []):
                        if len(r) > 1 and r[1].strip():
                            ocr_lines.append(r[1].strip())
                    if ocr_lines:
                        payload["metadata"]["used_ocr"] = True
            except Exception as e:
                log.warning(f"OCR error on page {page_num}: {e}")

        # Merge native + OCR text
        if is_valid_text(raw_text):
            payload["text"] = raw_text.strip()
            if ocr_lines:
                payload["text"] += "\n[PAGE_OCR]\n" + "\n".join(ocr_lines)
        elif ocr_lines:
            payload["text"] = "\n".join(ocr_lines)

        # Extract tables
        try:
            tables = pl_doc.pages[idx].extract_tables() or []
            if tables:
                payload["tables"] = tables
                payload["metadata"]["num_tables"] = len(tables)
        except Exception as e:
            log.warning(f"Table extraction error on page {page_num}: {e}")

        # Selective image extraction + per-image OCR
        page_w, page_h = fitz_page.rect.width, fitz_page.rect.height
        image_sources = embedded_images if embedded_images else (["FULLPAGE"],) if needs_render else []

        for img_idx, img_info in enumerate(image_sources):
            try:
                if img_info[0] == "FULLPAGE":
                    if pix is None:
                        continue
                    img_bytes = pix.tobytes("png")
                    img_cv = arr.copy()
                else:
                    xref = img_info[0]
                    pi = fitz.Pixmap(mp_doc, xref)
                    if pi.n == 4:
                        pi = fitz.Pixmap(fitz.csRGB, pi)
                    img_bytes = pi.tobytes("png")
                    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_cv = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                h, w = img_cv.shape[:2]
                if h < 100 or w < 100:
                    continue
                if (w * h) / (page_w * page_h) < 0.02:
                    continue

                # OCR on image
                ocr_img_text = ""
                try:
                    img_out = ocr(img_cv)
                    if img_out:
                        for r in (img_out[0] or []):
                            if len(r) > 1 and r[1].strip():
                                ocr_img_text += r[1].strip() + "\n"
                except Exception as e:
                    log.warning(f"OCR error on image p{page_num}#{img_idx+1}: {e}")
                    continue

                if len(ocr_img_text.strip()) < 10:
                    continue

                # Upload image
                img_key = f"{IMAGE_PREFIX}{doc_id}/page{page_num}_img{img_idx+1}.png"
                s3.put_object(Bucket=S3_BUCKET, Key=img_key,
                              Body=img_bytes, ContentType="image/png")
                payload["images"].append(f"s3://{S3_BUCKET}/{img_key}")
                payload["metadata"]["num_images"] += 1

                # Structured per-image OCR
                payload["image_ocr"].append({
                    "image_s3": f"s3://{S3_BUCKET}/{img_key}",
                    "text":      ocr_img_text.strip()
                })

                # Inline tag in page text
                payload["text"] += f"\n[IMG_OCR:{img_idx+1}]\n" + ocr_img_text.strip()

            except Exception as e:
                log.warning(f"Image process failed on page {page_num} idx {img_idx+1}: {e}")

        # record chunk duration
        page_end = datetime.utcnow()
        payload["metadata"]["parse_chunk_duration"] = int((page_end - page_start).total_seconds() * 1000)
        payload["start_time"] = None
        payload["end_time"]   = None

        # Upload JSON chunk
        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        key = f"{CHUNK_PREFIX}{chunk_id}.{ext}"
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode() if ext == "jsonl" \
               else json.dumps(payload, indent=2, ensure_ascii=False).encode()
        s3.put_object(Bucket=S3_BUCKET, Key=key,
                      Body=body, ContentType="application/json")
        log.info(f"Uploaded page {page_num} → s3://{S3_BUCKET}/{key}")
        saved_chunks += 1

    mp_doc.close()
    pl_doc.close()
    return {"saved_chunks": saved_chunks}
