import os
import json
import logging
import boto3
import fitz           # PyMuPDF
import pdfplumber
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from rapidocr_onnxruntime import RapidOCR

# --- ENV ---
S3_BUCKET        = os.environ["S3_BUCKET"]
CHUNK_FORMAT     = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl")
IMAGE_PREFIX     = os.getenv("S3_IMAGE_PREFIX", "data/images/")
CHUNK_PREFIX     = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")

log = logging.getLogger("pdf_parser")
log.setLevel(logging.INFO)
s3  = boto3.client("s3")

ocr = RapidOCR()  # native ONNX OCR

def parse_file(s3_key: str, manifest: dict):
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    source = f"s3://{S3_BUCKET}/{s3_key}"
    doc_id = manifest["sha256"]

    # open in both libraries
    pdf_stream = BytesIO(raw)
    mp_doc = fitz.open(stream=raw, filetype="pdf")
    pl_doc = pdfplumber.open(pdf_stream)

    saved = 0
    for page_index in range(len(mp_doc)):
        fitz_page    = mp_doc[page_index]
        plumber_page = pl_doc.pages[page_index]
        page_num     = page_index + 1

        chunk_id = f"{doc_id}_{page_index}"
        payload = {
            "document_id": doc_id,
            "chunk_id":   chunk_id,
            "page_number": page_num,
            "source_path": source,
            "text":       "",
            "tables":     [],
            "images":     [],
            "metadata": {
                "used_ocr": False,
                "num_tables": 0,
                "num_images": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }

        # 1) native text → OCR fallback
        text = fitz_page.get_text("text").strip()
        if text:
            payload["text"] = text
        else:
            # render → numpy → OCR
            pix = fitz_page.get_pixmap(dpi=300)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            ocr_res, _ = ocr(arr)
            lines = [r[1] for r in ocr_res if r[1].strip()]
            payload["text"] = "\n".join(lines)
            payload["metadata"]["used_ocr"] = True

        # 2) tables
        tables = plumber_page.extract_tables()
        if tables:
            payload["tables"] = tables
            payload["metadata"]["num_tables"] = len(tables)

        # 3) images
        for img_idx, img_info in enumerate(fitz_page.get_images(full=True)):
            xref = img_info[0]
            pix  = fitz.Pixmap(mp_doc, xref)
            if pix.n == 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_key  = f"{IMAGE_PREFIX}{doc_id}/page{page_num}_img{img_idx+1}.png"
            img_bytes= pix.tobytes("png")
            s3.put_object(Bucket=S3_BUCKET, Key=img_key, Body=img_bytes, ContentType="image/png")
            payload["images"].append(f"s3://{S3_BUCKET}/{img_key}")
            payload["metadata"]["num_images"] += 1

        # 4) upload chunk JSON
        ext   = "jsonl" if CHUNK_FORMAT=="jsonl" else "json"
        key   = f"{CHUNK_PREFIX}{chunk_id}.{ext}"
        body  = (json.dumps(payload, ensure_ascii=False)+"\n").encode() if ext=="jsonl" \
                else json.dumps(payload, indent=2, ensure_ascii=False).encode()
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
        log.info(f"Uploaded chunk page {page_num} → s3://{S3_BUCKET}/{key}")
        saved += 1

    pl_doc.close()
    mp_doc.close()
    return {"saved_chunks": saved}
