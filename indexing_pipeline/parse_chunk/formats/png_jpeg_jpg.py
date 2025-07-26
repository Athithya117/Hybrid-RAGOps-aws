import os, json, logging, boto3, warnings
from io import BytesIO
from datetime import datetime
from time import time as now
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

# --- ENV ---
S3_BUCKET = os.environ["S3_BUCKET"]
IS_MULTILINGUAL = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl")

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found",
                        category=UserWarning, module="torch.utils.data.dataloader")

log = logging.getLogger("pdf_parser")
log.setLevel(logging.INFO)
s3 = boto3.client("s3")
ocr = RapidOCR(multilingual=IS_MULTILINGUAL)

def parse_file(s3_key: str, manifest: dict):
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    source = f"s3://{S3_BUCKET}/{s3_key}"
    doc_id = manifest["sha256"]

    # PyMuPDF open
    doc = fitz.open(stream=raw, filetype="pdf")
    plumber = pdfplumber.open(BytesIO(raw))

    saved = 0
    for idx in range(len(doc)):
        page_num = idx + 1
        fitz_page = doc[idx]
        pdfpl_page = plumber.pages[idx]

        start_ms = int(now() * 1000)
        chunk_id = f"{doc_id}_{idx}"
        chunk = {
            "id": f"chunk_{chunk_id}",
            "payload": {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "page_number": page_num,
                "source_path": source,
                "source_hash": doc_id,
                "file_type": "pdf",
                "text": "",
                "tables": [],
                "images": [],
                "start_time": None,
                "end_time": None,
                "line_range": None,
                "metadata": {
                    "is_multilingual": IS_MULTILINGUAL,
                    "is_ocr": False,
                    "chunk_type": "page",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "errors": [],
                    "tags": [],
                    "layout_tags": [],
                    "confidence": None,
                    "decision_trace": [],
                    "chunk_duration_ms": None
                },
                "entities": [],
                "embedding": []
            }
        }

        # TEXT or OCR
        txt = fitz_page.get_text("text").strip()
        if txt:
            chunk["payload"]["text"] = txt
        else:
            pix = fitz_page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            ocr_res, _ = ocr(img)
            ocr_text = "\n".join([r[1] for r in ocr_res if r[1].strip()])
            chunk["payload"]["text"] = ocr_text
            chunk["payload"]["metadata"]["is_ocr"] = True

        # TABLES
        try:
            tables = pdfpl_page.extract_tables()
            if tables:
                chunk["payload"]["tables"] = tables
        except Exception as e:
            chunk["payload"]["metadata"]["errors"].append(f"table_extraction_error: {e}")

        # IMAGES
        imgs = fitz_page.get_images(full=True)
        for img_index, img_info in enumerate(imgs):
            xref = img_info[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n == 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_name = f"{chunk_id}_img{img_index+1}.png"
            img_bytes = pix.tobytes("png")
            key_img = os.path.join(os.getenv("S3_IMAGE_PREFIX", "data/images/"), img_name)
            s3.put_object(Bucket=S3_BUCKET, Key=key_img, Body=img_bytes,
                          ContentType="image/png")
            chunk["payload"]["images"].append(key_img)

        # metadata final
        chunk["payload"]["metadata"]["chunk_duration_ms"] = int(now() * 1000) - start_ms

        # SAVE chunk to S3
        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        key_chunk = f"{os.getenv('S3_CHUNKED_PREFIX','data/chunked/')}{chunk_id}.{ext}"
        body = (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8") if CHUNK_FORMAT == "jsonl" \
            else json.dumps(chunk, indent=2, ensure_ascii=False).encode("utf-8")

        s3.put_object(Bucket=S3_BUCKET, Key=key_chunk, Body=body, ContentType="application/json")
        log.info(f"Uploaded chunk page {page_num} to s3://{S3_BUCKET}/{key_chunk}")
        saved += 1

    doc.close()
    plumber.close()
    return {"saved_chunks": saved}
