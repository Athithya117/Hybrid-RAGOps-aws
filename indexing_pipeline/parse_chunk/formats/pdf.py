import os
import sys
import json
import logging
import boto3
import fitz  # PyMuPDF
import pdfplumber
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from PIL import Image
import pytesseract

# --- Determine OCR backend ---
OCR_BACKEND = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
if OCR_BACKEND not in ("rapidocr", "tesseract", "indicocr"):
    print(f"ERROR: OCR_ENGINE must be 'rapidocr', 'tesseract', or 'indicocr', got '{OCR_BACKEND}'", file=sys.stderr)
    sys.exit(1)

IS_MULTILINGUAL = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
if IS_MULTILINGUAL and OCR_BACKEND == "rapidocr":
    print("ERROR: multilingual OCR not supported with rapidocr. Use tesseract or indicocr", file=sys.stderr)
    sys.exit(1)

# --- AWS & Logging ---
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("pdf_parser")
s3 = boto3.client("s3")

# --- Tesseract config ---
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

raw_langs = os.getenv("TESSERACT_LANG", "").replace(",", " ").replace(";", " ")
langs_list = [l for l in raw_langs.split() if l]
TESS_LANGS = '+'.join(langs_list) if langs_list else 'eng'
TESS_CONFIG = f"-l {TESS_LANGS} --oem 1 --psm 6"
override = os.getenv("TESSERACT_CONFIG_OVERRIDE", "").strip()
if override:
    TESS_CONFIG = override

# --- Other configs ---
S3_BUCKET = os.getenv("S3_BUCKET") or sys.exit("Missing S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/")
S3_IMAGE_PREFIX = os.getenv("S3_IMAGE_PREFIX", "data/images/")
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").strip().lower()
if CHUNK_FORMAT not in ("json", "jsonl"):
    sys.exit("CHUNK_FORMAT must be 'json' or 'jsonl'")
DISABLE_OCR = os.getenv("DISABLE_OCR", "false").lower() == "true"
FORCE_OCR = os.getenv("FORCE_OCR", "false").lower() == "true"
RENDER_DPI = int(os.getenv("OCR_RENDER_DPI", "500"))
DEBUG_SAVE_IMG = os.getenv("DEBUG_SAVE_IMG", "false").lower() == "true"

# --- Utility functions ---
def is_valid_text(text: str) -> bool:
    t = text.strip()
    return len(t) > 20 and any(c.isalpha() for c in t)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(bi, cv2.MORPH_OPEN, kernel)
    return clean

def do_ocr_tesseract(img: np.ndarray) -> list[str]:
    pre = preprocess_image(img)
    pil = Image.fromarray(pre)
    raw = pytesseract.image_to_string(pil, config=TESS_CONFIG)
    lines = []
    for line in raw.splitlines():
        text = line.strip()
        if len(text) < 3:
            continue
        if not any(c.isalpha() for c in text):
            continue
        lines.append(text)
    return lines

# --- RapidOCR optional ---
if OCR_BACKEND == "rapidocr":
    from rapidocr_onnxruntime import RapidOCR
    ocr_rapid = RapidOCR()

def do_ocr_rapidocr(img: np.ndarray) -> list[str]:
    res = ocr_rapid(img)
    lines = []
    if res and res[0]:
        for item in res[0]:
            txt = item[1]
            if txt and txt.strip():
                lines.append(txt.strip())
    return lines

# --- Main parse function ---
def parse_file(s3_key: str, manifest: dict) -> dict:
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    source = f"s3://{S3_BUCKET}/{s3_key}"
    doc_id = manifest["sha256"]

    mp_doc = fitz.open(stream=raw, filetype="pdf")
    pl_doc = pdfplumber.open(BytesIO(raw))
    saved = 0

    for idx, page in enumerate(mp_doc):
        start = datetime.utcnow()
        page_num = idx + 1
        chunk_id = f"{doc_id}_{idx}"

        payload = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "page_number": page_num,
            "source_type": "pdf",
            "source_path": source,
            "parser": OCR_BACKEND,
            "line_range": None,
            "start_time": None,
            "end_time": None,
            "html_blocks": [],
            "markdown_blocks": [],
            "text": "",
            "tables": [],
            "images": [],
            "metadata": {
                "used_ocr": False,
                "is_multilingual": IS_MULTILINGUAL,
                "num_tables": 0,
                "num_images": 0,
                "timestamp": start.isoformat() + "Z",
                "parse_chunk_duration": None,
                "custom": {}
            }
        }

        raw_text = page.get_text("text") or ""
        imgs = page.get_images(full=True) or []
        needs = not DISABLE_OCR and (FORCE_OCR or not is_valid_text(raw_text) or bool(imgs))
        arr = None

        if needs:
            pix = page.get_pixmap(dpi=RENDER_DPI)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if DEBUG_SAVE_IMG:
                cv2.imwrite(f"/tmp/page_{page_num}.png", arr)

        ocr_lines = []
        if needs and arr is not None:
            try:
                if OCR_BACKEND in ("tesseract", "indicocr"):
                    ocr_lines = do_ocr_tesseract(arr)
                else:
                    ocr_lines = do_ocr_rapidocr(arr)
                if ocr_lines:
                    payload["metadata"]["used_ocr"] = True
            except Exception as e:
                log.warning(f"OCR error p{page_num}: {e}")

        if is_valid_text(raw_text):
            payload["text"] = raw_text.strip()
            if ocr_lines:
                payload["text"] += "\n[PAGE_OCR]\n" + "\n".join(ocr_lines)
        elif ocr_lines:
            payload["text"] = "\n".join(ocr_lines)

        try:
            tlist = pl_doc.pages[idx].extract_tables() or []
            if tlist:
                payload["tables"] = tlist
                payload["metadata"]["num_tables"] = len(tlist)
        except Exception as e:
            log.warning(f"Table error p{page_num}: {e}")

        pw, ph = page.rect.width, page.rect.height
        srcs = imgs or (["FULLPAGE"] if needs else [])

        for i, info in enumerate(srcs, start=1):
            try:
                if info == 'FULLPAGE':
                    img_bytes = pix.tobytes('png')
                    img_cv = arr.copy()
                else:
                    xref = info[0]
                    pi = fitz.Pixmap(mp_doc, xref)
                    if pi.n == 4:
                        pi = fitz.Pixmap(fitz.csRGB, pi)
                    img_bytes = pi.tobytes('png')
                    img_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                h, w = img_cv.shape[:2]
                if h < 100 or w < 100 or (w * h) / (pw * ph) < 0.02:
                    continue

                lines = []
                if not DISABLE_OCR:
                    if OCR_BACKEND in ("tesseract", "indicocr"):
                        lines = do_ocr_tesseract(img_cv)
                    else:
                        lines = do_ocr_rapidocr(img_cv)
                if not lines:
                    continue

                img_key = f"{S3_IMAGE_PREFIX}{doc_id}/page{page_num}_img{i}.png"
                s3.put_object(Bucket=S3_BUCKET, Key=img_key,
                              Body=img_bytes, ContentType='image/png')
                payload["images"].append(f"s3://{S3_BUCKET}/{img_key}")
                payload["metadata"]["num_images"] += 1

                payload["text"] += f"\n[IMG_OCR:{i}]\n" + "\n".join(lines)

            except Exception as e:
                log.warning(f"Image OCR error p{page_num}#{i}: {e}")

        duration = (datetime.utcnow() - start).total_seconds() * 1000
        payload["metadata"]["parse_chunk_duration"] = int(duration)

        ext = 'jsonl' if CHUNK_FORMAT == 'jsonl' else 'json'
        key = f"{S3_CHUNKED_PREFIX}{chunk_id}.{ext}"
        body = (json.dumps(payload, ensure_ascii=False) + '\n').encode() if ext == 'jsonl' \
            else json.dumps(payload, indent=2, ensure_ascii=False).encode()
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType='application/json')
        saved += 1

    mp_doc.close()
    pl_doc.close()
    return {"saved_chunks": saved}

# --- Entrypoint ---
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: script.py <s3_key> <manifest_json>', file=sys.stderr)
        sys.exit(1)
    key = sys.argv[1]
    manifest = json.loads(sys.argv[2])
    result = parse_file(key, manifest)
    print(json.dumps(result, indent=2))
