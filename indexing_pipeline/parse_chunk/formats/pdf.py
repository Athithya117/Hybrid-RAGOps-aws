import os
import io
import time
import logging
from datetime import datetime
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import fasttext
import boto3
import requests
import tarfile

from paddleocr import PaddleOCR
from langmap import get_paddle_lang

# Constants\ nMODEL_DIR = "models"
FASTTEXT_LID_PATH = os.path.join(MODEL_DIR, "lid.176.bin")
PADDLE_MODEL_URL_TEMPLATE = "https://paddleocr.bj.bcebos.com/ppocrv3/{lang}.tar"  # adjust if needed

# Environment vars\ nS3_BUCKET = os.environ["S3_BUCKET"]  # required
DEVICE = os.environ.get("DEVICE", "cpu")  # cpu or gpu

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Clients & models
s3 = boto3.client("s3")
lid_model = fasttext.load_model(FASTTEXT_LID_PATH)
oc r_cache = {}

# Helpers
def download_paddle_model(lang: str):
    url = PADDLE_MODEL_URL_TEMPLATE.format(lang=lang)
    dest_dir = os.path.join(MODEL_DIR, "paddleocr", lang)
    os.makedirs(dest_dir, exist_ok=True)

    tar_path = os.path.join(dest_dir, f"{lang}.tar")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tar_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dest_dir)
    os.remove(tar_path)
    return dest_dir


def get_ocr_model(lang: str):
    ocr_lang = get_paddle_lang(lang)
    if ocr_lang not in ocr_cache:
        model_dir = os.path.join(MODEL_DIR, "paddleocr", ocr_lang)
        if not os.path.exists(model_dir):
            logging.info(f"Downloading OCR model for '{ocr_lang}'")
            download_paddle_model(ocr_lang)
        ocr_cache[ocr_lang] = PaddleOCR(
            lang=ocr_lang,
            det_model_dir=None,
            rec_model_dir=None,
            cls_model_dir=None,
            use_angle_cls=True,
            use_space_char=True,
            device=DEVICE
        )
    return ocr_cache[ocr_lang]


def detect_language(text: str):
    if not text.strip():
        return "unknown", False
    labels, probs = lid_model.predict(text.replace("\n", " "), k=2)
    ft_lang = labels[0].replace("__label__", "")
    is_multi = len(probs) > 1 and (probs[1] / probs[0] > 0.2)
    return ft_lang, is_multi


def render_page_image(page, dpi=300):
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def ocr_image(image: Image.Image, lang: str) -> str:
    try:
        ocr = get_ocr_model(lang)
        res = ocr.ocr(np.array(image), cls=True)
        lines = [line[1][0] for line in res[0]]
        return "\n".join(lines)
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""


def normalize_tables(raw_tables, page_num, doc_hash):
    out = []
    for idx, tbl in enumerate(raw_tables):
        if not tbl or all(all(not cell for cell in row) for row in tbl):
            continue
        maxc = max(len(r) for r in tbl)
        rows = [r + [""] * (maxc - len(r)) for r in tbl]
        out.append({
            "table_id": f"{doc_hash}_page_{page_num}_table_{idx}",
            "table_index": idx,
            "page_number": page_num,
            "bbox": None,
            "rows": rows
        })
    return out


def parse_file(s3_key: str, manifest: dict) -> list:
    ts = datetime.utcnow().isoformat()
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    except Exception as e:
        logging.error(f"S3 fetch error: {e}")
        return []

    try:
        doc = fitz.open(stream=body, filetype="pdf")
        plumber = pdfplumber.open(io.BytesIO(body))
    except Exception as e:
        logging.error(f"PDF open error: {e}")
        return []

    if doc.is_encrypted:
        logging.warning(f"Encrypted PDF skipped: {s3_key}")
        doc.close()
        plumber.close()
        return []

    chunks = []
    size_bytes = len(body)

    for i, page in enumerate(doc):
        start = time.time()
        used_ocr = False
        text = page.get_text("text").strip()

        ft_lang, is_multi = detect_language(text)
        if len(text) < 15:
            img = render_page_image(page, dpi=200)
            text = ocr_image(img, ft_lang)
            used_ocr = True
            ft_lang, is_multi = detect_language(text)

        raw_tables = []
        try:
            raw_tables = plumber.pages[i].extract_tables()
        except Exception:
            pass

        tables = normalize_tables(raw_tables, i+1, manifest['sha256'])
        duration_ms = int((time.time() - start) * 1000)

        chunk = {
            "id": f"{manifest['sha256']}_page_{i+1}",
            "payload": {
                "document_id": manifest['sha256'],
                "chunk_id": f"{manifest['sha256'][:8]}_p{i+1}_c0",
                "chunk_index": i,
                "text": text,
                "parser": "paddleocr+pdfplumber+fitz",
                "pipeline_stage": "extracted",
                "source_path": f"s3://{S3_BUCKET}/{s3_key}",
                "source_hash": manifest['sha256'],
                "file_type": "pdf",
                "page_number": i+1,
                "bbox": [0, 0, page.rect.width, page.rect.height],
                "metadata": {
                    "language": get_paddle_lang(ft_lang),
                    "is_multilingual": is_multi,
                    "is_ocr": used_ocr,
                    "chunk_type": "page",
                    "timestamp": ts
                },
                "parse_duration_ms": duration_ms,
                "source_size_bytes": size_bytes,
                "parser_version": "pdf_parser_v1.3.0"
            },
            "tables": tables
        }
        chunks.append(chunk)

    doc.close()
    plumber.close()
    return chunks
