import fitz  # PyMuPDF
import pdfplumber
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import fasttext
import os
import io
import boto3
from datetime import datetime

s3 = boto3.client("s3")

# Load models once
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
lid_model = fasttext.load_model("/models/lid.176.bin")  # Ensure this path is correct

def render_page_image(fitz_page, dpi=300):
    pix = fitz.Page.get_pixmap(fitz_page, dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def ocr_image_paddle(image: Image.Image) -> str:
    ocr_result = ocr_model.ocr(np.array(image), cls=True)
    return "\n".join(line[1][0] for line in ocr_result[0])

def detect_language(text: str):
    if not text.strip():
        return "unknown", False
    labels, probs = lid_model.predict(text, k=2)
    lang1 = labels[0].replace("__label__", "")
    p1 = probs[0]
    p2 = probs[1] if len(probs) > 1 else 0.0
    is_multi = (p2 > 0.2) and (p2 / p1 > 0.2)
    return lang1, is_multi

def parse_file(s3_key: str, manifest: dict) -> list:
    bucket = os.environ["S3_BUCKET"]
    filename = s3_key.split("/")[-1]
    timestamp = datetime.utcnow().isoformat()

    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    pdf_bytes = obj["Body"].read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    plumber = pdfplumber.open(io.BytesIO(pdf_bytes))

    chunks = []

    for i in range(len(doc)):
        used_ocr = False
        fitz_page = doc[i]
        bbox = list(fitz_page.rect)
        text = fitz_page.get_text("text").strip()

        if not text:
            image = render_page_image(fitz_page)
            text = ocr_image_paddle(image)
            used_ocr = True

        try:
            tables = plumber.pages[i].extract_tables()
        except Exception:
            tables = []

        lang, is_multi = detect_language(text)

        chunk = {
            "id": f"{manifest['sha256']}_page_{i+1}",
            "payload": {
                "document_id": manifest['sha256'],
                "chunk_id": f"chunk_{i+1}",
                "chunk_index": i,
                "text": text,
                "parser": "paddleocr + pdfplumber + layout",
                "pipeline_stage": "extracted",
                "source": {
                    "path": f"s3://{bucket}/{s3_key}",
                    "hash": manifest['sha256'],
                    "file_type": "pdf",
                    "page_number": i + 1
                },
                "metadata": {
                    "language": lang,
                    "is_multilingual": is_multi,
                    "is_ocr": used_ocr,
                    "chunk_type": "page",
                    "timestamp": timestamp,
                    "tags": []
                },
                "entities": [],
                "triplets": []
            },
            "tables": tables
        }
        chunks.append(chunk)

    doc.close()
    plumber.close()
    return chunks
