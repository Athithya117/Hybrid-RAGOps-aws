import os
import json
import boto3
from datetime import datetime
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions
)
from indexing_pipeline.parse_chunk.router import log

# ENVIRONMENT VARIABLES
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl"), f"Unsupported CHUNK_FORMAT '{CHUNK_FORMAT}'"

S3_BUCKET = os.getenv("S3_BUCKET")
assert S3_BUCKET, "S3_BUCKET must be set"

s3 = boto3.client("s3")

def parse_file(s3_key, manifest):
    log(f"Parsing PDF page-by-page: {s3_key}")

    # Download PDF
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        pdf_bytes = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    except Exception as e:
        log(f"ERROR: Could not fetch PDF {s3_key}: {e}", level="ERROR")
        return []

    if not pdf_bytes:
        log(f"ERROR: PDF empty: {s3_key}", level="ERROR")
        return []

    # Wrap for in-memory conversion
    stream = BytesIO(pdf_bytes)
    doc_stream = DocumentStream(name=os.path.basename(s3_key), stream=stream)

    # Setup OCR correctly per examples :contentReference[oaicite:1]{index=1}
    ocr_opts = TesseractCliOcrOptions(force_full_page_ocr=True)
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    pdf_opts.ocr_options = ocr_opts

    # Apply correct format mapping
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions as _PDFOption
    from docling.document_converter import PdfFormatOption

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )

    try:
        dl_doc = converter.convert(doc_stream)
    except Exception as e:
        log(f"ERROR: PDF conversion failed for {s3_key}: {e}", level="ERROR")
        return []

    pages = getattr(dl_doc, "pages", [])
    if not pages:
        log(f"WARNING: No pages detected in {s3_key}", level="WARN")
        return []

    # Build chunks
    chunks = []
    for idx, page in enumerate(pages):
        text = (getattr(page, "text", "") or "").strip()
        if not text:
            log(f"WARNING: Page {idx+1} empty", level="WARN")

        page_no = getattr(page, "page_no", idx)
        is_ocr = getattr(page, "is_ocr", False)
        chunk_id = f"{manifest['sha256']}_{idx}"

        chunks.append({
            "id": f"chunk_{chunk_id}",
            "payload": {
                "document_id": manifest["sha256"],
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": text,
                "source_path": f"s3://{S3_BUCKET}/{s3_key}",
                "source_hash": manifest["sha256"],
                "file_type": "pdf",
                "page_number": page_no + 1,
                "start_time": None,
                "end_time": None,
                "line_range": None,
                "bbox": None,
                "metadata": {
                    "is_multilingual": False,
                    "is_ocr": is_ocr,
                    "chunk_type": "page",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "tags": [],
                    "layout_tags": [],
                    "confidence": None,
                },
                "entities": [],
                "embedding": []
            },
            "tables": None
        })

    log(f"Extracted {len(chunks)} pages, total chars: {sum(len(c['payload']['text']) for c in chunks)}")

    return [json.dumps(c, ensure_ascii=False) + "\n" for c in chunks] if CHUNK_FORMAT == "jsonl" else chunks
