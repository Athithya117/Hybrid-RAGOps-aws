# indexing_pipeline/parse_chunk/formats/pdf.py

import os
import boto3
import json
from io import BytesIO
from datetime import datetime

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from indexing_pipeline.parse_chunk.router import log

# ─── ENVIRONMENT ────────────────────────────────────────────────────────────────
S3_BUCKET       = os.environ["S3_BUCKET"]
IS_MULTILINGUAL = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
DOCLING_CACHE   = os.getenv("DOCLING_ARTIFACTS_PATH", None)
TABLE_MODE      = os.getenv("TABLE_MODE", "accurate").lower()

# ─── S3 CLIENT ──────────────────────────────────────────────────────────────────
s3 = boto3.client("s3")

# ─── DOCING PIPELINE SETUP ──────────────────────────────────────────────────────
# TableFormerMode.ACCURATE is more robust but slower; FAST is quicker
table_mode = (
    TableFormerMode.ACCURATE
    if TABLE_MODE != "fast"
    else TableFormerMode.FAST
)

# Build PdfPipelineOptions with required sub‑models
pdf_opts = PdfPipelineOptions(
    do_ocr=True,
    is_multilingual=IS_MULTILINGUAL,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=table_mode
    ),
    # artifacts_path tells Docling where to cache layout & table models
    artifacts_path=DOCLING_CACHE
)

converter = DocumentConverter({
    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
})

# ─── PARSER ENTRYPOINT ──────────────────────────────────────────────────────────
def parse_file(s3_key: str, manifest: dict) -> list[dict]:
    """
    Downloads the PDF at `s3_key`, runs Docling to detect layout, tables, OCR 
    and emits one JSON-serializable chunk per page.
    """
    log(f"Docling parsing PDF {s3_key}  multilingual={IS_MULTILINGUAL}", level="INFO")

    # 1) Download raw bytes
    try:
        raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    except Exception as e:
        log(f"Failed to download {s3_key}: {e}", level="ERROR")
        return []

    # 2) Wrap in DocumentStream and convert
    stream = DocumentStream(name=s3_key, stream=BytesIO(raw))
    try:
        result = converter.convert(source=stream)
    except Exception as e:
        log(f"Docling conversion error for {s3_key}: {e}", level="ERROR")
        return []

    doc = result.document
    chunks = []
    total_chars = 0

    # 3) Page‑by‑page chunking
    for page_no, page in enumerate(doc.pages):
        # Combine all text spans in reading order
        text = "\n".join(span.text for span in page.texts).strip()
        total_chars += len(text)

        # Extract tables on this page (if any)
        tables = [
            tbl.export_to_dict() 
            for tbl in doc.tables 
            if tbl.page_number == page_no + 1
        ] or None

        # Build the chunk payload
        cid = f"{manifest['sha256']}_{page_no}"
        payload = {
            "document_id": manifest["sha256"],
            "chunk_id": cid,
            "chunk_index": page_no,
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
                "is_multilingual": IS_MULTILINGUAL,
                "is_ocr": result.pipeline_steps.get("ocr_used", False),
                "chunk_type": "page",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tags": [],
                "layout_tags": [],
                "confidence": None,
                "errors": [],
                "decision_trace": []
            },
            "entities": [],
            "embedding": []
        }

        chunks.append({
            "id": f"chunk_{cid}",
            "payload": payload,
            "tables": tables
        })

        log(f"Page {page_no+1}: text_len={len(text)} tables={len(tables or [])}", level="DEBUG")

    log(f"Extracted {len(chunks)} pages, total_chars={total_chars}", level="INFO")
    return chunks
