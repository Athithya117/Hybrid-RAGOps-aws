import os, json
from io import BytesIO
from datetime import datetime
import boto3
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from indexing_pipeline.parse_chunk.router import log

# ENVIRONMENT
IS_MULTILINGUAL = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
DOCLING_CACHE = os.getenv("DOCLING_ARTIFACTS_PATH", None)
TABLE_MODE = os.getenv("TABLE_MODE", "accurate").lower()


# Setup PdfPipelineOptions with RapidOCR integration
pdf_opts = PdfPipelineOptions(do_table_structure=True, do_ocr=True)
if TABLE_MODE == "fast":
    pdf_opts.table_structure_options.mode = TableFormerMode.FAST
else:
    pdf_opts.table_structure_options.mode = TableFormerMode.ACCURATE

if DOCLING_CACHE:
    pdf_opts.artifacts_path = DOCLING_CACHE

# For multilingual OCR, supply RapidOcrOptions using ONNX paths
if os.getenv("RAPIDOCR_MODEL_DOWNLOAD_PATH"):
    from huggingface_hub import snapshot_download
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")
    det = os.path.join(download_path, "PP‑OCRv4", os.getenv("RAPIDOCR_DET_MODEL", "en_PP‑OCRv3_det_infer.onnx"))
    rec = os.path.join(download_path, "PP‑OCRv4", os.getenv("RAPIDOCR_REC_MODEL", IS_MULTILINGUAL and "ch_PP‑OCRv4_rec_server_infer.onnx" or "en_PP‑OCRv3_rec_infer.onnx"))
    cls = os.path.join(download_path, "PP‑OCRv3", "ch_ppocr_mobile_v2.0_cls_infer.onnx")
    pdf_opts.ocr_options = RapidOcrOptions(det_model_path=det, rec_model_path=rec, cls_model_path=cls)

conv = DocumentConverter({
    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
})

def parse_file(s3_key, manifest):
    log(f"Docling parsing PDF {s3_key}, multilingual={IS_MULTILINGUAL}", level="INFO")
    s3 = boto3.client("s3")
    raw = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=s3_key)["Body"].read()
    ds = DocumentStream(name=s3_key, stream=BytesIO(raw))
    result = conv.convert(source=ds)
    doc = result.document
    ocr_used = result.pipeline_steps.get("ocr_used", False)

    chunks = []
    total_chars = 0

    # pre-index tables per page
    tables_by_page = {}
    for table in doc.tables:
        tables_by_page.setdefault(table.page_number - 1, []).append(table.export_to_dict())

    for page_no, pg in enumerate(doc.pages):
        text = "\n".join(t.text for t in pg.texts)
        total_chars += len(text)
        cid = f"{manifest['sha256']}_{page_no}"

        tables = tables_by_page.get(page_no) or None

        payload = {
            "document_id": manifest["sha256"],
            "chunk_id": cid,
            "chunk_index": page_no,
            "text": text,
            "source_path": f"s3://{os.getenv('S3_BUCKET')}/{s3_key}",
            "source_hash": manifest["sha256"],
            "file_type": "pdf",
            "page_number": page_no + 1,
            "start_time": None, "end_time": None, "line_range": None, "bbox": None,
            "metadata": {
                "is_multilingual": IS_MULTILINGUAL,
                "is_ocr": ocr_used,
                "chunk_type": "page",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tags": [], "layout_tags": [],
                "confidence": None,
                "errors": [],
                "decision_trace": []
            },
            "entities": [], "embedding": []
        }

        chunks.append({"id": f"chunk_{cid}", "payload": payload, "tables": tables})
        log(f"Docling chunk page {page_no} text_len={len(text)} tables={(len(tables) if tables else 0)}", level="DEBUG")

    log(f"Extracted {len(chunks)} pages, total chars={total_chars}", level="INFO")
    return chunks
