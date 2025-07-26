import os, boto3, json, logging, time
from io import BytesIO
from datetime import datetime
from huggingface_hub import snapshot_download
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode

# ENV VARS
S3_BUCKET = os.environ["S3_BUCKET"]
IS_MULTILINGUAL = os.getenv("IS_MULTILINGUAL", "false").lower() == "true"
PDF_TABLE_MODE = os.getenv("PDF_TABLE_MODE", "accurate").lower()
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl")
os.environ["DOCLING_OCR_ENGINE"] = "rapidocr"
os.environ.setdefault("RAPIDOCR_CACHE", os.path.expanduser("~/.cache/rapidocr"))

logging.getLogger("pdf_parser").setLevel(logging.INFO)
log = logging.getLogger("pdf_parser")
s3 = boto3.client("s3")
_CONVERTER = None

def get_rapidocr_models():
    repo = snapshot_download("SWHL/RapidOCR", cache_dir=os.environ["RAPIDOCR_CACHE"])
    det = os.path.join(repo, "PP-OCRv4", "ch_PP-OCRv4_det_infer.onnx")
    rec = os.path.join(repo, "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx") if IS_MULTILINGUAL \
          else os.path.join(repo, "PP-OCRv3/en_PP-OCRv3_rec_infer.onnx")
    cls = os.path.join(repo, "PP-OCRv3/ch_ppocr_mobile_v2.0_cls_train.onnx")
    return det, rec, cls

def create_docling_converter():
    det, rec, cls = get_rapidocr_models()
    ocr_opts = RapidOcrOptions(det_model_path=det, rec_model_path=rec, cls_model_path=cls)
    pdf_opts = PdfPipelineOptions(do_ocr=True, do_table_structure=(PDF_TABLE_MODE == "accurate"))
    pdf_opts.ocr_options = ocr_opts
    pdf_opts.table_structure_options.mode = TableFormerMode.ACCURATE if PDF_TABLE_MODE == "accurate" else TableFormerMode.FAST
    return DocumentConverter({InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)})

def _get_converter():
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = create_docling_converter()
    return _CONVERTER

def parse_file(s3_key: str, manifest: dict):
    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    source = f"s3://{S3_BUCKET}/{s3_key}"
    doc_id = manifest["sha256"]
    result = _get_converter().convert(source=DocumentStream(source, BytesIO(raw)))

    if result.status != result.status.SUCCESS:
        raise RuntimeError(f"Docling conversion failed: {result.status}")

    full_text = result.document.export_to_text()
    pages = full_text.split("\f")

    saved = 0
    for idx, page_text in enumerate(pages):
        page_text = page_text.strip()
        if not page_text:
            continue

        start_t = time.perf_counter()

        chunk_id = f"{doc_id}_{idx}"
        obj = {
            "id": f"chunk_{chunk_id}",
            "payload": {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": page_text,
                "source_path": source,
                "source_hash": doc_id,
                "file_type": "pdf",
                "page_number": idx + 1,
                "start_time": None,
                "end_time": None,
                "line_range": None,
                "metadata": {
                    "is_multilingual": IS_MULTILINGUAL,
                    "is_ocr": True,
                    "chunk_type": "page",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "chunk_duration_ms": None,
                    "errors": [],
                    "tags": [],
                    "layout_tags": [],
                    "confidence": None,
                    "decision_trace": []
                },
                "entities": [],
                "embedding": []
            }
        }

        end_t = time.perf_counter()
        duration_ms = (end_t - start_t) * 1000
        obj["payload"]["metadata"]["chunk_duration_ms"] = round(duration_ms, 3)

        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        prefix = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")
        key = f"{prefix}{doc_id}_{idx}.{ext}"
        body = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8") if CHUNK_FORMAT == "jsonl" \
               else json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
        log.info(f"Uploaded chunk {idx} to s3://{S3_BUCKET}/{key}")
        saved += 1

    return {"saved_chunks": saved}
