import os
import sys
import boto3
import tempfile
import subprocess
import warnings
from botocore.exceptions import ClientError
from indexing_pipeline.parse_chunk.router import env_or_fail, log, retry, list_raw_files

# --- Suppress known safe warnings ---
warnings.filterwarnings("ignore")

# --- ENV ---
S3_BUCKET                = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX            = os.getenv("S3_RAW_PREFIX", "data/raw/")
OVERWRITE_OTHER_TO_PDF   = os.getenv("OVERWRITE_DOC_DOCX_TO_PDF", "true").lower() == "true"

# --- Supported Extensions ---
CONVERTIBLE = {"doc", "docx"}

# --- High-fidelity PDF export filter ---
PDF_FILTER = (
    "pdf:writer_pdf_Export:"
    "EmbedStandardFonts=true;"
    "SelectPdfVersion=1;"
    "ReduceImageResolution=false;"
    "Quality=100"
)

s3 = boto3.client("s3")

def download_from_s3(key, tmp_dir):
    local_path = os.path.join(tmp_dir, os.path.basename(key))
    retry(lambda: s3.download_file(S3_BUCKET, key, local_path))
    return local_path

def upload_to_s3(local_path, key):
    with open(local_path, "rb") as f:
        body = f.read()
    retry(lambda: s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/pdf"
    ))
    log(f"Uploaded PDF to s3://{S3_BUCKET}/{key}")

def libreoffice_convert(input_path, output_dir, target_ext):
    command = [
        "soffice",
        "--headless",
        "--invisible",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--nologo",
        "--convert-to", target_ext,
        "--outdir", output_dir,
        input_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.{target_ext.split(':')[0]}")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"{target_ext.split(':')[0].upper()} not created.")
    return out_path

def convert_and_upload(key):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local_input = download_from_s3(key, tmp)
            ext = key.split(".")[-1].lower()

            if ext == "doc":
                # Step 1: Convert .doc → .docx
                docx_path = libreoffice_convert(local_input, tmp, "docx")
                pdf_path = libreoffice_convert(docx_path, tmp, PDF_FILTER)
                pdf_key = f"{S3_RAW_PREFIX}{os.path.basename(key)}.pdf"
            else:
                # .docx → PDF
                pdf_path = libreoffice_convert(local_input, tmp, PDF_FILTER)
                pdf_key = f"{S3_RAW_PREFIX}{os.path.basename(key)}.pdf"

            upload_to_s3(pdf_path, pdf_key)

            if OVERWRITE_OTHER_TO_PDF:
                log(f"OVERWRITE_OTHER_TO_PDF=true → deleting original {key}")
                s3.delete_object(Bucket=S3_BUCKET, Key=key)

        except Exception as e:
            log(f"Conversion failed for {key}: {e}", level="ERROR")

def main():
    log("Running doc_to_pdf.py conversion pass")
    log(f"OVERWRITE_OTHER_TO_PDF={OVERWRITE_OTHER_TO_PDF}")
    keys = list(list_raw_files())
    log(f"Scanning {len(keys)} files in s3://{S3_BUCKET}/{S3_RAW_PREFIX}")

    for key in keys:
        ext = key.split(".")[-1].lower()
        if ext in CONVERTIBLE:
            log(f"Converting '{key}' to PDF with high-fidelity filter")
            convert_and_upload(key)

if __name__ == "__main__":
    main()
