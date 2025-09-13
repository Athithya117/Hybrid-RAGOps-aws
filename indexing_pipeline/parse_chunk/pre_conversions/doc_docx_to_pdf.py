import os
import sys
import time
import tempfile
import warnings
from pathlib import Path

import boto3
import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.connection import NoConnectException

from indexing_pipeline.parse_chunk.router import env_or_fail, log, retry, list_raw_files

warnings.filterwarnings("ignore")
S3_BUCKET = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/")
OVERWRITE_OTHER_TO_PDF = os.getenv("OVERWRITE_DOC_DOCX_TO_PDF", "true").lower() == "true"
LIBREOFFICE_PORTS = os.getenv("LIBREOFFICE_PORTS", "7003")
CONNECT_RETRIES = int(os.getenv("UNO_CONNECT_RETRIES", "6"))
CONNECT_DELAY_SECONDS = float(os.getenv("UNO_CONNECT_DELAY_SECONDS", "1.0"))
CONVERT_RETRIES = int(os.getenv("UNO_CONVERT_RETRIES", "2"))
s3 = boto3.client("s3")
PDF_FILTER_OPTIONS = "EmbedStandardFonts=true;SelectPdfVersion=1;ReduceImageResolution=false;Quality=100"


def _parse_ports(ports_str):
    return [p.strip() for p in ports_str.split(",") if p.strip()]


def connect_uno(ports=None, retries=CONNECT_RETRIES, delay=CONNECT_DELAY_SECONDS):
    ports = ports or _parse_ports(LIBREOFFICE_PORTS)
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local_ctx)
    last_exc = None
    for attempt in range(retries):
        for port in ports:
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                return ctx.ServiceManager
            except NoConnectException as e:
                last_exc = e
            except Exception as e:
                last_exc = e
        time.sleep(delay)
    raise RuntimeError(f"Unable to connect to UNO server on ports {ports}") from last_exc


def _make_prop(name, value):
    return PropertyValue(name, 0, value, 0)


def store_to_pdf(lo_service_mgr, input_path, output_path, options=PDF_FILTER_OPTIONS):
    desktop = lo_service_mgr.createInstanceWithContext("com.sun.star.frame.Desktop", lo_service_mgr)
    input_url = uno.systemPathToFileUrl(str(Path(input_path).resolve()))
    props = (PropertyValue("Hidden", 0, True, 0),)
    doc = desktop.loadComponentFromURL(input_url, "_blank", 0, props)
    try:
        pdf_props = (
            _make_prop("FilterName", "writer_pdf_Export"),
            _make_prop("FilterOptions", options),
        )
        out_url = uno.systemPathToFileUrl(str(Path(output_path).resolve()))
        doc.storeToURL(out_url, pdf_props)
    finally:
        try:
            doc.close(True)
        except Exception:
            pass


def download_from_s3(key, tmp_dir):
    local_path = os.path.join(tmp_dir, os.path.basename(key))
    retry(lambda: s3.download_file(S3_BUCKET, key, local_path))
    return local_path


def upload_to_s3(local_path, key):
    with open(local_path, "rb") as f:
        body = f.read()
    retry(lambda: s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/pdf"))
    log(f"Uploaded PDF to s3://{S3_BUCKET}/{key}")


def libreoffice_convert_uno(input_path, output_dir, lo_mgr):
    base_name = Path(input_path).stem
    out_pdf = os.path.join(output_dir, f"{base_name}.pdf")
    for attempt in range(1, CONVERT_RETRIES + 1):
        try:
            store_to_pdf(lo_mgr, input_path, out_pdf)
            if not os.path.exists(out_pdf):
                raise FileNotFoundError("PDF not produced by UNO conversion")
            return out_pdf
        except Exception as e:
            if attempt >= CONVERT_RETRIES:
                raise
            time.sleep(0.5 * attempt)
    raise RuntimeError("Failed to convert via UNO")


def fallback_subprocess_convert(input_path, output_dir):
    command = [
        "soffice",
        "--headless",
        "--invisible",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--nologo",
        "--convert-to",
        "pdf:writer_pdf_Export",
        "--outdir",
        output_dir,
        input_path,
    ]
    subprocess_env = os.environ.copy()
    import subprocess  # local import to avoid global requirement if UNO path missing
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.pdf")
    if not os.path.exists(out_path):
        raise FileNotFoundError("PDF not created by subprocess fallback")
    return out_path


def convert_and_upload(key):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local_input = download_from_s3(key, tmp)
            ext = key.split(".")[-1].lower()
            if ext not in {"doc", "docx"}:
                log(f"Skipping non-doc {key}")
                return
            lo_mgr = connect_uno()
            try:
                pdf_path = libreoffice_convert_uno(local_input, tmp, lo_mgr)
            except Exception as e:
                msg = str(e).lower()
                log(f"UNO conversion failed for {key}: {e}", level="WARNING")
                try:
                    pdf_path = fallback_subprocess_convert(local_input, tmp)
                except Exception as sb_e:
                    raise RuntimeError(f"Both UNO and subprocess conversion failed: {sb_e}") from sb_e
            pdf_key = f"{S3_RAW_PREFIX}{os.path.basename(key)}.pdf"
            upload_to_s3(pdf_path, pdf_key)
            if OVERWRITE_OTHER_TO_PDF:
                log(f"OVERWRITE_OTHER_TO_PDF=true → deleting original {key}")
                retry(lambda: s3.delete_object(Bucket=S3_BUCKET, Key=key))
        except Exception as e:
            log(f"Conversion failed for {key}: {e}", level="ERROR")


def main():
    log("Running doc_to_pdf_uno.py conversion pass")
    log(f"OVERWRITE_OTHER_TO_PDF={OVERWRITE_OTHER_TO_PDF}")
    keys = list(list_raw_files())
    log(f"Scanning {len(keys)} files in s3://{S3_BUCKET}/{S3_RAW_PREFIX}")
    for key in keys:
        ext = key.split(".")[-1].lower()
        if ext in {"doc", "docx"}:
            log(f"Converting '{key}' to PDF with UNO server")
            convert_and_upload(key)


if __name__ == "__main__":
    main()
