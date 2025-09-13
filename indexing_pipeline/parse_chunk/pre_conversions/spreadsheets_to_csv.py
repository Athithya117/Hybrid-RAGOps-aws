import os
import sys
import time
import tempfile
import zipfile
import warnings
from pathlib import Path
import io
import gc
from tempfile import NamedTemporaryFile
import subprocess
import botocore
import boto3
import uuid
from indexing_pipeline.parse_chunk.router import env_or_fail, log, retry, list_raw_files
warnings.filterwarnings("ignore")
S3_BUCKET = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/")
OVERWRITE_SPREADSHEETS_WITH_CSV = os.getenv("OVERWRITE_SPREADSHEETS_WITH_CSV", "false").lower() == "true"
CSV_DELIM_ASCII = os.getenv("CSV_DELIM_ASCII", "44")
CSV_TEXT_QUOTE_ASCII = os.getenv("CSV_TEXT_QUOTE_ASCII", "34")
CSV_ENCODING_TOKEN = os.getenv("CSV_ENCODING_TOKEN", "76")
CSV_QUOTE_FIELDS = os.getenv("CSV_QUOTE_FIELDS", "1")
CSV_SHEET_TOKEN = os.getenv("CSV_SHEET_TOKEN", "-1")
CONVERTIBLE = {"xls", "xlsx", "ods", "xlsm", "xlsb"}
CONVERT_TIMEOUT = int(os.getenv("CONVERT_TIMEOUT_SECONDS", "600"))
LIBREOFFICE_PORTS = os.getenv("LIBREOFFICE_PORTS", "7003")
UNO_CONNECT_RETRIES = int(os.getenv("UNO_CONNECT_RETRIES", "6"))
UNO_CONNECT_DELAY_SECONDS = float(os.getenv("UNO_CONNECT_DELAY_SECONDS", "1.0"))
CONVERT_RETRIES = int(os.getenv("UNO_CONVERT_RETRIES", "2"))
MAX_PER_WORKER = int(os.getenv("MAX_PER_WORKER", "1000"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(2 * 1024 * 1024 * 1024)))
DISK_REQUIRED_BUFFER = int(os.getenv("DISK_REQUIRED_BUFFER_BYTES", str(50 * 1024 * 1024)))
s3 = boto3.client("s3")
try:
    import uno
    from com.sun.star.beans import PropertyValue
    from com.sun.star.connection import NoConnectException
    UNO_AVAILABLE = True
except Exception:
    UNO_AVAILABLE = False

def _parse_ports(ports_str):
    return [p.strip() for p in ports_str.split(",") if p.strip()]

def connect_uno(ports=None, retries=UNO_CONNECT_RETRIES, delay=UNO_CONNECT_DELAY_SECONDS):
    ports = ports or _parse_ports(LIBREOFFICE_PORTS)
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local_ctx)
    last_exc = None
    for _ in range(retries):
        for port in ports:
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                return ctx
            except NoConnectException as e:
                last_exc = e
            except Exception as e:
                last_exc = e
        time.sleep(delay)
    raise RuntimeError(f"Unable to connect to UNO server on ports {ports}") from last_exc

def _pv(name, value):
    return PropertyValue(name, 0, value, 0)

def _s3_range_read(key, length=131072):
    try:
        resp = retry(lambda: s3.get_object(Bucket=S3_BUCKET, Key=key, Range=f"bytes=0-{length-1}"))
        return resp["Body"].read()
    except Exception:
        return b""

def is_ooxml_encrypted_s3(key):
    data = _s3_range_read(key, length=262144)
    if not data:
        return False
    try:
        with io.BytesIO(data) as b:
            try:
                with zipfile.ZipFile(b, "r") as z:
                    names = z.namelist()
                    if "EncryptedPackage" in names or "EncryptionInfo" in names:
                        return True
            except zipfile.BadZipFile:
                low = data.lower()
                if b"encryptedpackage" in low or b"encryptioninfo" in low:
                    return True
            return False
    except Exception:
        return False

def likely_binary_encrypted_s3(key):
    data = _s3_range_read(key, length=65536)
    if not data:
        return False
    low = data.lower()
    if b"encrypted" in low or b"password" in low or b"encryptioninfo" in low:
        return True
    return False

def is_password_protected_remote(key, ext):
    ext = ext.lower()
    if ext in {"xlsx", "xlsm", "xlsb", "ods"}:
        try:
            if is_ooxml_encrypted_s3(key):
                return True
        except Exception:
            pass
    if ext == "xls":
        try:
            if likely_binary_encrypted_s3(key):
                return True
        except Exception:
            pass
    return False

def has_enough_disk(path, required_bytes):
    try:
        st = os.statvfs(path)
        free = st.f_bavail * st.f_frsize
        return free >= required_bytes
    except Exception:
        return True

def download_from_s3_stream(key, tmp_dir):
    try:
        head = retry(lambda: s3.head_object(Bucket=S3_BUCKET, Key=key))
        size = int(head.get("ContentLength", 0))
    except Exception:
        size = None
    if size is not None:
        if size > MAX_FILE_SIZE_BYTES:
            raise RuntimeError(f"Skipping {key}: file too large ({size} bytes)")
        required = (size or 0) + DISK_REQUIRED_BUFFER
        if not has_enough_disk(tmp_dir, required):
            raise RuntimeError(f"Not enough disk space to download {key} (needs {required} bytes)")
    suffix = ("." + key.split(".")[-1]) if "." in key else ""
    tmp = NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    def _do():
        with open(tmp_path, "wb") as f:
            s3.download_fileobj(S3_BUCKET, key, f)
    retry(_do)
    return tmp_path

def upload_to_s3_atomic(local_path, key, original_key):
    metadata = {"converted-by": "libreoffice", "original-key": original_key}
    temp_key = f"{key}.tmp.{uuid.uuid4().hex}"
    def _put():
        s3.upload_file(local_path, S3_BUCKET, temp_key, ExtraArgs={"ContentType": "text/csv"})
    retry(_put)
    def _copy():
        copy_source = {"Bucket": S3_BUCKET, "Key": temp_key}
        s3.copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=key, Metadata=metadata, ContentType="text/csv", MetadataDirective='REPLACE')
    retry(_copy)
    def _del():
        s3.delete_object(Bucket=S3_BUCKET, Key=temp_key)
    retry(_del)
    log(f"Uploaded CSV to s3://{S3_BUCKET}/{key}")

def is_ooxml_encrypted(path):
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            if "EncryptedPackage" in names or "EncryptionInfo" in names:
                return True
    except zipfile.BadZipFile:
        return False
    return False

def likely_binary_encrypted(path):
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        low = head.lower()
        if b"encrypted" in low or b"password" in low or b"encryptioninfo" in low:
            return True
    except Exception:
        return False
    return False

def is_password_protected(local_path, ext):
    ext = ext.lower()
    if ext in {"xlsx", "xlsm", "xlsb"}:
        try:
            if is_ooxml_encrypted(local_path):
                return True
        except Exception:
            pass
    if ext == "ods":
        try:
            if is_ooxml_encrypted(local_path):
                return True
        except Exception:
            pass
    if ext == "xls":
        try:
            if likely_binary_encrypted(local_path):
                return True
        except Exception:
            pass
    return False

def _make_filter_tokens(sheet_index):
    tokens = [
        CSV_DELIM_ASCII,
        CSV_TEXT_QUOTE_ASCII,
        CSV_ENCODING_TOKEN,
        CSV_QUOTE_FIELDS,
        "",
        "0",
        "false",
        "true",
        "false",
        "false",
        "false",
        str(sheet_index),
    ]
    return ",".join(tokens)

def libreoffice_convert_to_csv_uno(input_path, output_dir, comp_ctx):
    smgr = comp_ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", comp_ctx)
    input_url = uno.systemPathToFileUrl(str(Path(input_path).resolve()))
    props = (PropertyValue("Hidden", 0, True, 0),)
    doc = desktop.loadComponentFromURL(input_url, "_blank", 0, props)
    produced = []
    try:
        sheets = doc.Sheets
        try:
            sheet_count = sheets.getCount()
        except Exception:
            sheet_count = 1
        for i in range(sheet_count):
            sheet_name = None
            try:
                sheet = sheets.getByIndex(i)
                sheet_name = getattr(sheet, "Name", None)
            except Exception:
                sheet_name = None
            filter_options = _make_filter_tokens(i)
            csv_props = (
                _pv("FilterName", "Text - txt - csv (StarCalc)"),
                _pv("FilterOptions", filter_options),
            )
            base = Path(input_path).stem
            suffix = f"_{i+1}"
            if sheet_name:
                safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in sheet_name).strip()
                suffix = f"_{safe_name}"
            out_name = f"{base}{suffix}.csv"
            out_path = os.path.join(output_dir, out_name)
            out_url = uno.systemPathToFileUrl(str(Path(out_path).resolve()))
            try:
                doc.storeToURL(out_url, csv_props)
            except Exception:
                continue
            if os.path.exists(out_path):
                produced.append(out_path)
    finally:
        try:
            doc.close(True)
        except Exception:
            pass
    if not produced:
        raise FileNotFoundError("No CSV produced by UNO conversion.")
    return produced

def _run_soffice_convert(cmd, output_dir, timeout):
    stdout_tmp = NamedTemporaryFile(delete=False, dir=output_dir, suffix=".stdout")
    stderr_tmp = NamedTemporaryFile(delete=False, dir=output_dir, suffix=".stderr")
    stdout_path = stdout_tmp.name
    stderr_path = stderr_tmp.name
    stdout_tmp.close()
    stderr_tmp.close()
    try:
        with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se, timeout=timeout)
        if proc.returncode != 0:
            msg = ""
            try:
                with open(stderr_path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    to_read = 4000
                    if size > to_read:
                        f.seek(-to_read, os.SEEK_END)
                    else:
                        f.seek(0)
                    msg = f.read().decode(errors="replace").strip()
            except Exception:
                msg = "soffice failed (stderr unavailable)"
            raise RuntimeError(f"LibreOffice conversion failed rc={proc.returncode} stderr={msg}")
    finally:
        try:
            os.unlink(stdout_path)
        except Exception:
            pass
        try:
            os.unlink(stderr_path)
        except Exception:
            pass

def libreoffice_convert_to_csv_subprocess(input_path, output_dir, filter_tokens, timeout=CONVERT_TIMEOUT):
    base_name = Path(input_path).stem
    target = f'csv:Text - txt - csv (StarCalc):{filter_tokens}'
    cmd = [
        "soffice",
        "--headless",
        "--invisible",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--nologo",
        "--convert-to",
        target,
        "--outdir",
        output_dir,
        input_path,
    ]
    _run_soffice_convert(cmd, output_dir, timeout)
    produced = sorted(Path(output_dir).glob(f"{base_name}*.csv"))
    if not produced:
        fallback_cmd = [
            "soffice",
            "--headless",
            "--infilter", f"CSV:{filter_tokens}",
            "--convert-to", "csv",
            "--outdir", output_dir,
            input_path,
        ]
        _run_soffice_convert(fallback_cmd, output_dir, timeout)
        produced = sorted(Path(output_dir).glob(f"{base_name}*.csv"))
    if not produced:
        raise FileNotFoundError("No CSV produced by LibreOffice conversion.")
    return [str(p) for p in produced]

def libreoffice_convert_to_csv(input_path, output_dir, filter_tokens, timeout=CONVERT_TIMEOUT):
    if UNO_AVAILABLE:
        try:
            comp_ctx = connect_uno()
        except Exception as e:
            log(f"UNO connect failed, falling back to subprocess conversion: {e}")
            return libreoffice_convert_to_csv_subprocess(input_path, output_dir, filter_tokens, timeout)
        for attempt in range(1, CONVERT_RETRIES + 1):
            try:
                return libreoffice_convert_to_csv_uno(input_path, output_dir, comp_ctx)
            except Exception:
                if attempt >= CONVERT_RETRIES:
                    raise
                time.sleep(0.5 * attempt)
    else:
        return libreoffice_convert_to_csv_subprocess(input_path, output_dir, filter_tokens, timeout)

def convert_and_upload(key):
    with tempfile.TemporaryDirectory() as tmp:
        local_input = None
        try:
            start_ts = time.time()
            ext = key.split(".")[-1].lower()
            if ext not in CONVERTIBLE:
                log(f"Skipping non-spreadsheet {key}")
                return True
            if is_password_protected_remote(key, ext):
                log(f"Skipping password-protected file {key}")
                return True
            local_input = download_from_s3_stream(key, tmp)
            if not local_input or not os.path.exists(local_input):
                log(f"Download failed or returned no path for {key}")
                return False
            if is_password_protected(local_input, ext):
                log(f"Skipping password-protected file {key}")
                try:
                    os.unlink(local_input)
                except Exception:
                    pass
                return True
            tokens = _make_filter_tokens(CSV_SHEET_TOKEN)
            try:
                csv_paths = libreoffice_convert_to_csv(local_input, tmp, tokens)
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ("password", "encrypted", "encryption", "password required")):
                    log(f"Detected password-protected during conversion; skipping {key}")
                    try:
                        os.unlink(local_input)
                    except Exception:
                        pass
                    return True
                raise
            original_basename = os.path.basename(key)
            base_name = Path(local_input).stem
            for csv_path in csv_paths:
                produced_stem = Path(csv_path).stem
                if produced_stem.startswith(base_name):
                    suffix = produced_stem[len(base_name):]
                else:
                    suffix = f"_{produced_stem}"
                target_basename = original_basename + suffix + ".csv"
                csv_key = f"{S3_RAW_PREFIX}{target_basename}"
                upload_to_s3_atomic(csv_path, csv_key, key)
            if OVERWRITE_SPREADSHEETS_WITH_CSV:
                log(f"OVERWRITE_SPREADSHEETS_WITH_CSV=true → deleting original {key}")
                retry(lambda: s3.delete_object(Bucket=S3_BUCKET, Key=key))
            try:
                os.unlink(local_input)
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            dur = time.time() - start_ts
            log(f"Converted {key} in {dur:.2f}s")
            return True
        except Exception as e:
            log(f"Conversion failed for {key}: {e}", level="ERROR")
            try:
                if local_input and os.path.exists(local_input):
                    os.unlink(local_input)
            except Exception:
                pass
            return False

def main():
    log("Running sheet_to_csv_uno.py conversion pass")
    log(f"OVERWRITE_SPREADSHEETS_WITH_CSV={OVERWRITE_SPREADSHEETS_WITH_CSV}")
    keys = list(list_raw_files())
    log(f"Scanning {len(keys)} files in s3://{S3_BUCKET}/{S3_RAW_PREFIX}")
    processed = 0
    for key in keys:
        ext = key.split(".")[-1].lower()
        if ext in CONVERTIBLE:
            log(f"Converting '{key}' to CSV(S) with LibreOffice UNO")
            ok = convert_and_upload(key)
            processed += 1
            if processed >= MAX_PER_WORKER:
                log("Max conversions reached, exiting to allow orchestrator to restart")
                sys.exit(0)

if __name__ == "__main__":
    main()
