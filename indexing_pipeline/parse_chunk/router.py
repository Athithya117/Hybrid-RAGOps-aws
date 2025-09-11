import os
import sys
import time
import json
import uuid
import boto3
import hashlib
import importlib
import mimetypes
import logging
from datetime import datetime
from botocore.exceptions import ClientError

try:
    import colorama
    colorama.init()
except Exception:
    pass

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[36m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;41m"
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"

logger = logging.getLogger("router")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(ColorFormatter(fmt._fmt))
logger.handlers[:] = [handler]

def env_or_fail(var, default=None, mandatory=True):
    val = os.environ.get(var, default)
    if mandatory and val is None:
        print(f"ERROR: Required env var '{var}' not set.", file=sys.stderr)
        sys.exit(1)
    return val

S3_BUCKET = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
FORCE_PROCESS = os.getenv("FORCE_PROCESS", "false").lower() == "true"

assert CHUNK_FORMAT in ("json", "jsonl"), f"Invalid CHUNK_FORMAT '{CHUNK_FORMAT}'"

s3 = boto3.client("s3")

def log(*args, level="INFO", **kwargs):
    msg = " ".join(str(a) for a in args)
    lvl = level.upper()
    if lvl == "WARN":
        lvl = "WARNING"
    levelno = getattr(logging, lvl, logging.INFO)
    logger.log(levelno, msg, **kwargs)

def retry(func, retries=3, delay=2, backoff=2):
    for attempt in range(retries):
        try:
            return func()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                raise
            if attempt == retries - 1:
                raise
            log(f"Retry {attempt + 1}/{retries} after error: {e}", level="WARN")
            time.sleep(delay)
            delay *= backoff
        except Exception as e:
            if attempt == retries - 1:
                raise
            log(f"Retry {attempt + 1}/{retries} after error: {e}", level="WARN")
            time.sleep(delay)
            delay *= backoff

def list_raw_files():
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if key.lower().endswith(".manifest.json"):
                continue
            yield key

def file_sha256(s3_key):
    hasher = hashlib.blake2b(digest_size=16)
    obj = retry(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
    stream = obj["Body"]
    for chunk in iter(lambda: stream.read(8192), b""):
        hasher.update(chunk)
    return hasher.hexdigest()

def manifest_path(s3_key, file_hash=None):
    # keep manifest next to original file (but router will skip any .manifest.json keys)
    return f"{s3_key}.manifest.json"

def is_already_processed(file_hash):
    if FORCE_PROCESS:
        return False
    base_prefix = S3_CHUNKED_PREFIX.rstrip("/") + "/"
    search_prefix = f"{base_prefix}{file_hash}_"
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=search_prefix, PaginationConfig={"MaxItems": 1})
        for page in pages:
            if page.get("Contents"):
                return True
    except ClientError as e:
        log(f"S3 list_objects_v2 error while checking {search_prefix}: {e}", level="WARN")
    for ext in ("json", "jsonl"):
        test_key = f"{base_prefix}{file_hash}_1.{ext}"
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=test_key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "404":
                continue
            raise
    return False

def save_manifest(s3_key, manifest):
    key = manifest_path(s3_key, manifest.get("file_hash"))
    try:
        retry(lambda: s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json"
        ))
        log(f"Saved manifest to s3://{S3_BUCKET}/{key}")
        return True
    except Exception as e:
        log(f"Failed to save manifest: {e}", level="ERROR")
        return False

def get_format_module(ext):
    return {
        "pdf": "pdf",
        "doc": "doc_docx",
        "docx": "doc_docx",
        "ppt": "ppt_pptx",
        "pptx": "ppt_pptx",
        "html": "html",
        "md": "md",
        "txt": "txt",
        "mp3": "mp3",
        "jpg": "png_jpeg_jpg",
        "jpeg": "png_jpeg_jpg",
        "png": "png_jpeg_jpg",
        "csv": "csv",
        "jsonl": "jsonl",
        "ndjson": "jsonl"
    }.get(ext.lower())

def detect_mime(key):
    mime, _ = mimetypes.guess_type(key)
    return mime or "application/octet-stream"

def main():
    log("Router pipeline started")
    run_id = str(uuid.uuid4())
    parser_version = "2.42.1"
    keys = list(list_raw_files())
    log(f"Found {len(keys)} files")
    for key in keys:
        if key.lower().endswith(".manifest.json"):
            log(f"Skipping manifest file {key}")
            continue

        ext = key.split(".")[-1]
        module_name = get_format_module(ext)
        if not module_name:
            log(f"Skipping unsupported '{ext}': {key}", level="WARN")
            continue
        try:
            mod = importlib.import_module(
                f"indexing_pipeline.parse_chunk.formats.{module_name}"
            )
            if not hasattr(mod, "parse_file"):
                log(f"No parse_file() in {module_name}, skipping {key}", level="WARN")
                continue
        except Exception as e:
            log(f"Import error in module {module_name}: {e}", level="ERROR")
            continue
        try:
            file_hash = file_sha256(key)
        except Exception as e:
            log(f"Hash error for {key}: {e}", level="ERROR")
            continue
        if is_already_processed(file_hash):
            log(f"Already processed {file_hash}, skipping")
            continue
        manifest = {
            "file_hash": file_hash,
            "s3_key": key,
            "pipeline_run_id": run_id,
            "mime_type": detect_mime(key),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        try:
            result = mod.parse_file(key, manifest)
            if not isinstance(result, dict) or "saved_chunks" not in result:
                raise ValueError(
                    "Invalid parse_file() return. Expected dict with 'saved_chunks'."
                )
        except Exception as e:
            log(f"Parse error for {key}: {e}", level="ERROR")
            continue
        log(f"Parsed and stored {result['saved_chunks']} chunks for {key}")
        save_manifest(key, manifest)

if __name__ == "__main__":
    main()
