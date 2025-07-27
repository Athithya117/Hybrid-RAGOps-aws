import os
import sys
import time
import json
import uuid
import boto3
import hashlib
import importlib
import mimetypes
from datetime import datetime
from botocore.exceptions import ClientError

# --- ENV ---
def env_or_fail(var, default=None, mandatory=True):
    val = os.environ.get(var, default)
    if mandatory and val is None:
        print(f"ERROR: Required env var '{var}' not set.", file=sys.stderr)
        sys.exit(1)
    return val

S3_BUCKET         = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX     = os.getenv("S3_RAW_PREFIX", "data/raw/")
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")
CHUNK_FORMAT      = os.getenv("CHUNK_FORMAT", "json").lower()
FORCE_PROCESS     = os.getenv("FORCE_PROCESS", "false").lower() == "true"

assert CHUNK_FORMAT in ("json", "jsonl"), f"Invalid CHUNK_FORMAT '{CHUNK_FORMAT}'"

s3 = boto3.client("s3")

def log(*args, level="INFO", **kwargs):
    print(f"[{level}]", *args, flush=True, **kwargs)

def retry(func, retries=3, delay=2, backoff=2):
    for attempt in range(retries):
        try:
            return func()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
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
            if not key.endswith("/"):
                yield key

def file_sha256(s3_key):
    hasher = hashlib.sha256()
    obj = retry(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
    stream = obj["Body"]
    for chunk in iter(lambda: stream.read(8192), b""):
        hasher.update(chunk)
    return hasher.hexdigest()

def manifest_path(s3_key, sha):
    base = os.path.splitext(s3_key)[0]
    return f"{base}.manifest.json"

def is_already_processed(sha):
    if FORCE_PROCESS:
        return False
    test_key = f"{S3_CHUNKED_PREFIX}{sha}_0.{CHUNK_FORMAT}"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=test_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def save_manifest(s3_key, manifest):
    key = manifest_path(s3_key, manifest["sha256"])
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
        "json": "json"
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
        ext = key.split(".")[-1]
        module_name = get_format_module(ext)
        if not module_name:
            log(f"Skipping unsupported '{ext}': {key}", level="WARN")
            continue

        try:
            mod = importlib.import_module(f"indexing_pipeline.parse_chunk.formats.{module_name}")
            if not hasattr(mod, "parse_file"):
                log(f"No parse_file() in {module_name}, skipping {key}", level="WARN")
                continue
        except Exception as e:
            log(f"Import error in module {module_name}: {e}", level="ERROR")
            continue

        try:
            sha = file_sha256(key)
        except Exception as e:
            log(f"SHA256 error for {key}: {e}", level="ERROR")
            continue

        if is_already_processed(sha):
            log(f"Already processed {sha}, skipping")
            continue

        manifest = {
            "sha256": sha,
            "s3_key": key,
            "pipeline_run_id": run_id,
            "mime_type": detect_mime(key),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            result = mod.parse_file(key, manifest)
            if not isinstance(result, dict) or "saved_chunks" not in result:
                raise ValueError("Invalid parse_file() return. Expected dict with 'saved_chunks'.")
        except Exception as e:
            log(f"Parse error for {key}: {e}", level="ERROR")
            continue

        log(f"Parsed and stored {result['saved_chunks']} chunks for {key}")
        save_manifest(key, manifest)

if __name__ == "__main__":
    main()
