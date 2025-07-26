import os
import sys
import time
import json
import uuid
import boto3
import hashlib
import importlib
import mimetypes
from io import BytesIO
from datetime import datetime
from botocore.exceptions import ClientError

# Env loader
def env_or_fail(var, default=None, mandatory=True):
    val = os.environ.get(var, default)
    if mandatory and val is None:
        print(f"ERROR: Required env var '{var}' not set.", file=sys.stderr)
        sys.exit(1)
    return val

# Load environment
S3_BUCKET        = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX    = os.getenv("S3_RAW_PREFIX", "data/raw/")
S3_CHUNKED_PREFIX= os.getenv("S3_CHUNKED_PREFIX", "data/chunked/")
CHUNK_FORMAT     = os.getenv("CHUNK_FORMAT", "jsonl").lower()
FORCE_PROCESS    = os.getenv("FORCE_PROCESS", "false").lower() == "true"

assert CHUNK_FORMAT in ("jsonl", "json"), f"Unsupported CHUNK_FORMAT '{CHUNK_FORMAT}'"

# S3 client
s3 = boto3.client("s3")

# Logging
def log(*args, level="INFO", **kwargs):
    print(f"[{level}]", *args, flush=True, **kwargs)

# Retry helper
def retry(func, retries=3, delay=2, backoff=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log(f"Retry {attempt+1}/{retries} after error: {e}", level="WARN")
            time.sleep(delay)
            delay *= backoff

# List raw files
def list_raw_files():
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                yield key

# Compute SHA256
def file_sha256(s3_key):
    hasher = hashlib.sha256()
    obj = retry(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
    stream = obj['Body']
    for chunk in iter(lambda: stream.read(8192), b''):
        hasher.update(chunk)
    return hasher.hexdigest()

# Manifest path
def manifest_path(s3_key, hash_str):
    base = os.path.splitext(s3_key)[0]
    return f"{base}.manifest.json"

# Skip if already processed
def is_already_processed(hash_str):
    if FORCE_PROCESS:
        return False
    # check any one chunk exists
    sample_key = f"{S3_CHUNKED_PREFIX}{hash_str}_0.{CHUNK_FORMAT}"
    try:
        retry(lambda: s3.head_object(Bucket=S3_BUCKET, Key=sample_key))
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

# Save each chunk separately
def save_chunks(hash_str, chunks):
    success = True
    for chunk in chunks:
        idx = chunk["payload"]["chunk_index"]
        ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
        out_key = f"{S3_CHUNKED_PREFIX}{hash_str}_{idx}.{ext}"

        if CHUNK_FORMAT == "jsonl":
            body = (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
        else:
            body = json.dumps(chunk, indent=2, ensure_ascii=False).encode("utf-8")

        try:
            retry(lambda: s3.put_object(
                Bucket=S3_BUCKET,
                Key=out_key,
                Body=body,
                ContentType="application/json"
            ))
            log(f"Saved chunk {idx} to s3://{S3_BUCKET}/{out_key}", level="DEBUG")
        except Exception as e:
            log(f"ERROR: Failed to save chunk {idx}: {e}", level="ERROR")
            success = False
    return success


# Save manifest
def save_manifest(s3_key, manifest):
    manifest_key = manifest_path(s3_key, manifest['sha256'])
    try:
        retry(lambda: s3.put_object(
            Bucket=S3_BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json"
        ))
        log(f"Saved manifest to s3://{S3_BUCKET}/{manifest_key}")
        return True
    except Exception as e:
        log(f"ERROR: Failed to save manifest: {e}", level="ERROR")
        return False

# Extension → parser module
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
        "jpeg":"png_jpeg_jpg",
        "png": "png_jpeg_jpg"
    }.get(ext.lower())

# MIME detection
def detect_mime(key):
    mime, _ = mimetypes.guess_type(key)
    return mime or "application/octet-stream"

# Main
def main():
    log("Starting routing pipeline...")
    run_id = str(uuid.uuid4())
    version = "2.42.1"

    keys = list(list_raw_files())
    log(f"Found {len(keys)} files under s3://{S3_BUCKET}/{S3_RAW_PREFIX}")

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
            log(f"Error importing {module_name}: {e}", level="ERROR")
            continue

        log(f"Processing: {key} with module {module_name}")
        try:
            sha = file_sha256(key)
        except Exception as e:
            log(f"SHA256 error for {key}: {e}", level="ERROR")
            continue

        if is_already_processed(sha):
            log(f"Already processed {sha}, skipping {key}")
            continue

        mime_type = detect_mime(key)
        manifest = {
            "sha256": sha,
            "s3_key": key,
            "parser_version": version,
            "pipeline_run_id": run_id,
            "mime_type": mime_type,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            chunks = mod.parse_file(key, manifest)
        except Exception as e:
            log(f"Error parsing {key}: {e}", level="ERROR")
            continue

        if save_chunks(sha, chunks):
            save_manifest(key, manifest)

if __name__ == "__main__":
    main()

