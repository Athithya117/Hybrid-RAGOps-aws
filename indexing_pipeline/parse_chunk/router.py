# indexing_pipeline/parse_chunk/router.py

import os
import importlib
import boto3
import hashlib
import json
from tqdm import tqdm
from io import BytesIO
import sys
import time
import mimetypes
import uuid
from datetime import datetime
from botocore.exceptions import ClientError

def env_or_fail(var, default=None, mandatory=True):
    val = os.environ.get(var, default)
    if mandatory and val is None:
        print(f"ERROR: Required env var '{var}' not set. Exiting.", file=sys.stderr)
        sys.exit(1)
    return val

S3_BUCKET = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX = os.environ.get("S3_RAW_PREFIX", "data/raw/")
S3_CHUNKED_PREFIX = os.environ.get("S3_CHUNKED_PREFIX", "data/chunked/")
CHUNK_FORMAT = os.environ.get("CHUNK_FORMAT", "jsonl").lower()
assert CHUNK_FORMAT in ("jsonl", "json"), f"Unsupported CHUNK_FORMAT '{CHUNK_FORMAT}'"

s3 = boto3.client("s3")

def log(*args, level="INFO", **kwargs):
    print(f"[{level}]", *args, flush=True, **kwargs)

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

def list_raw_files():
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            yield key

def file_sha256(s3_key):
    hasher = hashlib.sha256()
    try:
        obj = retry(lambda: s3.get_object(Bucket=S3_BUCKET, Key=s3_key))
        stream = obj['Body']
        for chunk in iter(lambda: stream.read(8192), b''):
            hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        log(f"ERROR: Failed to compute SHA256 for {s3_key}: {e}", level="ERROR")
        raise

def chunked_path(hash_str):
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    return f"{S3_CHUNKED_PREFIX}{hash_str}.{ext}"

def manifest_path(s3_key, hash_str):
    return f"{os.path.splitext(s3_key)[0]}.manifest.json"

def is_already_processed(hash_str):
    try:
        retry(lambda: s3.head_object(Bucket=S3_BUCKET, Key=chunked_path(hash_str)))
        return True
    except ClientError:
        return False

def save_chunks(hash_str, chunks):
    out_key = chunked_path(hash_str)
    try:
        if CHUNK_FORMAT == "jsonl":
            buffer = BytesIO()
            for chunk in chunks:
                buffer.write((json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8"))
            buffer.seek(0)
            retry(lambda: s3.put_object(
                Bucket=S3_BUCKET,
                Key=out_key,
                Body=buffer,
                ContentType="application/json"
            ))
        else:
            retry(lambda: s3.put_object(
                Bucket=S3_BUCKET,
                Key=out_key,
                Body=json.dumps(chunks, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json"
            ))
        log(f"Saved parsed chunks to s3://{S3_BUCKET}/{out_key}")
        return True
    except Exception as e:
        log(f"ERROR: Failed to save chunks for {hash_str}: {e}", level="ERROR")
        return False

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
        log(f"ERROR: Failed to save manifest for {s3_key}: {e}", level="ERROR")
        return False

def get_format_module(ext):
    ext = ext.lower()
    ext_map = {
        "pdf": "pdf",
        "doc": "doc_docx",
        "docx": "doc_docx",
        "html": "html",
        "md": "md",
        "mp3": "mp3",
        "png": "png_jpeg_jpg",
        "jpg": "png_jpeg_jpg",
        "jpeg": "png_jpeg_jpg",
        "ppt": "ppt_pptx",
        "pptx": "ppt_pptx",
        "txt": "txt",
    }
    return ext_map.get(ext)

def detect_mime(key):
    mime, _ = mimetypes.guess_type(key)
    return mime or "application/octet-stream"

def main():
    log("Starting routing pipeline...")
    pipeline_run_id = str(uuid.uuid4())
    parser_version = "2.42.1"

    keys = list(list_raw_files())
    log(f"Found {len(keys)} raw files in s3://{S3_BUCKET}/{S3_RAW_PREFIX}")

    for key in tqdm(keys):
        ext = key.split(".")[-1]
        module_name = get_format_module(ext)
        if not module_name:
            log(f"Skipping unsupported extension '{ext}': {key}", level="WARN")
            continue

        try:
            module = importlib.import_module(f"indexing_pipeline.parse_chunk.formats.{module_name}")
            if not hasattr(module, "parse_file"):
                log(f"Skipping: indexing_pipeline.parse_chunk.formats.{module_name} has no parse_file()", level="WARN")
                continue
        except Exception as e:
            log(f"Skipping: Error importing indexing_pipeline.parse_chunk.formats.{module_name} → {e}", level="ERROR")
            continue

        log(f"Processing: {key} (module: {module_name})")

        try:
            sha256 = file_sha256(key)
        except Exception:
            log(f"Skipping {key} due to SHA256 error", level="ERROR")
            continue

        if is_already_processed(sha256):
            log(f"Already processed: {sha256} - skipping {key}")
            continue

        mime_type = detect_mime(key)

        manifest = {
            "sha256": sha256,
            "s3_key": key,
            "parser_version": parser_version,
            "pipeline_run_id": pipeline_run_id,
            "mime_type": mime_type,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            chunks = module.parse_file(key, manifest)
        except Exception as e:
            log(f"Error parsing {key} using {module_name}: {e}", level="ERROR")
            continue

        if not save_chunks(sha256, chunks):
            continue

        save_manifest(key, manifest)

if __name__ == "__main__":
    main()
