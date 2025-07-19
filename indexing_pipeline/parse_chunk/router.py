import os
import importlib
import boto3
import hashlib
import json
from tqdm import tqdm

s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]
raw_prefix = os.environ.get("S3_RAW_PREFIX", "data/raw/")
chunked_prefix = os.environ.get("S3_CHUNKED_PREFIX", "data/chunked/")
chunk_format = os.environ.get("CHUNK_FORMAT", "jsonl")  # supports jsonl or json

def list_raw_files():
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=raw_prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            yield key

def file_sha256(s3_key):
    # Stream hash from S3 without downloading full file
    hasher = hashlib.sha256()
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    stream = obj['Body']
    for chunk in iter(lambda: stream.read(8192), b''):
        hasher.update(chunk)
    return hasher.hexdigest()

def chunked_path(hash_str):
    ext = "jsonl" if chunk_format == "jsonl" else "json"
    return f"{chunked_prefix}{hash_str}.{ext}"

def is_already_processed(hash_str):
    try:
        s3.head_object(Bucket=bucket, Key=chunked_path(hash_str))
        return True
    except s3.exceptions.ClientError:
        return False

def save_chunks(hash_str, chunks):
    out_key = chunked_path(hash_str)
    if chunk_format == "jsonl":
        # Upload JSONL (one JSON object per line)
        lines = (json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
        s3.put_object(Bucket=bucket, Key=out_key, Body="".join(lines).encode("utf-8"))
    else:
        # Upload a single JSON array
        s3.put_object(Bucket=bucket, Key=out_key, Body=json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
    print(f"Saved parsed chunks to s3://{bucket}/{out_key}")

def get_format_module(ext):
    ext = ext.lower()
    # Map extensions to module names - add your mapping here
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

def main():
    print("Starting routing pipeline...")
    keys = list(list_raw_files())
    print(f"Found {len(keys)} raw files in s3://{bucket}/{raw_prefix}")

    for key in tqdm(keys):
        ext = key.split(".")[-1]
        module_name = get_format_module(ext)
        if not module_name:
            print(f"Skipping unsupported extension '{ext}': {key}")
            continue

        print(f"Processing: {key} (using module: {module_name})")
        sha256 = file_sha256(key)
        if is_already_processed(sha256):
            print(f"Already processed: {sha256} - skipping {key}")
            continue

        manifest = {"sha256": sha256, "s3_key": key}

        try:
            module = importlib.import_module(f"formats.{module_name}")
            chunks = module.parse_file(key, manifest)
            save_chunks(sha256, chunks)
        except Exception as e:
            print(f"Error processing {key} with {module_name}: {e}")

if __name__ == "__main__":
    main()
