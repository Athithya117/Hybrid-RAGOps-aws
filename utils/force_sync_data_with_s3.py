"""
THIS FORCE SYNC FILE IS FOR DEV PURPOSE ONLY. IF USING FOR PROD, MAKE SURE UPLOAD/DOWNLOAD ARGUMENT IS CORRECT.

python3 utils/force_sync_data_with_s3.py --upload
(OR)
python3 utils/force_sync_data_with_s3.py --download
"""


import os
import sys
import argparse
from pathlib import Path

import boto3
import ray
from botocore.exceptions import ClientError

# Initialize Ray with env-controlled settings
ray.init(ignore_reinit_error=True)

# Get concurrency from env or fallback
CONCURRENCY = int(os.environ.get("RAY_CONCURRENT_FILES", 4))

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

@ray.remote(max_retries=3)
def upload_file(path: str, rel_path: str, bucket: str, prefix: str):
    s3 = get_s3_client()
    s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
    s3.upload_file(path, bucket, s3_key)
    return rel_path

@ray.remote(max_retries=3)
def download_file(s3_key: str, rel_path: str, bucket: str, local_base: str):
    s3 = get_s3_client()
    target = Path(local_base) / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(target))
    return rel_path

def list_local_files(base_dir):
    base = Path(base_dir)
    return [(str(p.resolve()), p.relative_to(base).as_posix())
            for p in base.rglob("*") if p.is_file()]

def list_s3_objects(bucket, prefix):
    s3 = get_s3_client()
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                rel = key[len(prefix)+1:]
                out.append((key, rel))
    return out

def upload_directory(base_dir, bucket, prefix, max_concurrency):
    files = list_local_files(base_dir)
    futures = []
    for path, rel in files:
        futures.append(upload_file.options(num_cpus=1).remote(path, rel, bucket, prefix))
        if len(futures) >= max_concurrency:
            done, futures = ray.wait(futures, num_returns=1)
    ray.get(futures)
    print("Upload complete")

def download_directory(bucket, base_dir, prefix, max_concurrency):
    objects = list_s3_objects(bucket, prefix)
    print(f"Found {len(objects)} object(s) under prefix '{prefix}'")
    if not objects:
        print("Nothing to download; check S3_PREFIX or bucket contents.")
        return
    futures = []
    for key, rel in objects:
        futures.append(download_file.options(num_cpus=1).remote(key, rel, bucket, base_dir))
        if len(futures) >= max_concurrency:
            done, futures = ray.wait(futures, num_returns=1)
    ray.get(futures)
    print("Download complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    bucket = os.environ.get("S3_BUCKET_NAME")
    prefix = "data"  # Fixed prefix
    concurrency = CONCURRENCY

    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME env variable is not set.")

    if args.upload:
        upload_directory("data", bucket, prefix, concurrency)
    elif args.download:
        download_directory(bucket, "data", prefix, concurrency)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
