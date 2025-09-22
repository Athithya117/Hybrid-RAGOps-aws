"""
sync:

python3 utils/force_sync_data_with_s3.py --upload
(OR)
python3 utils/force_sync_data_with_s3.py --download

force sync: deletes what's not in local data/ or deletes what's not in s3://<bucket_name>/data/

python3 utils/force_sync_data_with_s3.py --upload --force   
(OR)
python3 utils/force_sync_data_with_s3.py --download --force

aws s3 ls "s3://$S3_BUCKET/data/" --recursive | tail -n 100

"""

from __future__ import annotations
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
import ray

DEFAULT_PREFIX = "data"
LOCAL_BASE = "data"
DEFAULT_CONCURRENCY = 4
DEFAULT_CHUNKSIZE_MB = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Force sync local data/ with S3 data/ using Ray.")
    p.add_argument("--upload", action="store_true", help="Upload local -> S3")
    p.add_argument("--download", action="store_true", help="Download S3 -> local")
    p.add_argument("--force", action="store_true", help="Delete files not present on the other side")
    p.add_argument("--max-concurrency", type=int, default=0, help="Override concurrency (0 = auto/env)")
    p.add_argument("--multipart-chunksize-mb", type=int, default=DEFAULT_CHUNKSIZE_MB, help="Multipart chunk size in MiB")
    p.add_argument("--region", default=None, help="AWS region (optional)")
    return p.parse_args()


def compute_concurrency(override: int = 0) -> int:
    if override and override > 0:
        return max(1, override)
    env_val = os.environ.get("RAY_CONCURRENT_FILES")
    if env_val:
        try:
            v = int(env_val)
            return max(1, v)
        except Exception:
            pass
    return DEFAULT_CONCURRENCY


def get_s3_client(region: Optional[str] = None):
    boto_conf = BotoConfig(retries={"max_attempts": 10, "mode": "standard"})
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=region or os.environ.get("AWS_REGION"),
        config=boto_conf,
    )


def compute_md5(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_sha256(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@ray.remote(max_retries=3)
def upload_file_remote(local_path: str, rel_path: str, bucket: str, prefix: str, chunksize_mb: int, max_concurrency: int, region: Optional[str], sha256: Optional[str]):
    s3 = get_s3_client(region)
    key = f"{prefix.rstrip('/')}/{rel_path}".lstrip('/')
    config = TransferConfig(
        multipart_chunksize=chunksize_mb * 1024 * 1024,
        multipart_threshold=chunksize_mb * 1024 * 1024,
        max_concurrency=max(1, max_concurrency),
        use_threads=True,
    )
    extra = {"ContentType": "application/octet-stream"}
    if sha256:
        extra["Metadata"] = {"sha256": sha256}
    start = time.time()
    s3.upload_file(Filename=local_path, Bucket=bucket, Key=key, Config=config, ExtraArgs=extra)
    end = time.time()
    return {"rel_path": rel_path, "s3_key": key, "duration_seconds": int(end - start)}


@ray.remote(max_retries=3)
def download_file_remote(bucket: str, s3_key: str, rel_path: str, local_base: str, region: Optional[str], chunksize_mb: int, max_concurrency: int):
    s3 = get_s3_client(region)
    target = Path(local_base) / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    config = TransferConfig(
        multipart_chunksize=chunksize_mb * 1024 * 1024,
        multipart_threshold=chunksize_mb * 1024 * 1024,
        max_concurrency=max(1, max_concurrency),
        use_threads=True,
    )
    start = time.time()
    s3.download_file(Bucket=bucket, Key=s3_key, Filename=str(target), Config=config)
    end = time.time()
    return {"rel_path": rel_path, "s3_key": s3_key, "duration_seconds": int(end - start)}


@ray.remote(max_retries=3)
def delete_s3_file_remote(bucket: str, s3_key: str, region: Optional[str]):
    s3 = get_s3_client(region)
    s3.delete_object(Bucket=bucket, Key=s3_key)
    return s3_key


def list_local_files(base_dir: str) -> List[Tuple[str, str]]:
    base = Path(base_dir)
    if not base.exists():
        return []
    out: List[Tuple[str, str]] = []
    for p in base.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(base).as_posix()
            except Exception:
                rel = p.name
            out.append((str(p.resolve()), rel))
    return out


def list_s3_objects(bucket: str, prefix: str, region: Optional[str]) -> List[Tuple[str, str, int, str]]:
    s3 = get_s3_client(region)
    out: List[Tuple[str, str, int, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    prefix_key = prefix.rstrip("/") + "/"
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix_key):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix_key):] if key.startswith(prefix_key) else key
                size = int(obj.get("Size", 0))
                etag = obj.get("ETag", "").strip('"')
                out.append((key, rel, size, etag))
    except ClientError:
        raise
    return out


def head_object_metadata(bucket: str, key: str, region: Optional[str]) -> Dict[str, str]:
    s3 = get_s3_client(region)
    try:
        ho = s3.head_object(Bucket=bucket, Key=key)
        meta = ho.get("Metadata", {}) or {}
        return {k.lower(): v for k, v in meta.items()}
    except ClientError:
        return {}


def upload_directory(base_dir: str, bucket: str, prefix: str, concurrency: int, chunksize_mb: int, force: bool, region: Optional[str]):
    local = dict(list_local_files(base_dir))
    remote_entries = list_s3_objects(bucket, prefix, region)
    remote_map: Dict[str, Dict[str, object]] = {}
    for key, rel, size, etag in remote_entries:
        remote_map[rel] = {"key": key, "size": size, "etag": etag, "metadata_sha256": None}
    for rel, info in list(remote_map.items()):
        meta = head_object_metadata(bucket, info["key"], region)
        if meta.get("sha256"):
            remote_map[rel]["metadata_sha256"] = meta.get("sha256")
    if force:
        stale_keys = [info["key"] for rel, info in remote_map.items() if rel not in local.values()]
        if stale_keys:
            deletes = [delete_s3_file_remote.remote(bucket, key, region) for key in stale_keys]
            ray.get(deletes)
        remote_entries = list_s3_objects(bucket, prefix, region)
        remote_map = {}
        for key, rel, size, etag in remote_entries:
            remote_map[rel] = {"key": key, "size": size, "etag": etag, "metadata_sha256": None}
        for rel, info in list(remote_map.items()):
            meta = head_object_metadata(bucket, info["key"], region)
            if meta.get("sha256"):
                remote_map[rel]["metadata_sha256"] = meta.get("sha256")
    ongoing = []
    submitted = 0
    skipped = 0
    successes = 0
    failures = 0
    errors: List[str] = []
    for local_path, rel in local.items():
        remote_info = remote_map.get(rel)
        should_upload = True
        local_sha = None
        if remote_info is not None:
            remote_size = int(remote_info["size"])
            remote_etag = str(remote_info.get("etag", ""))
            remote_meta_sha = remote_info.get("metadata_sha256")
            try:
                local_size = Path(local_path).stat().st_size
            except Exception:
                local_size = None
            if remote_meta_sha:
                local_sha = compute_sha256(local_path)
                if local_sha == remote_meta_sha:
                    should_upload = False
            elif local_size is not None and local_size == remote_size and "-" not in remote_etag and remote_etag != "":
                local_md5 = compute_md5(local_path)
                if local_md5 == remote_etag:
                    should_upload = False
        if not should_upload:
            skipped += 1
            continue
        if local_sha is None:
            local_sha = compute_sha256(local_path)
        fut = upload_file_remote.options(num_cpus=1).remote(local_path, rel, bucket, prefix, chunksize_mb, concurrency, region, local_sha)
        ongoing.append(fut)
        submitted += 1
        if len(ongoing) >= concurrency:
            done, ongoing = ray.wait(ongoing, num_returns=1)
            for d in done:
                try:
                    _ = ray.get(d)
                    successes += 1
                except Exception as e:
                    failures += 1
                    errors.append(str(e))
    if ongoing:
        try:
            done_results = ray.get(ongoing)
            successes += len(done_results)
        except Exception as e:
            failures += 1
            errors.append(str(e))
    print(f"Upload scheduled: {submitted}, skipped: {skipped}, succeeded: {successes}, failed: {failures}")
    if errors:
        for e in errors[:20]:
            print(e)


def download_directory(bucket: str, base_dir: str, prefix: str, concurrency: int, force: bool, region: Optional[str], chunksize_mb: int):
    remote_entries = list_s3_objects(bucket, prefix, region)
    remote_map: Dict[str, Dict[str, object]] = {}
    for key, rel, size, etag in remote_entries:
        remote_map[rel] = {"key": key, "size": size, "etag": etag, "metadata_sha256": None}
    for rel, info in list(remote_map.items()):
        meta = head_object_metadata(bucket, info["key"], region)
        if meta.get("sha256"):
            remote_map[rel]["metadata_sha256"] = meta.get("sha256")
    local_map = dict(list_local_files(base_dir))
    if force:
        stale_paths = [str(Path(base_dir) / rel) for rel in local_map.values() if rel not in remote_map]
        removed = 0
        for path in stale_paths:
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
    ongoing = []
    submitted = 0
    skipped = 0
    successes = 0
    failures = 0
    errors: List[str] = []
    for rel, info in remote_map.items():
        key = info["key"]
        remote_size = int(info["size"])
        remote_etag = str(info.get("etag", ""))
        remote_meta_sha = info.get("metadata_sha256")
        local_path = Path(base_dir) / rel
        should_download = True
        if local_path.exists():
            try:
                local_size = local_path.stat().st_size
            except Exception:
                local_size = None
            if remote_meta_sha:
                local_sha = compute_sha256(str(local_path))
                if local_sha == remote_meta_sha:
                    should_download = False
            elif local_size is not None and local_size == remote_size and "-" not in remote_etag and remote_etag != "":
                local_md5 = compute_md5(str(local_path))
                if local_md5 == remote_etag:
                    should_download = False
        if not should_download:
            skipped += 1
            continue
        fut = download_file_remote.options(num_cpus=1).remote(bucket, key, rel, base_dir, region, chunksize_mb, concurrency)
        ongoing.append(fut)
        submitted += 1
        if len(ongoing) >= concurrency:
            done, ongoing = ray.wait(ongoing, num_returns=1)
            for d in done:
                try:
                    _ = ray.get(d)
                    successes += 1
                except Exception as e:
                    failures += 1
                    errors.append(str(e))
    if ongoing:
        try:
            done_results = ray.get(ongoing)
            successes += len(done_results)
        except Exception as e:
            failures += 1
            errors.append(str(e))
    print(f"Download scheduled: {submitted}, skipped: {skipped}, succeeded: {successes}, failed: {failures}")
    if errors:
        for e in errors[:20]:
            print(e)


def try_init_ray():
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        started = "connected to existing cluster (address='auto')"
    except Exception:
        ray.init(ignore_reinit_error=True)
        started = "started local Ray instance"
    return started


def main() -> None:
    args = parse_args()
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print("error: S3_BUCKET env variable is not set", file=sys.stderr)
        sys.exit(1)
    concurrency = compute_concurrency(args.max_concurrency)
    prefix = DEFAULT_PREFIX
    region = args.region or os.environ.get("AWS_REGION")
    started = try_init_ray()
    print(f"Ray init status: {started}")
    if args.upload:
        upload_directory(LOCAL_BASE, bucket, prefix, concurrency, args.multipart_chunksize_mb, args.force, region)
    elif args.download:
        download_directory(bucket, LOCAL_BASE, prefix, concurrency, args.force, region, args.multipart_chunksize_mb)
    else:
        print("Please specify --upload or --download", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
