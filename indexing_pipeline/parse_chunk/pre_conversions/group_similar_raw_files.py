
import os
import argparse
import boto3
import botocore
from collections import defaultdict
from indexing_pipeline.parse_chunk.router import env_or_fail, log, retry

S3_BUCKET = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/")
if not S3_RAW_PREFIX.endswith("/"):
    S3_RAW_PREFIX += "/"

s3 = boto3.client("s3")

FORMATS = [
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm", ".amr", ".wma", ".aiff", ".aif",
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif",
    ".pdf", ".doc", ".docx", ".pptx", ".txt", ".csv", ".md", ".html", ".jsonl", "other"
]

ALLOW_OTHER = "other" in FORMATS

EXT_TO_PREFIX = {
    "mp3": "audio/",
    "m4a": "audio/",
    "aac": "audio/",
    "wav": "audio/",
    "flac": "audio/",
    "ogg": "audio/",
    "opus": "audio/",
    "webm": "audio/",
    "amr": "audio/",
    "wma": "audio/",
    "aiff": "audio/",
    "aif": "audio/",
    "jpg": "images/",
    "jpeg": "images/",
    "png": "images/",
    "webp": "images/",
    "tif": "images/",
    "tiff": "images/",
    "bmp": "images/",
    "gif": "images/",
    "pdf": "pdfs/",
    "doc": "docs/",
    "docx": "docs/",
    "pptx": "ppts/",
    "txt": "txts/",
    "csv": "csvs/",
    "md": "mds/",
    "html": "htmls/",
    "jsonl": "jsonls/"
}


ALLOWED_EXTS = {e.lstrip(".") for e in FORMATS if e != "other"}
SKIP_SUFFIXES = ("/",)

def _list_objects(prefix):
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": S3_BUCKET, "Prefix": prefix}
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            yield obj

def _exists(key):
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound", "404 Not Found"):
            return False
        raise

def _determine_target_for_name(name_lower):
    for ext in sorted(EXT_TO_PREFIX.keys(), key=len, reverse=True):
        if name_lower.endswith("." + ext):
            return EXT_TO_PREFIX[ext]
    parts = name_lower.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in ALLOWED_EXTS:
        return EXT_TO_PREFIX.get(parts[1], "others/")
    if ALLOW_OTHER:
        return "others/"
    return None

def _determine_target_subprefix(key, keys_set):
    base = key[len(S3_RAW_PREFIX):] if key.startswith(S3_RAW_PREFIX) else key
    if "/" in base:
        first = base.split("/", 1)[0]
        known_prefixes = {v.rstrip("/") for v in set(EXT_TO_PREFIX.values())}
        known_prefixes.update({"others", "manifests"})
        if first in known_prefixes:
            return None
    name = os.path.basename(key)
    name_lower = name.lower()
    if name_lower.endswith(".manifest.json"):
        raw_key = key[:-len(".manifest.json")]
        raw_name = os.path.basename(raw_key).lower()
        if raw_key in keys_set or _exists(raw_key):
            target = _determine_target_for_name(raw_name)
            return target
        if ALLOW_OTHER:
            return "manifests/"
        return None
    return _determine_target_for_name(name_lower)

def _copy_and_delete(src_key, dst_key):
    copy_source = {"Bucket": S3_BUCKET, "Key": src_key}
    def _copy():
        try:
            s3.copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=dst_key, MetadataDirective="COPY")
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound", "404 Not Found"):
                log(f"Source not found, skipping copy: s3://{S3_BUCKET}/{src_key}", level="WARNING")
                return False
            raise
        return True
    copied = retry(_copy)
    if not copied:
        return False
    def _del():
        s3.delete_object(Bucket=S3_BUCKET, Key=src_key)
    retry(_del)
    return True

def group_objects(dry_run=False, prefix_filter=None):
    all_keys = []
    for obj in _list_objects(S3_RAW_PREFIX):
        k = obj["Key"]
        if k.endswith(SKIP_SUFFIXES):
            continue
        if prefix_filter and not k.startswith(prefix_filter):
            continue
        all_keys.append(k)
    keys_set = set(all_keys)
    mapping = defaultdict(list)
    for key in all_keys:
        sub = _determine_target_subprefix(key, keys_set)
        if not sub:
            continue
        mapping[sub].append(key)
    moved = 0
    skipped = 0
    processed = set()
    for sub in sorted(mapping.keys()):
        for key in sorted(mapping[sub]):
            if key in processed:
                continue
            if key.startswith(S3_RAW_PREFIX + sub):
                processed.add(key)
                skipped += 1
                continue
            name = os.path.basename(key)
            dst_key = S3_RAW_PREFIX + sub + name
            if dst_key == key:
                processed.add(key)
                skipped += 1
                continue
            if _exists(dst_key):
                base, ext = os.path.splitext(name)
                i = 1
                while True:
                    candidate = f"{base}_{i}{ext}"
                    dst_key = S3_RAW_PREFIX + sub + candidate
                    if not _exists(dst_key):
                        break
                    i += 1
            if dry_run:
                log(f"DRY-RUN: would move s3://{S3_BUCKET}/{key} -> s3://{S3_BUCKET}/{dst_key}")
                moved += 1
                processed.add(key)
                manifest_src = key + ".manifest.json"
                if manifest_src in keys_set:
                    processed.add(manifest_src)
                continue
            try:
                moved_ok = _copy_and_delete(key, dst_key)
                if moved_ok:
                    moved += 1
                    processed.add(key)
                    manifest_src = key + ".manifest.json"
                    if manifest_src in keys_set or _exists(manifest_src):
                        dst_manifest = dst_key + ".manifest.json"
                        try:
                            _copy_and_delete(manifest_src, dst_manifest)
                            processed.add(manifest_src)
                        except Exception as e:
                            log(f"Failed to move manifest {manifest_src}: {e}", level="ERROR")
                else:
                    skipped += 1
                    processed.add(key)
            except Exception as e:
                log(f"Failed to move {key}: {e}", level="ERROR")
                skipped += 1
                processed.add(key)
    log(f"Grouping complete. moved={moved} skipped={skipped}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--filter", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    group_objects(dry_run=args.dry_run, prefix_filter=args.filter)
