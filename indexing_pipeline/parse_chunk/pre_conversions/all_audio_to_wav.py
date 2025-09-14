#!/usr/bin/env python3
import os
import sys
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "data/raw/").lstrip("/").rstrip("/") + "/"
FORCE = os.getenv("FORCE_CONVERT", "0").lower() in ("1", "true", "yes")
OVERWRITE_ALL_AUDIO_FILES = os.getenv("OVERWRITE_ALL_AUDIO_FILES", "false").lower() in ("1", "true", "yes")
AWS_REGION = os.getenv("AWS_REGION")
FFMPEG_BIN = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_PATH", "ffprobe")
TMP_DIR = os.getenv("TMP_DIR", "/tmp")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("all_audio_to_wav")
if not S3_BUCKET:
    logger.error("S3_BUCKET env var is required")
    sys.exit(2)
botocore_cfg = BotoConfig(connect_timeout=30, read_timeout=300, retries={"max_attempts": 3, "mode": "standard"})
session = boto3.Session(region_name=AWS_REGION) if AWS_REGION else boto3.Session()
s3 = session.client("s3", config=botocore_cfg)
TRANSFER_CFG = TransferConfig(multipart_threshold=8 * 1024 * 1024, max_concurrency=4, use_threads=False)
AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm", ".amr", ".wma", ".aiff", ".aif"}

def run_cmd(cmd: list) -> Tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    rc = proc.returncode
    return rc, (out or "") + (err or "")

def list_audio_keys(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            _, ext = os.path.splitext(key.lower())
            if ext in AUDIO_EXTS:
                yield key

def s3_download(bucket: str, key: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Downloading s3://%s/%s -> %s", bucket, key, out_path)
    s3.download_file(bucket, key, str(out_path))

def s3_upload(bucket: str, key: str, src_path: Path) -> None:
    logger.debug("Uploading %s -> s3://%s/%s", src_path, bucket, key)
    try:
        s3.upload_file(str(src_path), bucket, key, Config=TRANSFER_CFG)
    except KeyboardInterrupt:
        logger.warning("Upload interrupted by KeyboardInterrupt for %s -> s3://%s/%s", src_path, bucket, key)
        raise
    except Exception as e:
        logger.exception("s3.upload_file failed for %s -> s3://%s/%s: %s", src_path, bucket, key, e)
        raise

def probe_audio(path: Path) -> Tuple[int, int, str]:
    cmd = [FFPROBE_BIN, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels,codec_name", "-of", "json", str(path)]
    rc, out = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out}")
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"no audio stream found in {path}")
    st = streams[0]
    sr = int(st.get("sample_rate", 0))
    ch = int(st.get("channels", 0))
    codec = st.get("codec_name", "")
    return sr, ch, codec

def convert_to_wav(src: Path, dst: Path, sample_rate: int = 16000) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-ar", str(sample_rate), "-ac", "1", "-sample_fmt", "s16", str(dst)]
    rc, out = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed converting {src} -> {dst}: {out}")

def target_key_for_wav(original_key: str) -> str:
    key = original_key
    while key.lower().endswith(".wav"):
        key = key[:-4]
    return f"{key}.wav"

def already_ok_wav(path: Path) -> bool:
    try:
        sr, ch, codec = probe_audio(path)
        if sr == 16000 and ch == 1:
            return True
    except Exception:
        return False
    return False

def delete_s3_key(bucket: str, key: str) -> bool:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted original s3://%s/%s", bucket, key)
        return True
    except Exception as e:
        logger.exception("Failed to delete s3://%s/%s: %s", bucket, key, e)
        return False

def process_key(bucket: str, key: str, tmp_root: Path) -> bool:
    target_key = target_key_for_wav(key)
    logger.info("Processing s3://%s/%s (target=%s OVERWRITE_ALL_AUDIO_FILES=%s FORCE=%s)", bucket, key, target_key, OVERWRITE_ALL_AUDIO_FILES, FORCE)
    target_exists = False
    try:
        s3.head_object(Bucket=bucket, Key=target_key)
        target_exists = True
    except ClientError:
        target_exists = False
    if target_exists and not FORCE:
        logger.info("Target s3://%s/%s already exists and FORCE not set. Skipping.", bucket, target_key)
        if OVERWRITE_ALL_AUDIO_FILES and target_key != key:
            try:
                delete_s3_key(bucket, key)
                return True
            except Exception:
                logger.exception("Failed to delete original after skipping upload for %s", key)
                return False
        return False
    local_src = tmp_root / "src" / Path(key).name
    local_dst = tmp_root / "dst" / Path(target_key).name
    try:
        s3_download(bucket, key, local_src)
    except Exception as e:
        logger.exception("Failed to download s3://%s/%s -> %s: %s", bucket, key, local_src, e)
        return False
    try:
        if local_src.suffix.lower() == ".wav" and already_ok_wav(local_src):
            logger.info("Source is already 16kHz mono WAV: %s", local_src)
            if target_key == key:
                logger.info("Original is the target WAV; nothing to upload.")
                return False
            try:
                s3_upload(bucket, target_key, local_src)
            except Exception as e:
                logger.exception("Upload failed for %s -> s3://%s/%s: %s", local_src, bucket, target_key, e)
                return False
            logger.info("Uploaded existing WAV to s3://%s/%s", bucket, target_key)
            if OVERWRITE_ALL_AUDIO_FILES and target_key != key:
                delete_s3_key(bucket, key)
            return True
    except Exception as e:
        logger.warning("Probe/handling failed for %s: %s (will attempt conversion)", local_src, e)
    try:
        convert_to_wav(local_src, local_dst, sample_rate=16000)
    except Exception as e:
        logger.exception("Conversion failed for %s: %s", local_src, e)
        return False
    try:
        s3_upload(bucket, target_key, local_dst)
    except Exception as e:
        logger.exception("Upload failed for %s -> s3://%s/%s: %s", local_dst, bucket, target_key, e)
        return False
    logger.info("Converted and uploaded s3://%s/%s -> s3://%s/%s", bucket, key, bucket, target_key)
    if OVERWRITE_ALL_AUDIO_FILES and target_key != key:
        delete_s3_key(bucket, key)
    return True

def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="audioconv_", dir=TMP_DIR))
    logger.info("Using temp dir %s", tmp_root)
    processed = 0
    skipped = 0
    failed = 0
    rc, out = run_cmd([FFPROBE_BIN, "-version"])
    if rc != 0:
        logger.warning("ffprobe not found or failing: %s", out)
    rc, out = run_cmd([FFMPEG_BIN, "-version"])
    if rc != 0:
        logger.warning("ffmpeg not found or failing: %s", out)
    try:
        for key in list_audio_keys(S3_BUCKET, S3_PREFIX):
            try:
                ok = process_key(S3_BUCKET, key, tmp_root)
                if ok:
                    processed += 1
                else:
                    skipped += 1
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received — aborting processing loop")
                break
            except Exception:
                failed += 1
                logger.exception("Failed processing %s", key)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            logger.exception("Failed to remove tmp dir %s", tmp_root)
    logger.info("Done. processed=%d skipped=%d failed=%d", processed, skipped, failed)
if __name__ == "__main__":
    main()
