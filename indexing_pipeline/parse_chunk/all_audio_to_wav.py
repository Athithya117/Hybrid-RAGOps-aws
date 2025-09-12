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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("all_audio_to_wav")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "data/raw/").lstrip("/").rstrip("/") + "/"
FORCE = os.getenv("FORCE_CONVERT", "0").lower() in ("1", "true", "yes")
OVERWRITE_ALL_AUDIO_FILES = os.getenv("OVERWRITE_ALL_AUDIO_FILES", "false").lower() in ("1", "true", "yes")
AWS_REGION = os.getenv("AWS_REGION")
FFMPEG_BIN = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_PATH", "ffprobe")
TMP_DIR = os.getenv("TMP_DIR", "/tmp")

if not S3_BUCKET:
    logger.error("S3_BUCKET env var is required")
    sys.exit(2)

session = boto3.Session(region_name=AWS_REGION) if AWS_REGION else boto3.Session()
s3 = session.client("s3")

AUDIO_EXTS = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm", ".amr", ".wma", ".aiff", ".aif"
}

def run_cmd(cmd: list) -> Tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    rc = proc.returncode
    return rc, out + err


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
    s3.download_file(bucket, key, str(out_path))


def s3_upload(bucket: str, key: str, src_path: Path) -> None:
    s3.upload_file(str(src_path), bucket, key)


def probe_audio(path: Path) -> Tuple[int, int, str]:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name",
        "-of",
        "json",
        str(path),
    ]
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
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    rc, out = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed converting {src} -> {dst}: {out}")


def target_key_for_wav(original_key: str) -> str:
    base, _ = os.path.splitext(original_key)
    return f"{base}.wav"


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
    logger.info("Processing s3://%s/%s (OVERWRITE_ALL_AUDIO_FILES=%s FORCE=%s)", bucket, key, OVERWRITE_ALL_AUDIO_FILES, FORCE)
    target_key = target_key_for_wav(key)
    target_exists = False
    try:
        s3.head_object(Bucket=bucket, Key=target_key)
        target_exists = True
    except ClientError:
        target_exists = False

    # If target exists and FORCE not set, we normally skip uploading.
    if target_exists and not FORCE:
        logger.info("Target %s already exists and FORCE not set.", target_key)
        # If user requested overwrite semantics (delete originals), do it anyway (but avoid deleting if same key)
        if OVERWRITE_ALL_AUDIO_FILES and target_key != key:
            try:
                delete_s3_key(bucket, key)
                return True
            except Exception:
                logger.exception("Failed to delete original after skipping upload for %s", key)
                return False
        logger.info("Skipping %s", key)
        return False

    local_src = tmp_root / "src" / Path(key).name
    local_dst = tmp_root / "dst" / (Path(key).stem + ".wav")
    try:
        s3_download(bucket, key, local_src)
    except Exception as e:
        logger.exception("Failed to download s3://%s/%s -> %s: %s", bucket, key, local_src, e)
        return False

    # If source is already a good 16k mono wav, upload as-is (unless it's same key)
    try:
        if local_src.suffix.lower() == ".wav" and already_ok_wav(local_src):
            logger.info("Source is already 16kHz mono WAV: %s", local_src)
            # If original key equals target_key, nothing to change; but if OVERWRITE requested, we don't delete it because it's the same file.
            if local_src.name == Path(target_key).name and key.endswith(".wav"):
                logger.info("Original is target WAV; nothing to upload.")
                # If OVERWRITE_ALL_AUDIO_FILES true and target_key != key, handled earlier; here target==original so nothing to delete.
                return False
            try:
                s3_upload(bucket, target_key, local_src)
            except Exception as e:
                logger.exception("Upload failed for %s -> s3://%s/%s: %s", local_src, bucket, target_key, e)
                return False
            logger.info("Uploaded existing WAV to %s", target_key)
            # If requested, delete original (only if different key)
            if OVERWRITE_ALL_AUDIO_FILES and target_key != key:
                delete_s3_key(bucket, key)
            return True
    except Exception as e:
        logger.warning("Probe/handling failed for %s: %s (will attempt conversion)", local_src, e)

    # Convert and upload
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

    logger.info("Converted and uploaded %s -> s3://%s/%s", key, bucket, target_key)

    # If requested, delete the original (only if different key)
    if OVERWRITE_ALL_AUDIO_FILES and target_key != key:
        delete_s3_key(bucket, key)

    return True


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="audioconv_", dir=TMP_DIR))
    logger.info("Using temp dir %s", tmp_root)
    processed = 0
    skipped = 0
    failed = 0
    try:
        for key in list_audio_keys(S3_BUCKET, S3_PREFIX):
            try:
                ok = process_key(S3_BUCKET, key, tmp_root)
                if ok:
                    processed += 1
                else:
                    skipped += 1
            except Exception:
                failed += 1
                logger.exception("Failed processing %s", key)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

    logger.info("Done. processed=%d skipped=%d failed=%d", processed, skipped, failed)


if __name__ == "__main__":
    main()
