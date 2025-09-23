# Don't create python venv this is meant to run in a container

#!/usr/bin/env python3
# Don't create python venv — this is intended to run in a container
import os
import time
import tempfile
import warnings
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List

import boto3
import uno
from com.sun.star.connection import NoConnectException

# NOTE:
# Use uno.createUnoStruct to build PropertyValue structs in a robust way
# rather than relying on different bindings/constructors across platforms.
warnings.filterwarnings("ignore")

# Import your project's helpers (unchanged)
from indexing_pipeline.parse_chunk.router import env_or_fail, log, retry, list_raw_files

S3_BUCKET = env_or_fail("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "data/raw/")
OVERWRITE_OTHER_TO_PPTX = os.getenv("OVERWRITE_PPT_WITH_PPTS", "true").lower() == "true"
LIBREOFFICE_PORTS = os.getenv("LIBREOFFICE_PORTS", "7003")
CONNECT_RETRIES = int(os.getenv("UNO_CONNECT_RETRIES", "6"))
CONNECT_DELAY_SECONDS = float(os.getenv("UNO_CONNECT_DELAY_SECONDS", "1.0"))
CONVERT_RETRIES = int(os.getenv("UNO_CONVERT_RETRIES", "2"))
ALLOW_START_SOFFICE = os.getenv("ALLOW_START_SOFFICE", "true").lower() == "true"
SOFFICE_BIN = os.getenv("LIBREOFFICE_BIN", shutil.which("soffice") or "/usr/bin/soffice")

s3 = boto3.client("s3")

PPTX_FILTERS = [
    "Impress MS PowerPoint 2007 XML",
    "Impress Office Open XML",
    "Impress MS PowerPoint 2007 XML VBA",
]


def _parse_ports(ports_str: str) -> List[str]:
    return [p.strip() for p in ports_str.split(",") if p.strip()]


def _make_prop(name: str, value):
    """Create a com.sun.star.beans.PropertyValue struct reliably."""
    p = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    p.Name = name
    p.Value = value
    return p


def _start_soffice_headless(port: str, user_profile_dir: str, soffice_bin: str = SOFFICE_BIN):
    if not os.path.exists(soffice_bin):
        raise FileNotFoundError(f"soffice binary not found at '{soffice_bin}'")
    accept_arg = f"socket,host=127.0.0.1,port={port};urp;StarOffice.ServiceManager"
    cmd = [
        soffice_bin,
        "--headless",
        "--invisible",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--nolockcheck",
        f'--accept={accept_arg}',
        f'--env:UserInstallation=file://{user_profile_dir}',
    ]
    # run detached, suppressing output (container logs will capture soffice if needed)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def connect_uno(ports: Optional[List[str]] = None,
                retries: int = CONNECT_RETRIES,
                delay: float = CONNECT_DELAY_SECONDS,
                allow_start: bool = ALLOW_START_SOFFICE) -> Optional[object]:
    ports = ports or _parse_ports(LIBREOFFICE_PORTS)
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local_ctx)
    last_exc = None
    for attempt in range(1, retries + 1):
        for p in ports:
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={p};urp;StarOffice.ComponentContext")
                log(f"Connected to UNO on port {p}")
                return ctx
            except NoConnectException as e:
                last_exc = e
                log(f"UNO not available on port {p} (attempt {attempt}/{retries})", level="DEBUG")
            except Exception as e:
                last_exc = e
                log(f"UNO connect error on port {p} (attempt {attempt}/{retries}): {e}", level="DEBUG")
        time.sleep(delay)

    if not allow_start:
        raise RuntimeError(f"Unable to connect to UNO server on ports {ports}") from last_exc

    # Try spawning soffice processes and connect again
    for p in ports:
        user_profile_dir = f"/tmp/libreoffice_profile_{p}_{int(time.time())}"
        os.makedirs(user_profile_dir, exist_ok=True)
        try:
            proc = _start_soffice_headless(p, user_profile_dir)
        except Exception as e:
            last_exc = e
            log(f"Failed to start soffice on port {p}: {e}", level="WARNING")
            continue
        # wait up to retries*2 attempts for connection
        for wait in range(1, int(retries * 2) + 1):
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={p};urp;StarOffice.ComponentContext")
                log(f"Connected to UNO (spawned soffice) on port {p}")
                return ctx
            except Exception:
                time.sleep(delay)
        # give up on this proc
        try:
            proc.kill()
        except Exception:
            pass

    raise RuntimeError(f"Unable to connect to UNO server on ports {ports}") from last_exc


def store_to_pptx(component_ctx, input_path: str, output_path: str, filter_name: str):
    """
    Load document with UNO and store to output_path using filter_name.
    This function raises if the document couldn't be loaded (doc is None)
    or storeToURL fails.
    """
    smgr = component_ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", component_ctx)
    in_url = uno.systemPathToFileUrl(str(Path(input_path).resolve()))
    out_url = uno.systemPathToFileUrl(str(Path(output_path).resolve()))

    # Prefer Hidden + ReadOnly when loading unknown/old binary docs
    load_props = (
        _make_prop("Hidden", True),
        _make_prop("ReadOnly", True),
    )

    # loadComponentFromURL may return None if loading fails (see LibreOffice bug reports).
    doc = desktop.loadComponentFromURL(in_url, "_blank", 0, load_props)
    if doc is None:
        raise RuntimeError(f"UNO failed to load document (loadComponentFromURL returned None) for {input_path}")

    try:
        # For export filters use storeToURL (XStorable::storeToURL) — this is the recommended method.
        store_props = (_make_prop("FilterName", filter_name),)
        # If exporter needs additional props they could be added here (e.g. 'Overwrite' etc).
        doc.storeToURL(out_url, store_props)
    finally:
        try:
            doc.close(True)
        except Exception:
            # best-effort close; ignore errors here
            pass


def download_from_s3(key: str, tmp_dir: str) -> str:
    local_path = os.path.join(tmp_dir, os.path.basename(key))
    retry(lambda: s3.download_file(S3_BUCKET, key, local_path))
    return local_path


def upload_to_s3(local_path: str, key: str):
    content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    with open(local_path, "rb") as f:
        body = f.read()
    retry(lambda: s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type))
    log(f"Uploaded PPTX to s3://{S3_BUCKET}/{key}")


def libreoffice_convert_uno(component_ctx, input_path: str, output_dir: str) -> str:
    """
    Attempt to convert using UNO. Returns the path to created pptx on success,
    raises on failure.
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]  # e.g. CT from CT.ppt
    desired_out = os.path.join(output_dir, f"{basename}.pptx")
    last_exc = None

    for filter_name in PPTX_FILTERS:
        for attempt in range(1, CONVERT_RETRIES + 1):
            try:
                log(f"UNO: attempt store with filter '{filter_name}' (attempt {attempt})", level="DEBUG")
                store_to_pptx(component_ctx, input_path, desired_out, filter_name)
                if os.path.exists(desired_out) and os.path.getsize(desired_out) > 512:
                    return desired_out
                last_exc = RuntimeError("UNO produced file missing or too small")
                log(f"UNO produced file but it was missing or too small: {desired_out}", level="WARNING")
            except Exception as e:
                last_exc = e
                log(f"UNO conversion failed with filter '{filter_name}' (attempt {attempt}): {e}", level="WARNING")
            time.sleep(0.5 * attempt)

    raise RuntimeError("UNO conversion failed") from last_exc


def fallback_subprocess_convert(input_path: str, output_dir: str) -> str:
    """
    Use soffice --convert-to as a last-resort fallback. Produce <basename>.pptx.
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    soffice_out = os.path.join(output_dir, f"{base}.pptx")
    desired_out = soffice_out

    if not SOFFICE_BIN or not os.path.exists(SOFFICE_BIN):
        raise FileNotFoundError(f"soffice binary not found at '{SOFFICE_BIN}'")

    cmd = [
        SOFFICE_BIN,
        "--headless",
        "--invisible",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--nologo",
        "--convert-to",
        "pptx",
        "--outdir",
        output_dir,
        input_path,
    ]
    log(f"Running subprocess fallback: {' '.join(cmd)}", level="DEBUG")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(desired_out):
        return desired_out

    # Some versions emit a slightly different filename — try to locate any pptx starting with base
    for fname in os.listdir(output_dir):
        if fname.lower().endswith(".pptx") and fname.startswith(base):
            found = os.path.join(output_dir, fname)
            # normalize name
            if found != desired_out:
                os.replace(found, desired_out)
            return desired_out

    raise FileNotFoundError("PPTX not created by subprocess fallback")


def convert_and_upload(key: str):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local_input = download_from_s3(key, tmp)
            ext = key.split(".")[-1].lower()
            if ext != "ppt":
                log(f"Skipping non-ppt {key}")
                return

            # Attempt UNO connection (best-effort). If not available, component_ctx will be None.
            component_ctx = None
            try:
                component_ctx = connect_uno()
            except Exception as e:
                log(f"Unable to connect to UNO server: {e}", level="WARNING")
                component_ctx = None

            pptx_path = None
            if component_ctx is not None:
                try:
                    pptx_path = libreoffice_convert_uno(component_ctx, local_input, tmp)
                except Exception as e:
                    log(f"UNO conversion attempts failed for {key}: {e}", level="WARNING")
                    pptx_path = None

            if pptx_path is None:
                try:
                    pptx_path = fallback_subprocess_convert(local_input, tmp)
                except Exception as sb_e:
                    raise RuntimeError(f"Both UNO and subprocess conversion failed: {sb_e}") from sb_e

            # Build consistent S3 key: <prefix>/ppts/<basename>.pptx
            base = os.path.splitext(os.path.basename(key))[0]
            out_prefix = S3_RAW_PREFIX.rstrip("/") + "/ppts/"
            out_key = f"{out_prefix}{base}.pptx"
            upload_to_s3(pptx_path, out_key)

            if OVERWRITE_OTHER_TO_PPTX:
                log(f"OVERWRITE_OTHER_TO_PPTX=true → deleting original {key}")
                retry(lambda: s3.delete_object(Bucket=S3_BUCKET, Key=key))

        except Exception as e:
            log(f"Conversion failed for {key}: {e}", level="ERROR")


def main():
    log("Running ppt_to_pptx_uno.py conversion pass")
    log(f"OVERWRITE_OTHER_TO_PPTX={OVERWRITE_OTHER_TO_PPTX}")
    keys = list(list_raw_files())
    log(f"Scanning {len(keys)} files in s3://{S3_BUCKET}/{S3_RAW_PREFIX}")
    for key in keys:
        ext = key.split(".")[-1].lower()
        if ext == "ppt":
            log(f"Converting '{key}' to PPTX with UNO server")
            convert_and_upload(key)


if __name__ == "__main__":
    main()
