#!/usr/bin/env python3
"""
Robust downloader for model artifacts from the Hugging Face Hub.

Usage:
  python3 download_hf_models.py

Important:
- If a repo is private, set HUGGINGFACE_HUB_TOKEN in the environment.
- Set WORKSPACE_MODELS and FORCE_DOWNLOAD via env if desired.
"""
import os
import sys
import logging
import shutil
import time
from pathlib import Path
from typing import List

try:
    from huggingface_hub import hf_hub_download, HfApi
except Exception as e:
    raise ImportError(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from e

# Basic config
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("download_hf")

WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "/opt/models"))
FORCE = os.getenv("FORCE_DOWNLOAD", "true").lower() in ("1", "true", "yes")
MAX_RETRIES = int(os.getenv("HF_DL_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("HF_DL_BACKOFF", "2.0"))  # seconds

# Models list: you can add an "items": ["file1", "subdir/file2"] if you want explicit files.
# If "items" is missing or empty, the script will auto-detect likely files in the repo.
MODELS = [
    {
        "repo_id": "Qwen/Qwen3-4B-AWQ",
        "name": "Qwen3-4B-AWQ",
        "base": "Qwen3-4B-AWQ",
        # "items": ["config.json", "pytorch_model.safetensors", "tokenizer.json"]
    },
    {
        "repo_id": "Orion-zhen/Qwen3-1.7B-AWQ",
        "name": "Qwen3-1.7B-AWQ",
        "base": "Qwen3-1.7B-AWQ",
    },
    {
        "repo_id": "Orion-zhen/Qwen3-0.6B-AWQ",
        "name": "Qwen3-0.6B-AWQ",
        "base": "Qwen3-0.6B-AWQ",
    },
]

# Patterns considered "interesting" to download when auto-detecting
INTERESTING_SUFFIXES = (
    ".safetensors",
    ".pt",
    ".bin",
    ".bin.index.json",
    ".json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    ".model",
)


def list_repo_files(repo_id: str) -> List[str]:
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id)
        return files or []
    except Exception as e:
        logger.warning("Could not list files for %s: %s", repo_id, e)
        return []


def choose_files_from_repo(all_files: List[str]) -> List[str]:
    # Select files that match interesting suffixes, prefer top-level items (no extremely deep paths).
    chosen = []
    for f in all_files:
        for s in INTERESTING_SUFFIXES:
            if f.lower().endswith(s):
                chosen.append(f)
                break
    # If nothing chosen, fall back to downloading everything (careful with huge repos).
    if not chosen:
        logger.info("No 'interesting' files detected, defaulting to all files (count=%d)", len(all_files))
        return all_files
    # Deduplicate and keep order
    seen = set()
    ordered = []
    for f in chosen:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def download_one(repo_id: str, remote: str, target: Path) -> bool:
    """Download single remote file into target path. Returns True on success."""
    # skip if exists and not forced
    if target.exists() and not FORCE:
        logger.info("SKIP exists %s", target)
        return True

    tmp_dir = Path("/tmp") / "hf_download" / repo_id.replace("/", "_")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple times with backoff
    attempt = 0
    last_exc = None
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            got = hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                local_dir=str(tmp_dir),
                local_dir_use_symlinks=False,
                force_download=FORCE,
            )
            got_path = Path(got)
            if not got_path.exists():
                raise RuntimeError(f"hf_hub_download returned path that does not exist: {got}")

            # create parent and move file to final location
            target.parent.mkdir(parents=True, exist_ok=True)
            # if a file exists, remove (we'll replace)
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
            shutil.move(str(got_path), str(target))
            try:
                os.chmod(str(target), 0o444)
            except Exception:
                pass
            logger.info("Downloaded %s -> %s", remote, target)
            return True
        except Exception as e:
            last_exc = e
            logger.warning("Attempt %d/%d: failed to download %s:%s (%s)", attempt, MAX_RETRIES, repo_id, remote, e)
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.debug("Sleeping %.1f sec before retry", sleep_time)
                time.sleep(sleep_time)
    logger.error("Failed to download %s:%s after %d attempts. Last error: %s", repo_id, remote, MAX_RETRIES, last_exc)
    return False


def ensure_model(model: dict) -> bool:
    repo_id = model["repo_id"]
    name = model.get("name", repo_id.replace("/", "_"))
    base = model.get("base", "llm")
    model_root = WORKSPACE_MODELS / base / name

    items = model.get("items")
    if not items:
        logger.info("No explicit 'items' for %s. Listing repo files from Hugging Face...", repo_id)
        all_files = list_repo_files(repo_id)
        if not all_files:
            logger.error("No files found in repo %s (or listing failed). Skipping.", repo_id)
            return False
        items = choose_files_from_repo(all_files)
        logger.info("Autodetected %d files to download for %s", len(items), repo_id)

    ok = True
    for item in items:
        # item may be a Path or str
        remote_rel = str(item)
        target = model_root / Path(remote_rel)
        required = not remote_rel.endswith("special_tokens_map.json")
        success = download_one(repo_id, remote_rel, target)
        if not success and required:
            ok = False
            logger.error("Missing required %s:%s", name, remote_rel)
    return ok


def main() -> None:
    if os.getenv("HUGGINGFACE_HUB_TOKEN") is None:
        logger.info("No HUGGINGFACE_HUB_TOKEN found in env. Public repos should work; private repos require a token.")
    all_ok = True
    for m in MODELS:
        logger.info("Ensuring model %s (repo=%s)", m.get("name"), m["repo_id"])
        if not ensure_model(m):
            all_ok = False
    if not all_ok:
        logger.error("Some required files failed to download")
        sys.exit(2)
    logger.info("All model artifacts are present under %s", WORKSPACE_MODELS)


if __name__ == "__main__":
    main()
