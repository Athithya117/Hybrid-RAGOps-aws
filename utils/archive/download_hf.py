#!/usr/bin/env python3
"""download_hf_fixed.py

Fixed version of the user's downloader script. Resolves:
 - missing imports
 - MODELS items unpacking error (accepts both str and (remote,target) tuples)
 - safer path handling and clearer logging
"""
import os
import sys
import logging
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    raise ImportError("huggingface_hub is required for this script: pip install huggingface_hub") from e

# configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("download_hf")

WORKSPACE_MODELS = os.getenv("WORKSPACE_MODELS", "/workspace/models")
FORCE = os.getenv("FORCE_DOWNLOAD", "0").lower() in ("1", "true", "yes")

# NOTE: items may be either:
#  - a string (remote path) -> saved with the same relative path locally, or
#  - a 2-tuple/list (remote_rel, target_rel)
MODELS = [
    {"repo_id": "RAG8s/gte-modernbert-base-onnx-int8", "name": "gte-modernbert-base-onnx-int8", "base": "onnx", "items": ["model.onnx"]},
    # Example (kept commented) of how to specify remote->target mapping explicitly:
    # {"repo_id": "RAG8s/gte-reranker-modernbert-base-onnx-int8", "name": "gte-reranker-modernbert-base-onnx-int8", "base": "onnx", "items": [("onnx/model_int8.onnx","onnx/model_int8.onnx"), ("config.json","config.json")]},
]


def download_one(repo_id: str, name: str, remote_candidates: list, target_rel: str, base: str) -> bool:
    """Attempt to download one file from a list of remote candidate paths.

    remote_candidates: list of remote filename strings to try in order.
    target_rel: relative path under WORKSPACE_MODELS/<base>/<name>/ to place the file.
    """
    target = Path(WORKSPACE_MODELS) / base / name / Path(target_rel)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not FORCE:
        logger.info("SKIP exists %s", target)
        return True

    last_exc = None
    tmp_dir = Path("/tmp") / "hf_download" / base / name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for remote in remote_candidates:
        try:
            logger.info("Attempting %s -> %s", remote, str(target))
            got = hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                local_dir=str(tmp_dir),
                local_dir_use_symlinks=False,
                force_download=FORCE,
            )
            got_path = Path(got)
            if got_path.exists():
                # Move into place (avoid overwriting if already exactly the same file)
                if got_path.resolve() != target.resolve():
                    # ensure parent exists
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # If target exists and FORCE is True, remove it first
                    if target.exists():
                        try:
                            target.unlink()
                        except Exception:
                            pass
                    shutil.move(str(got_path), str(target))
                try:
                    os.chmod(str(target), 0o444)
                except Exception:
                    # not critical
                    pass
                logger.info("Downloaded %s to %s", remote, str(target))
                return True
        except Exception as e:
            last_exc = e
            logger.debug("failed candidate %s: %s", remote, e, exc_info=True)

    logger.error("All candidates failed for %s:%s last=%s", repo_id, target_rel, last_exc)
    return False


def ensure_model(model: dict) -> bool:
    """Ensure all items for a given model are present locally.

    Supports items as either strings or (remote_rel, target_rel) pairs.
    """
    repo_id = model["repo_id"]
    name = model["name"]
    base = model.get("base", "llm")

    ok = True
    for item in model.get("items", []):
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                logger.error("Invalid item entry for model %s: %r (expected 2-tuple or string)", name, item)
                ok = False
                continue
            remote_rel, target_rel = item
        else:
            remote_rel = item
            target_rel = item

        # decide whether this file is required (special_tokens_map.json is optional in original script)
        required = not str(remote_rel).endswith("special_tokens_map.json")

        candidates = [str(remote_rel)]
        if "/" in str(remote_rel):
            candidates.append(str(remote_rel).split("/", 1)[1])
        if str(remote_rel).startswith("onnx/"):
            candidates.append(str(remote_rel).split("onnx/", 1)[1])

        success = download_one(repo_id, name, candidates, target_rel, base)
        if not success and required:
            ok = False
            logger.error("Required file missing for %s: %s", name, target_rel)

    return ok


def remove_stray_onnx_duplicates(models: list) -> None:
    """Remove stray ONNX folders that duplicate non-onnx model folders.

    For each model that is *not* an onnx base, if there is a folder in
    <WORKSPACE_MODELS>/onnx/ that matches the model name or repo name, remove it.
    """
    onnx_root = Path(WORKSPACE_MODELS) / "onnx"
    if not onnx_root.exists() or not onnx_root.is_dir():
        return

    for m in models:
        # only consider models that are not already 'onnx' base
        if m.get("base") == "onnx":
            continue

        llm_dir = Path(WORKSPACE_MODELS) / m.get("base", "llm") / m["name"]
        if not llm_dir.exists():
            continue

        repo_last = m["repo_id"].split("/", 1)[1] if "/" in m["repo_id"] else m["repo_id"]
        candidates = set([m["name"].lower(), repo_last.lower()])

        try:
            for child in onnx_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name.lower() in candidates:
                    try:
                        shutil.rmtree(child)
                        logger.info("Removed stray onnx duplicate %s", str(child))
                    except Exception as e:
                        logger.warning("Failed to remove %s: %s", str(child), e)
        except Exception as e:
            logger.warning("Scanning onnx root failed: %s", e)


def main() -> None:
    all_ok = True
    for m in MODELS:
        if not ensure_model(m):
            all_ok = False

    if not all_ok:
        logger.error("One or more required files failed to download")
        sys.exit(2)

    remove_stray_onnx_duplicates(MODELS)
    logger.info("All model artifacts present under %s", WORKSPACE_MODELS)


if __name__ == "__main__":
    main()
