#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, whoami

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_HOME = Path(os.getenv("MODEL_HOME", "/opt/models"))
ONNX_ROOT = MODEL_HOME / "hf" / "onnx"

MODEL_CONFIG = {
    "gte-modernbert-base": "gte-modernbert-base-onnx-int8",
    "gte-reranker-modernbert-base": "gte-reranker-modernbert-base-onnx-int8"
}

ALLOWED_FILES = {
    "onnx/model_int8.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md"
}

def get_username():
    info = whoami()
    username = info.get("name")
    if not username:
        raise RuntimeError("Login to HF CLI first: `huggingface-cli login`.")
    return username

def upload_model(model_dirname: str, repo_suffix: str, username: str):
    model_dir = ONNX_ROOT / model_dirname
    repo_id = f"{username}/{repo_suffix}"
    create_repo(repo_id, repo_type="model", exist_ok=True)

    for relpath in ALLOWED_FILES:
        src_path = model_dir / relpath
        if not src_path.exists():
            logger.warning(f"Skipping missing file: {src_path}")
            continue
        upload_file(
            path_or_fileobj=str(src_path),
            path_in_repo=relpath,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {relpath}"
        )
        logger.info(f" Uploaded {relpath} to {repo_id}")

if __name__ == "__main__":
    try:
        username = get_username()
        for model, suffix in MODEL_CONFIG.items():
            upload_model(model, suffix, username)
    except Exception as e:
        logger.error(str(e))
        exit(1)
