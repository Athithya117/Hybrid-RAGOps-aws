# python3 -m spacy download en_core_web_sm --target /models/spacy

import os
import sys
import logging
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub") from e

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("download_hf")

WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "/workspace/indexing_pipeline/models"))
FORCE = os.getenv("FORCE_DOWNLOAD", "0").lower() in ("1", "true", "yes")

MODELS = [
    {  
        "repo_id": "Systran/faster-whisper-base",
        "name": "faster-whisper-base",
        "base": "faster_whisper",
        "items": [
            "model.bin",
            "config.json",
            "tokenizer.json",
            "vocabulary.txt",
            "README.md",
        ],
    },
]

def download_one(repo_id: str, remote: str, target: Path) -> bool:
    if target.exists() and not FORCE:
        logger.info("SKIP exists %s", target)
        return True
    tmp_dir = Path("/tmp") / "hf_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        got = hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            local_dir=str(tmp_dir),
            local_dir_use_symlinks=False,
            force_download=FORCE,
        )
        got_path = Path(got)
        if got_path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
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
        logger.warning("Failed to download %s:%s (%s)", repo_id, remote, e)
    return False


def ensure_model(model: dict) -> bool:
    repo_id = model["repo_id"]
    name = model["name"]
    base = model.get("base", "llm")
    model_root = WORKSPACE_MODELS / base / name
    ok = True
    for item in model.get("items", []):
        remote_rel = str(item)
        target = model_root / Path(remote_rel)
        required = not remote_rel.endswith("special_tokens_map.json")
        success = download_one(repo_id, remote_rel, target)
        if not success and required:
            ok = False
            logger.error("Missing required %s:%s", name, remote_rel)
    return ok


def main() -> None:
    all_ok = True
    for m in MODELS:
        if not ensure_model(m):
            all_ok = False
    if not all_ok:
        logger.error("Some required files failed to download")
        sys.exit(2)
    logger.info("All model artifacts are present under %s", WORKSPACE_MODELS)


if __name__ == "__main__":
    main()