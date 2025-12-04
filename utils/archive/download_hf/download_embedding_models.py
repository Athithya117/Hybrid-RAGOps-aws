from __future__ import annotations  
import os  
import sys  
import logging  
import shutil  
import subprocess  
from pathlib import Path  
from typing import Optional, List, Dict  
  
try:  
    # huggingface_hub 0.15+ recommended  
    from huggingface_hub import hf_hub_download  
except Exception as exc:  # pragma: no cover - dependency error  
    raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub") from exc  
  
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")  
logger = logging.getLogger("download_models")  
  
WORKSPACE_MODELS = Path(os.getenv("WORKSPACE_MODELS", "infra/embedding-models-int8"))  
FORCE = os.getenv("FORCE_DOWNLOAD", "0").lower() in ("1", "true", "yes")  
TMP_DIR_BASE = Path(os.getenv("TMP_DIR_BASE", "/tmp/hf_download")).resolve()  
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None  
  
TMP_DIR_BASE.mkdir(parents=True, exist_ok=True)  
WORKSPACE_MODELS.mkdir(parents=True, exist_ok=True)  
  
# Repos to fetch; keep order of preference for onnx + vocab/tokenizer candidates.  
REPOS = [  
    {  
        "repo_id": "nixiesearch/e5-small-v2-onnx",  
        "name": "e5-small-v2-onnx",  
        "onnx_candidates": [  
            "model_opt2_QInt8.onnx",  
            "model_quantized.onnx",  
            "model.onnx",  
        ],  
        "vocab_candidates": [  
            "tokenizer.json",  
            "tokenizer_config.json",  
            "special_tokens_map.json",  
            "config.json",  
        ],  
        # alt_repos can be used to fetch tokenizer artifacts from compatible repos  
        "alt_repos": ["intfloat/e5-small-v2", "intfloat/e5-small-v2-onnx"],  
    },  
    {  
        "repo_id": "castorini/splade-v3-onnx",  
        "name": "splade-v3-onnx",  
        "onnx_candidates": [  
            "splade-v3-8bit.onnx",  
            "onnx/splade-v3-8bit.onnx",  
            "model.onnx",  
            "model_int8.onnx",  
        ],  
        "vocab_candidates": [  
            "splade-v3-vocab.txt",  
            "vocab.txt",  
            "config.json",  
        ],  
        # splade tokenizer often lives under naver/splade-v3  
        "alt_repos": ["naver/splade-v3"],  
    }  
]  
  
# Optional additional small files to download where present  
COMMON_ITEMS = [  
    "README.md",  
    "config.json",  
    "tokenizer.json",  
    "tokenizer_config.json",  
    "special_tokens_map.json",  
]  
  
# hf_hub_download kwargs  
HF_DOWNLOAD_KW = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}  
  
# Helper ---------------------------------------------------------------------  
def download_one(repo_id: str, remote: str, target: Path) -> bool:  
    """  
    Download single file `remote` from `repo_id` into final `target` path.  
    Returns True on success (file exists at target), False otherwise.  
    """  
    if target.exists() and not FORCE:  
        logger.debug("SKIP exists: %s", target)  
        return True  
  
    tmp_dir = TMP_DIR_BASE / repo_id.replace("/", "_")  
    tmp_dir.mkdir(parents=True, exist_ok=True)  
    try:  
        logger.debug("Attempting hf_hub_download(%s, %s)", repo_id, remote)  
        got = hf_hub_download(repo_id=repo_id, filename=remote, local_dir=str(tmp_dir),  
                              local_dir_use_symlinks=False, force_download=FORCE, **HF_DOWNLOAD_KW)  
        got_path = Path(got)  
        if got_path.exists():  
            target.parent.mkdir(parents=True, exist_ok=True)  
            # move into final location (overwrite if exists)  
            try:  
                if target.exists():  
                    target.unlink()  
            except Exception:  
                pass  
            shutil.move(str(got_path), str(target))  
            try:  
                os.chmod(str(target), 0o444)  
            except Exception:  
                pass  
            logger.info("Downloaded %s:%s -> %s", repo_id, remote, target)  
            return True  
    except Exception as e:  
        logger.debug("hf_hub_download failed for %s:%s (%s)", repo_id, remote, e)  
        logger.warning("Failed to download %s:%s (%s)", repo_id, remote, e)  
    return False  
  
def try_alternates_for_item(item: str, alt_repos: List[str], target: Path) -> bool:  
    """  
    Try to download `item` from any alternate repo in alt_repos.  
    """  
    for alt in alt_repos or []:  
        if download_one(alt, item, target):  
            logger.info("Downloaded alternate %s from %s", item, alt)  
            return True  
    return False  
  
def ensure_model(repo: Dict[str, any], model_root: Path) -> bool:  
    """  
    Ensure required files for a repo exist under model_root.  
    Returns True if success (onnx + at least one vocab/tokenizer), False otherwise.  
    """  
    repo_id = repo["repo_id"]  
    name = repo["name"]  
    logger.info("Ensuring model %s under %s", repo_id, model_root)  
  
    model_root.mkdir(parents=True, exist_ok=True)  
  
    # common items best-effort  
    for item in COMMON_ITEMS:  
        target = model_root / item  
        # attempt from primary repo  
        _ = download_one(repo_id, item, target)  
        # attempt alternates if not present  
        if not target.exists() and repo.get("alt_repos"):  
            try_alternates_for_item(item, repo.get("alt_repos"), target)  
  
    # vocab/tokenizer candidates (need at least one)  
    vocab_ok = False  
    for v in repo.get("vocab_candidates", []):  
        target = model_root / v  
        if download_one(repo_id, v, target):  
            vocab_ok = True  
            break  
        # try alt repos for vocab  
        if repo.get("alt_repos"):  
            if try_alternates_for_item(v, repo.get("alt_repos"), target):  
                vocab_ok = True  
                break  
  
    # ONNX candidates: try specified order, also try common fallbacks  
    onnx_ok = False  
    onnx_target: Optional[Path] = None  
    for candidate in repo.get("onnx_candidates", []):  
        target = model_root / candidate  
        if download_one(repo_id, candidate, target):  
            onnx_ok = True  
            onnx_target = target  
            break  
        # try under onnx/ subdir if candidate is not prefixed  
        if not candidate.startswith("onnx/"):  
            alt = "onnx/" + candidate  
            target2 = model_root / alt  
            if download_one(repo_id, alt, target2):  
                onnx_ok = True  
                onnx_target = target2  
                break  
        # try alternates for candidate  
        if repo.get("alt_repos"):  
            for alt_repo in repo["alt_repos"]:  
                if download_one(alt_repo, candidate, model_root / candidate):  
                    onnx_ok = True  
                    onnx_target = model_root / candidate  
                    break  
        if onnx_ok:  
            break  
  
    # fallback ONNX names  
    if not onnx_ok:  
        fallback_names = ["onnx/model_int8.onnx", "onnx/model.onnx", "model.onnx", "model_quantized.onnx", "model_int8.onnx"]  
        for f in fallback_names:  
            t = model_root / f  
            if download_one(repo_id, f, t):  
                onnx_ok = True  
                onnx_target = t  
                break  
            # try alternates  
            if repo.get("alt_repos"):  
                for alt_repo in repo["alt_repos"]:  
                    if download_one(alt_repo, f, model_root / f):  
                        onnx_ok = True  
                        onnx_target = model_root / f  
                        break  
            if onnx_ok:  
                break  
  
    if not onnx_ok:  
        logger.error("Missing ONNX for %s. Tried candidates: %s", name, repo.get("onnx_candidates", []))  
    else:  
        logger.debug("Found ONNX for %s -> %s", name, onnx_target)  
  
    # If vocab not found but there is a splade-specific vocab-like file, attempt to create vocab.txt  
    if not vocab_ok and repo["name"].startswith("splade"):  
        # search for any *vocab*.txt that may have been downloaded via common items  
        candidates = list(model_root.glob("*vocab*.txt")) + list(model_root.glob("*.txt"))  
        for c in candidates:  
            # prefer files with 'vocab' in name  
            if "vocab" in c.name.lower():  
                try:  
                    target_vocab = model_root / "vocab.txt"  
                    if not target_vocab.exists():  
                        shutil.copyfile(str(c), str(target_vocab))  
                        try:  
                            os.chmod(str(target_vocab), 0o444)  
                        except Exception:  
                            pass  
                        logger.info("Copied %s -> %s to help tokenizer loading", c, target_vocab)  
                    vocab_ok = True  
                    break  
                except Exception as e:  
                    logger.warning("Failed to copy vocab file %s -> %s: %s", c, target_vocab, e)  
        if not vocab_ok:  
            # try alternates (naver splade)  
            if repo.get("alt_repos"):  
                for alt in repo["alt_repos"]:  
                    for alt_name in ("tokenizer.json", "vocab.txt", "tokenizer_config.json"):  
                        if download_one(alt, alt_name, model_root / alt_name):  
                            vocab_ok = True  
                            break  
                    if vocab_ok:  
                        break  
  
    if not vocab_ok:  
        logger.warning("No tokenizer/vocab file found for %s (attempted %s). Some tooling may fail.", name, repo.get("vocab_candidates", []))  
  
    ok = onnx_ok and vocab_ok  
    if ok:  
        logger.info("Model %s ready under %s (onnx=%s, vocab_ok=%s)", name, model_root, str(onnx_target) if onnx_target else "unknown", vocab_ok)  
    else:  
        logger.error("Model %s is incomplete (onnx_ok=%s, vocab_ok=%s).", name, onnx_ok, vocab_ok)  
    return ok  
  
  
def main() -> None:  
    all_ok = True  
    logger.info("Workspace models root: %s (force=%s)", WORKSPACE_MODELS, FORCE)  
    for repo in REPOS:  
        model_root = WORKSPACE_MODELS / repo["name"]  
        model_root.mkdir(parents=True, exist_ok=True)  
        ok = ensure_model(repo, model_root)  
        if not ok:  
            all_ok = False  
  
    if not all_ok:  
        logger.error("Some required files failed to download under %s", WORKSPACE_MODELS)  
        sys.exit(2)  
  
    logger.info("All requested ONNX model artifacts are present under %s", WORKSPACE_MODELS)  
    sys.exit(0)  
  
if __name__ == "__main__":  
    main()  