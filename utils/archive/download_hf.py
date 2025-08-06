import os
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed paths from env or fallback defaults
EMBEDDER_ONNX_PATH = os.getenv(
    "EMBEDDER_ONNX_PATH", "/opt/models/hf/onnx/gte-modernbert-base/model.onnx"
)
RERANKER_ONNX_PATH = os.getenv(
    "RERANKER_ONNX_PATH", "/opt/models/hf/onnx/gte-reranker-modernbert-base/model.onnx"
)

MODELS = [
    {
        "repo_id": "RAG8s/gte-modernbert-base-onnx",
        "filename": "model.onnx",
        "target_path": EMBEDDER_ONNX_PATH,
    },
    {
        "repo_id": "RAG8s/gte-reranker-modernbert-base-onnx",
        "filename": "model.onnx",
        "target_path": RERANKER_ONNX_PATH,
    },
]


def download_model(repo_id, filename, target_path):
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    logger.info(f"Downloading {filename} from {repo_id} to {target_path}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        force_download=True,
    )

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Download failed: {target_path}")
    logger.info(f"Model ready at: {target_path}")


if __name__ == "__main__":
    for model in MODELS:
        download_model(**model)
