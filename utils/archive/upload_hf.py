#!/usr/bin/env bash
set -euo pipefail

# 1. Install dependencies (CPU-only)
pip install --no-cache-dir \
    torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    huggingface_hub==0.34.3 \
    optimum[onnxruntime]==1.27.0 \
    hf-transfer==0.1.9 \
    numpy==2.2.6 \
    onnx==1.18.0 \
    accelerate==1.9.0 \
    onnxruntime==1.22.1 \
    onnxoptimizer==0.3.13 \
    onnxruntime-tools==1.7.0 \
    transformers==4.53.3 \
    sentencepiece==0.2.0

sudo pip3 install huggingface_hub==0.34.3

# 2. Prepare model directories
sudo mkdir -p /opt/models/hf/onnx/gte-modernbert-base
sudo mkdir -p /opt/models/hf/onnx/gte-reranker-modernbert-base
sudo chown -R $USER:$USER /opt/models

# 3. Clone the full repos (with all quantization variants)
git -C /opt/models/hf/onnx/gte-modernbert-base pull 2>/dev/null || \
  git clone https://huggingface.co/Alibaba-NLP/gte-modernbert-base /opt/models/hf/onnx/gte-modernbert-base

git -C /opt/models/hf/onnx/gte-reranker-modernbert-base pull 2>/dev/null || \
  git clone https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base /opt/models/hf/onnx/gte-reranker-modernbert-base

# 4. For each model, keep ONLY the INT8 ONNX file and rename to model.onnx
for MODEL in gte-modernbert-base gte-reranker-modernbert-base; do
  DIR="/opt/models/hf/onnx/${MODEL}/onnx"
  if [[ ! -d "$DIR" ]]; then
    DIR="/opt/models/hf/onnx/${MODEL}"
  fi

  echo "Processing $MODEL in $DIR"

  # Find the int8 file
  INT8=$(find "$DIR" -type f -iname "*int8*.onnx" | head -n1)
  if [[ -z "$INT8" ]]; then
    echo "❌ No INT8 ONNX found in $DIR"
    exit 1
  fi

  # Remove all other .onnx files
  find "$DIR" -type f -name "*.onnx" ! -path "$INT8" -delete

  # Rename the int8 file to model.onnx
  mv -v "$INT8" "$DIR/model.onnx"
done

echo "✅ INT8 models ready at:"
echo "   /opt/models/hf/onnx/gte-modernbert-base/onnx/model.onnx"
echo "   /opt/models/hf/onnx/gte-reranker-modernbert-base/onnx/model.onnx"

# 5. Upload the INT8 models to your Hugging Face account as repos ending with -onnx-int8
python3 - << 'EOF'
#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, whoami

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_HOME = Path(os.getenv("MODEL_HOME", "/opt/models"))
HF_HOME = MODEL_HOME / "hf"
ONNX_ROOT = HF_HOME / "onnx"

# Model repo suffixes (with -onnx-int8)
EMBED_REPO_SUFFIX  = "gte-modernbert-base-onnx-int8"
RERANK_REPO_SUFFIX = "gte-reranker-modernbert-base-onnx-int8"

def find_model_file(model_name: str) -> Path:
    base_dir = ONNX_ROOT / model_name
    for path in base_dir.rglob("model.onnx"):
        return path
    raise FileNotFoundError(f"model.onnx not found under {base_dir}")

def get_username() -> str:
    info = whoami()
    name = info.get("name")
    if not name:
        raise RuntimeError("Could not determine HF username. Please run `huggingface-cli login`.")
    return name

def upload_onnx(model_name: str, repo_suffix: str):
    file_path = find_model_file(model_name)
    username = get_username()
    repo_id = f"{username}/{repo_suffix}"
    create_repo(repo_id, repo_type="model", exist_ok=True)
    upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo="model.onnx",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add INT8 ONNX model"
    )
    logger.info(f"✅ Uploaded {file_path} to {repo_id}/model.onnx")

if __name__ == "__main__":
    try:
        upload_onnx("gte-modernbert-base", EMBED_REPO_SUFFIX)
        upload_onnx("gte-reranker-modernbert-base", RERANK_REPO_SUFFIX)
    except Exception as e:
        logger.error(e)
        exit(1)
EOF
