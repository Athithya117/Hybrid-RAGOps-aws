#!/usr/bin/env bash
set -euo pipefail

# Map local directory names to HF repo suffixes
declare -A MODELS=(
  ["gte-modernbert-base-onnx-int8"]="gte-modernbert-base-onnx-int8"
  ["gte-reranker-modernbert-base-onnx-int8"]="gte-reranker-modernbert-base-onnx-int8"
)
BASE_DIR="/opt/models/hf/onnx"

# Ensure git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    sudo apt-get update && sudo apt-get install -y git-lfs
fi
git lfs install

# Install Python dependencies
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

# Clone or update each ONNX-int8 repo
for DIR_NAME in "${!MODELS[@]}"; do
    REPO_SUFFIX=${MODELS[$DIR_NAME]}
    REPO_URL="https://huggingface.co/RAG8s/${REPO_SUFFIX}"
    MODEL_DIR="${BASE_DIR}/${DIR_NAME}"

    echo "Preparing ${DIR_NAME} in ${MODEL_DIR}..."
    sudo mkdir -p "$MODEL_DIR"
    sudo chown -R $USER:$USER "$MODEL_DIR"

    if [ -d "${MODEL_DIR}/.git" ]; then
        echo "Updating existing repo ${REPO_SUFFIX}..."
        git -C "$MODEL_DIR" pull
    else
        echo "Cloning ${REPO_SUFFIX}..."
        git clone "$REPO_URL" "$MODEL_DIR"
    fi
    echo "Pulling LFS files for ${REPO_SUFFIX}..."
    git -C "$MODEL_DIR" lfs pull
done

# Write and run test script
cat << 'EOF' > test_onnx_models.py
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
import time

def test_model(model_path, tokenizer_name):
    print(f"\nTesting: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sample = "ONNX Runtime is a high-performance inference engine." * 10

    t0 = time.time()
    inputs = tokenizer(sample, return_tensors="np", padding=True, truncation=True)
    t1 = time.time()
    print(f"Tokenization: {t1-t0:.4f}s")

    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    t2 = time.time()
    out = sess.run(None, ort_inputs)
    t3 = time.time()
    print(f"Inference: {t3-t2:.4f}s, Output shape: {out[0].shape}")

if __name__ == "__main__":
    base = "/opt/models/hf/onnx"
    test_model(f"{base}/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx",
               "RAG8s/gte-modernbert-base-onnx-int8")
    test_model(f"{base}/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx",
               "RAG8s/gte-reranker-modernbert-base-onnx-int8")
EOF

python3 test_onnx_models.py
