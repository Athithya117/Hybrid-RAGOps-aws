#!/usr/bin/env bash
set -euo pipefail

# Define models and paths
MODEL_NAMES=("gte-modernbert-base" "gte-reranker-modernbert-base")
BASE_DIR="/opt/models/hf/onnx"

# Ensure git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    sudo apt-get update && sudo apt-get install -y git-lfs
fi

git lfs install

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

# Create directories and clone or update repos
for MODEL in "${MODEL_NAMES[@]}"; do
    MODEL_DIR="${BASE_DIR}/${MODEL}"
    echo "Preparing ${MODEL} in ${MODEL_DIR}..."

    # Ensure base directory exists and ownership is correct
    sudo mkdir -p "$MODEL_DIR"
    sudo chown -R $USER:$USER "$MODEL_DIR"

    if [ -d "${MODEL_DIR}/.git" ]; then
        echo "Repository already exists. Pulling latest changes for ${MODEL}..."
        git -C "$MODEL_DIR" pull
    else
        echo "Cloning ${MODEL} repository..."
        git clone https://huggingface.co/Alibaba-NLP/${MODEL} "$MODEL_DIR"
    fi

    echo "Pulling Git LFS files..."
    git -C "$MODEL_DIR" lfs pull
done
cat << 'EOF' > test_onnx_models.py
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
import time

def test_model(model_path, tokenizer_name, text):
    print(f"\nTesting: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenization timing
    t0 = time.time()
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    t1 = time.time()
    print(f"Tokenization time: {t1 - t0:.4f} sec")

    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    ort_session = onnxruntime.InferenceSession(model_path)

    # Inference timing
    t2 = time.time()
    outputs = ort_session.run(None, ort_inputs)
    t3 = time.time()

    print(f"Inference time: {t3 - t2:.4f} sec")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output (first 5 dims): {outputs[0][0][:5]}")

if __name__ == "__main__":
    # Simulate a long document (500+ tokens)
    paragraph = (
        "ONNX Runtime is a high-performance inference engine for machine learning models. "
        "In this test, we want to observe how ONNX handles long input sequences compared to traditional PyTorch models. "
        "This is particularly useful in large-scale applications such as semantic search, question answering, and reranking tasks. "
        "By testing on a longer paragraph, we can measure improvements in speed and efficiency, which are crucial for real-time systems. "
        "Repeat this paragraph to simulate a larger input. "
    ) * 10  # Repeating to increase length

    test_model("/opt/models/hf/onnx/gte-modernbert-base/onnx/model_int8.onnx", "Alibaba-NLP/gte-modernbert-base", paragraph)
    test_model("/opt/models/hf/onnx/gte-reranker-modernbert-base/onnx/model_int8.onnx", "Alibaba-NLP/gte-reranker-modernbert-base", paragraph)
EOF

# Run the test script
python3 test_onnx_models.py
