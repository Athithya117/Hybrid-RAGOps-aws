#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/opt/models/hf/hub
export TRANSFORMERS_CACHE=/opt/models/hf/hub
export HUGGINGFACE_HUB_CACHE=/opt/models/hf/hub

SRC_DIR="/opt/models/hf/hub/models--relik-ie--relik-cie-small/snapshots/aefb0ec0a58c493a9d981d47eec4a10341ff15b3"
DST_DIR="/opt/models/hf/onnx/relik-cie-small"

mkdir -p "$DST_DIR"

pip install --no-cache-dir \
    transformers==4.53.3 \
    onnxruntime==1.22.1 \
    huggingface_hub==0.34.3 \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

python3 <<'EOF'
import os, shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

src = Path("/opt/models/hf/hub/models--relik-ie--relik-cie-small/snapshots/aefb0ec0a58c493a9d981d47eec4a10341ff15b3")
dst = Path("/opt/models/hf/onnx/relik-cie-small")
dst.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer + model from:", src)
tokenizer = AutoTokenizer.from_pretrained(src.as_posix())
model = AutoModel.from_pretrained(src.as_posix())

# Export to ONNX
onnx_path = dst / "model.onnx"
dummy = tokenizer("Relik is extracting triplets and entities.", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    onnx_path.as_posix(),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                  "attention_mask": {0: "batch", 1: "seq"},
                  "last_hidden_state": {0: "batch", 1: "seq"}},
    opset_version=17
)
print(f" Exported {onnx_path}")

# Copy tokenizer/config files
keep = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt", "vocab.json",
    "merges.txt",
    "added_tokens.json",
]
for f in keep:
    src_file = src / f
    if src_file.exists():
        shutil.copy2(src_file, dst / f)
        print(f"  copied {src_file} -> {dst}")
EOF
echo " All files are in $DST_DIR:"