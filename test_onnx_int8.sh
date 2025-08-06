#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PYCODE'
import onnxruntime
import numpy as np
from transformers import AutoTokenizer
import time

declare_and_test() {
    local model_path="\$1"
    local tokenizer_repo="\$2"
    local text="\$3"
    printf "\nTesting %s\n" "\$model_path"
    local tok=\$(python3 - << 'CODE'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$tokenizer_repo", use_fast=True)
print(tok)  # just to load
CODE
)
    t0=\$(date +%s.%N)
    inputs=\$(python3 - << 'CODE'
from transformers import AutoTokenizer
import numpy as np
tok = AutoTokenizer.from_pretrained("$tokenizer_repo", use_fast=True)
inputs = tok("$text", return_tensors="np", padding=True, truncation=True)
print(inputs)
CODE
)
    t1=\$(date +%s.%N)
    echo "Tokenization time: \$(echo "\$t1-\$t0" | bc) sec"

    session=onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    t2=\$(date +%s.%N)
    outputs=session.run(None, inputs)
    t3=\$(date +%s.%N)
    echo "Inference time: \$(echo "\$t3-\$t2" | bc) sec"
    echo "Output shape: ", outputs[0].shape
    echo "First five dims: ", outputs[0][0][:5]
}

# Simulate long text
PARA="ONNX Runtime is a high-performance inference engine for ML models. "
PARA="\${PARA} \${PARA} \${PARA} \${PARA}"  # repeat

declare_and_test "/opt/models/hf/onnx/gte-modernbert-base-onnx-int8/model_int8.onnx" "RAG8s/gte-modernbert-base-onnx-int8" "\$PARA"
declare_and_test "/opt/models/hf/onnx/gte-reranker-modernbert-base-onnx-int8/model_int8.onnx" "RAG8s/gte-reranker-modernbert-base-onnx-int8" "\$PARA"
PYCODE
