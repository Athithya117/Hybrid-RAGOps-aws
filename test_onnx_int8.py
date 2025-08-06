import onnxruntime
import numpy as np
from transformers import AutoTokenizer
import time

def test_model(model_path: str, tokenizer_repo: str, text: str):
    print(f"\nTesting {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True)

    # Tokenize
    t0 = time.time()
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    t1 = time.time()
    print(f"Tokenization time: {t1 - t0:.4f} sec")

    # Inference
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]
    t2 = time.time()
    outputs = session.run(output_names, inputs)  # two args: list[str], dict[str, array]
    t3 = time.time()
    print(f"Inference time: {t3 - t2:.4f} sec")

    array = outputs[0]
    print(f"Output shape: {array.shape}")
    print(f"First five values: {array.flatten()[:5].tolist()}")

if __name__ == "__main__":
    paragraph = (
        "ONNX Runtime is a high-performance inference engine for machine learning models. "
    ) * 4

    test_model(
        "/opt/models/hf/onnx/gte-modernbert-base-onnx-int8/onnx/model_int8.onnx",
        "RAG8s/gte-modernbert-base-onnx-int8",
        paragraph
    )
    test_model(
        "/opt/models/hf/onnx/gte-reranker-modernbert-base-onnx-int8/onnx/model_int8.onnx",
        "RAG8s/gte-reranker-modernbert-base-onnx-int8",
        paragraph
    )

