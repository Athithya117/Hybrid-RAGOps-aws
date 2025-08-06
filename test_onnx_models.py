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
