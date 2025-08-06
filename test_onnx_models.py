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
