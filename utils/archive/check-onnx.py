import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

def test_onnx_model(model_path: str, tokenizer_path: str, input_text: str | list):
    print(f"\nTesting ONNX model at: {model_path}")

    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer(input_text, return_tensors="np", padding=True, truncation=True)

    # Load ONNX model
    session = InferenceSession(model_path)
    input_names = {inp.name for inp in session.get_inputs()}
    ort_inputs = {k: v for k, v in inputs.items() if k in input_names}

    # Inference
    outputs = session.run(None, ort_inputs)

    # Output info
    for idx, out in enumerate(outputs):
        print(f"Output {idx}: shape={out.shape}, dtype={out.dtype}")
    print("✓ Model executed successfully.")

# Fixed local paths
EMBEDDER_MODEL_PATH = "/opt/models/hf/onnx/gte-modernbert-base/onnx/model.onnx"
RERANKER_MODEL_PATH = "/opt/models/hf/onnx/gte-reranker-modernbert-base/model.onnx"

EMBEDDER_TOKENIZER_PATH = "/opt/models/hf/onnx/gte-modernbert-base"
RERANKER_TOKENIZER_PATH = "/opt/models/hf/onnx/gte-reranker-modernbert-base"

test_text = "The capital of France is Paris."

test_onnx_model(EMBEDDER_MODEL_PATH, EMBEDDER_TOKENIZER_PATH, test_text)
test_onnx_model(RERANKER_MODEL_PATH, RERANKER_TOKENIZER_PATH, [test_text, test_text])
