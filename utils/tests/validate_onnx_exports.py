import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = os.path.join(os.getcwd(), "models")

MODELS = [
    "gte-modernbert-base",
    "gte-reranker-modernbert-base",
    sapienzanlp/relik-relation-extraction-nyt-large
]

def mean_pool(last_hidden_state, attention_mask):
    """Applies mean pooling on token embeddings using attention mask."""
    mask = attention_mask.astype(np.float32)
    masked_hidden = last_hidden_state * np.expand_dims(mask, axis=-1)
    sum_hidden = masked_hidden.sum(axis=1)
    lengths = mask.sum(axis=1, keepdims=True)
    return sum_hidden / lengths

def validate_onnx_model(model_name):
    model_dir = os.path.join(BASE_DIR, model_name)
    model_path = os.path.join(model_dir, "model.onnx")
    tokenizer_path = model_dir

    print(f"\n[INFO] Validating: {model_name}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] ONNX model not found at: {model_path}")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Prepare dummy sentence pair input
    inputs = tokenizer(
        "What is the capital of France?",
        "Paris is the capital city of France.",
        return_tensors="np",
        padding=True,
        truncation=True
    )

    # Load ONNX model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    ort_inputs = {name: inputs[name] for name in input_names}

    # Run inference
    outputs = session.run(None, ort_inputs)
    token_embeddings = outputs[0]  # (batch_size, seq_len, hidden_size)

    # Apply mean pooling
    pooled = mean_pool(token_embeddings, inputs['attention_mask'])

    print(f"[SUCCESS] Pooled output shape: {pooled.shape}")
    print(f"[CHECK] First 5 values: {pooled[0][:5]}")

if __name__ == "__main__":
    for model in MODELS:
        validate_onnx_model(model)
