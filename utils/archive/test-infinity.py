from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = ORTModelForFeatureExtraction.from_pretrained(
    "RAG8s/gte-modernbert-base-onnx-int8",
    file_name="onnx/model.onnx"   # explicitly point to your int8
)
tok = AutoTokenizer.from_pretrained("RAG8s/gte-modernbert-base-onnx-int8")

inputs = tok("test embedding sentence", return_tensors="pt")
outputs = model(**inputs)
print("Embedding shape:", outputs.last_hidden_state.shape)
