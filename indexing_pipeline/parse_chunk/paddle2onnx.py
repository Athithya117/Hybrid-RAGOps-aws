import paddle
from paddleocr import PaddleOCR
from paddle2onnx import export  # built-in conversion
import os

# --- SETTINGS ---
lang = "ta"  # change this to desired language code
save_dir = f"./onnx_export/{lang}"
os.makedirs(save_dir, exist_ok=True)

# Load PaddleOCR recognition model
ocr = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=False, det=False, rec=True)

# Get model paths
rec_model = ocr.rec_model  # Contains model + params

# Paths
model_file = rec_model.predictor.model_file
params_file = rec_model.predictor.params_file
save_path = os.path.join(save_dir, f"{lang}_rec.onnx")

# Convert to ONNX
export(
    model_file=model_file,
    params_file=params_file,
    save_file=save_path,
    opset_version=11,  # Paddle default
    enable_onnx_checker=True
)

print(f"ONNX model saved to: {save_path}")

