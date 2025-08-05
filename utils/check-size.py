from relik import Relik
import os
import psutil
import time

# Load model to trigger weight download
model = Relik.from_pretrained("relik-ie/relik-cie-tiny")

# Path where HuggingFace stores the model (after from_pretrained)
model_dir = model.model_path if hasattr(model, "model_path") else model.config_dir

# Sum all files under model_dir
total_size = 0
for dirpath, _, filenames in os.walk(model_dir):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        total_size += os.path.getsize(fp)

print("Total downloaded model size:", round(total_size / (1024 * 1024), 2), "MiB")

# Optional: RAM usage after loading
process = psutil.Process()
print("RAM used:", round(process.memory_info().rss / (1024 * 1024), 2), "MiB")


