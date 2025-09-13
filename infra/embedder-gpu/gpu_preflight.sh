#!/usr/bin/env bash
# gpu_preflight.sh
# Run on the GPU VM to quickly validate drivers + container runtime + basic onnx GPU provider availability.
set -euo pipefail

echo "1) Host nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "ERROR: nvidia-smi not found on host. Install NVIDIA driver."
  exit 1
fi

echo
echo "2) Docker version:"
docker --version || { echo "ERROR: docker missing"; exit 2; }

echo
echo "3) Run CUDA container to verify GPU access (CUDA 12.2 runtime used as default):"
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi || { echo "ERROR: Docker container cannot access GPU"; exit 3; }

echo
echo "4) Quick onnxruntime-gpu provider test in scratch container (installs wheel). This may take a minute."
docker run --rm --gpus all --entrypoint bash python:3.10-slim -c "\
  python3 -m pip install --no-cache-dir onnxruntime-gpu==1.22.1 >/dev/null 2>&1 || exit 4; \
  python3 - <<'PY'\nimport onnxruntime as ort\nprint('onnx providers:', ort.get_available_providers())\nPY" || { echo 'WARN: onnxruntime-gpu provider check failed'; exit 4; }

echo
echo "Preflight passed (if all above checks succeeded). If any step failed, match driver/CUDA version and NVIDIA container toolkit."
