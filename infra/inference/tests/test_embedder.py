#!/usr/bin/env python3
"""
Test script for embedding with the gte-modernbert-base-onnx-int8 model in KubeAI.

This version hardcodes the model name, so you can just provide text inputs:
    python3 test_embedder.py "Your text here" "Another text"

Environment variables:
- KUBEAI_BASE: URL of your KubeAI inference endpoint (default: http://localhost:8081/openai/v1)
- KUBEAI_API_KEY: optional API key
- KUBEAI_TIMEOUT: request timeout in seconds (default: 60)
"""

import os
import sys
import json
import time
from typing import List
import requests

# -----------------------------
# Config
# -----------------------------
BASE_URL = os.environ.get("KUBEAI_BASE", "http://localhost:8081/openai/v1")
API_KEY = os.environ.get("KUBEAI_API_KEY", "")
TIMEOUT = float(os.environ.get("KUBEAI_TIMEOUT", "60"))

# Hardcoded model
MODEL_NAME = "gte-modernbert-base-onnx-int8"

# -----------------------------
# Embedding function
# -----------------------------
def embed(inputs: List[str]) -> List[List[float]]:
    """Send inputs to KubeAI embeddings endpoint and return embeddings."""
    url = f"{BASE_URL}/embeddings"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {"model": MODEL_NAME, "input": inputs}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return [item["embedding"] for item in data.get("data", [])]

# -----------------------------
# Main
# -----------------------------
def main(argv: List[str]) -> None:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <text> [<text> ...]", file=sys.stderr)
        sys.exit(2)

    inputs = argv[1:]
    t0 = time.time()
    vecs = embed(inputs)
    dt = time.time() - t0

    print(f"model: {MODEL_NAME}")
    print(f"count: {len(vecs)}")
    print(f"dimension: {len(vecs[0]) if vecs else 0}")
    print(f"time_ms: {int(dt * 1000)}")
    if vecs:
        print(f"first_vector_slice: {vecs[0][:8]}")

if __name__ == "__main__":
    main(sys.argv)
