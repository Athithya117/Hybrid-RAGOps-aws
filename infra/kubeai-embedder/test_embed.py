#!/usr/bin/env python3
import os
import sys
import json
import time
from typing import List
import requests

BASE_URL = os.environ.get("KUBEAI_BASE", "http://localhost:8081/openai/v1")
API_KEY = os.environ.get("KUBEAI_API_KEY", "")
TIMEOUT = 60.0

def embed(model: str, inputs: List[str]) -> List[List[float]]:
    url = f"{BASE_URL}/embeddings"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {"model": model, "input": inputs}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return [item["embedding"] for item in j["data"]]

def main(argv):
    if len(argv) < 3:
        print("usage: test_embed.py <model-name> <text> [<text> ...]", file=sys.stderr)
        sys.exit(2)
    model = argv[1]
    inputs = argv[2:]
    t0 = time.time()
    vecs = embed(model, inputs)
    dt = time.time() - t0
    print("model", model, "count", len(vecs), "dim", len(vecs[0]) if vecs else 0, "time_ms", int(dt*1000))
    print("first_vector_slice", vecs[0][:8])
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)
