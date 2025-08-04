#!/usr/bin/env python3
import os
import subprocess
import venv
import time
import shutil
import sys

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VENV_DIR     = os.path.join(PROJECT_ROOT, ".venv_refined")
CLONE_DIR    = os.path.join(PROJECT_ROOT, "refined_repo")
BENCH_SCRIPT = os.path.join(PROJECT_ROOT, "benchmark_refined.py")
SELF         = os.path.abspath(__file__)

PYTHON = os.path.join(VENV_DIR, "bin", "python")
PIP    = os.path.join(VENV_DIR, "bin", "pip")

# ── Helper ────────────────────────────────────────────────────
def run(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

# ── 1) Virtualenv & base deps ─────────────────────────────────
def setup_env():
    if os.path.isdir(VENV_DIR):
        print(f"[INFO] Reusing existing venv at {VENV_DIR}; reinstalling torch…")
        run([PIP, "uninstall", "-y", "torch"])
    else:
        print(f"[INFO] Creating venv at {VENV_DIR}…")
        venv.create(VENV_DIR, with_pip=True)

    print("[INFO] Upgrading pip…")
    run([PIP, "install", "--upgrade", "pip"])

    print("[INFO] Installing CPU-only torch 2.7.1…")
    run([
        PIP, "install", "--force-reinstall",
        "torch==2.7.1+cpu",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

    print("[INFO] Installing ReFinED runtime dependencies…")
    run([
        PIP, "install", "--no-deps",
        # Core
        "ujson",
        "transformers>=4.5.0",
        "numpy",
        "lmdb",
        "nltk",
        "regex",
        "huggingface-hub",
        "typing-extensions>=4.10.0",
        # ReFinED internals
        "Cython",
        "Unidecode>=1.1.1",
        "boto3",
        "botocore",
        "tensorboard",
        "prettyprint",
        # Transitive
        "jmespath",
        "s3transfer",
        "python-dateutil",
        "protobuf<4",
        "absl-py",
        "werkzeug",
        "grpcio",
        "six",
        "markdown",
        "setuptools",
    ])

# ── 2) Clone & install ReFinED ────────────────────────────────
def install_refined():
    if not os.path.isdir(CLONE_DIR):
        print("[INFO] Cloning ReFinED repo…")
        run(["git", "clone", "https://github.com/amazon-science/Refined.git", CLONE_DIR])
    print("[INFO] Installing ReFinED (editable, no-deps)…")
    run([PIP, "install", "-e", ".", "--no-deps"], cwd=CLONE_DIR)

# ── 3) Write benchmark script ─────────────────────────────────
def write_benchmark_script():
    print(f"[INFO] Writing benchmark script to {BENCH_SCRIPT}…")
    code = '''\
import time
import torch
from refined.inference.processor import Refined

# force single-threaded CPU for consistency
torch.set_num_threads(1)

def benchmark(texts, runs=5, warmup=2):
    model = Refined.from_pretrained("wikipedia_model_with_numbers", entity_set="wikipedia")
    print("[INFO] Model loaded.")

    # Warm-up
    for _ in range(warmup):
        _ = model.process_batch(texts)

    # Timed runs
    times = []
    for i in range(runs):
        start = time.time()
        _ = model.process_batch(texts)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f}s")

    avg = sum(times) / runs
    print(f"\\n[RESULT] Avg per batch: {avg:.4f}s")
    print(f"[RESULT] Avg per doc: {avg / len(texts):.4f}s")

if __name__ == "__main__":
    docs = ["Barack Obama was born in Hawaii."] * 4
    benchmark(docs)
'''
    with open(BENCH_SCRIPT, "w") as f:
        f.write(code)

# ── 4) Execute benchmark ───────────────────────────────────────
def run_benchmark():
    print("[INFO] Running benchmark…")
    run([PYTHON, BENCH_SCRIPT])

# ── 5) Cleanup ────────────────────────────────────────────────
def cleanup():
    print("[INFO] Waiting 10 seconds before cleanup…")
    time.sleep(10)
    for path in (VENV_DIR, CLONE_DIR, BENCH_SCRIPT, SELF):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
            print(f"[CLEAN] Removed: {path}")
        except Exception as e:
            print(f"[WARN] Could not remove {path}: {e}")

# ── Main Entry ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        setup_env()
        install_refined()
        write_benchmark_script()
        run_benchmark()
        cleanup()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}", file=sys.stderr)
        sys.exit(1)
