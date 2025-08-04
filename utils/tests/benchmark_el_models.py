import time
import torch
import random

from refined.inference.processor import Refined
from relik import Relik

# Sample input
sentences = [
    "Barack Obama was born in Hawaii.",
    "Apple Inc. is based in Cupertino, California.",
    "The Eiffel Tower is located in Paris.",
    "Albert Einstein developed the theory of relativity.",
    "Amazon was founded by Jeff Bezos."
]

# Generate a batch of N duplicated inputs
def generate_batch(texts, batch_size):
    return [random.choice(texts) for _ in range(batch_size)]

def benchmark_model(name, model_fn, batch_size=8, warmup=1, runs=10):
    inputs = generate_batch(sentences, batch_size)

    # Warmup
    for _ in range(warmup):
        model_fn(inputs)

    start = time.perf_counter()
    for _ in range(runs):
        model_fn(inputs)
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / runs
    throughput = batch_size / avg_time

    print(f"\n[RESULT] {name}")
    print(f"  Avg latency per batch: {avg_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")


# ReFinED wrapper
def run_refined(texts):
    global refined_model
    refined_model.process_text_batch(texts)

# ReLiK wrapper
def run_relik(texts):
    global relik_model
    relik_model.annotate(texts)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load models
    refined_model = Refined.from_pretrained(device=device)
    relik_model = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")

    # Benchmark settings
    batch_size = 8
    warmup = 1
    runs = 10

    # Run benchmarks
    benchmark_model("ReFinED", run_refined, batch_size, warmup, runs)
    benchmark_model("ReLiK", run_relik, batch_size, warmup, runs)
