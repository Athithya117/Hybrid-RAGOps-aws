# pip install onnx
# python3 utils/archive/test_onnx_arm_support.py models/onnx/gte-modernbert-base-onnx-int8/model.onnx
# python3 utils/archive/test_onnx_arm_support.py models/onnx/ms-marco-TinyBERT-L2-v2/onnx/model_qint8_arm64.onnx


import sys
from collections import Counter, defaultdict
from pathlib import Path
import onnx
from onnx import numpy_helper

QUANT_OPS = {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}
INT_MATMUL_OPS = {"MatMulInteger", "QLinearMatMul"}
STANDARD_DOMAINS = {"", "ai.onnx"}

def inspect_model(path: Path) -> dict:
    model = onnx.load_model(str(path))
    node_types = Counter()
    node_domains = Counter()
    domain_nodes = defaultdict(list)
    initializers = {init.name: init for init in model.graph.initializer}
    tensor_value_info = {vi.name: vi for vi in model.graph.value_info}
    for node in model.graph.node:
        node_types[node.op_type] += 1
        domain = node.domain or ""
        node_domains[domain] += 1
        domain_nodes[domain].append(node.op_type)
    quant_present = {op: node_types.get(op, 0) for op in QUANT_OPS}
    int_matmul_present = {op: node_types.get(op, 0) for op in INT_MATMUL_OPS}
    nonstandard_domains = {d: cnt for d, cnt in node_domains.items() if d not in STANDARD_DOMAINS}
    weights_info = []
    for name, tensor in initializers.items():
        arr = numpy_helper.to_array(tensor)
        weights_info.append((name, arr.dtype, arr.shape))
    return {
        "op_counts": dict(node_types),
        "domain_counts": dict(node_domains),
        "nonstandard_domains": nonstandard_domains,
        "quant_ops": quant_present,
        "int_matmul_ops": int_matmul_present,
        "weights": weights_info,
        "producer": getattr(model, "producer_name", None),
        "ir_version": getattr(model, "ir_version", None),
        "opset_imports": [(op.domain or "", op.version) for op in model.opset_import],
    }

def pretty_print(report: dict, path: Path) -> None:
    print(f"INSPECTION: {path}")
    print("producer:", report.get("producer"))
    print("ir_version:", report.get("ir_version"))
    print("opset imports:", report.get("opset_imports"))
    print("\nTop op types (sample):")
    ops = sorted(report["op_counts"].items(), key=lambda x: -x[1])[:30]
    for op, cnt in ops:
        print(f"  {op}: {cnt}")
    print("\nQuantization-related ops:")
    for op, cnt in report["quant_ops"].items():
        print(f"  {op}: {cnt}")
    print("\nInteger-matmul-related ops:")
    for op, cnt in report["int_matmul_ops"].items():
        print(f"  {op}: {cnt}")
    print("\nDomains present:")
    for d, cnt in report["domain_counts"].items():
        label = d if d else "<empty>"
        print(f"  {label}: {cnt}")
    if report["nonstandard_domains"]:
        print("\nNon-standard/custom domains detected (may reduce portability):")
        for d, cnt in report["nonstandard_domains"].items():
            print(f"  {d}: {cnt}")
            sample = report['op_counts']
    else:
        print("\nNo non-standard domains detected (good).")
    print("\nWeights summary (first 10):")
    for w in report["weights"][:10]:
        print(f"  name={w[0]} dtype={w[1]} shape={w[2]}")
    print("\nHeuristic verdict:")
    if any(report["quant_ops"].values()):
        print("  Model contains quantization ops.")
    else:
        print("  No quantization ops found (may be float model or post-export quantized differently).")
    if report["nonstandard_domains"]:
        print("  WARNING: custom/nonstandard ops present; portability to ARM runtimes may require custom kernels or reprofiling.")
    else:
        print("  OK: uses standard ops/domains; portable to ONNXRuntime on ARM in principle. Performance depends on EP/kernels.")
    print("\nDone.")

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_onnx_ops.py path/to/model.onnx", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print("Model file not found:", path, file=sys.stderr)
        sys.exit(2)
    report = inspect_model(path)
    pretty_print(report, path)

if __name__ == "__main__":
    main()

