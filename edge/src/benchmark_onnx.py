"""Benchmark ONNX model: measure size, latency, and throughput."""

import os
import sys
import time
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'
REPORTS_DIR = PROJECT_ROOT / 'reports'


def get_model_size(model_path: str) -> Dict[str, float]:
    size_bytes = os.path.getsize(model_path)
    external_data_path = model_path + '.data'
    if os.path.exists(external_data_path):
        size_bytes += os.path.getsize(external_data_path)
    return {
        'size_bytes': float(size_bytes),
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / (1024 * 1024)
    }


def measure_latency(session: ort.InferenceSession, input_data: np.ndarray,
                     num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
    input_name = session.get_inputs()[0].name

    for _ in range(num_warmup):
        session.run(None, {input_name: input_data})

    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        session.run(None, {input_name: input_data})
        latencies.append((time.perf_counter() - start_time) * 1000)  # to ms

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99))
    }


def measure_throughput(session: ort.InferenceSession, input_data: np.ndarray,
                        duration_seconds: float = 5.0) -> float:
    input_name = session.get_inputs()[0].name

    start_time = time.perf_counter()
    count = 0
    while (time.perf_counter() - start_time) < duration_seconds:
        session.run(None, {input_name: input_data})
        count += 1

    elapsed = time.perf_counter() - start_time
    return count / elapsed


def benchmark_model(model_path: str, config_path=None) -> Dict:
    config = load_config(config_path or DEFAULT_CONFIG_PATH)
    bench_cfg = config.get('benchmark', {})

    size_info = get_model_size(model_path)

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_shape = session.get_inputs()[0].shape
    shape = [1 if type(dim) == str else dim for dim in input_shape]
    input_data = np.random.randn(*shape).astype(np.float32)

    num_warmup = bench_cfg.get('num_warmup_runs', 10)
    num_runs = bench_cfg.get('num_benchmark_runs', 100)

    latency_info = measure_latency(session, input_data, num_warmup, num_runs)
    throughput = measure_throughput(session, input_data)

    metadata = {
        'input_name': session.get_inputs()[0].name,
        'input_shape': shape,
        'output_name': session.get_outputs()[0].name,
        'output_shape': [1 if type(dim) == str else dim for dim in session.get_outputs()[0].shape]
    }

    results = {
        'model_path': model_path,
        'size': size_info,
        'latency': latency_info,
        'throughput': throughput,
        'metadata': metadata,
        'ort_version': ort.__version__,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print("\n" + "=" * 50)
    print("ONNX Benchmark Report")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Size:  {size_info['size_mb']:.2f} MB")
    print(f"Shape: Input {metadata['input_shape']} -> Output {metadata['output_shape']}")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Latency:")
    print(f"  Mean:   {latency_info['mean_ms']:.2f} ms")
    print(f"  Std:    {latency_info['std_ms']:.2f} ms")
    print(f"  Min:    {latency_info['min_ms']:.2f} ms")
    print(f"  Max:    {latency_info['max_ms']:.2f} ms")
    print(f"  P95:    {latency_info['p95_ms']:.2f} ms")
    print(f"  P99:    {latency_info['p99_ms']:.2f} ms")
    print("=" * 50 + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / 'benchmark_fp32.md'
    with open(report_path, 'w') as f:
        f.write("# ONNX Benchmark Report\n\n")
        f.write(f"**Date:** {results['timestamp']}  \n")
        f.write(f"**ONNX Runtime Version:** {results['ort_version']}  \n")
        f.write(f"**Model Path:** `{model_path}`  \n\n")

        f.write("## Model Info\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Size | {size_info['size_mb']:.2f} MB ({size_info['size_kb']:.2f} KB) |\n")
        f.write(f"| Input Shape | `{metadata['input_shape']}` |\n")
        f.write(f"| Output Shape | `{metadata['output_shape']}` |\n\n")

        f.write("## Performance\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Throughput | {throughput:.2f} samples/sec |\n")
        f.write(f"| Latency (Mean) | {latency_info['mean_ms']:.2f} ms |\n")
        f.write(f"| Latency (Std) | {latency_info['std_ms']:.2f} ms |\n")
        f.write(f"| Latency (Min) | {latency_info['min_ms']:.2f} ms |\n")
        f.write(f"| Latency (Max) | {latency_info['max_ms']:.2f} ms |\n")
        f.write(f"| Latency (p95) | {latency_info['p95_ms']:.2f} ms |\n")
        f.write(f"| Latency (p99) | {latency_info['p99_ms']:.2f} ms |\n")

    print(f"Saved markdown report to {report_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .onnx file")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()
    benchmark_model(args.model_path, args.config)