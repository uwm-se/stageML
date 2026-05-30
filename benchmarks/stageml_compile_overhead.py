from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from benchmarks.lora_baselines_bench import LoraLibStyleLinear
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(device, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--compile-repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", default="out/compile_overhead.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    model = LoraLibStyleLinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)

    trace_times = []
    rewrite_times = []
    specialize_times = []
    total_times = []
    rewrite_counts = []
    before_ops = []
    after_ops = []

    for _ in range(args.compile_repeats):
        total_start = time.perf_counter()
        (gm, annotations), trace_ms = timed(device, lambda: trace_and_annotate(model, {"x": "stage1"}))
        before_ops.append(count_compute_ops(gm))
        (rewrite_result, rewrite_ms) = timed(device, lambda: optimize_evaluation_order(gm, annotations))
        gm2, annotations2, stats = rewrite_result
        rewrite_counts.append(stats.total_rewrites)
        residual, specialize_ms = timed(device, lambda: specialize(gm2, annotations2))
        after_ops.append(count_compute_ops(residual))
        sync(device)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        trace_times.append(trace_ms)
        rewrite_times.append(rewrite_ms)
        specialize_times.append(specialize_ms)
        total_times.append(total_ms)

    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    gm, annotations, stats = optimize_evaluation_order(gm, annotations)
    residual = specialize(gm, annotations).to(device).eval()

    with torch.no_grad():
        max_diff = (model(x) - residual(x)).abs().max().item()

    eager_ms = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
    stageml_ms = benchmark_latency_ms(residual, x, warmup=args.warmup, iterations=args.iterations)
    saved_ms = eager_ms - stageml_ms
    payback = float("inf") if saved_ms <= 0 else median(total_times) / saved_ms

    row = {
        "benchmark": "stageml_compile_overhead_loralib_style_lora",
        "device": str(device),
        "dtype": str(dtype),
        "dim": args.dim,
        "rank": args.rank,
        "batch": args.batch,
        "compile_repeats": args.compile_repeats,
        "trace_ms_median": median(trace_times),
        "rewrite_ms_median": median(rewrite_times),
        "specialize_ms_median": median(specialize_times),
        "total_compile_ms_median": median(total_times),
        "eager_ms": eager_ms,
        "stageml_ms": stageml_ms,
        "saved_ms_per_call": saved_ms,
        "payback_inferences": payback,
        "compute_ops_before_median": median(before_ops),
        "compute_ops_after_median": median(after_ops),
        "rewrite_count_median": median(rewrite_counts),
        "max_diff": max_diff,
    }
    write_csv(args.out, [row])
    print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
