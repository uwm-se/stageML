from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.evaluator import specialize
from stageml.tracer import trace_and_annotate


class QuantizedWeightRuntime(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        q = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        self.register_buffer("w_int", q)
        self.register_buffer("scale", torch.rand(out_features, 1, dtype=torch.float32) * 0.02 + 0.001)
        self.register_buffer("zero_point", torch.zeros(out_features, 1, dtype=torch.float32))
        self.register_buffer("bias", torch.randn(out_features, dtype=torch.float32) * 0.01)

    def forward(self, x):
        # Dequantization is intentionally written in the forward path so StageML
        # has a non-LoRA static pattern to fold. Cast to x.dtype so fp16/bf16
        # inputs do not fail on CUDA matmul dtype checks.
        w_dequant = ((self.w_int.float() - self.zero_point.float()) * self.scale.float()).to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype)
        return x @ w_dequant.t() + bias


class QuantizedWeightManualMerged(nn.Module):
    def __init__(self, src: QuantizedWeightRuntime):
        super().__init__()
        with torch.no_grad():
            w_dequant = (src.w_int.float() - src.zero_point) * src.scale
        self.register_buffer("w_dequant", w_dequant.detach().clone())
        self.register_buffer("bias", src.bias.detach().clone())

    def forward(self, x):
        return x @ self.w_dequant.to(dtype=x.dtype).t() + self.bias.to(dtype=x.dtype)


def time_specialization(model: nn.Module, x: torch.Tensor):
    start = time.perf_counter()
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    traced = time.perf_counter()
    residual = specialize(gm, annotations)
    specialized = time.perf_counter()
    return residual, (traced - start) * 1000.0, (specialized - traced) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", default="out/research/quantized_weight_bench.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    eager = QuantizedWeightRuntime(args.in_features, args.out_features).to(device).eval()
    manual = QuantizedWeightManualMerged(eager).to(device).eval()
    x = torch.randn(args.batch, args.in_features, device=device, dtype=dtype)
    eager = eager.to(dtype=dtype)
    manual = manual.to(dtype=dtype)

    residual, trace_ms, specialize_ms = time_specialization(eager, x)
    residual = residual.to(device=device, dtype=dtype).eval()

    rows = []
    variants = [("eager_runtime_dequant", eager), ("manual_predequant", manual), ("stageml_static_dequant", residual)]
    times = {}
    for name, model in variants:
        latency = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
        with torch.no_grad():
            diff = (model(x) - eager(x)).abs().max().item()
        times[name] = latency
        rows.append({
            "variant": name,
            "device": str(device),
            "dtype": str(dtype),
            "in_features": args.in_features,
            "out_features": args.out_features,
            "batch": args.batch,
            "latency_ms": latency,
            "speedup_vs_eager": None,
            "compute_ops": count_compute_ops(model) if hasattr(model, "graph") else "module",
            "trace_ms": trace_ms if name == "stageml_static_dequant" else "",
            "specialize_ms": specialize_ms if name == "stageml_static_dequant" else "",
            "max_diff_vs_eager": diff,
        })
    for row in rows:
        row["speedup_vs_eager"] = times["eager_runtime_dequant"] / row["latency_ms"]
    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
