from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.annotations import stage0
from stageml.evaluator import specialize
from stageml.tracer import trace_and_annotate


class LoRALinear(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, rank=16, alpha=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        merged = self.W + (self.B @ self.A) * self.scaling
        return x @ merged.t()


class MoERouterOnly(nn.Module):
    def __init__(self, dim=4096, num_experts=8):
        super().__init__()
        self.router_weight = nn.Parameter(torch.randn(num_experts, dim) * 0.02)
        self.router_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        w_norm = F.normalize(self.router_weight, dim=-1)
        return F.softmax(x @ w_norm.t() + self.router_bias, dim=-1)


def run_one(name: str, model: nn.Module, x: torch.Tensor, warmup: int, iterations: int, include_torch_compile: bool) -> dict:
    model.eval()
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    compute_before = count_compute_ops(gm)
    static_compute = sum(1 for n in gm.graph.nodes if annotations.get(n) == stage0 and n.op in {"call_function", "call_method", "call_module"})
    residual = specialize(gm, annotations).to(x.device).eval()
    compute_after = count_compute_ops(residual)

    with torch.no_grad():
        y0 = model(x)
        y1 = residual(x)
    max_diff = (y0 - y1).abs().max().item()

    eager_ms = benchmark_latency_ms(model, x, warmup=warmup, iterations=iterations)
    residual_ms = benchmark_latency_ms(residual, x, warmup=warmup, iterations=iterations)

    compiled_ms = float("nan")
    compile_ok = False
    if include_torch_compile:
        try:
            compiled = torch.compile(model)
            compiled(x)
            compiled_ms = benchmark_latency_ms(compiled, x, warmup=warmup, iterations=iterations)
            compile_ok = True
        except Exception:
            compiled_ms = float("nan")

    return {
        "benchmark": name,
        "device": str(x.device),
        "dtype": str(x.dtype),
        "compute_before": compute_before,
        "compute_after": compute_after,
        "static_compute_ops": static_compute,
        "max_diff": max_diff,
        "eager_ms": eager_ms,
        "stageml_ms": residual_ms,
        "torch_compile_ms": compiled_ms,
        "torch_compile_ok": compile_ok,
        "speedup_vs_eager": eager_ms / residual_ms,
        "speedup_vs_torch_compile": compiled_ms / residual_ms if compiled_ms and compiled_ms == compiled_ms else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", default="out/gpu_lora_moe_results.csv")
    parser.add_argument("--include-torch-compile", action="store_true")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)
    rows = []

    lora_model = LoRALinear(args.dim, args.dim, args.rank).to(device=device, dtype=dtype)
    rows.append(run_one("lora_merge", lora_model, x, args.warmup, args.iterations, args.include_torch_compile))

    moe_model = MoERouterOnly(args.dim, num_experts=8).to(device=device, dtype=dtype)
    rows.append(run_one("moe_router_norm", moe_model, x, args.warmup, args.iterations, args.include_torch_compile))

    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
