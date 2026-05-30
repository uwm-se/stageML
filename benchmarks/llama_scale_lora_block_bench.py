from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.annotations import stage0
from stageml.evaluator import specialize
from stageml.tracer import trace_and_annotate


class LlamaScaleLoRAProjection(nn.Module):
    def __init__(self, hidden_size=4096, rank=16, alpha=16.0):
        super().__init__()
        self.q_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.q_lora_a = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
        self.q_lora_b = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
        self.k_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.k_lora_a = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
        self.k_lora_b = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
        self.v_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.v_lora_a = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
        self.v_lora_b = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
        self.scaling = alpha / rank

    def _merge(self, w, a, b):
        return w + (b @ a) * self.scaling

    def forward(self, x):
        q_w = self._merge(self.q_weight, self.q_lora_a, self.q_lora_b)
        k_w = self._merge(self.k_weight, self.k_lora_a, self.k_lora_b)
        v_w = self._merge(self.v_weight, self.v_lora_a, self.v_lora_b)
        q = x @ q_w.t()
        k = x @ k_w.t()
        v = x @ v_w.t()
        return q + k + v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", default="out/llama_scale_lora_results.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    model = LlamaScaleLoRAProjection(args.hidden_size, args.rank).to(device=device, dtype=dtype).eval()
    x = torch.randn(args.batch, args.seq_len, args.hidden_size, device=device, dtype=dtype)

    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    before = count_compute_ops(gm)
    static_compute = sum(1 for n in gm.graph.nodes if annotations.get(n) == stage0 and n.op in {"call_function", "call_method", "call_module"})
    residual = specialize(gm, annotations).to(device).eval()
    after = count_compute_ops(residual)

    with torch.no_grad():
        diff = (model(x) - residual(x)).abs().max().item()

    eager_ms = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
    stageml_ms = benchmark_latency_ms(residual, x, warmup=args.warmup, iterations=args.iterations)

    row = {
        "benchmark": "llama_scale_qkv_lora_projection",
        "device": str(device),
        "dtype": str(dtype),
        "hidden_size": args.hidden_size,
        "rank": args.rank,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "compute_before": before,
        "compute_after": after,
        "static_compute_ops": static_compute,
        "max_diff": diff,
        "eager_ms": eager_ms,
        "stageml_ms": stageml_ms,
        "speedup_vs_eager": eager_ms / stageml_ms,
    }
    write_csv(args.out, [row])
    print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
