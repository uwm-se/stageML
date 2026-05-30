from __future__ import annotations

import argparse
import operator
from pathlib import Path

import torch
import torch.nn as nn
from torch._dynamo import export

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.evaluator import specialize
from stageml.tracer import trace_and_annotate


class LoRALinear(nn.Module):
    def __init__(self, dim=1024, rank=16, alpha=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        merged = self.W + (self.B @ self.A) * self.scaling
        return x @ merged.t()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", default="out/torch_compile_comparison.csv")
    parser.add_argument("--graph-out", default="out/torch_compile_exported_graph.txt")
    args = parser.parse_args()

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)
    model = LoRALinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)

    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    residual = specialize(gm, annotations).to(device).eval()

    compiled = torch.compile(model)
    compiled(x)

    eager_ms = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
    stageml_ms = benchmark_latency_ms(residual, x, warmup=args.warmup, iterations=args.iterations)
    compiled_ms = benchmark_latency_ms(compiled, x, warmup=args.warmup, iterations=args.iterations)

    try:
        exported = export(model, x)
        exported_gm = exported.graph_module
        graph_text = str(exported_gm.graph)
    except Exception as exc:
        exported_gm = None
        graph_text = f"EXPORT FAILED: {exc}"

    Path(args.graph_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.graph_out).write_text(graph_text, encoding="utf-8")

    matmul_count_export = graph_text.count("matmul") + graph_text.count("mm")
    row = {
        "device": str(device),
        "dtype": str(dtype),
        "dim": args.dim,
        "rank": args.rank,
        "eager_ms": eager_ms,
        "stageml_ms": stageml_ms,
        "torch_compile_ms": compiled_ms,
        "speedup_vs_eager": eager_ms / stageml_ms,
        "speedup_vs_torch_compile": compiled_ms / stageml_ms,
        "stage_ml_compute_ops": count_compute_ops(residual),
        "export_graph_matmul_mentions": matmul_count_export,
        "export_graph_file": args.graph_out,
    }
    write_csv(args.out, [row])
    print(row)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.graph_out}")


if __name__ == "__main__":
    main()
