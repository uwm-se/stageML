from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.annotations import stage0
from stageml.evaluator import specialize
from stageml.real_mlir_lower import write_parseable_mlir, verify_mlir_with_mlir_opt
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


class LoraLibStyleLinear(nn.Module):
    def __init__(self, dim=4096, rank=16, alpha=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        base = x @ self.W.t()
        lora_branch = (x @ self.A.t()) @ self.B.t()
        return base + lora_branch * self.scaling


def static_compute_count(gm, annotations):
    return sum(
        1
        for n in gm.graph.nodes
        if annotations.get(n) == stage0 and n.op in {"call_function", "call_method", "call_module"}
    )


def run_variant(model, x, enable_rewrite, artifact_dir, name, warmup, iterations):
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    original_graph = str(gm.graph)
    rewrite_count = 0

    if enable_rewrite:
        gm, annotations, stats = optimize_evaluation_order(gm, annotations)
        rewrite_count = stats.total_rewrites

    graph_after_rewrite = str(gm.graph)
    mlir_path = artifact_dir / f"{name}.mlir"
    write_parseable_mlir(gm, annotations, mlir_path, fn_name=name)
    parse_ok, parse_message = verify_mlir_with_mlir_opt(mlir_path)

    compute_ops_after_analysis = count_compute_ops(gm)
    static_compute_ops_after_analysis = static_compute_count(gm, annotations)
    residual = specialize(gm, annotations).to(x.device).eval()
    residual_graph = str(residual.graph)

    with torch.no_grad():
        max_diff = (model(x) - residual(x)).abs().max().item()

    ms = benchmark_latency_ms(residual, x, warmup=warmup, iterations=iterations)

    (artifact_dir / f"{name}_original_fx.txt").write_text(original_graph, encoding="utf-8")
    (artifact_dir / f"{name}_rewritten_fx.txt").write_text(graph_after_rewrite, encoding="utf-8")
    (artifact_dir / f"{name}_residual_fx.txt").write_text(residual_graph, encoding="utf-8")
    (artifact_dir / f"{name}_mlir_parse.txt").write_text(str(parse_ok) + "\n" + parse_message, encoding="utf-8")

    return {
        "variant": name,
        "device": str(x.device),
        "dtype": str(x.dtype),
        "enable_rewrite": enable_rewrite,
        "rewrite_count": rewrite_count,
        "compute_ops_after_analysis": compute_ops_after_analysis,
        "static_compute_ops_after_analysis": static_compute_ops_after_analysis,
        "residual_compute_ops": count_compute_ops(residual),
        "latency_ms": ms,
        "max_diff": max_diff,
        "parseable_mlir_file": str(mlir_path),
        "mlir_opt_parse_ok": parse_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", default="out/rewrite_ablation.csv")
    parser.add_argument("--artifacts", default="out/rewrite_ablation_artifacts")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    model = LoraLibStyleLinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    eager_ms = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
    rows = []
    no_rewrite = run_variant(model, x, False, artifact_dir, "stageml_no_rewrite", args.warmup, args.iterations)
    with_rewrite = run_variant(model, x, True, artifact_dir, "stageml_with_rewrite", args.warmup, args.iterations)

    for row in [no_rewrite, with_rewrite]:
        row["eager_ms"] = eager_ms
        row["speedup_vs_eager"] = eager_ms / row["latency_ms"]
        rows.append(row)

    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")
    print(f"Wrote artifacts under {artifact_dir}")


if __name__ == "__main__":
    main()
