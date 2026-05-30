from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


class CanonicalLoRALinear(nn.Module):
    def __init__(self, dim=4096, rank=16, alpha=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        merged = self.W + (self.B @ self.A) * self.scaling
        return x @ merged.t()


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


class ManuallyMergedLinear(nn.Module):
    def __init__(self, source):
        super().__init__()
        with torch.no_grad():
            merged = source.W + (source.B @ source.A) * source.scaling
        self.register_buffer("merged_weight", merged.detach())

    def forward(self, x):
        return x @ self.merged_weight.t()


class ManuallyMergedFromLoraLib(nn.Module):
    def __init__(self, source):
        super().__init__()
        with torch.no_grad():
            merged = source.W + (source.B @ source.A) * source.scaling
        self.register_buffer("merged_weight", merged.detach())

    def forward(self, x):
        return x @ self.merged_weight.t()


def stageml_residual(model, rewrite=False):
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    rewrite_count = 0
    if rewrite:
        gm, annotations, stats = optimize_evaluation_order(gm, annotations)
        rewrite_count = stats.total_rewrites
    residual = specialize(gm, annotations)
    return residual, rewrite_count


def run_one(label, model, x, warmup, iterations):
    ms = benchmark_latency_ms(model, x, warmup=warmup, iterations=iterations)
    return {"variant": label, "latency_ms": ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", default="out/lora_baselines.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)

    canonical = CanonicalLoRALinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    canonical_manual = ManuallyMergedLinear(canonical).to(device=device, dtype=dtype).eval()
    canonical_stageml, canonical_rewrites = stageml_residual(canonical, rewrite=False)
    canonical_stageml = canonical_stageml.to(device).eval()

    loralib = LoraLibStyleLinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    loralib_manual = ManuallyMergedFromLoraLib(loralib).to(device=device, dtype=dtype).eval()
    loralib_no_rewrite, loralib_no_rewrite_count = stageml_residual(loralib, rewrite=False)
    loralib_rewrite, loralib_rewrite_count = stageml_residual(loralib, rewrite=True)
    loralib_no_rewrite = loralib_no_rewrite.to(device).eval()
    loralib_rewrite = loralib_rewrite.to(device).eval()

    rows = []
    variants = [
        ("canonical_eager", canonical, canonical, 0),
        ("canonical_manual_merge", canonical_manual, canonical, 0),
        ("canonical_stageml", canonical_stageml, canonical, canonical_rewrites),
        ("loralib_eager", loralib, loralib, 0),
        ("loralib_manual_merge", loralib_manual, loralib, 0),
        ("loralib_stageml_no_rewrite", loralib_no_rewrite, loralib, loralib_no_rewrite_count),
        ("loralib_stageml_with_rewrite", loralib_rewrite, loralib, loralib_rewrite_count),
    ]

    baseline_times = {}
    for label, model, reference, rewrite_count in variants:
        row = run_one(label, model, x, args.warmup, args.iterations)
        with torch.no_grad():
            row["max_diff_vs_reference"] = (model(x) - reference(x)).abs().max().item()
        row["device"] = str(device)
        row["dtype"] = str(dtype)
        row["dim"] = args.dim
        row["rank"] = args.rank
        row["batch"] = args.batch
        row["rewrite_count"] = rewrite_count
        try:
            if hasattr(model, "graph"):
                row["compute_ops"] = count_compute_ops(model)
            else:
                row["compute_ops"] = "module"
        except Exception:
            row["compute_ops"] = "unknown"
        rows.append(row)
        baseline_times[label] = row["latency_ms"]

    for row in rows:
        if row["variant"].startswith("canonical"):
            row["speedup_vs_family_eager"] = baseline_times["canonical_eager"] / row["latency_ms"]
        else:
            row["speedup_vs_family_eager"] = baseline_times["loralib_eager"] / row["latency_ms"]

    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
