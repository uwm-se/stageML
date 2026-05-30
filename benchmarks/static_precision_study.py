from __future__ import annotations

import argparse

import torch

from benchmarks.common import get_device, write_csv, count_compute_ops
from benchmarks.lora_baselines_bench import CanonicalLoRALinear, LoraLibStyleLinear
from benchmarks.llama_scale_lora_block_bench import LlamaScaleLoRAProjection
from stageml.annotations import stage0, stage1
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


def compute_rows(name, model, example, rewrite):
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    before_ops = count_compute_ops(gm)
    rewrite_count = 0
    if rewrite:
        gm, annotations, stats = optimize_evaluation_order(gm, annotations)
        rewrite_count = stats.total_rewrites
    static_ops = sum(1 for n in gm.graph.nodes if annotations.get(n) == stage0 and n.op in {"call_function", "call_method", "call_module"})
    dynamic_ops = sum(1 for n in gm.graph.nodes if annotations.get(n) == stage1 and n.op in {"call_function", "call_method", "call_module"})
    residual = specialize(gm, annotations)
    after_ops = count_compute_ops(residual)
    with torch.no_grad():
        diff = (model(example) - residual.to(example.device)(example)).abs().max().item()
    return {
        "pattern": name,
        "rewrite": rewrite,
        "compute_ops_before": before_ops,
        "stage0_compute_ops": static_ops,
        "stage1_compute_ops": dynamic_ops,
        "compute_ops_after": after_ops,
        "folded_compute_ops": before_ops - after_ops,
        "rewrite_count": rewrite_count,
        "max_diff": diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--out", default="out/static_precision_study.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(0)
    rows = []

    x = torch.randn(args.batch, args.dim, device=device, dtype=dtype)
    canonical = CanonicalLoRALinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    rows.append(compute_rows("canonical_lora", canonical, x, False))

    loralib = LoraLibStyleLinear(args.dim, args.rank).to(device=device, dtype=dtype).eval()
    rows.append(compute_rows("loralib_style_no_rewrite", loralib, x, False))
    rows.append(compute_rows("loralib_style_with_rewrite", loralib, x, True))

    x2 = torch.randn(args.batch, args.seq_len, args.hidden_size, device=device, dtype=dtype)
    llama = LlamaScaleLoRAProjection(args.hidden_size, args.rank).to(device=device, dtype=dtype).eval()
    rows.append(compute_rows("llama_scale_qkv_lora_projection", llama, x2, False))

    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
