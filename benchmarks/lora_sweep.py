from __future__ import annotations

import argparse
import itertools

import torch

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from benchmarks.lora_baselines_bench import LoraLibStyleLinear, ManuallyMergedFromLoraLib
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_strings(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_stageml(model, rewrite):
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    rewrite_count = 0
    if rewrite:
        gm, annotations, stats = optimize_evaluation_order(gm, annotations)
        rewrite_count = stats.total_rewrites
    residual = specialize(gm, annotations)
    return residual, rewrite_count, count_compute_ops(gm), count_compute_ops(residual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="1024,2048,4096")
    parser.add_argument("--ranks", default="8,16,32")
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--out", default="out/lora_sweep.csv")
    args = parser.parse_args()

    device = get_device()
    dims = parse_ints(args.dims)
    ranks = parse_ints(args.ranks)
    batches = parse_ints(args.batches)
    dtype_names = parse_strings(args.dtypes)

    rows = []
    for dim, rank, batch, dtype_name in itertools.product(dims, ranks, batches, dtype_names):
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
        if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32

        torch.manual_seed(0)
        x = torch.randn(batch, dim, device=device, dtype=dtype)
        eager = LoraLibStyleLinear(dim, rank).to(device=device, dtype=dtype).eval()
        manual = ManuallyMergedFromLoraLib(eager).to(device=device, dtype=dtype).eval()
        stageml_no, rewrite_no, before_no, after_no = build_stageml(eager, rewrite=False)
        stageml_yes, rewrite_yes, before_yes, after_yes = build_stageml(eager, rewrite=True)
        stageml_no = stageml_no.to(device).eval()
        stageml_yes = stageml_yes.to(device).eval()

        variants = [
            ("eager", eager, 0, "module"),
            ("manual_merge", manual, 0, "module"),
            ("stageml_no_rewrite", stageml_no, rewrite_no, after_no),
            ("stageml_with_rewrite", stageml_yes, rewrite_yes, after_yes),
        ]
        times = {}
        for name, model, rewrite_count, ops_after in variants:
            latency = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
            with torch.no_grad():
                max_diff = (model(x) - eager(x)).abs().max().item()
            times[name] = latency
            rows.append({
                "variant": name,
                "device": str(device),
                "dtype": str(dtype),
                "dim": dim,
                "rank": rank,
                "batch": batch,
                "latency_ms": latency,
                "speedup_vs_eager": None,
                "rewrite_count": rewrite_count,
                "compute_ops_before": before_yes,
                "compute_ops_after": ops_after,
                "max_diff_vs_eager": max_diff,
            })
        for row in rows[-len(variants):]:
            row["speedup_vs_eager"] = times["eager"] / row["latency_ms"]
        print(f"finished dim={dim} rank={rank} batch={batch} dtype={dtype}")

    write_csv(args.out, rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
