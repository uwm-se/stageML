"""
Benchmark StageML adapter-bank residualization for multi-adapter serving.

This benchmark is the research extension beyond single-adapter LoRA merge.
It asks: if requests in one batch use different adapters, can StageML
pre-specialize each adapter's residual weight and avoid runtime adapter math?

Compared variants:
    dynamic_lora          : base matmul + LoRA branch at runtime
    on_the_fly_merge      : bad baseline, merges W + BA during runtime
    stageml_adapter_bank  : precomputed residual weights, grouped runtime matmul
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks.common import benchmark_latency_ms, get_device, write_csv
from stageml.adapter_bank import (
    DynamicLoRALinear,
    OnTheFlyMergeLinear,
    StageMLAdapterBankLinear,
    make_adapter_ids,
    make_random_adapters,
)


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-adapters", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--pattern", type=str, default="round_robin", choices=["single", "round_robin", "clustered"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", type=str, default="out/research/multi_adapter_bank_bench.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    torch.manual_seed(0)

    W = torch.randn(args.out_features, args.in_features, device=device, dtype=dtype) / (args.in_features ** 0.5)
    bias = torch.randn(args.out_features, device=device, dtype=dtype) / (args.out_features ** 0.5)
    adapters = make_random_adapters(
        num_adapters=args.num_adapters,
        in_features=args.in_features,
        out_features=args.out_features,
        rank=args.rank,
        dtype=dtype,
        device=device,
    )
    x = torch.randn(args.batch, args.in_features, device=device, dtype=dtype)
    adapter_ids = make_adapter_ids(args.batch, args.num_adapters, args.pattern)

    dynamic = DynamicLoRALinear(W, adapters, bias).to(device).eval()
    on_the_fly = OnTheFlyMergeLinear(W, adapters, bias).to(device).eval()
    bank = StageMLAdapterBankLinear.from_lora_factors(W, adapters, bias).to(device).eval()

    with torch.no_grad():
        y_dyn = dynamic(x, adapter_ids)
        y_bank = bank(x, adapter_ids)
        max_diff = float((y_dyn - y_bank).abs().max().detach().cpu())

    rows = []
    variants = [
        ("dynamic_lora", dynamic),
        ("on_the_fly_merge", on_the_fly),
        ("stageml_adapter_bank", bank),
    ]
    base_latency = None
    for name, module in variants:
        latency = benchmark_latency_ms(lambda z: module(z, adapter_ids), x, warmup=args.warmup, iterations=args.iterations)
        if name == "dynamic_lora":
            base_latency = latency
        rows.append({
            "variant": name,
            "latency_ms": latency,
            "speedup_vs_dynamic_lora": (base_latency / latency) if base_latency else 1.0,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "batch": args.batch,
            "num_adapters": args.num_adapters,
            "dtype": args.dtype,
            "pattern": args.pattern,
            "max_diff_vs_dynamic": max_diff if name == "stageml_adapter_bank" else 0.0,
            "stage_model": "base+adapter+request",
        })

    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
