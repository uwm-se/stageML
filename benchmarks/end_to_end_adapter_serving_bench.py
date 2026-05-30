"""
End-to-end adapter serving-style benchmark for StageML.

This benchmark is designed to answer the professor's question:

    If inference servers hide single-request LoRA overhead, where is StageML useful?

It does not claim to replace LoRAX or vLLM.  It creates a controlled serving-like
request stream with many adapters, repeated hot adapters, cold adapters, mixed
batches, and a memory budget.  It compares:

    dynamic_lora                PEFT-like dynamic adapter branch
    stageml_all_cached          residualize every adapter
    stageml_cost_based_cache    residualize only hot adapters under budget
    on_the_fly_merge            intentionally bad runtime merge baseline

Metrics include p50/p95 batch latency, requests/sec, adapter-cache memory, and
accuracy difference against dynamic LoRA.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import torch

from stageml.adapter_bank import (
    DynamicLoRALinear,
    OnTheFlyMergeLinear,
    StageMLAdapterBankLinear,
    make_random_adapters,
)
from stageml.adapter_cache import StageMLCostBasedAdapterCache, choose_hot_adapters


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_zipf_request_stream(num_adapters: int, num_batches: int, batch: int, hotness: float, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** hotness) for i in range(num_adapters)]
    total = sum(weights)
    probs = [w / total for w in weights]
    names = [f"adapter_{i}" for i in range(num_adapters)]
    stream: list[list[str]] = []
    for _ in range(num_batches):
        stream.append(rng.choices(names, weights=probs, k=batch))
    return stream


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def bench_stream(module, x_batches: list[torch.Tensor], id_batches: list[list[str]], warmup: int) -> dict[str, float]:
    module.eval()
    with torch.no_grad():
        for i in range(min(warmup, len(x_batches))):
            _ = module(x_batches[i], id_batches[i])
        synchronize()

        latencies: list[float] = []
        total_requests = 0
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for x, ids in zip(x_batches, id_batches):
                start_event.record()
                _ = module(x, ids)
                end_event.record()
                synchronize()
                latencies.append(float(start_event.elapsed_time(end_event)))
                total_requests += x.shape[0]
        else:
            import time
            for x, ids in zip(x_batches, id_batches):
                t0 = time.perf_counter()
                _ = module(x, ids)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                total_requests += x.shape[0]

    total_ms = sum(latencies)
    return {
        "mean_batch_latency_ms": statistics.mean(latencies),
        "p50_batch_latency_ms": percentile(latencies, 50),
        "p95_batch_latency_ms": percentile(latencies, 95),
        "p99_batch_latency_ms": percentile(latencies, 99),
        "requests_per_second": total_requests / (total_ms / 1000.0),
        "num_measured_batches": len(latencies),
    }


def memory_mb_from_tensor(t: torch.Tensor) -> float:
    return float(t.numel() * t.element_size()) / (1024.0 * 1024.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-adapters", type=int, default=32)
    p.add_argument("--num-batches", type=int, default=200)
    p.add_argument("--warmup-batches", type=int, default=20)
    p.add_argument("--hotness", type=float, default=1.2)
    p.add_argument("--cache-budget-mb", type=float, default=512.0)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/research/end_to_end_adapter_serving_bench.csv")
    args = p.parse_args()

    device = get_device()
    dtype = parse_dtype(args.dtype)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    W = torch.randn(args.out_features, args.in_features, device=device, dtype=dtype) / math.sqrt(args.in_features)
    bias = torch.randn(args.out_features, device=device, dtype=dtype) / math.sqrt(args.out_features)
    adapters = make_random_adapters(
        num_adapters=args.num_adapters,
        in_features=args.in_features,
        out_features=args.out_features,
        rank=args.rank,
        dtype=dtype,
        device=device,
    )

    id_batches = make_zipf_request_stream(args.num_adapters, args.num_batches, args.batch, args.hotness, args.seed)
    request_counts = Counter(a for batch_ids in id_batches for a in batch_ids)
    cached_names, decisions = choose_hot_adapters(
        request_counts,
        adapters,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=dtype,
        memory_budget_mb=args.cache_budget_mb,
    )

    x_batches = [torch.randn(args.batch, args.in_features, device=device, dtype=dtype) for _ in range(args.num_batches)]

    dynamic = DynamicLoRALinear(W, adapters, bias).to(device).eval()
    all_cached = StageMLAdapterBankLinear.from_lora_factors(W, adapters, bias).to(device).eval()
    cost_cache = StageMLCostBasedAdapterCache(W, adapters, cached_adapter_names=cached_names, bias=bias).to(device).eval()
    on_the_fly = OnTheFlyMergeLinear(W, adapters, bias).to(device).eval()

    with torch.no_grad():
        y_ref = dynamic(x_batches[0], id_batches[0])
        diff_all = float((y_ref - all_cached(x_batches[0], id_batches[0])).abs().max().detach().cpu())
        diff_cache = float((y_ref - cost_cache(x_batches[0], id_batches[0])).abs().max().detach().cpu())

    base_weight_mb = memory_mb_from_tensor(W)
    adapter_factor_mb = sum(memory_mb_from_tensor(a.A) + memory_mb_from_tensor(a.B) for a in adapters)
    all_cached_mb = memory_mb_from_tensor(all_cached.merged_weights)
    cost_cache_mb = float(cost_cache.cached_memory_bytes()) / (1024.0 * 1024.0)

    variants = [
        ("dynamic_lora", dynamic, 0.0, 0.0),
        ("on_the_fly_merge", on_the_fly, 0.0, 0.0),
        ("stageml_all_cached", all_cached, all_cached_mb, diff_all),
        ("stageml_cost_based_cache", cost_cache, cost_cache_mb, diff_cache),
    ]

    rows = []
    dynamic_rps = None
    for name, module, cache_mb, max_diff in variants:
        stats = bench_stream(module, x_batches, id_batches, args.warmup_batches)
        if name == "dynamic_lora":
            dynamic_rps = stats["requests_per_second"]
        rows.append({
            "variant": name,
            **stats,
            "speedup_rps_vs_dynamic_lora": (stats["requests_per_second"] / dynamic_rps) if dynamic_rps else 1.0,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "batch": args.batch,
            "num_adapters": args.num_adapters,
            "num_batches": args.num_batches,
            "hotness": args.hotness,
            "dtype": args.dtype,
            "cache_budget_mb": args.cache_budget_mb,
            "cached_adapters": len(cached_names) if name == "stageml_cost_based_cache" else (args.num_adapters if name == "stageml_all_cached" else 0),
            "adapter_factor_memory_mb": adapter_factor_mb,
            "base_weight_memory_mb": base_weight_mb,
            "cache_memory_mb": cache_mb,
            "max_diff_vs_dynamic": max_diff,
            "stage_model": "base+adapter+request",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    decision_path = out.with_name(out.stem + "_cache_decisions.csv")
    with decision_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "cached", "estimated_bytes", "use_count"])
        writer.writeheader()
        for d in decisions:
            writer.writerow(d.__dict__)

    print(f"wrote {out}")
    print(f"wrote {decision_path}")
    print(f"cached adapters: {cached_names[:10]}{'...' if len(cached_names) > 10 else ''}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
