#!/usr/bin/env python3
"""
Real PEFT multi-adapter benchmark for StageML.

This benchmark replaces the earlier synthetic adapter matrices with real PEFT
LoRA modules attached to a real Hugging Face model.  It uses Qwen by default,
creates several PEFT LoRA adapters on the same base model, extracts one LoRA
linear projection, and evaluates mixed-adapter request batches.

It compares:
    1. peft_dynamic_proxy       = runtime LoRA factors A/B
    2. stageml_all_cached       = residualized weight for every adapter
    3. stageml_cost_based_cache = residualized hot adapters under memory budget

This is a projection-level serving benchmark.  It is intended to test the
compiler/cache policy on real PEFT adapter modules before moving to a full
vLLM/LoRAX serving experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from statistics import median, quantiles
from typing import Iterable

import torch

from stageml.adapter_bank import AdapterSpec, DynamicLoRALinear, StageMLAdapterBankLinear
from stageml.adapter_cache import StageMLCostBasedAdapterCache
from stageml.policy_optimizer import choose_adapters_by_benefit_density


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def make_zipf_trace(num_adapters: int, num_requests: int, hotness: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** hotness) for i in range(num_adapters)]
    total = sum(weights)
    probs = [w / total for w in weights]
    names = [f"adapter_{i}" for i in range(num_adapters)]
    prompts = [
        "Explain photosynthesis in simple terms.",
        "Write a short Python function for binary search.",
        "Summarize the main idea in one sentence.",
        "Convert this message into professional English.",
        "Give a concise answer to a technical question.",
        "Classify this support ticket by topic.",
        "Generate a small SQL query example.",
        "Explain this code snippet briefly.",
    ]
    trace = []
    for i in range(num_requests):
        adapter = rng.choices(names, weights=probs, k=1)[0]
        trace.append({"request_id": i, "adapter": adapter, "prompt": rng.choice(prompts)})
    return trace


def load_or_create_trace(path: Path, *, num_adapters: int, num_requests: int, hotness: float, seed: int) -> list[dict]:
    if path.exists():
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
    trace = make_zipf_trace(num_adapters, num_requests, hotness, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in trace:
            f.write(json.dumps(item) + "\n")
    return trace


def batched_adapter_ids(trace: list[dict], batch_size: int) -> list[list[str]]:
    ids = [item["adapter"] for item in trace]
    return [ids[i : i + batch_size] for i in range(0, len(ids), batch_size) if ids[i : i + batch_size]]


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_module(module, xs: list[torch.Tensor], id_batches: list[list[str]], *, warmup_batches: int) -> tuple[list[float], int]:
    latencies_ms: list[float] = []
    total_requests = 0
    with torch.inference_mode():
        for i, (x, adapter_ids) in enumerate(zip(xs, id_batches)):
            sync_if_cuda()
            t0 = time.perf_counter()
            _ = module(x, adapter_ids)
            sync_if_cuda()
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= warmup_batches:
                latencies_ms.append(dt)
                total_requests += len(adapter_ids)
    return latencies_ms, total_requests


def summarize(system: str, latencies_ms: list[float], total_requests: int, cache_memory_mb: float, peak_memory_mb: float, max_diff: float, cached_adapters: int) -> dict:
    total_ms = sum(latencies_ms)
    rps = total_requests / (total_ms / 1000.0) if total_ms > 0 else 0.0
    p50 = median(latencies_ms) if latencies_ms else 0.0
    if len(latencies_ms) >= 20:
        p95 = quantiles(latencies_ms, n=20)[18]
    else:
        p95 = max(latencies_ms) if latencies_ms else 0.0
    return {
        "system": system,
        "num_measured_batches": len(latencies_ms),
        "num_measured_requests": total_requests,
        "requests_per_sec": rps,
        "p50_ms_per_batch": p50,
        "p95_ms_per_batch": p95,
        "mean_ms_per_batch": (total_ms / len(latencies_ms)) if latencies_ms else 0.0,
        "cache_memory_mb": cache_memory_mb,
        "peak_gpu_memory_mb": peak_memory_mb,
        "max_diff_vs_dynamic": max_diff,
        "cached_adapters": cached_adapters,
    }


def find_lora_module_with_adapters(model, adapter_names: list[str]):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer"):
            if all(a in module.lora_A and a in module.lora_B for a in adapter_names):
                base = getattr(module, "base_layer", None)
                if hasattr(base, "weight") and base.weight.ndim == 2:
                    return name, module
    raise RuntimeError("could not find PEFT LoRA Linear module with all adapters")


def build_peft_model_and_extract_adapters(args, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model

    load_kwargs = {"torch_dtype": dtype}
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    adapter_names = [f"adapter_{i}" for i in range(args.num_adapters)]
    cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg, adapter_name=adapter_names[0])
    for name in adapter_names[1:]:
        model.add_adapter(name, cfg)
    model.eval()

    adapter_dir = Path(args.adapter_out_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Saving is useful for vLLM/LoRAX follow-up.  If a PEFT version refuses to
    # save inactive adapters, this is non-fatal for the benchmark.
    for name in adapter_names:
        try:
            model.set_adapter(name)
            model.save_pretrained(adapter_dir / name, selected_adapters=[name])
        except Exception as e:
            print(f"warning: could not save adapter {name}: {e}")

    layer_name, module = find_lora_module_with_adapters(model, adapter_names)
    base = module.base_layer
    weight = base.weight.detach().to(device=device, dtype=dtype).contiguous()
    bias = None
    if getattr(base, "bias", None) is not None:
        bias = base.bias.detach().to(device=device, dtype=dtype).contiguous()

    specs: list[AdapterSpec] = []
    for name in adapter_names:
        A = module.lora_A[name].weight.detach().to(device=device, dtype=dtype).contiguous()
        B = module.lora_B[name].weight.detach().to(device=device, dtype=dtype).contiguous()
        scaling = float(module.scaling[name])
        specs.append(AdapterSpec(name=name, A=A, B=B, scaling=scaling))
    return model, layer_name, weight, bias, specs, adapter_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--num-adapters", type=int, default=4)
    p.add_argument("--num-requests", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--target-modules", default="q_proj,v_proj")
    p.add_argument("--hotness", type=float, default=1.2)
    p.add_argument("--cache-budget-mb", type=float, default=64.0)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace", default="benchmarks/workloads/adapter_request_trace.jsonl")
    p.add_argument("--adapter-out-dir", default="out/real_peft_adapters")
    p.add_argument("--out", default="out/research/real_peft_multi_adapter_bench.csv")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
        print("warning: CPU detected, using float32 instead of float16")

    trace = load_or_create_trace(Path(args.trace), num_adapters=args.num_adapters, num_requests=args.num_requests, hotness=args.hotness, seed=args.seed)
    id_batches = batched_adapter_ids(trace, args.batch)
    request_counts = Counter(item["adapter"] for item in trace)

    model, layer_name, weight, bias, adapters, adapter_names = build_peft_model_and_extract_adapters(args, dtype, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    in_features = weight.shape[1]
    out_features = weight.shape[0]
    xs = [torch.randn(len(ids), in_features, device=device, dtype=dtype) for ids in id_batches]

    dynamic = DynamicLoRALinear(weight, adapters, bias).to(device).eval()
    all_cached = StageMLAdapterBankLinear(weight, adapters, bias).to(device).eval()
    policy = choose_adapters_by_benefit_density(
        request_counts,
        adapters,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        memory_budget_mb=args.cache_budget_mb,
    )
    cost_cached = StageMLCostBasedAdapterCache(weight, adapters, cached_adapter_names=policy.cached_adapters, bias=bias).to(device).eval()

    # Correctness check on first few batches.
    with torch.inference_mode():
        max_diff_all = 0.0
        max_diff_cost = 0.0
        for x, ids in list(zip(xs, id_batches))[: min(5, len(xs))]:
            ref = dynamic(x, ids)
            max_diff_all = max(max_diff_all, float((ref - all_cached(x, ids)).abs().max().item()))
            max_diff_cost = max(max_diff_cost, float((ref - cost_cached(x, ids)).abs().max().item()))

    rows = []
    for system, module, cache_mb, diff, cached_count in [
        ("peft_dynamic_proxy", dynamic, 0.0, 0.0, 0),
        ("stageml_all_cached", all_cached, float(all_cached.merged_weights.numel() * all_cached.merged_weights.element_size()) / (1024 ** 2), max_diff_all, len(adapters)),
        ("stageml_cost_based_cache", cost_cached, float(cost_cached.cached_memory_bytes()) / (1024 ** 2), max_diff_cost, len(policy.cached_adapters)),
    ]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        lat, total = time_module(module, xs, id_batches, warmup_batches=args.warmup_batches)
        peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        row = summarize(system, lat, total, cache_mb, peak, diff, cached_count)
        row.update({
            "model": args.model,
            "layer_name": layer_name,
            "num_adapters": args.num_adapters,
            "rank": args.rank,
            "num_requests": args.num_requests,
            "batch": args.batch,
            "hotness": args.hotness,
            "cache_budget_mb": args.cache_budget_mb,
            "in_features": in_features,
            "out_features": out_features,
            "dtype": str(dtype).replace("torch.", ""),
        })
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    decisions_path = out_path.with_name(out_path.stem + "_policy.csv")
    with decisions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "request_count", "cached", "residual_bytes"])
        writer.writeheader()
        cached = set(policy.cached_adapters)
        for prof in sorted(policy.profiles, key=lambda p: (-p.request_count, p.name)):
            writer.writerow({
                "name": prof.name,
                "request_count": prof.request_count,
                "cached": prof.name in cached,
                "residual_bytes": prof.residual_bytes,
            })

    print(f"model={args.model}")
    print(f"layer={layer_name}")
    print(f"wrote={out_path}")
    print(f"policy={decisions_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
