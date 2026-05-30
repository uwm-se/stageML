from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer



def disable_incompatible_torchao_for_peft():
    """PEFT may see an old torchao install in Colab and fail before using LoRA.

    StageML does not need torchao for this benchmark. If torchao is present but
    older than PEFT expects, force PEFT's torchao dispatcher to report unavailable.
    This keeps the benchmark on standard PEFT LoRA layers.
    """
    try:
        import importlib.metadata as importlib_metadata
        version = importlib_metadata.version("torchao")
    except Exception:
        return

    def major_minor(v: str):
        parts = []
        for x in v.split(".")[:2]:
            num = "".join(ch for ch in x if ch.isdigit())
            parts.append(int(num) if num else 0)
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts)

    if major_minor(version) < (0, 16):
        try:
            import peft.import_utils as peft_import_utils
            peft_import_utils.is_torchao_available = lambda: False
        except Exception:
            pass
        try:
            import peft.tuners.lora.torchao as peft_lora_torchao
            peft_lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass
        print(f"Disabled incompatible torchao {version} for this PEFT benchmark")

from benchmarks.common import get_device, write_csv


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def gpu_peak_mb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def auto_targets(model):
    names = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
            names.add(leaf)
        if leaf in {"c_attn", "c_proj"}:
            names.add(leaf)
    if names:
        return sorted(names)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.add(name.split(".")[-1])
    return sorted(names)[:4]


def benchmark_generate(model, tokenizer, prompt, device, max_new_tokens, warmup, iterations):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        sync(device)
        start = time.perf_counter()
        total_new_tokens = 0
        for _ in range(iterations):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            total_new_tokens += max(0, out.shape[-1] - inputs["input_ids"].shape[-1])
        sync(device)
    elapsed = time.perf_counter() - start
    return {
        "latency_ms_per_request": elapsed * 1000.0 / iterations,
        "tokens_per_second": total_new_tokens / elapsed if elapsed > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="Explain LoRA in one sentence.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--target-modules", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--out", default="out/peft_real_model_generate.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    targets = auto_targets(base) if args.target_modules == "auto" else [x.strip() for x in args.target_modules.split(",")]
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    disable_incompatible_torchao_for_peft()
    peft_model = get_peft_model(base, config).to(device).eval()

    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    unmerged = benchmark_generate(peft_model, tokenizer, args.prompt, device, args.max_new_tokens, args.warmup, args.iterations)
    rows.append({
        "variant": "peft_unmerged_lora",
        "model": args.model,
        "target_modules": ",".join(targets),
        "rank": args.rank,
        "alpha": args.alpha,
        "dtype": str(dtype),
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "latency_ms_per_request": unmerged["latency_ms_per_request"],
        "tokens_per_second": unmerged["tokens_per_second"],
        "peak_gpu_memory_mb": gpu_peak_mb(device),
    })

    merged_model = peft_model.merge_and_unload().to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    merged = benchmark_generate(merged_model, tokenizer, args.prompt, device, args.max_new_tokens, args.warmup, args.iterations)
    rows.append({
        "variant": "peft_merge_and_unload",
        "model": args.model,
        "target_modules": ",".join(targets),
        "rank": args.rank,
        "alpha": args.alpha,
        "dtype": str(dtype),
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "latency_ms_per_request": merged["latency_ms_per_request"],
        "tokens_per_second": merged["tokens_per_second"],
        "peak_gpu_memory_mb": gpu_peak_mb(device),
    })

    rows[1]["speedup_vs_unmerged_latency"] = rows[0]["latency_ms_per_request"] / rows[1]["latency_ms_per_request"]
    rows[0]["speedup_vs_unmerged_latency"] = 1.0
    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
