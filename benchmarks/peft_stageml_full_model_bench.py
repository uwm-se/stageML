from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.common import get_device, write_csv
from stageml.peft_bridge import replace_lora_layers_with_stageml


def disable_incompatible_torchao_for_peft():
    """Disable old torchao integrations that can break PEFT in Colab."""
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def gpu_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def auto_targets(model) -> list[str]:
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


def benchmark_generate(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int, warmup: int, iterations: int) -> dict[str, float]:
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


def logits_for_prompt(model, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits.detach().float().cpu()


def load_peft_model(args, device: torch.device, dtype: torch.dtype, targets: list[str] | None = None):
    torch.manual_seed(args.seed)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    if targets is None:
        targets = auto_targets(base) if args.target_modules == "auto" else [x.strip() for x in args.target_modules.split(",") if x.strip()]
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    disable_incompatible_torchao_for_peft()
    torch.manual_seed(args.seed)
    peft_model = get_peft_model(base, config).to(device).eval()
    return peft_model, targets


def run_variant(args, variant: str, tokenizer, device: torch.device, dtype: torch.dtype, targets: list[str] | None, reference_logits: torch.Tensor | None):
    cleanup_cuda()
    model, targets = load_peft_model(args, device, dtype, targets)
    replacement_stats: dict[str, Any] = {
        "replaced_layers": 0,
        "skipped_layers": 0,
        "total_rewrites": 0,
        "total_compute_ops_before": None,
        "total_compute_ops_after": None,
    }

    if variant == "peft_merge_and_unload":
        model = model.merge_and_unload().to(device).eval()
    elif variant == "stageml_rewrite_replacement":
        stats = replace_lora_layers_with_stageml(model, enable_rewrite=True, max_layers=args.max_layers)
        replacement_stats = stats.__dict__
    elif variant == "stageml_no_rewrite_replacement":
        stats = replace_lora_layers_with_stageml(model, enable_rewrite=False, max_layers=args.max_layers)
        replacement_stats = stats.__dict__
    elif variant != "peft_unmerged_lora":
        raise ValueError(f"unknown variant: {variant}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    latency = benchmark_generate(model, tokenizer, args.prompt, device, args.max_new_tokens, args.warmup, args.iterations)
    peak = gpu_peak_mb(device)

    max_diff = None
    if reference_logits is None:
        current_logits = logits_for_prompt(model, tokenizer, args.prompt, device)
        reference_logits = current_logits
        max_diff = 0.0
    else:
        current_logits = logits_for_prompt(model, tokenizer, args.prompt, device)
        max_diff = (current_logits - reference_logits).abs().max().item()

    row = {
        "variant": variant,
        "model": args.model,
        "target_modules": ",".join(targets),
        "rank": args.rank,
        "alpha": args.alpha,
        "dtype": str(dtype),
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "latency_ms_per_request": latency["latency_ms_per_request"],
        "tokens_per_second": latency["tokens_per_second"],
        "speedup_vs_peft_unmerged": None,
        "peak_gpu_memory_mb": peak,
        "max_diff_logits_vs_peft_unmerged": max_diff,
        **replacement_stats,
    }

    del model
    cleanup_cuda()
    return row, targets, reference_logits


def main():
    parser = argparse.ArgumentParser(description="End-to-end PEFT generation benchmark with StageML LoRA layer replacement.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="Explain LoRA in one sentence.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--target-modules", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-layers", type=int, default=None, help="Optional limit for debugging. Default replaces all supported LoRA layers.")
    parser.add_argument("--include-no-rewrite", action="store_true", help="Also benchmark StageML replacement without the rewrite pass.")
    parser.add_argument("--out", default="out/research/peft_stageml_full_model.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    variants = ["peft_unmerged_lora", "peft_merge_and_unload", "stageml_rewrite_replacement"]
    if args.include_no_rewrite:
        variants.append("stageml_no_rewrite_replacement")

    rows = []
    targets = None
    reference_logits = None
    for variant in variants:
        print(f"\n=== Running {variant} ===")
        row, targets, reference_logits = run_variant(args, variant, tokenizer, device, dtype, targets, reference_logits)
        rows.append(row)
        print(row)

    base_latency = rows[0]["latency_ms_per_request"]
    for row in rows:
        row["speedup_vs_peft_unmerged"] = base_latency / row["latency_ms_per_request"]

    write_csv(args.out, rows)
    print(f"Wrote {args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
