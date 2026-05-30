from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM



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

from benchmarks.common import benchmark_latency_ms, count_compute_ops, get_device, write_csv
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


class ExtractedLoRALayer(nn.Module):
    def __init__(self, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("W", W.detach().clone())
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.scaling = float(scaling)

    def forward(self, x):
        y = x @ self.W.t() + self.scaling * (x @ self.A.t() @ self.B.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class ManualMergedLayer(nn.Module):
    def __init__(self, src: ExtractedLoRALayer):
        super().__init__()
        with torch.no_grad():
            merged = src.W + src.scaling * (src.B @ src.A)
        self.register_buffer("merged", merged.detach().clone())
        if src.bias is not None:
            self.register_buffer("bias", src.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        y = x @ self.merged.t()
        if self.bias is not None:
            y = y + self.bias
        return y


def auto_targets(model: nn.Module) -> list[str]:
    names = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj"}:
            names.add(leaf)
    if names:
        return sorted(names)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(name.split(".")[-1])
    return sorted(names)[:4]


def first_lora_module(model: nn.Module) -> tuple[str, Any]:
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer"):
            return name, module
    raise RuntimeError("No PEFT LoRA layer found. Try a different target module list.")


def extract_lora_tensors(module: Any, adapter_name: str = "default"):
    base = module.base_layer
    W = base.weight.detach()
    bias = base.bias.detach() if getattr(base, "bias", None) is not None else None
    A_mod = module.lora_A[adapter_name]
    B_mod = module.lora_B[adapter_name]
    A = A_mod.weight.detach()
    B = B_mod.weight.detach()
    scaling = module.scaling[adapter_name]
    return W, A, B, float(scaling), bias


def build_stageml(src: ExtractedLoRALayer, rewrite: bool):
    gm, annotations = trace_and_annotate(src, {"x": "stage1"})
    rewrite_count = 0
    if rewrite:
        gm, annotations, stats = optimize_evaluation_order(gm, annotations)
        rewrite_count = stats.total_rewrites
    residual = specialize(gm, annotations)
    return residual, rewrite_count, count_compute_ops(gm), count_compute_ops(residual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--target-modules", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--out", default="out/research/peft_lora_layer_bench.csv")
    args = parser.parse_args()

    device = get_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    targets = auto_targets(base) if args.target_modules == "auto" else [x.strip() for x in args.target_modules.split(",")]
    task_type = TaskType.CAUSAL_LM
    config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=targets, lora_dropout=0.0, bias="none", task_type=task_type)
    disable_incompatible_torchao_for_peft()
    peft_model = get_peft_model(base, config).to(device).eval()
    layer_name, lora_layer = first_lora_module(peft_model)
    W, A, B, scaling, bias = extract_lora_tensors(lora_layer)
    extracted = ExtractedLoRALayer(W, A, B, scaling, bias).to(device=device, dtype=dtype).eval()
    manual = ManualMergedLayer(extracted).to(device=device, dtype=dtype).eval()
    stageml_no, rew_no, before_no, after_no = build_stageml(extracted, rewrite=False)
    stageml_yes, rew_yes, before_yes, after_yes = build_stageml(extracted, rewrite=True)
    stageml_no = stageml_no.to(device=device, dtype=dtype).eval()
    stageml_yes = stageml_yes.to(device=device, dtype=dtype).eval()

    in_features = W.shape[1]
    x = torch.randn(args.batch * args.seq_len, in_features, device=device, dtype=dtype)
    rows = []
    variants = [
        ("extracted_peft_lora_math", extracted, 0, before_yes),
        ("manual_merge", manual, 0, 1),
        ("stageml_no_rewrite", stageml_no, rew_no, after_no),
        ("stageml_with_rewrite", stageml_yes, rew_yes, after_yes),
    ]
    times = {}
    for name, model, rewrite_count, compute_ops in variants:
        latency = benchmark_latency_ms(model, x, warmup=args.warmup, iterations=args.iterations)
        with torch.no_grad():
            diff = (model(x) - extracted(x)).abs().max().item()
        times[name] = latency
        rows.append({
            "variant": name,
            "model": args.model,
            "peft_layer": layer_name,
            "target_modules": ",".join(targets),
            "device": str(device),
            "dtype": str(dtype),
            "rank": args.rank,
            "alpha": args.alpha,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "in_features": int(W.shape[1]),
            "out_features": int(W.shape[0]),
            "latency_ms": latency,
            "speedup_vs_extracted_peft_lora_math": None,
            "rewrite_count": rewrite_count,
            "compute_ops_after": compute_ops,
            "max_diff_vs_extracted": diff,
        })
    for row in rows:
        row["speedup_vs_extracted_peft_lora_math"] = times["extracted_peft_lora_math"] / row["latency_ms"]
    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
