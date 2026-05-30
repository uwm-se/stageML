from __future__ import annotations

"""Small perplexity evaluation for StageML residualization.

The script supports two modes:

1. direct: use Transformers and Datasets in-process.  This can evaluate a base
   model and, when an expert-specific adapter is supplied, a model whose first
   Mixtral w1 expert projection has been residualized by StageML.
2. lm_eval: call the lm-evaluation-harness CLI for the base model or for an
   already-saved compiled model directory.

The direct path is the useful StageML quality guard because it can patch the
loaded model in memory after the quantization abstract interpreter accepts the
rewrite.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from benchmarks.real_moe_lora_residual_bench import (
    _load_optional_transformers,
    extract_expert_weight,
    load_expert_lora_adapter,
)
from stageml.quant_absint import QuantizationConfig, analyze_residualization, quant_dequant
from stageml.residual_planner import AdapterQuantInfo, PlannerConfig, choose_residual_plan
from stageml.moe_ir import MoEAdapterMeta


def find_first_mixtral_experts_module(model: torch.nn.Module):
    for name, module in model.named_modules():
        if not name.endswith(".mlp.experts"):
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        gate_up = getattr(gate_up, "weight", gate_up)
        if isinstance(gate_up, torch.Tensor) and gate_up.ndim == 3:
            return name, module, gate_up
    raise RuntimeError("could not find Mixtral .mlp.experts.gate_up_proj tensor")


def apply_stage_w1_residual_to_first_layer(model: torch.nn.Module, adapter_dir: Path, *, theta: float, max_experts: int | None = None, quant_bits: int | None = None, quant_per_channel: bool = True) -> dict[str, Any]:
    """Apply accepted StageML residualization to the first Mixtral expert w1 block.

    This is an evaluation hook, not a complete production model compiler.  It is
    enough to test whether an accepted residual rewrite changes perplexity on a
    small dataset.
    """
    name, module, gate_up = find_first_mixtral_experts_module(model)
    if gate_up.shape[1] % 2 != 0:
        raise RuntimeError("gate_up_proj does not pack w1 and w3 evenly")
    num_experts = int(gate_up.shape[0] if max_experts is None else min(gate_up.shape[0], max_experts))
    in_features = int(gate_up.shape[2])
    out_features = int(gate_up.shape[1] // 2)

    # Use the existing adapter loader to ensure the same expert-specific tensor
    # selection as the main benchmark.
    adapter = load_expert_lora_adapter(adapter_dir, num_experts=num_experts, in_features=in_features, out_features=out_features)

    base_w1 = gate_up[:num_experts, :out_features, :].detach().cpu().to(torch.float32).contiguous()
    delta = adapter.scaling * torch.einsum("eor,eri->eoi", adapter.B, adapter.A)

    if quant_bits is None:
        # A direct high precision residualization case has zero projection error.
        eps = 0.0
        safe = True
        q_report = {"mode": "exact_high_precision", "epsilon": eps, "safe": safe}
        fused_cpu = base_w1 + delta
    else:
        qcfg = QuantizationConfig(bits=int(quant_bits), per_channel=bool(quant_per_channel), channel_dim=0)
        bound = analyze_residualization(base_w1, delta, theta=theta, config=qcfg)
        eps = float(bound.epsilon_weight_fro)
        safe = bool(bound.safe)
        q_report = {
            "mode": "quantized_residual",
            "bits": int(quant_bits),
            "per_channel": bool(quant_per_channel),
            "epsilon_weight_fro": float(bound.epsilon_weight_fro),
            "epsilon_output_fro": bound.epsilon_output_fro,
            "theta": float(theta),
            "safe": safe,
            "decision": "accept" if safe else "reject",
        }
        fused_cpu = quant_dequant(base_w1 + delta, qcfg)

    meta = MoEAdapterMeta(
        name=adapter.name,
        rank=adapter.rank,
        num_experts=adapter.num_experts,
        in_features=adapter.in_features,
        out_features=adapter.out_features,
        dtype="bf16",
        request_count=512,
    )
    plan = choose_residual_plan(
        [meta],
        config=PlannerConfig(memory_budget_mb=2048, theta=theta, dtype="bf16"),
        quant_info={adapter.name: AdapterQuantInfo(epsilon=eps, safe=safe)},
    )

    if not safe or eps > theta or plan.by_kind().get("materialized_residual", 0) == 0:
        return {"accepted": False, "module": name, "quantization": q_report, "plan": plan.to_dict()}

    with torch.no_grad():
        fused = fused_cpu.to(device=gate_up.device, dtype=gate_up.dtype)
        # gate_up can be a Parameter-like tensor.  In-place copy preserves model structure.
        gate_up[:num_experts, :out_features, :].copy_(fused)

    return {
        "accepted": True,
        "module": name,
        "adapter": adapter.name,
        "quantization": q_report,
        "plan": plan.to_dict(),
        "note": "Applied StageML exact materialized residual to first Mixtral w1 expert block in memory.",
    }


def load_wikitext_texts(dataset_name: str, split: str, limit_docs: int) -> list[str]:
    from datasets import load_dataset

    # Hugging Face exposes WikiText under the namespaced repository
    # Salesforce/wikitext. The older unnamespaced id, wikitext, can fail in
    # recent huggingface_hub/datasets combinations when it is resolved as an
    # hf:// URI. Keep the CLI option flexible, but canonicalize the default so
    # artifact evaluation does not fail before the perplexity guard runs.
    canonical_name = "Salesforce/wikitext" if dataset_name == "wikitext" else dataset_name
    try:
        ds = load_dataset(canonical_name, "wikitext-2-raw-v1", split=split)
    except Exception as exc:
        if canonical_name == "Salesforce/wikitext":
            parquet_url = (
                "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
                f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
            )
            try:
                ds = load_dataset("parquet", data_files={split: parquet_url}, split=split)
            except Exception as parquet_exc:
                raise RuntimeError(
                    "Failed to load WikiText using both the canonical "
                    "Salesforce/wikitext dataset id and the direct parquet fallback. "
                    "Check HF_TOKEN, network access, and datasets/huggingface_hub versions."
                ) from parquet_exc
        else:
            raise RuntimeError(f"Failed to load dataset {dataset_name!r}") from exc
    texts = []
    for row in ds:
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= limit_docs:
            break
    return texts


def direct_perplexity(model, tokenizer, texts: list[str], *, device: str, max_length: int, stride: int) -> dict[str, float]:
    model.eval()
    joined = "\n\n".join(texts)
    enc = tokenizer(joined, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(ids, labels=target_ids, use_cache=False)
            nlls.append(out.loss.detach().float() * trg_len)
        prev_end = end
        if end == seq_len:
            break
    total_nll = torch.stack(nlls).sum()
    total_tokens = max(seq_len - 1, 1)
    ppl = float(torch.exp(total_nll / total_tokens).detach().cpu())
    return {"perplexity": ppl, "tokens": int(total_tokens), "windows": len(nlls)}


def run_lm_eval(model: str, *, task: str, output_path: Path, limit: int | None, batch_size: str) -> dict[str, Any]:
    exe = shutil.which("lm_eval")
    if exe is None:
        # Most installs also support python -m lm_eval.
        cmd = ["python", "-m", "lm_eval"]
    else:
        cmd = [exe]
    cmd += [
        "--model", "hf",
        "--model_args", f"pretrained={model},trust_remote_code=True,dtype=bfloat16,device_map=auto",
        "--tasks", task,
        "--batch_size", batch_size,
        "--output_path", str(output_path),
    ]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"returncode": proc.returncode, "command": cmd, "stdout_tail": proc.stdout[-4000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--out-dir", default="paper_outputs/perplexity_eval")
    ap.add_argument("--backend", default="direct", choices=["direct", "lm_eval"])
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-docs", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--gpu-memory-gb", type=int, default=82)
    ap.add_argument("--cpu-memory-gb", type=int, default=180)
    ap.add_argument("--offload-folder", default="/data/stageml_h100_run/hf_offload")
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--quantized-check", action="store_true", help="apply the quantization safety gate before the in-memory residual rewrite")
    ap.add_argument("--quant-bits", type=int, default=8)
    ap.add_argument("--quant-per-channel", action="store_true", default=True)
    ap.add_argument("--max-experts", type=int, default=8)
    ap.add_argument("--lm-eval-task", default="wikitext")
    ap.add_argument("--lm-eval-limit", type=int, default=20)
    ap.add_argument("--lm-eval-batch-size", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "lm_eval":
        base = run_lm_eval(args.model, task=args.lm_eval_task, output_path=out_dir / "lm_eval_base", limit=args.lm_eval_limit, batch_size=args.lm_eval_batch_size)
        result = {
            "backend": "lm_eval",
            "base": base,
            "compiled": {
                "status": "not_run",
                "reason": "lm-eval runs a model path. Use --backend direct for in-memory StageML residualization, or save a compiled model directory first.",
            },
        }
        (out_dir / "perplexity_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    AutoModelForCausalLM, AutoTokenizer = _load_optional_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "device_map": args.device_map,
        "offload_folder": args.offload_folder,
        "offload_state_dict": True,
    }
    if args.device_map == "auto":
        kwargs["max_memory"] = {0: f"{args.gpu_memory_gb}GiB", "cpu": f"{args.cpu_memory_gb}GiB"}

    texts = load_wikitext_texts(args.dataset, args.split, args.limit_docs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    base = direct_perplexity(model, tokenizer, texts, device=args.device, max_length=args.max_length, stride=args.stride)

    compile_report: dict[str, Any] = {"status": "not_requested"}
    compiled = None
    if args.adapter_dir:
        compile_report = apply_stage_w1_residual_to_first_layer(
            model,
            Path(args.adapter_dir),
            theta=args.theta,
            max_experts=args.max_experts,
            quant_bits=args.quant_bits if args.quantized_check else None,
            quant_per_channel=args.quant_per_channel,
        )
        if compile_report.get("accepted"):
            compiled = direct_perplexity(model, tokenizer, texts, device=args.device, max_length=args.max_length, stride=args.stride)

    result = {
        "backend": "direct",
        "model": args.model,
        "dataset": "wikitext-2-raw-v1",
        "split": args.split,
        "limit_docs": args.limit_docs,
        "base": base,
        "compile_report": compile_report,
        "compiled_after_stage_residual": compiled,
        "claim_boundary": "This is a small quality guard for an in-memory StageML w1 residual rewrite, not a full lm-eval replacement unless backend=lm_eval is used on saved models.",
    }
    if compiled is not None:
        result["relative_ppl_change"] = (compiled["perplexity"] - base["perplexity"]) / max(base["perplexity"], 1e-12)

    (out_dir / "perplexity_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (out_dir / "perplexity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "perplexity", "tokens", "windows"])
        writer.writeheader()
        writer.writerow({"system": "base", **base})
        if compiled is not None:
            writer.writerow({"system": "stage_residual", **compiled})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
