from __future__ import annotations

"""Same-boundary diagnostic benchmark for StageML and vLLM MoE kernels.

This script avoids the vLLM HTTP server.  It loads the same real prompt hidden
states used by the StageML benchmark, discovers the Mixtral router, and then
attempts to call vLLM's internal fused MoE Python entry point directly.

Important boundary note:
- vLLM's public supported interface is serving; its internal fused MoE function
  is version-dependent.
- The direct fused MoE path below measures the base MoE expert block if the
  internal vLLM function is available.
- StageML measures the materialized MoE LoRA residual block at the same hidden
  state/routing boundary.

The JSON output records whether the vLLM internal call was available and whether
it was semantically comparable.  This is meant to prevent accidental overclaiming.
"""

import argparse
import csv
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from stageml.moe_lora_layers import MoEAdapterSpec

from benchmarks.real_moe_lora_residual_bench import (
    _gpu_metadata,
    _load_optional_transformers,
    _read_prompts,
    collect_real_hidden_states,
    load_expert_lora_adapter,
    make_routing,
    percentile,
)


class StageMLMaterializedMixtralMoELoRA(nn.Module):
    """Full Mixtral expert MLP boundary with w1 residualized by StageML.

    For each routed token this computes:
        down( silu(x @ (w1 + BA).T) * (x @ w3.T) )
    and applies the router weights.
    """

    def __init__(
        self,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        adapters: list[MoEAdapterSpec],
    ) -> None:
        super().__init__()
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("gate_up_proj and down_proj must be [experts, features, hidden]")
        intermediate2 = int(gate_up_proj.shape[1])
        if intermediate2 % 2 != 0:
            raise ValueError("gate_up_proj must pack w1 and w3")
        intermediate = intermediate2 // 2
        w1 = gate_up_proj[:, :intermediate, :].detach()
        w3 = gate_up_proj[:, intermediate:, :].detach()
        self.adapter_names = [a.name for a in adapters]
        self.name_to_index = {n: i for i, n in enumerate(self.adapter_names)}
        self.register_buffer("w3", w3.contiguous())
        self.register_buffer("down_proj", down_proj.detach().contiguous())

        merged = []
        for adapter in adapters:
            delta = adapter.scaling * torch.einsum("eor,eri->eoi", adapter.B.detach(), adapter.A.detach())
            merged.append((w1 + delta).contiguous())
        self.register_buffer("w1_residual", torch.stack(merged, dim=0).contiguous())

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor, adapter_ids: list[str]) -> torch.Tensor:
        out = torch.zeros((x.shape[0], self.down_proj.shape[1]), device=x.device, dtype=x.dtype)
        adapter_index = torch.tensor([self.name_to_index[str(a)] for a in adapter_ids], device=x.device, dtype=torch.long)
        num_experts = int(self.down_proj.shape[0])
        for k in range(expert_ids.shape[1]):
            eids = expert_ids[:, k]
            gates = routing_weights[:, k].to(dtype=x.dtype)
            for aidx in range(len(self.adapter_names)):
                mask_a = adapter_index == aidx
                for e in range(num_experts):
                    idx = torch.nonzero(mask_a & (eids == e), as_tuple=False).flatten()
                    if idx.numel() == 0:
                        continue
                    xs = x.index_select(0, idx)
                    up = F.linear(xs, self.w1_residual[aidx, e])
                    gate = F.linear(xs, self.w3[e])
                    hidden = F.silu(up) * gate
                    y = F.linear(hidden, self.down_proj[e])
                    y = y * gates.index_select(0, idx).unsqueeze(1)
                    out.index_add_(0, idx, y)
        return out


class TorchGroupedBaseMixtralMoE(nn.Module):
    """PyTorch grouped implementation of the base Mixtral MoE MLP."""

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        super().__init__()
        half = int(gate_up_proj.shape[1] // 2)
        self.register_buffer("w1", gate_up_proj[:, :half, :].detach().contiguous())
        self.register_buffer("w3", gate_up_proj[:, half:, :].detach().contiguous())
        self.register_buffer("down_proj", down_proj.detach().contiguous())

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((x.shape[0], self.down_proj.shape[1]), device=x.device, dtype=x.dtype)
        num_experts = int(self.down_proj.shape[0])
        for k in range(expert_ids.shape[1]):
            eids = expert_ids[:, k]
            gates = routing_weights[:, k].to(dtype=x.dtype)
            for e in range(num_experts):
                idx = torch.nonzero(eids == e, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                xs = x.index_select(0, idx)
                up = F.linear(xs, self.w1[e])
                gate = F.linear(xs, self.w3[e])
                hidden = F.silu(up) * gate
                y = F.linear(hidden, self.down_proj[e])
                y = y * gates.index_select(0, idx).unsqueeze(1)
                out.index_add_(0, idx, y)
        return out


def extract_mixtral_moe_tensors(model: torch.nn.Module, *, max_experts: int | None = None) -> tuple[torch.Tensor, torch.Tensor, str]:
    for mod_name, module in model.named_modules():
        if not mod_name.endswith(".mlp.experts"):
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        gate_up = getattr(gate_up, "weight", gate_up)
        down = getattr(down, "weight", down)
        if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
            continue
        if gate_up.ndim != 3 or down.ndim != 3:
            continue
        n = int(gate_up.shape[0])
        if max_experts is not None:
            n = min(n, int(max_experts))
        return (
            gate_up[:n].detach().cpu().to(torch.float32).contiguous(),
            down[:n].detach().cpu().to(torch.float32).contiguous(),
            mod_name,
        )
    raise RuntimeError("could not find Mixtral fused experts tensors")


def try_import_vllm_fused_experts() -> tuple[Callable[..., Any] | None, str]:
    """Resolve the callable vLLM fused experts entry point.

    Recent vLLM versions expose the same-boundary MoE callable as
    fused_experts rather than fused_moe.  This helper deliberately resolves a
    callable function and refuses to return a module object.
    """
    candidates = [
        ("vllm.model_executor.layers.fused_moe.fused_moe", "fused_experts"),
        ("vllm.model_executor.layers.fused_moe", "fused_experts"),
    ]
    errors = []
    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr)
            if callable(fn):
                return fn, f"{module_name}.{attr}"
            errors.append(f"{module_name}.{attr} was found but was not callable")
        except Exception as exc:
            errors.append(f"{module_name}.{attr}: {exc}")
    return None, "; ".join(errors)


def call_vllm_fused_experts_best_effort(
    fn: Callable[..., Any],
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Call vLLM fused_experts across known internal signatures.

    The callable computes the Mixtral fused experts block at the hidden state
    boundary.  The function is internal to vLLM and may change across versions,
    so failures are reported in the output JSON rather than hidden.
    """
    topk_ids = topk_ids.to(dtype=torch.long).contiguous()
    topk_weights = topk_weights.contiguous()
    attempts = [
        lambda: fn(
            hidden_states=hidden,
            w1=gate_up,
            w2=down,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            global_num_experts=int(gate_up.shape[0]),
        ),
        lambda: fn(hidden, gate_up, down, topk_weights, topk_ids, False),
    ]
    errors = []
    for attempt in attempts:
        try:
            out = attempt()
            if isinstance(out, torch.Tensor):
                return out
            raise RuntimeError(f"vLLM fused_experts returned non tensor output {type(out)}")
        except Exception as exc:
            errors.append(str(exc))
    try:
        sig = str(inspect.signature(fn))
    except Exception:
        sig = "<signature unavailable>"
    raise RuntimeError(
        "Could not call vLLM fused_experts. signature=" + sig + " errors=" + " || ".join(errors[-4:])
    )


def make_stage_residual_gate_up(stage_layer: StageMLMaterializedMixtralMoELoRA, gate_up: torch.Tensor, adapter_index: int = 0) -> torch.Tensor:
    """Pack StageML residualized w1 with the original w3 for vLLM fused_experts.

    vLLM expects a Mixtral gate_up tensor with shape experts by two times
    intermediate by hidden.  StageML residualizes only the w1 projection in this
    artifact.  This function builds the Trojan horse bridge tensor that allows
    the StageML residual plan to run on vLLM's optimized fused experts backend.
    """
    residual_gate_up = gate_up.detach().clone().contiguous()
    intermediate = int(residual_gate_up.shape[1] // 2)
    residual_gate_up[:, :intermediate, :] = stage_layer.w1_residual[adapter_index].detach().to(
        device=residual_gate_up.device,
        dtype=residual_gate_up.dtype,
    )
    return residual_gate_up.contiguous()

def time_cuda(fn: Callable[[], Any], *, warmups: int, repeats: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        else:
            import time
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / (len(times) - 1)) ** 0.5 if len(times) > 1 else 0.0
    return {
        "runs": len(times),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--adapter-dirs", nargs="+", required=True)
    ap.add_argument("--prompts-jsonl", required=True)
    ap.add_argument("--out", default="paper_outputs/vllm_layer_bench.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--gpu-memory-gb", type=int, default=82)
    ap.add_argument("--cpu-memory-gb", type=int, default=180)
    ap.add_argument("--offload-folder", default="/data/stageml_h100_run/hf_offload")
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--bench-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--skip-vllm-op", action="store_true")
    args = ap.parse_args()

    AutoModelForCausalLM, AutoTokenizer = _load_optional_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload = Path(args.offload_folder)
    offload.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "device_map": args.device_map,
        "offload_folder": str(offload),
        "offload_state_dict": True,
    }
    if args.device_map == "auto":
        kwargs["max_memory"] = {0: f"{args.gpu_memory_gb}GiB", "cpu": f"{args.cpu_memory_gb}GiB"}

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    rows = _read_prompts(Path(args.prompts_jsonl), args.max_prompts)
    hidden = collect_real_hidden_states(model, tokenizer, rows, layer_index=args.layer_index, device=args.device, max_tokens=args.max_tokens)
    gate_up, down, experts_name = extract_mixtral_moe_tensors(model, max_experts=args.max_experts)
    num_experts = int(gate_up.shape[0])
    intermediate = int(gate_up.shape[1] // 2)
    in_features = int(gate_up.shape[2])
    adapters = [load_expert_lora_adapter(Path(p), num_experts=num_experts, in_features=in_features, out_features=intermediate) for p in args.adapter_dirs]
    expert_ids, routing_weights, gate_name = make_routing(model, hidden, num_experts, args.top_k, args.device)

    # Router logits for direct vLLM fused_moe. Recompute from top-k weights is not exact, so find gate through make_routing's model path indirectly by using selected ids for StageML and by passing a dense approximation to vLLM only as a diagnostic.
    router_logits = torch.zeros((hidden.shape[0], num_experts), dtype=torch.float32)
    router_logits.scatter_(1, expert_ids, routing_weights)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.bench_dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hidden = hidden[:, :in_features].to(device=device, dtype=dtype).contiguous()
    gate_up = gate_up.to(device=device, dtype=dtype).contiguous()
    down = down.to(device=device, dtype=dtype).contiguous()
    expert_ids = expert_ids.to(device=device)
    routing_weights = routing_weights.to(device=device, dtype=dtype)
    router_logits = router_logits.to(device=device, dtype=dtype)
    adapter_ids = [adapters[0].name for _ in range(hidden.shape[0])]
    adapters_b = [MoEAdapterSpec(a.name, a.A.to(device=device, dtype=dtype), a.B.to(device=device, dtype=dtype), a.scaling) for a in adapters]

    stage_layer = StageMLMaterializedMixtralMoELoRA(gate_up, down, adapters_b).to(device)
    base_layer = TorchGroupedBaseMixtralMoE(gate_up, down).to(device)

    result: dict[str, Any] = {
        "model": args.model,
        "experts_name": experts_name,
        "gate_name": gate_name,
        "num_tokens": int(hidden.shape[0]),
        "num_experts": num_experts,
        "top_k": args.top_k,
        "dtype": args.bench_dtype,
        "stage_boundary": "hidden_state_to_moe_mlp_output",
        "environment": _gpu_metadata(),
    }

    with torch.no_grad():
        result["torch_grouped_base_moe"] = time_cuda(lambda: base_layer(hidden, expert_ids, routing_weights), warmups=args.warmups, repeats=args.repeats)
        result["stageml_materialized_moe_lora"] = time_cuda(lambda: stage_layer(hidden, expert_ids, routing_weights, adapter_ids), warmups=args.warmups, repeats=args.repeats)

    if args.skip_vllm_op:
        result["vllm_fused_moe"] = {"status": "skipped", "reason": "--skip-vllm-op was set"}
    else:
        fn, source = try_import_vllm_fused_experts()
        if fn is None:
            result["vllm_fused_moe"] = {"status": "unavailable", "reason": source}
        else:
            try:
                with torch.no_grad():
                    # vLLM expects packed gate_up, down weights, topk weights, and topk ids. This call is internal and version dependent.
                    y_vllm = call_vllm_fused_experts_best_effort(fn, hidden, gate_up, down, routing_weights, expert_ids)
                    y_base = base_layer(hidden, expert_ids, routing_weights)
                    max_err = float((y_vllm.float() - y_base.float()).abs().max().detach().cpu()) if isinstance(y_vllm, torch.Tensor) else None
                    timing = time_cuda(lambda: call_vllm_fused_experts_best_effort(fn, hidden, gate_up, down, routing_weights, expert_ids), warmups=args.warmups, repeats=args.repeats)

                    residual_gate_up = make_stage_residual_gate_up(stage_layer, gate_up, adapter_index=0)
                    y_vllm_residual = call_vllm_fused_experts_best_effort(fn, hidden, residual_gate_up, down, routing_weights, expert_ids)
                    y_stage = stage_layer(hidden, expert_ids, routing_weights, adapter_ids)
                    residual_max_err = float((y_vllm_residual.float() - y_stage.float()).abs().max().detach().cpu())
                    residual_timing = time_cuda(lambda: call_vllm_fused_experts_best_effort(fn, hidden, residual_gate_up, down, routing_weights, expert_ids), warmups=args.warmups, repeats=args.repeats)
                result["vllm_fused_moe"] = {"status": "ok", "source": source, "max_abs_error_vs_torch_grouped_base": max_err, **timing}
                result["stageml_residual_on_vllm_fused_experts"] = {"status": "ok", "source": source, "max_abs_error_vs_stageml_pytorch_residual": residual_max_err, **residual_timing}
            except Exception as exc:
                result["vllm_fused_moe"] = {"status": "call_failed", "source": source, "reason": str(exc)}

    result["claim_boundary"] = (
        "This benchmark isolates the hidden-state MoE layer boundary. vLLM internal fused_moe is version-dependent and normally measures the base MoE block. "
        "StageML materialized_moe_lora includes the accepted LoRA residual in w1. The optional stageml_residual_on_vllm_fused_experts row feeds StageML residualized weights into the vLLM fused experts backend. Use the recorded status fields before making speedup claims."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
