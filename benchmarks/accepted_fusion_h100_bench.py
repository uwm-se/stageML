from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stageml.moe_ir import MoEAdapterMeta
from stageml.moe_lora_layers import (
    DynamicMoELoRALayer,
    MaterializedMoELoRALayer,
    MoEAdapterSpec,
)
from stageml.moe_plan_export import write_plan
from stageml.residual_planner import AdapterQuantInfo, PlannerConfig, choose_residual_plan

from benchmarks.real_moe_lora_residual_bench import (
    _gpu_metadata,
    _load_optional_transformers,
    _read_prompts,
    collect_real_hidden_states,
    extract_expert_weight,
    load_expert_lora_adapter,
    make_routing,
    percentile,
)


class AcceptedFusedMaterializedMoELoRA(nn.Module):
   

    def __init__(self, expert_weight: torch.Tensor, adapters: list[MoEAdapterSpec]):
        super().__init__()
        self.adapter_names = [a.name for a in adapters]
        self.name_to_index = {n: i for i, n in enumerate(self.adapter_names)}

        merged = []
        for adapter in adapters:
            delta = adapter.scaling * torch.einsum(
                "eor,eri->eoi",
                adapter.B.detach(),
                adapter.A.detach(),
            )
            merged.append(expert_weight.detach() + delta)

        self.register_buffer("merged_weight", torch.stack(merged, dim=0).contiguous())

    def forward(self, x, expert_ids, routing_weights, adapter_ids):
        out = torch.zeros(
            (x.shape[0], self.merged_weight.shape[-2]),
            device=x.device,
            dtype=x.dtype,
        )

        adapter_index = torch.tensor(
            [self.name_to_index[str(a)] for a in adapter_ids],
            device=x.device,
            dtype=torch.long,
        )

        num_experts = self.merged_weight.shape[1]

        for k in range(expert_ids.shape[1]):
            eids = expert_ids[:, k]
            gates = routing_weights[:, k].to(dtype=x.dtype)

            for aidx in range(len(self.adapter_names)):
                mask_a = adapter_index == aidx

                for e in range(num_experts):
                    idx = torch.nonzero(mask_a & (eids == e), as_tuple=False).flatten()
                    if idx.numel() == 0:
                        continue

                    x_sel = x.index_select(0, idx)
                    y = F.linear(x_sel, self.merged_weight[aidx, e])
                    y = y * gates.index_select(0, idx).unsqueeze(1)
                    out.index_add_(0, idx, y)

        return out


def move_adapter(adapter: MoEAdapterSpec, device: torch.device, dtype: torch.dtype) -> MoEAdapterSpec:
    return MoEAdapterSpec(
        name=adapter.name,
        A=adapter.A.to(device=device, dtype=dtype).contiguous(),
        B=adapter.B.to(device=device, dtype=dtype).contiguous(),
        scaling=adapter.scaling,
    )


def time_layer(layer, x, expert_ids, routing_weights, adapter_ids, repeats: int):
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = layer(x, expert_ids, routing_weights, adapter_ids)
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / (len(times) - 1)) ** 0.5 if len(times) > 1 else 0.0
    return {
        "runs": len(times),
        "p50_ms": percentile(times, 50) * 1000.0,
        "p95_ms": percentile(times, 95) * 1000.0,
        "mean_ms": mean * 1000.0,
        "std_ms": std * 1000.0,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
    }


def load_vllm_p50(path: Path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    return float(rows[0]["p50_ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--adapter-dirs", nargs="+", required=True)
    ap.add_argument("--prompts-jsonl", required=True)
    ap.add_argument("--out-dir", default="paper_outputs/accepted_fusion")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--gpu-memory-gb", type=int, default=82)
    ap.add_argument("--cpu-memory-gb", type=int, default=180)
    ap.add_argument("--offload-folder", default="/data/stageml_h100_run/hf_offload")
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-experts", type=int, default=8)
    ap.add_argument("--memory-budget-mb", type=float, default=2048.0)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--bench-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--vllm-csv", default="paper_outputs/vllm_lora_baseline.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    AutoModelForCausalLM, AutoTokenizer = _load_optional_transformers()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload_folder = Path(args.offload_folder)
    offload_folder.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "device_map": args.device_map,
        "offload_folder": str(offload_folder),
        "offload_state_dict": True,
    }

    if args.device_map == "auto":
        model_kwargs["max_memory"] = {
            0: f"{args.gpu_memory_gb}GiB",
            "cpu": f"{args.cpu_memory_gb}GiB",
        }

    print("loading high precision model")
    print(json.dumps({k: str(v) for k, v in model_kwargs.items()}, indent=2))

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    rows = _read_prompts(Path(args.prompts_jsonl), args.max_prompts)

    hidden = collect_real_hidden_states(
        model,
        tokenizer,
        rows,
        layer_index=args.layer_index,
        device=args.device,
        max_tokens=args.max_tokens,
    )

    expert_weight, expert_name = extract_expert_weight(
        model,
        max_experts=args.max_experts,
    )

    num_experts, out_features, in_features = expert_weight.shape

    adapters = []
    adapter_load_modes = {}
    for p in args.adapter_dirs:
        spec = load_expert_lora_adapter(
            Path(p),
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
        )
        adapters.append(spec)
        adapter_load_modes[spec.name] = "expert_specific"

    names = {a.name for a in adapters}

    encoded = tokenizer(
        [r["prompt"] for r in rows],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_tokens,
    )

    adapter_ids = []
    for i, row in enumerate(rows):
        adapter_name = row.get("adapter") or adapters[0].name
        if adapter_name not in names:
            adapter_name = adapters[0].name
        count = int(encoded["attention_mask"][i].sum().item())
        adapter_ids.extend([adapter_name] * count)

    adapter_ids = adapter_ids[: hidden.shape[0]]

    expert_ids, routing_weights, gate_name = make_routing(
        model,
        hidden,
        num_experts,
        args.top_k,
        args.device,
    )

    print("unloading model before fused benchmark")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hidden = hidden[:, :in_features].to(torch.float32).contiguous()
    expert_weight = expert_weight.to(torch.float32).contiguous()

    print("running CPU exact correctness check")
    dyn_cpu = DynamicMoELoRALayer(expert_weight, adapters)
    mat_cpu = MaterializedMoELoRALayer(expert_weight, adapters)

    with torch.no_grad():
        y_dyn = dyn_cpu(hidden, expert_ids, routing_weights, adapter_ids)
        y_mat = mat_cpu(hidden, expert_ids, routing_weights, adapter_ids)

    max_error = float((y_dyn - y_mat).abs().max().item())

    del dyn_cpu, mat_cpu, y_dyn, y_mat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.bench_dtype == "bf16":
        bench_dtype = torch.bfloat16
    elif args.bench_dtype == "fp16":
        bench_dtype = torch.float16
    else:
        bench_dtype = torch.float32

    bench_device = torch.device(args.device)

    hidden_b = hidden.to(device=bench_device, dtype=bench_dtype).contiguous()
    expert_weight_b = expert_weight.to(device=bench_device, dtype=bench_dtype).contiguous()
    expert_ids_b = expert_ids.to(device=bench_device)
    routing_weights_b = routing_weights.to(device=bench_device, dtype=bench_dtype)
    adapters_b = [move_adapter(a, bench_device, bench_dtype) for a in adapters]

    print("benchmarking dynamic reference")
    dyn = DynamicMoELoRALayer(expert_weight_b, adapters_b)
    dynamic_time = time_layer(
        dyn,
        hidden_b,
        expert_ids_b,
        routing_weights_b,
        adapter_ids,
        args.repeats,
    )

    del dyn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("benchmarking accepted fused materialized residual")
    fused = AcceptedFusedMaterializedMoELoRA(expert_weight_b, adapters_b)
    fused_time = time_layer(
        fused,
        hidden_b,
        expert_ids_b,
        routing_weights_b,
        adapter_ids,
        args.repeats,
    )

    counts = {name: adapter_ids.count(name) for name in names}
    metas = [
        MoEAdapterMeta(
            a.name,
            a.rank,
            a.num_experts,
            a.in_features,
            a.out_features,
            dtype="bf16" if args.bench_dtype == "bf16" else "fp16",
            request_count=counts[a.name],
        )
        for a in adapters
    ]

    qinfo = {
        a.name: AdapterQuantInfo(epsilon=0.0, safe=True)
        for a in adapters
    }

    plan = choose_residual_plan(
        metas,
        config=PlannerConfig(
            memory_budget_mb=args.memory_budget_mb,
            theta=0.0,
            dtype="bf16" if args.bench_dtype == "bf16" else "fp16",
        ),
        quant_info=qinfo,
    )

    plan_path = out_dir / "accepted_residual_plan.json"
    write_plan(plan, plan_path)

    vllm_p50 = load_vllm_p50(Path(args.vllm_csv))
    fused_p50 = fused_time["p50_ms"]

    result = {
        "model": args.model,
        "expert_weight_name": expert_name,
        "gate_name": gate_name,
        "num_tokens": int(hidden.shape[0]),
        "num_experts": int(num_experts),
        "top_k": int(args.top_k),
        "adapter_load_modes": adapter_load_modes,
        "adapter_counts": counts,
        "exact_accept_mode": True,
        "quantization": {
            name: {"epsilon": 0.0, "safe": True, "decision": "accept"}
            for name in names
        },
        "plan_summary": plan.by_kind(),
        "plan_path": str(plan_path),
        "max_abs_error_dynamic_vs_materialized": max_error,
        "dynamic_reference": dynamic_time,
        "accepted_fused_materialized": fused_time,
        "vllm_p50_ms": vllm_p50,
        "stage_fused_vs_vllm_speedup": (vllm_p50 / fused_p50) if vllm_p50 else None,
        "environment": _gpu_metadata(),
    }

    result_path = out_dir / "accepted_fusion_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    table_path = out_dir / "table_accepted_fusion_vs_vllm.tex"
    table_path.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Accepted StageML residualization versus vLLM baseline}\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "System & P50 ms & P95 ms \\\\\n"
        "\\midrule\n"
        f"StageML dynamic reference & {dynamic_time['p50_ms']:.3f} & {dynamic_time['p95_ms']:.3f} \\\\\n"
        f"StageML accepted fused residual & {fused_time['p50_ms']:.3f} & {fused_time['p95_ms']:.3f} \\\\\n"
        + (
            f"vLLM LoRA serving baseline & {vllm_p50:.3f} & -- \\\\\n"
            if vllm_p50 is not None
            else ""
        )
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    claim_path = out_dir / "accepted_fusion_claim.md"
    lines = []
    lines.append("# Accepted fusion benchmark claim status\n")
    lines.append(f"- Plan summary: `{plan.by_kind()}`")
    lines.append("- Quantization decision: `accept`")
    lines.append("- Epsilon: `0.0`")
    lines.append(f"- StageML accepted fused P50 ms: `{fused_time['p50_ms']:.3f}`")
    lines.append(f"- StageML accepted fused P95 ms: `{fused_time['p95_ms']:.3f}`")
    lines.append(f"- Dynamic reference P50 ms: `{dynamic_time['p50_ms']:.3f}`")
    lines.append(f"- Max abs error: `{max_error}`")
    if vllm_p50 is not None:
        lines.append(f"- vLLM P50 ms: `{vllm_p50:.3f}`")
        lines.append(f"- StageML fused speedup over vLLM P50: `{vllm_p50 / fused_p50:.3f}x`")
        if fused_p50 < vllm_p50:
            lines.append("- Result: StageML accepted fused residual path is faster than the recorded vLLM P50 baseline.")
        else:
            lines.append("- Result: StageML accepted fused residual path is not faster than the recorded vLLM P50 baseline.")
    lines.append("")
    lines.append(
        "Interpretation note: this is a hidden state MoE LoRA residualization fragment benchmark. "
        "The vLLM number is an OpenAI compatible serving baseline. Use care when wording the paper."
    )
    claim_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    print(claim_path.read_text())


if __name__ == "__main__":
    main()
