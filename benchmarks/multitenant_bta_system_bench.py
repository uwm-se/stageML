from __future__ import annotations

"""StageML multi tenant binding time analysis benchmark.

This is the main benchmark for the StageML paper direction where the paper asks
whether tenant known adapter computation can be moved out of token time in a
repeated multi tenant MoE LoRA workload.

The benchmark is intentionally request level rather than single layer only. It
builds a stream of tenant requests, applies a memory bounded materialization
policy, measures latency with repeated runs, reports standard deviation, and
reports operation counters that explain why the optimization matters.
"""

import argparse
import importlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from stageml.benchmark_stats import attach_speedups, time_ms, write_json_csv
from stageml.benchmark_env import attach_environment, assert_same_environment, capture_environment
from stageml.h100_guard import require_h100
from stageml.moe_lora_layers import MoEAdapterSpec

try:
    from benchmarks.real_moe_lora_residual_bench import (
        _gpu_metadata,
        _load_optional_transformers,
        _read_prompts,
        collect_real_hidden_states,
        load_expert_lora_adapter,
        make_routing,
    )
except Exception:
    _gpu_metadata = None
    _load_optional_transformers = None
    _read_prompts = None
    collect_real_hidden_states = None
    load_expert_lora_adapter = None
    make_routing = None


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    tenant_id: int
    adapter_id: int
    start: int
    end: int


@dataclass(frozen=True)
class WorkloadPlan:
    requests: list[RequestSpec]
    tenant_request_counts: dict[int, int]
    tenant_to_adapter: dict[int, int]
    materialized_adapter_ids: set[int]
    materialization_policy: str
    estimated_memory_per_adapter_bytes: int
    selected_materialized_memory_bytes: int


def choose_dtype(name: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name}")
    return mapping[name]


def zipf_probs(n: int, alpha: float) -> list[float]:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(weights)
    return [x / total for x in weights]


def sample_tenants(n: int, tenants: int, distribution: str, alpha: float, burst: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if distribution == "uniform":
        return [rng.randrange(tenants) for _ in range(n)]
    if distribution == "zipf":
        return rng.choices(list(range(tenants)), weights=zipf_probs(tenants, alpha), k=n)
    if distribution == "round_robin":
        return [i % tenants for i in range(n)]
    if distribution == "bursty":
        out: list[int] = []
        probs = zipf_probs(tenants, alpha)
        while len(out) < n:
            tenant = rng.choices(list(range(tenants)), weights=probs, k=1)[0]
            out.extend([tenant] * max(1, burst))
        return out[:n]
    raise ValueError(f"unknown tenant distribution {distribution}")


def estimate_residual_memory_bytes(gate_up: torch.Tensor, dtype: torch.dtype) -> int:
    experts = int(gate_up.shape[0])
    intermediate = int(gate_up.shape[1] // 2)
    hidden = int(gate_up.shape[2])
    elem = torch.empty((), dtype=dtype).element_size()
    return experts * intermediate * hidden * elem


def make_workload_plan(
    *,
    num_requests: int,
    num_tenants: int,
    num_adapters: int,
    tokens_available: int,
    tokens_per_request: int,
    distribution: str,
    zipf_alpha: float,
    burst_size: int,
    materialization_policy: str,
    materialize_top_k: int,
    materialize_min_requests: int,
    memory_budget_mb: float,
    memory_per_adapter_bytes: int,
    seed: int,
) -> WorkloadPlan:
    tenants = sample_tenants(num_requests, num_tenants, distribution, zipf_alpha, burst_size, seed)
    tenant_to_adapter = {tenant: tenant % num_adapters for tenant in range(num_tenants)}
    counts = Counter(tenants)
    adapter_counts: Counter[int] = Counter(tenant_to_adapter[t] for t in tenants)

    selected: list[int] = []
    if materialization_policy == "none":
        selected = []
    elif materialization_policy == "all":
        selected = sorted(adapter_counts)
    elif materialization_policy in {"top_k", "memory_budget"}:
        candidates = [
            aid for aid, count in adapter_counts.most_common()
            if count >= materialize_min_requests
        ]
        if materialize_top_k >= 0:
            candidates = candidates[:materialize_top_k]
        if materialization_policy == "top_k":
            selected = candidates
        else:
            used = 0
            budget = int(memory_budget_mb * 1024 * 1024)
            for aid in candidates:
                if used + memory_per_adapter_bytes <= budget:
                    selected.append(aid)
                    used += memory_per_adapter_bytes
    else:
        raise ValueError(f"unknown materialization policy {materialization_policy}")

    if tokens_available < tokens_per_request:
        raise ValueError("tokens_available must be at least tokens_per_request")
    span = max(1, tokens_available - tokens_per_request + 1)
    requests = []
    for rid, tenant in enumerate(tenants):
        start = (rid * tokens_per_request) % span
        requests.append(RequestSpec(rid, tenant, tenant_to_adapter[tenant], start, start + tokens_per_request))

    return WorkloadPlan(
        requests=requests,
        tenant_request_counts=dict(counts),
        tenant_to_adapter=tenant_to_adapter,
        materialized_adapter_ids=set(selected),
        materialization_policy=materialization_policy,
        estimated_memory_per_adapter_bytes=memory_per_adapter_bytes,
        selected_materialized_memory_bytes=len(selected) * memory_per_adapter_bytes,
    )


def make_synthetic_inputs(
    *,
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[MoEAdapterSpec], str, str]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g).to(device=device, dtype=dtype)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, generator=g).to(device=device, dtype=dtype) * 0.02
    down = torch.randn(experts, hidden, intermediate, generator=g).to(device=device, dtype=dtype) * 0.02
    topk_ids = torch.randint(0, experts, (tokens, top_k), generator=g).to(device=device, dtype=torch.long)
    logits = torch.randn(tokens, top_k, generator=g).to(device=device, dtype=dtype)
    topk_weights = torch.softmax(logits.float(), dim=-1).to(device=device, dtype=dtype)
    A = torch.randn(experts, rank, hidden, generator=g).to(device=device, dtype=dtype) * 0.02
    B = torch.randn(experts, intermediate, rank, generator=g).to(device=device, dtype=dtype) * 0.02
    return x, gate_up, down, topk_ids, topk_weights, [MoEAdapterSpec("synthetic", A, B, 1.0 / max(rank, 1))], "synthetic.experts", "synthetic.router"


def extract_mixtral_moe_tensors(model: torch.nn.Module, *, max_experts: int | None) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Extract MoE expert tensors from Mixtral-style fused experts or Qwen-style
    per-expert modules.

    Returned shapes:
      gate_up: [experts, 2 * intermediate, hidden]
      down:    [experts, hidden, intermediate]
    """

    # Case 1: Mixtral fused experts.
    for name, module in model.named_modules():
        if not name.endswith(".mlp.experts"):
            continue

        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        gate_up = getattr(gate_up, "weight", gate_up)
        down = getattr(down, "weight", down)

        if isinstance(gate_up, torch.Tensor) and isinstance(down, torch.Tensor) and gate_up.ndim == 3 and down.ndim == 3:
            n = int(gate_up.shape[0]) if max_experts is None else min(int(gate_up.shape[0]), max_experts)
            return gate_up[:n].detach().cpu().float().contiguous(), down[:n].detach().cpu().float().contiguous(), name

    # Case 2: Generic/Qwen-style expert ModuleList.
    for name, module in model.named_modules():
        lname = name.lower()
        if "experts" not in lname:
            continue

        expert_tuples = []

        for child_name, expert in module.named_children():
            if not child_name.isdigit():
                continue

            gate = getattr(expert, "gate_proj", None)
            up = getattr(expert, "up_proj", None)
            down = getattr(expert, "down_proj", None)

            # Mixtral naming in some implementations.
            if gate is None:
                gate = getattr(expert, "w1", None)
            if up is None:
                up = getattr(expert, "w3", None)
            if down is None:
                down = getattr(expert, "w2", None)

            gate_w = getattr(gate, "weight", gate)
            up_w = getattr(up, "weight", up)
            down_w = getattr(down, "weight", down)

            if not (
                isinstance(gate_w, torch.Tensor)
                and isinstance(up_w, torch.Tensor)
                and isinstance(down_w, torch.Tensor)
                and gate_w.ndim == 2
                and up_w.ndim == 2
                and down_w.ndim == 2
            ):
                continue

            # gate/up: [intermediate, hidden]
            # down:    [hidden, intermediate]
            if gate_w.shape != up_w.shape:
                continue
            if down_w.shape[1] != gate_w.shape[0]:
                continue

            packed = torch.cat([gate_w.detach().cpu().float(), up_w.detach().cpu().float()], dim=0).contiguous()
            down_cpu = down_w.detach().cpu().float().contiguous()
            expert_tuples.append((packed, down_cpu))

        if expert_tuples:
            n = len(expert_tuples) if max_experts is None else min(len(expert_tuples), max_experts)
            gate_up = torch.stack([x[0] for x in expert_tuples[:n]], dim=0).contiguous()
            down = torch.stack([x[1] for x in expert_tuples[:n]], dim=0).contiguous()
            return gate_up, down, name

    raise RuntimeError("could not find supported MoE expert tensors")


def clone_adapters(base: MoEAdapterSpec, count: int, *, device: torch.device, dtype: torch.dtype) -> list[MoEAdapterSpec]:
    adapters: list[MoEAdapterSpec] = []
    for i in range(count):
        scale = 1.0 + 0.001 * i
        adapters.append(
            MoEAdapterSpec(
                name=f"{base.name}_tenant_{i}",
                A=base.A.to(device=device, dtype=dtype).contiguous(),
                B=(base.B.to(device=device, dtype=dtype) * scale).contiguous(),
                scaling=base.scaling,
            )
        )
    return adapters


def expand_adapters(
    loaded: list[MoEAdapterSpec],
    requested_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
) -> tuple[list[MoEAdapterSpec], bool, str]:
    """Build the adapter bank used by the multi tenant workload.

    mode=scale_copy preserves the older single adapter behavior by deriving
    virtual adapters from the first real adapter.  mode=strict_real requires
    the caller to provide enough genuinely independent adapter checkpoints.
    mode=cycle_real uses all provided real adapters and repeats them only when
    more virtual slots are requested than available checkpoints.
    """
    if not loaded:
        raise ValueError("at least one adapter must be loaded")
    if requested_count <= 0:
        raise ValueError("requested adapter count must be positive")
    if mode == "scale_copy":
        if requested_count > len(loaded):
            return clone_adapters(loaded[0], requested_count, device=device, dtype=dtype), len(loaded) == 1, "scale_copy_first_adapter"
        return [MoEAdapterSpec(a.name, a.A.to(device=device, dtype=dtype).contiguous(), a.B.to(device=device, dtype=dtype).contiguous(), a.scaling) for a in loaded[:requested_count]], False, "real_subset"
    if mode == "strict_real":
        if len(loaded) < requested_count:
            raise ValueError(f"strict_real requested {requested_count} adapters but only {len(loaded)} adapter checkpoints were provided")
        return [MoEAdapterSpec(a.name, a.A.to(device=device, dtype=dtype).contiguous(), a.B.to(device=device, dtype=dtype).contiguous(), a.scaling) for a in loaded[:requested_count]], False, "strict_real"
    if mode == "cycle_real":
        out: list[MoEAdapterSpec] = []
        for i in range(requested_count):
            src = loaded[i % len(loaded)]
            out.append(MoEAdapterSpec(f"{src.name}_slot_{i}", src.A.to(device=device, dtype=dtype).contiguous(), src.B.to(device=device, dtype=dtype).contiguous(), src.scaling))
        return out, len(loaded) == 1, "cycle_real"
    raise ValueError(f"unknown adapter expansion mode {mode}")


def try_import_vllm_fused_experts() -> tuple[Callable[..., Any] | None, str]:
    """Resolve vLLM's internal fused_experts callable with useful diagnostics.

    vLLM has moved internal modules across releases. The benchmark only needs
    the fused experts callable, so we try a small set of known module locations
    and return the exact import/attribute errors when resolution fails. This
    prevents a silent ``unavailable`` row with no actionable reason.
    """
    candidates = [
        ("vllm.model_executor.layers.fused_moe.fused_moe", "fused_experts"),
        ("vllm.model_executor.layers.fused_moe", "fused_experts"),
    ]
    errors: list[str] = []
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            errors.append(f"{mod_name}: import failed: {type(exc).__name__}: {exc}")
            continue
        try:
            fn = getattr(mod, attr)
        except Exception as exc:
            exported = ",".join(sorted(x for x in dir(mod) if "fused" in x.lower() or "expert" in x.lower()))
            errors.append(f"{mod_name}.{attr}: missing: {type(exc).__name__}: {exc}; exported={exported}")
            continue
        if callable(fn):
            return fn, f"{mod_name}.{attr}"
        errors.append(f"{mod_name}.{attr}: found but not callable: {type(fn).__name__}")
    return None, "; ".join(errors)


def call_vllm_fused_experts(fn: Callable[..., Any], x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return fn(
        hidden_states=x,
        w1=gate_up.contiguous(),
        w2=down.contiguous(),
        topk_weights=weights.contiguous(),
        topk_ids=ids.long().contiguous(),
        inplace=False,
        global_num_experts=int(gate_up.shape[0]),
    )


class RequestExecutor:
    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, adapters: list[MoEAdapterSpec], materialized_ids: set[int]) -> None:
        self.gate_up = gate_up.contiguous()
        self.down = down.contiguous()
        self.adapters = adapters
        self.materialized_ids = set(materialized_ids)
        self.intermediate = int(gate_up.shape[1] // 2)
        self.w1 = self.gate_up[:, : self.intermediate, :].contiguous()
        self.w3 = self.gate_up[:, self.intermediate :, :].contiguous()
        self.residual_w1: dict[int, torch.Tensor] = {}
        t0 = time.perf_counter()
        for aid in sorted(self.materialized_ids):
            ad = adapters[aid]
            delta = ad.scaling * torch.einsum("eor,eri->eoi", ad.B, ad.A)
            self.residual_w1[aid] = (self.w1 + delta).contiguous()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.compile_time_ms = (time.perf_counter() - t0) * 1000.0

    def dynamic_one(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, adapter_id: int) -> torch.Tensor:
        ad = self.adapters[adapter_id]
        out = torch.zeros((x.shape[0], self.down.shape[1]), device=x.device, dtype=x.dtype)
        for k in range(ids.shape[1]):
            eids = ids[:, k]
            gates = weights[:, k].to(dtype=x.dtype)
            for expert in range(int(self.down.shape[0])):
                index = torch.nonzero(eids == expert, as_tuple=False).flatten()
                if index.numel() == 0:
                    continue
                xs = x.index_select(0, index)
                lora = ad.scaling * (xs @ ad.A[expert].t() @ ad.B[expert].t())
                up = F.linear(xs, self.w1[expert]) + lora
                gate = F.linear(xs, self.w3[expert])
                y = F.linear(F.silu(up) * gate, self.down[expert])
                out.index_add_(0, index, y * gates.index_select(0, index).unsqueeze(1))
        return out

    def bta_one(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, adapter_id: int) -> torch.Tensor:
        if adapter_id not in self.residual_w1:
            return self.dynamic_one(x, ids, weights, adapter_id)
        w1 = self.residual_w1[adapter_id]
        out = torch.zeros((x.shape[0], self.down.shape[1]), device=x.device, dtype=x.dtype)
        for k in range(ids.shape[1]):
            eids = ids[:, k]
            gates = weights[:, k].to(dtype=x.dtype)
            for expert in range(int(self.down.shape[0])):
                index = torch.nonzero(eids == expert, as_tuple=False).flatten()
                if index.numel() == 0:
                    continue
                xs = x.index_select(0, index)
                up = F.linear(xs, w1[expert])
                gate = F.linear(xs, self.w3[expert])
                y = F.linear(F.silu(up) * gate, self.down[expert])
                out.index_add_(0, index, y * gates.index_select(0, index).unsqueeze(1))
        return out

    def residual_gate_up(self, adapter_id: int) -> torch.Tensor:
        packed = self.gate_up.detach().clone().contiguous()
        if adapter_id in self.residual_w1:
            packed[:, : self.intermediate, :] = self.residual_w1[adapter_id]
        else:
            ad = self.adapters[adapter_id]
            delta = ad.scaling * torch.einsum("eor,eri->eoi", ad.B, ad.A)
            packed[:, : self.intermediate, :] = self.w1 + delta
        return packed.contiguous()


def run_sequence(executor: RequestExecutor, plan: WorkloadPlan, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, mode: str, *, vllm_fn: Callable[..., Any] | None = None) -> torch.Tensor:
    checksum = torch.zeros((), device=hidden.device, dtype=torch.float32)
    packed_cache: dict[int, torch.Tensor] = {}
    for req in plan.requests:
        x = hidden[req.start:req.end]
        eids = ids[req.start:req.end]
        wgts = weights[req.start:req.end]
        if mode == "dynamic":
            y = executor.dynamic_one(x, eids, wgts, req.adapter_id)
        elif mode == "bta":
            y = executor.bta_one(x, eids, wgts, req.adapter_id)
        elif mode == "vllm":
            if vllm_fn is None:
                raise RuntimeError("vLLM function was not provided")
            if req.adapter_id not in packed_cache:
                packed_cache[req.adapter_id] = executor.residual_gate_up(req.adapter_id)
            y = call_vllm_fused_experts(vllm_fn, x, packed_cache[req.adapter_id], executor.down, wgts, eids)
        else:
            raise ValueError(mode)
        checksum = checksum + y.float().mean()
    return checksum


def build_counters(plan: WorkloadPlan, *, top_k: int, tokens_per_request: int) -> dict[str, Any]:
    total_routed_token_expert_pairs = len(plan.requests) * tokens_per_request * top_k
    materialized_requests = sum(1 for r in plan.requests if r.adapter_id in plan.materialized_adapter_ids)
    materialized_pairs = materialized_requests * tokens_per_request * top_k
    dynamic_pairs_remaining = total_routed_token_expert_pairs - materialized_pairs
    return {
        "total_requests": len(plan.requests),
        "materialized_requests": materialized_requests,
        "materialized_request_fraction": materialized_requests / max(1, len(plan.requests)),
        "total_routed_token_expert_pairs": total_routed_token_expert_pairs,
        "adapter_lora_pairs_removed_from_token_time": materialized_pairs,
        "adapter_lora_pairs_remaining_dynamic": dynamic_pairs_remaining,
        "adapter_lora_pair_removal_fraction": materialized_pairs / max(1, total_routed_token_expert_pairs),
    }


def generate_latex_summary(result: dict[str, Any], out_json: Path) -> None:
    tex = out_json.with_suffix(".tex")
    rows = []
    for key, label in [
        ("dynamic_runtime_adapters", "Dynamic runtime adapters"),
        ("stageml_bta_materialized_cache", "StageML BTA materialized cache"),
        ("stageml_bta_vllm_fused_backend", "StageML BTA on vLLM fused backend"),
    ]:
        b = result.get(key, {})
        if not isinstance(b, dict) or b.get("status") not in (None, "ok"):
            rows.append(f"{label} & {b.get('status', 'missing')} & & & & & & \\")
            continue
        rows.append(
            f"{label} & {b.get('runs', '')} & {float(b.get('mean_ms', float('nan'))):.3f} & "
            f"{float(b.get('std_ms', float('nan'))):.3f} & {float(b.get('p50_ms', float('nan'))):.3f} & "
            f"{float(b.get('p95_ms', float('nan'))):.3f} & {float(b.get('speedup_over_dynamic_p50', 1.0)):.2f} & "
            f"{float(result.get('selected_materialized_memory_mb', 0.0)):.1f} \\")
    body = "\n".join(rows)
    tex.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Multi tenant request level benchmark. Runs is the number of measured repetitions after warmup. Std is the sample standard deviation over measured repetitions. Speedup uses P50 latency relative to dynamic runtime adapter execution.}\n"
        "\\label{tab:multitenant-bta-main}\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "System & Runs & Mean ms & Std ms & P50 ms & P95 ms & Speedup & Mem MB \\\\\n"
        "\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
        "\\paragraph{How to read the table.} The dynamic row is the baseline that recomputes LoRA factors during token execution. The StageML BTA row uses the tenant request distribution to materialize hot adapter residuals before token execution and falls back to the dynamic path for cold adapters. The vLLM backend row is reported only when the installed vLLM package exposes the internal fused experts callable.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--adapter-dirs", nargs="+", default=[])
    ap.add_argument("--prompts-jsonl", default="/data/stageml_h100_run/data/real_trace.jsonl")
    ap.add_argument("--out", default="paper_outputs/multitenant_bta_system_bench.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--model-load-mode", default="bf16_offload", choices=["bf16_offload", "8bit", "4bit"], help="base model loading mode for real benchmark")
    ap.add_argument("--require-full-gpu-resident", action="store_true", help="fail if Hugging Face places any model shard on CPU or disk")
    ap.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"], help="bitsandbytes 4-bit quantization type")
    ap.add_argument("--gpu-memory-gb", type=int, default=82)
    ap.add_argument("--cpu-memory-gb", type=int, default=180)
    ap.add_argument("--offload-folder", default="/data/stageml_h100_run/hf_offload")
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num-tenants", type=int, default=16)
    ap.add_argument("--virtual-adapters", type=int, default=8)
    ap.add_argument("--adapter-expansion-mode", default="scale_copy", choices=["scale_copy", "cycle_real", "strict_real"], help="scale_copy preserves old single adapter virtual adapter behavior; strict_real requires enough independent adapter dirs; cycle_real repeats provided real adapters when needed")
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--tokens-per-request", type=int, default=16)
    ap.add_argument("--tenant-distribution", default="zipf", choices=["zipf", "uniform", "round_robin", "bursty"])
    ap.add_argument("--zipf-alpha", type=float, default=1.2)
    ap.add_argument("--burst-size", type=int, default=8)
    ap.add_argument("--materialization-policy", default="memory_budget", choices=["none", "all", "top_k", "memory_budget"])
    ap.add_argument("--materialize-top-k", type=int, default=4)
    ap.add_argument("--materialize-min-requests", type=int, default=2)
    ap.add_argument("--memory-budget-mb", type=float, default=2048.0)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--enable-vllm-backend", action="store_true")
    ap.add_argument("--synthetic-hidden-size", type=int, default=256)
    ap.add_argument("--synthetic-intermediate-size", type=int, default=512)
    ap.add_argument("--synthetic-rank", type=int, default=8)
    ap.add_argument("--require-h100", action="store_true", help="fail before benchmarking unless the active CUDA device is an H100 class GPU")
    args = ap.parse_args()

    if args.require_h100:
        require_h100()
    dtype = choose_dtype(args.dtype)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    row_environment = capture_environment(str(device))

    if args.synthetic:
        hidden, gate_up, down, ids, weights, adapters_loaded, experts_name, gate_name = make_synthetic_inputs(
            tokens=max(args.max_tokens, args.num_requests * args.tokens_per_request),
            hidden=args.synthetic_hidden_size,
            intermediate=args.synthetic_intermediate_size,
            experts=args.max_experts,
            top_k=args.top_k,
            rank=args.synthetic_rank,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
        virtual_from_single = True
    else:
        if _load_optional_transformers is None:
            raise SystemExit("real benchmark helpers are unavailable")
        if not args.adapter_dirs:
            raise SystemExit("provide at least one adapter dir or use --synthetic")
        AutoModelForCausalLM, AutoTokenizer = _load_optional_transformers()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        offload = Path(args.offload_folder)
        offload.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "device_map": args.device_map,
            "offload_folder": str(offload),
            "offload_state_dict": True,
        }
        quantization_label = "none_bf16"
        if args.model_load_mode in {"8bit", "4bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError("Install bitsandbytes support with: pip install -U bitsandbytes accelerate") from exc
            if args.model_load_mode == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                quantization_label = "bitsandbytes_8bit"
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=True,
                )
                quantization_label = f"bitsandbytes_4bit_{args.bnb_4bit_quant_type}"
        if str(args.device_map).lower() in {"cuda", "cuda:0", "single_gpu", "gpu0"}:
            kwargs["device_map"] = {"": 0}
            kwargs.pop("max_memory", None)
            kwargs.pop("offload_folder", None)
            kwargs["offload_state_dict"] = False
        elif args.device_map == "auto":
            kwargs["max_memory"] = {0: f"{args.gpu_memory_gb}GiB", "cpu": f"{args.cpu_memory_gb}GiB"}
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
        hf_device_map = getattr(model, "hf_device_map", {}) or {}
        bad_device_map = {str(k): str(v) for k, v in hf_device_map.items() if str(v).lower() in {"cpu", "disk"} or "cpu" in str(v).lower() or "disk" in str(v).lower()}
        if args.require_full_gpu_resident and bad_device_map:
            raise RuntimeError(f"Model is not fully GPU resident. CPU/disk mapped modules: {bad_device_map}")
        try:
            model_memory_footprint_bytes = int(model.get_memory_footprint())
        except Exception:
            model_memory_footprint_bytes = None
        rows = _read_prompts(Path(args.prompts_jsonl), args.max_prompts)
        hidden = collect_real_hidden_states(model, tokenizer, rows, layer_index=args.layer_index, device=str(device), max_tokens=args.max_tokens)
        gate_up, down, experts_name = extract_mixtral_moe_tensors(model, max_experts=args.max_experts)
        num_experts = int(gate_up.shape[0])
        inter = int(gate_up.shape[1] // 2)
        in_features = int(gate_up.shape[2])
        adapters_loaded = []
        for path in args.adapter_dirs:
            path_s = str(path)
            if path_s.lower() in {"synthetic", "synthetic_lora"} or path_s.lower().startswith("synthetic:"):
                rank = int(os.environ.get("SYNTHETIC_LORA_RANK", "8"))
                g = torch.Generator(device="cpu").manual_seed(int(os.environ.get("SYNTHETIC_LORA_SEED", "1234")))
                A = torch.randn(num_experts, rank, in_features, generator=g, dtype=torch.float32) * 0.02
                B = torch.randn(num_experts, inter, rank, generator=g, dtype=torch.float32) * 0.02
                adapters_loaded.append(MoEAdapterSpec(name=f"synthetic_rank_{rank}", A=A, B=B, scaling=1.0 / max(rank, 1)))
                print("using_synthetic_lora_adapter", "rank", rank, "num_experts", num_experts, "in_features", in_features, "out_features", inter)
            else:
                adapters_loaded.append(
                    load_expert_lora_adapter(Path(path), num_experts=num_experts, in_features=in_features, out_features=inter)
                )
        ids, weights, gate_name = make_routing(model, hidden, num_experts, args.top_k, str(device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hidden = hidden[:, :in_features].to(device=device, dtype=dtype).contiguous()
        gate_up = gate_up.to(device=device, dtype=dtype).contiguous()
        down = down.to(device=device, dtype=dtype).contiguous()
        ids = ids.to(device=device, dtype=torch.long).contiguous()
        weights = weights.to(device=device, dtype=dtype).contiguous()
        adapters_loaded = [
            MoEAdapterSpec(a.name, a.A.to(device=device, dtype=dtype).contiguous(), a.B.to(device=device, dtype=dtype).contiguous(), a.scaling)
            for a in adapters_loaded
        ]
        virtual_from_single = len(adapters_loaded) == 1 and args.virtual_adapters > 1

    adapters, virtual_from_single, adapter_source_mode = expand_adapters(
        adapters_loaded,
        args.virtual_adapters,
        device=device,
        dtype=dtype,
        mode=args.adapter_expansion_mode,
    )

    memory_per_adapter = estimate_residual_memory_bytes(gate_up, dtype)
    plan = make_workload_plan(
        num_requests=args.num_requests,
        num_tenants=args.num_tenants,
        num_adapters=len(adapters),
        tokens_available=int(hidden.shape[0]),
        tokens_per_request=args.tokens_per_request,
        distribution=args.tenant_distribution,
        zipf_alpha=args.zipf_alpha,
        burst_size=args.burst_size,
        materialization_policy=args.materialization_policy,
        materialize_top_k=args.materialize_top_k,
        materialize_min_requests=args.materialize_min_requests,
        memory_budget_mb=args.memory_budget_mb,
        memory_per_adapter_bytes=memory_per_adapter,
        seed=args.seed,
    )
    executor = RequestExecutor(gate_up, down, adapters, plan.materialized_adapter_ids)

    result: dict[str, Any] = {
        "benchmark": "multitenant_bta_system_bench",
        "research_question": "Does binding time analysis reduce repeated multi tenant MoE LoRA request latency by moving tenant known adapter computation out of token time",
        "model": args.model if not args.synthetic else "synthetic",
        "experts_name": experts_name,
        "gate_name": gate_name,
        "dtype": args.dtype,
        "device": str(device),
        "model_load_mode": args.model_load_mode if not args.synthetic else "synthetic",
        "base_model_quantization": quantization_label if not args.synthetic else "synthetic",
        "require_full_gpu_resident": bool(args.require_full_gpu_resident) if not args.synthetic else False,
        "hf_device_map": {str(k): str(v) for k, v in (hf_device_map.items() if not args.synthetic else [])},
        "bad_device_map": bad_device_map if not args.synthetic else {},
        "model_memory_footprint_bytes": model_memory_footprint_bytes if not args.synthetic else None,
        "num_tenants": args.num_tenants,
        "num_adapters": len(adapters),
        "num_requests": args.num_requests,
        "tokens_per_request": args.tokens_per_request,
        "total_request_tokens": args.num_requests * args.tokens_per_request,
        "num_experts": int(gate_up.shape[0]),
        "top_k": args.top_k,
        "tenant_distribution": args.tenant_distribution,
        "zipf_alpha": args.zipf_alpha,
        "burst_size": args.burst_size,
        "warmups": args.warmups,
        "measured_runs": args.repeats,
        "materialization_policy": args.materialization_policy,
        "memory_budget_mb": args.memory_budget_mb,
        "estimated_memory_per_adapter_mb": memory_per_adapter / (1024 * 1024),
        "selected_materialized_memory_mb": plan.selected_materialized_memory_bytes / (1024 * 1024),
        "materialized_adapter_ids": sorted(plan.materialized_adapter_ids),
        "materialized_adapter_count": len(plan.materialized_adapter_ids),
        "tenant_request_counts": plan.tenant_request_counts,
        "virtual_adapters_from_single_real_adapter": virtual_from_single,
        "adapter_expansion_mode": args.adapter_expansion_mode,
        "adapter_source_mode": adapter_source_mode,
        "adapter_names": [a.name for a in adapters],
        "compile_time_ms": executor.compile_time_ms,
        "workload_counters": build_counters(plan, top_k=args.top_k, tokens_per_request=args.tokens_per_request),
        "environment": row_environment,
        "claim_boundary": "Request level benchmark over a generated tenant request stream using real Mixtral hidden states when synthetic is false. It is not an HTTP server benchmark.",
    }

    result["dynamic_runtime_adapters"] = attach_environment(time_ms(
        lambda: run_sequence(executor, plan, hidden, ids, weights, "dynamic"),
        warmups=args.warmups,
        repeats=args.repeats,
    ), row_environment)
    result["stageml_bta_materialized_cache"] = attach_environment(time_ms(
        lambda: run_sequence(executor, plan, hidden, ids, weights, "bta"),
        warmups=args.warmups,
        repeats=args.repeats,
    ), row_environment)

    if args.enable_vllm_backend:
        fn, source = try_import_vllm_fused_experts()
        if fn is None:
            result["stageml_bta_vllm_fused_backend"] = {"status": "unavailable", "reason": source}
        else:
            try:
                stats = time_ms(
                    lambda: run_sequence(executor, plan, hidden, ids, weights, "vllm", vllm_fn=fn),
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                stats["status"] = "ok"
                stats["source"] = source
                attach_environment(stats, row_environment)
                result["stageml_bta_vllm_fused_backend"] = stats
            except Exception as exc:
                result["stageml_bta_vllm_fused_backend"] = {"status": "call_failed", "reason": str(exc), "source": source}
    else:
        result["stageml_bta_vllm_fused_backend"] = {"status": "skipped", "reason": "enable vllm backend was not set"}

    systems = ["dynamic_runtime_adapters", "stageml_bta_materialized_cache", "stageml_bta_vllm_fused_backend"]
    assert_same_environment(result, systems)
    result["same_environment_assertion"] = "passed for all ok rows in this table"
    attach_speedups(result, "dynamic_runtime_adapters", systems)

    counters = result["workload_counters"]
    materialized = result["stageml_bta_materialized_cache"]
    if isinstance(materialized, dict):
        materialized["materialized_request_fraction"] = counters["materialized_request_fraction"]
        materialized["adapter_lora_pair_removal_fraction"] = counters["adapter_lora_pair_removal_fraction"]
        materialized["compile_time_ms"] = executor.compile_time_ms
        p50_gain = materialized.get("speedup_over_dynamic_p50", float("nan"))
        result["main_takeaway"] = (
            f"StageML materialized {len(plan.materialized_adapter_ids)} adapters, removed "
            f"{100.0 * counters['adapter_lora_pair_removal_fraction']:.2f} percent of adapter LoRA routed token expert pairs from token time, "
            f"and achieved {float(p50_gain):.3f}x P50 speedup over dynamic runtime adapters."
        )

    out_json, _ = write_json_csv(
        result,
        args.out,
        systems=systems,
        extra_columns=["materialized_request_fraction", "adapter_lora_pair_removal_fraction", "compile_time_ms"],
    )
    generate_latex_summary(result, out_json)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out_json}")
    print(f"wrote {out_json.with_suffix('.csv')}")
    print(f"wrote {out_json.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
