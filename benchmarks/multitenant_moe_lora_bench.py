from __future__ import annotations

"""Multi tenant MoE LoRA benchmark for StageML.

This benchmark is designed to answer one clear systems question.

Does binding time analysis help when tenant identity and adapter weights are
known before token execution, but MoE routing and hidden states remain dynamic.

The benchmark compares three request level paths.

1. dynamic_stage_moe_lora
   The baseline executes the LoRA factors dynamically for every routed token.

2. stageml_materialized_hot_tenants
   StageML materializes hot tenant adapter residuals into expert w1 weights and
   falls back to dynamic LoRA for cold tenants.

3. stageml_residual_on_vllm_fused_experts
   Optional. For hot tenants, StageML feeds residualized expert weights into
   vLLM's internal fused_experts backend. This path requires running from a vLLM
   environment and is reported as unavailable or call_failed if the internal API
   cannot be used.

The JSON output reports mean, P50, P95, standard deviation, minimum, maximum,
number of measured runs, compile time, materialized memory, and speedups.
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


@dataclass
class Workload:
    requests: list[RequestSpec]
    tenant_request_counts: dict[int, int]
    materialized_adapter_ids: set[int]
    tenant_to_adapter: dict[int, int]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(values[lo])
    weight = rank - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def summarize_times(times: list[float]) -> dict[str, float | int]:
    if not times:
        return {
            "runs": 0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
        }
    mean = sum(times) / len(times)
    if len(times) > 1:
        var = sum((x - mean) ** 2 for x in times) / (len(times) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "runs": len(times),
        "mean_ms": float(mean),
        "std_ms": float(std),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
    }


def time_workload(fn: Callable[[], Any], *, warmups: int, repeats: int) -> dict[str, float | int]:
    with torch.no_grad():
        for _ in range(warmups):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times: list[float] = []
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
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1000.0)
    return summarize_times(times)


def build_zipf_probs(num_tenants: int, alpha: float) -> list[float]:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(num_tenants)]
    total = sum(weights)
    return [w / total for w in weights]


def sample_tenants(num_requests: int, num_tenants: int, distribution: str, alpha: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    if distribution == "uniform":
        return [rng.randrange(num_tenants) for _ in range(num_requests)]
    if distribution == "zipf":
        probs = build_zipf_probs(num_tenants, alpha)
        population = list(range(num_tenants))
        return rng.choices(population, weights=probs, k=num_requests)
    if distribution == "round_robin":
        return [i % num_tenants for i in range(num_requests)]
    raise ValueError(f"unknown tenant distribution {distribution}")


def make_request_workload(
    *,
    num_requests: int,
    num_tenants: int,
    num_adapters: int,
    tokens_available: int,
    tokens_per_request: int,
    distribution: str,
    zipf_alpha: float,
    materialize_top_k: int,
    materialize_min_requests: int,
    seed: int,
) -> Workload:
    tenants = sample_tenants(num_requests, num_tenants, distribution, zipf_alpha, seed)
    tenant_to_adapter = {t: t % num_adapters for t in range(num_tenants)}
    counts = Counter(tenants)
    hot_tenants = [t for t, c in counts.most_common() if c >= materialize_min_requests]
    if materialize_top_k >= 0:
        hot_tenants = hot_tenants[:materialize_top_k]
    materialized_adapter_ids = {tenant_to_adapter[t] for t in hot_tenants}

    if tokens_available < tokens_per_request:
        raise ValueError("tokens_available must be at least tokens_per_request")
    requests: list[RequestSpec] = []
    span = max(1, tokens_available - tokens_per_request + 1)
    for rid, tenant in enumerate(tenants):
        start = (rid * tokens_per_request) % span
        end = start + tokens_per_request
        requests.append(RequestSpec(rid, tenant, tenant_to_adapter[tenant], start, end))

    return Workload(
        requests=requests,
        tenant_request_counts=dict(counts),
        materialized_adapter_ids=materialized_adapter_ids,
        tenant_to_adapter=tenant_to_adapter,
    )


def clone_virtual_adapters(base: MoEAdapterSpec, count: int, *, dtype: torch.dtype, device: torch.device) -> list[MoEAdapterSpec]:
    adapters = []
    for idx in range(count):
        # Deterministic small scale change creates distinct tenant adapters while
        # preserving the tensor shape and rank of the real adapter.
        scale = 1.0 + 0.002 * idx
        adapters.append(
            MoEAdapterSpec(
                name=f"{base.name}__tenant{idx}",
                A=base.A.to(device=device, dtype=dtype).contiguous(),
                B=(base.B.to(device=device, dtype=dtype) * scale).contiguous(),
                scaling=base.scaling,
            )
        )
    return adapters


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


def make_synthetic_inputs(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[MoEAdapterSpec], str, str]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    hidden = torch.randn(tokens, hidden_size, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    gate_up = torch.randn(num_experts, 2 * intermediate_size, hidden_size, generator=g, dtype=torch.float32).to(device=device, dtype=dtype) * 0.02
    down = torch.randn(num_experts, hidden_size, intermediate_size, generator=g, dtype=torch.float32).to(device=device, dtype=dtype) * 0.02
    expert_ids = torch.randint(0, num_experts, (tokens, top_k), generator=g, dtype=torch.long).to(device)
    routing_logits = torch.randn(tokens, top_k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    routing_weights = torch.softmax(routing_logits, dim=-1).to(dtype=dtype)
    A = torch.randn(num_experts, rank, hidden_size, generator=g, dtype=torch.float32).to(device=device, dtype=dtype) * 0.02
    B = torch.randn(num_experts, intermediate_size, rank, generator=g, dtype=torch.float32).to(device=device, dtype=dtype) * 0.02
    adapters = [MoEAdapterSpec("synthetic_adapter", A, B, 1.0 / max(rank, 1))]
    return hidden, gate_up, down, expert_ids, routing_weights, adapters, "synthetic.experts", "synthetic.gate"


def try_import_vllm_fused_experts() -> tuple[Callable[..., Any] | None, str]:
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


def call_vllm_fused_experts(
    fn: Callable[..., Any],
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
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
    raise RuntimeError("could not call vLLM fused_experts. errors=" + " || ".join(errors[-4:]))


class MultiTenantMoELoRAExecutor:
    def __init__(
        self,
        *,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        adapters: list[MoEAdapterSpec],
        materialized_adapter_ids: set[int],
    ) -> None:
        self.gate_up = gate_up.contiguous()
        self.down = down.contiguous()
        self.adapters = adapters
        self.materialized_adapter_ids = set(materialized_adapter_ids)
        self.intermediate = int(gate_up.shape[1] // 2)
        self.w1 = self.gate_up[:, : self.intermediate, :].contiguous()
        self.w3 = self.gate_up[:, self.intermediate :, :].contiguous()
        self.residual_w1: dict[int, torch.Tensor] = {}
        t0 = time.perf_counter()
        for aid in sorted(self.materialized_adapter_ids):
            ad = self.adapters[aid]
            delta = ad.scaling * torch.einsum("eor,eri->eoi", ad.B, ad.A)
            self.residual_w1[aid] = (self.w1 + delta).contiguous()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.compile_time_ms = (time.perf_counter() - t0) * 1000.0

    @property
    def materialized_memory_bytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.residual_w1.values()))

    def dynamic_request(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor, adapter_id: int) -> torch.Tensor:
        adapter = self.adapters[adapter_id]
        out = torch.zeros((x.shape[0], self.down.shape[1]), device=x.device, dtype=x.dtype)
        num_experts = int(self.down.shape[0])
        for k in range(expert_ids.shape[1]):
            eids = expert_ids[:, k]
            gates = routing_weights[:, k].to(dtype=x.dtype)
            for e in range(num_experts):
                idx = torch.nonzero(eids == e, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                xs = x.index_select(0, idx)
                up = F.linear(xs, self.w1[e])
                lora = adapter.scaling * (xs @ adapter.A[e].t() @ adapter.B[e].t())
                gate = F.linear(xs, self.w3[e])
                hidden = F.silu(up + lora) * gate
                y = F.linear(hidden, self.down[e])
                out.index_add_(0, idx, y * gates.index_select(0, idx).unsqueeze(1))
        return out

    def materialized_or_dynamic_request(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor, adapter_id: int) -> torch.Tensor:
        if adapter_id not in self.residual_w1:
            return self.dynamic_request(x, expert_ids, routing_weights, adapter_id)
        w1_residual = self.residual_w1[adapter_id]
        out = torch.zeros((x.shape[0], self.down.shape[1]), device=x.device, dtype=x.dtype)
        num_experts = int(self.down.shape[0])
        for k in range(expert_ids.shape[1]):
            eids = expert_ids[:, k]
            gates = routing_weights[:, k].to(dtype=x.dtype)
            for e in range(num_experts):
                idx = torch.nonzero(eids == e, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                xs = x.index_select(0, idx)
                up = F.linear(xs, w1_residual[e])
                gate = F.linear(xs, self.w3[e])
                hidden = F.silu(up) * gate
                y = F.linear(hidden, self.down[e])
                out.index_add_(0, idx, y * gates.index_select(0, idx).unsqueeze(1))
        return out

    def residual_gate_up(self, adapter_id: int) -> torch.Tensor:
        gate_up = self.gate_up.detach().clone().contiguous()
        if adapter_id in self.residual_w1:
            gate_up[:, : self.intermediate, :] = self.residual_w1[adapter_id]
        else:
            ad = self.adapters[adapter_id]
            delta = ad.scaling * torch.einsum("eor,eri->eoi", ad.B, ad.A)
            gate_up[:, : self.intermediate, :] = self.w1 + delta
        return gate_up.contiguous()


def run_request_sequence(
    executor: MultiTenantMoELoRAExecutor,
    workload: Workload,
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    mode: str,
    *,
    vllm_fn: Callable[..., Any] | None = None,
) -> torch.Tensor:
    checksum = torch.zeros((), device=hidden.device, dtype=torch.float32)
    residual_gate_up_cache: dict[int, torch.Tensor] = {}
    for req in workload.requests:
        h = hidden[req.start : req.end]
        eids = expert_ids[req.start : req.end]
        weights = routing_weights[req.start : req.end]
        if mode == "dynamic":
            y = executor.dynamic_request(h, eids, weights, req.adapter_id)
        elif mode == "materialized":
            y = executor.materialized_or_dynamic_request(h, eids, weights, req.adapter_id)
        elif mode == "vllm_bridge":
            if vllm_fn is None:
                raise RuntimeError("vLLM bridge requested but vllm_fn is None")
            if req.adapter_id not in residual_gate_up_cache:
                residual_gate_up_cache[req.adapter_id] = executor.residual_gate_up(req.adapter_id)
            y = call_vllm_fused_experts(vllm_fn, h, residual_gate_up_cache[req.adapter_id], executor.down, weights, eids)
        else:
            raise ValueError(f"unknown mode {mode}")
        checksum = checksum + y.float().mean()
    return checksum


def speedup(base: dict[str, Any], other: dict[str, Any], key: str = "p50_ms") -> float:
    try:
        return float(base[key]) / float(other[key])
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--adapter-dirs", nargs="+", default=[])
    ap.add_argument("--prompts-jsonl", default="/data/stageml_h100_run/data/real_trace.jsonl")
    ap.add_argument("--out", default="paper_outputs/multitenant_moe_lora_bench.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--gpu-memory-gb", type=int, default=82)
    ap.add_argument("--cpu-memory-gb", type=int, default=180)
    ap.add_argument("--offload-folder", default="/data/stageml_h100_run/hf_offload")
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--bench-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num-tenants", type=int, default=8)
    ap.add_argument("--virtual-adapters", type=int, default=8)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--tokens-per-request", type=int, default=16)
    ap.add_argument("--tenant-distribution", default="zipf", choices=["zipf", "uniform", "round_robin"])
    ap.add_argument("--zipf-alpha", type=float, default=1.2)
    ap.add_argument("--materialize-top-k", type=int, default=2)
    ap.add_argument("--materialize-min-requests", type=int, default=2)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--enable-vllm-bridge", action="store_true")
    ap.add_argument("--synthetic-hidden-size", type=int, default=256)
    ap.add_argument("--synthetic-intermediate-size", type=int, default=512)
    ap.add_argument("--synthetic-rank", type=int, default=8)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.bench_dtype]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    if args.synthetic:
        hidden, gate_up, down, expert_ids, routing_weights, adapters_loaded, experts_name, gate_name = make_synthetic_inputs(
            tokens=max(args.max_tokens, args.num_requests * args.tokens_per_request),
            hidden_size=args.synthetic_hidden_size,
            intermediate_size=args.synthetic_intermediate_size,
            num_experts=args.max_experts,
            top_k=args.top_k,
            rank=args.synthetic_rank,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
    else:
        if not args.adapter_dirs:
            raise SystemExit("--adapter-dirs is required unless --synthetic is used")
        if _load_optional_transformers is None:
            raise SystemExit("real benchmark helpers are unavailable")
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
        hidden = collect_real_hidden_states(model, tokenizer, rows, layer_index=args.layer_index, device=str(device), max_tokens=args.max_tokens)
        gate_up, down, experts_name = extract_mixtral_moe_tensors(model, max_experts=args.max_experts)
        num_experts = int(gate_up.shape[0])
        intermediate = int(gate_up.shape[1] // 2)
        in_features = int(gate_up.shape[2])
        adapters_loaded = [
            load_expert_lora_adapter(Path(p), num_experts=num_experts, in_features=in_features, out_features=intermediate)
            for p in args.adapter_dirs
        ]
        expert_ids, routing_weights, gate_name = make_routing(model, hidden, num_experts, args.top_k, str(device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hidden = hidden[:, :in_features].to(device=device, dtype=dtype).contiguous()
        gate_up = gate_up.to(device=device, dtype=dtype).contiguous()
        down = down.to(device=device, dtype=dtype).contiguous()
        expert_ids = expert_ids.to(device=device, dtype=torch.long).contiguous()
        routing_weights = routing_weights.to(device=device, dtype=dtype).contiguous()
        adapters_loaded = [
            MoEAdapterSpec(a.name, a.A.to(device=device, dtype=dtype).contiguous(), a.B.to(device=device, dtype=dtype).contiguous(), a.scaling)
            for a in adapters_loaded
        ]

    if args.virtual_adapters > len(adapters_loaded):
        adapters = clone_virtual_adapters(adapters_loaded[0], args.virtual_adapters, dtype=dtype, device=device)
        virtual_from_single = True
    else:
        adapters = adapters_loaded[: args.virtual_adapters]
        virtual_from_single = False

    tokens_needed = args.tokens_per_request
    if int(hidden.shape[0]) < tokens_needed:
        raise SystemExit("not enough hidden tokens for one request")

    workload = make_request_workload(
        num_requests=args.num_requests,
        num_tenants=args.num_tenants,
        num_adapters=len(adapters),
        tokens_available=int(hidden.shape[0]),
        tokens_per_request=args.tokens_per_request,
        distribution=args.tenant_distribution,
        zipf_alpha=args.zipf_alpha,
        materialize_top_k=args.materialize_top_k,
        materialize_min_requests=args.materialize_min_requests,
        seed=args.seed,
    )

    executor = MultiTenantMoELoRAExecutor(
        gate_up=gate_up,
        down=down,
        adapters=adapters,
        materialized_adapter_ids=workload.materialized_adapter_ids,
    )

    result: dict[str, Any] = {
        "benchmark": "multitenant_moe_lora_request_level",
        "model": args.model if not args.synthetic else "synthetic",
        "experts_name": experts_name,
        "gate_name": gate_name,
        "dtype": args.bench_dtype,
        "device": str(device),
        "num_tokens_available": int(hidden.shape[0]),
        "num_requests": args.num_requests,
        "tokens_per_request": args.tokens_per_request,
        "num_tenants": args.num_tenants,
        "num_adapters": len(adapters),
        "virtual_adapters_from_single_real_adapter": virtual_from_single,
        "tenant_distribution": args.tenant_distribution,
        "zipf_alpha": args.zipf_alpha,
        "top_k": args.top_k,
        "num_experts": int(gate_up.shape[0]),
        "warmups": args.warmups,
        "measured_runs": args.repeats,
        "tenant_request_counts": workload.tenant_request_counts,
        "materialized_adapter_ids": sorted(workload.materialized_adapter_ids),
        "materialized_adapter_count": len(workload.materialized_adapter_ids),
        "materialized_memory_bytes": executor.materialized_memory_bytes,
        "materialized_memory_mb": executor.materialized_memory_bytes / (1024.0 * 1024.0),
        "compile_time_ms": executor.compile_time_ms,
        "environment": _gpu_metadata() if _gpu_metadata is not None else {},
        "claim_boundary": (
            "This is a request level StageML harness benchmark for multi tenant MoE LoRA. "
            "It is not a live HTTP serving benchmark. The vLLM bridge path is optional and measures the fused experts backend with StageML residualized weights when available."
        ),
    }

    with torch.no_grad():
        dynamic_stats = time_workload(
            lambda: run_request_sequence(executor, workload, hidden, expert_ids, routing_weights, "dynamic"),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        materialized_stats = time_workload(
            lambda: run_request_sequence(executor, workload, hidden, expert_ids, routing_weights, "materialized"),
            warmups=args.warmups,
            repeats=args.repeats,
        )
    materialized_stats["speedup_over_dynamic_p50"] = speedup(dynamic_stats, materialized_stats, "p50_ms")
    materialized_stats["speedup_over_dynamic_mean"] = speedup(dynamic_stats, materialized_stats, "mean_ms")
    result["dynamic_stage_moe_lora"] = dynamic_stats
    result["stageml_materialized_hot_tenants"] = materialized_stats

    if args.enable_vllm_bridge:
        fn, source = try_import_vllm_fused_experts()
        if fn is None:
            result["stageml_residual_on_vllm_fused_experts"] = {"status": "unavailable", "reason": source}
        else:
            try:
                with torch.no_grad():
                    bridge_stats = time_workload(
                        lambda: run_request_sequence(executor, workload, hidden, expert_ids, routing_weights, "vllm_bridge", vllm_fn=fn),
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                bridge_stats["status"] = "ok"
                bridge_stats["source"] = source
                bridge_stats["speedup_over_dynamic_p50"] = speedup(dynamic_stats, bridge_stats, "p50_ms")
                bridge_stats["speedup_over_dynamic_mean"] = speedup(dynamic_stats, bridge_stats, "mean_ms")
                bridge_stats["speedup_over_stage_pytorch_materialized_p50"] = speedup(materialized_stats, bridge_stats, "p50_ms")
                result["stageml_residual_on_vllm_fused_experts"] = bridge_stats
            except Exception as exc:
                result["stageml_residual_on_vllm_fused_experts"] = {"status": "call_failed", "reason": str(exc), "source": source}
    else:
        result["stageml_residual_on_vllm_fused_experts"] = {"status": "skipped", "reason": "--enable-vllm-bridge was not set"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    csv_out = out.with_suffix(".csv")
    rows = []
    for name in ["dynamic_stage_moe_lora", "stageml_materialized_hot_tenants", "stageml_residual_on_vllm_fused_experts"]:
        block = result.get(name, {})
        if not isinstance(block, dict) or block.get("status") not in (None, "ok"):
            rows.append({"system": name, "status": block.get("status", "missing"), "mean_ms": "", "std_ms": "", "p50_ms": "", "p95_ms": "", "runs": "", "speedup_over_dynamic_p50": ""})
        else:
            rows.append({
                "system": name,
                "status": block.get("status", "ok"),
                "mean_ms": block.get("mean_ms", ""),
                "std_ms": block.get("std_ms", ""),
                "p50_ms": block.get("p50_ms", ""),
                "p95_ms": block.get("p95_ms", ""),
                "runs": block.get("runs", ""),
                "speedup_over_dynamic_p50": block.get("speedup_over_dynamic_p50", 1.0 if name == "dynamic_stage_moe_lora" else ""),
            })
    with csv_out.open("w") as f:
        headers = ["system", "status", "mean_ms", "std_ms", "p50_ms", "p95_ms", "runs", "speedup_over_dynamic_p50"]
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
