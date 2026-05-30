"""
StageML policy optimizer for adapter-specialized inference.

This is the research-facing component.  It separates the simple algebraic
operation "merge LoRA weights" from the harder system decision:

    Which adapters should be residualized under a memory budget and a request
    distribution?

The policy is intentionally transparent.  It can be explained in a compiler
paper and replaced later by a learned or more detailed cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from stageml.adapter_bank import AdapterSpec


@dataclass(frozen=True)
class AdapterProfile:
    name: str
    request_count: int
    factor_bytes: int
    residual_bytes: int
    estimated_dynamic_cost: float
    estimated_cached_cost: float
    estimated_saving: float


@dataclass(frozen=True)
class PolicyResult:
    cached_adapters: list[str]
    profiles: list[AdapterProfile]
    total_residual_bytes: int
    budget_bytes: int


def tensor_bytes(numel: int, dtype: torch.dtype) -> int:
    return int(numel * torch.empty((), dtype=dtype).element_size())


def profile_adapters(
    request_counts: Mapping[str, int],
    adapters: Sequence[AdapterSpec],
    *,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    dynamic_cost_per_request: float = 3.0,
    cached_cost_per_request: float = 1.0,
) -> list[AdapterProfile]:
    """Build a simple cost profile for every adapter.

    The default cost units are relative, not wall-clock times.  Dynamic LoRA is
    modeled as roughly three matmul-like operations and cached residual execution
    as one matmul-like operation.  The request count scales the expected benefit.
    """

    profiles: list[AdapterProfile] = []
    for a in adapters:
        count = int(request_counts.get(a.name, 0))
        factor_numel = int(a.A.numel() + a.B.numel())
        residual_numel = int(in_features * out_features)
        factor_bytes = tensor_bytes(factor_numel, dtype)
        residual_bytes = tensor_bytes(residual_numel, dtype)
        dynamic_cost = count * dynamic_cost_per_request
        cached_cost = count * cached_cost_per_request
        saving = dynamic_cost - cached_cost
        profiles.append(
            AdapterProfile(
                name=a.name,
                request_count=count,
                factor_bytes=factor_bytes,
                residual_bytes=residual_bytes,
                estimated_dynamic_cost=dynamic_cost,
                estimated_cached_cost=cached_cost,
                estimated_saving=saving,
            )
        )
    return profiles


def choose_adapters_by_benefit_density(
    request_counts: Mapping[str, int],
    adapters: Sequence[AdapterSpec],
    *,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    memory_budget_mb: float,
) -> PolicyResult:
    """Choose cached adapters by saving per residual byte.

    This is a compiler-style policy: estimate benefit, estimate memory cost, then
    select the best residualizations under a budget.  It is deliberately simple
    enough to reproduce and inspect.
    """

    profiles = profile_adapters(
        request_counts,
        adapters,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
    )
    budget_bytes = int(memory_budget_mb * 1024 * 1024)
    ranked = sorted(
        profiles,
        key=lambda p: (-(p.estimated_saving / max(p.residual_bytes, 1)), -p.request_count, p.name),
    )
    cached: list[str] = []
    used = 0
    for p in ranked:
        if p.request_count <= 0 or p.estimated_saving <= 0:
            continue
        if used + p.residual_bytes <= budget_bytes:
            cached.append(p.name)
            used += p.residual_bytes
    return PolicyResult(cached_adapters=cached, profiles=profiles, total_residual_bytes=used, budget_bytes=budget_bytes)
