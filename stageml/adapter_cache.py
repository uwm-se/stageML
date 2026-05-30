"""
Cost-based adapter residual cache for StageML.

This module is the industry-facing extension of StageML.  It does not try to
replace LoRAX, vLLM, S-LoRA, or Punica as a serving engine.  Instead, it models
an optimizer decision those systems care about:

    Which adapter computations should be residualized and cached, and which
    should remain dynamic because they are cold or memory is limited?

The abstraction is intentionally small and pure PyTorch so it can be tested in
Colab and explained in a compiler paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from stageml.adapter_bank import AdapterSpec


@dataclass(frozen=True)
class AdapterCacheDecision:
    """Decision for one adapter in the StageML cache policy."""

    name: str
    cached: bool
    estimated_bytes: int
    use_count: int


class StageMLCostBasedAdapterCache(nn.Module):
    """Mixed cached/dynamic LoRA layer.

    Hot adapters are residualized once:

        W_i = W + scaling_i * (B_i @ A_i)

    Cold adapters keep the dynamic LoRA form:

        y = x W^T + scaling_i * (x A_i^T) B_i^T

    This is the smallest useful model of a pre-serving StageML optimizer.  It
    exposes a tradeoff between latency and GPU memory instead of blindly merging
    every adapter.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        adapters: Sequence[AdapterSpec],
        *,
        cached_adapter_names: Sequence[str],
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        self.adapters = {a.name: a for a in adapters}
        self.cached_adapter_names = list(cached_adapter_names)
        self.cached_name_to_index = {name: i for i, name in enumerate(self.cached_adapter_names)}

        merged = []
        for name in self.cached_adapter_names:
            adapter = self.adapters[name]
            merged_weight = weight.detach() + adapter.scaling * (adapter.B.detach() @ adapter.A.detach())
            merged.append(merged_weight)

        if merged:
            self.register_buffer("cached_merged_weights", torch.stack(merged, dim=0).contiguous())
        else:
            self.register_buffer("cached_merged_weights", torch.empty(0, *weight.shape, device=weight.device, dtype=weight.dtype))

    def forward(self, x: torch.Tensor, adapter_ids: Sequence[str]) -> torch.Tensor:
        if x.shape[0] != len(adapter_ids):
            raise ValueError("batch dimension must match number of adapter ids")

        out = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=x.dtype)
        for name in sorted(set(adapter_ids)):
            idx = [i for i, a in enumerate(adapter_ids) if a == name]
            xi = x[idx]

            if name in self.cached_name_to_index:
                merged_weight = self.cached_merged_weights[self.cached_name_to_index[name]]
                out[idx] = F.linear(xi, merged_weight, self.bias)
            else:
                adapter = self.adapters[name]
                base = F.linear(xi, self.weight, self.bias)
                update = adapter.scaling * (xi @ adapter.A.t() @ adapter.B.t())
                out[idx] = base + update
        return out

    def cached_memory_bytes(self) -> int:
        return int(self.cached_merged_weights.numel() * self.cached_merged_weights.element_size())


def choose_hot_adapters(
    request_counts: Mapping[str, int],
    adapters: Sequence[AdapterSpec],
    *,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    memory_budget_mb: float,
) -> tuple[list[str], list[AdapterCacheDecision]]:
    """Choose adapters to cache under a simple memory budget.

    The policy is deliberately transparent for a research artifact: sort by
    observed request count and cache the hottest adapters until the budget is
    exhausted.  This is a baseline cost model that can later be replaced by a
    learned or profile-guided policy.
    """

    bytes_per_weight = torch.empty((), dtype=dtype).element_size()
    per_adapter_bytes = int(in_features * out_features * bytes_per_weight)
    budget_bytes = int(memory_budget_mb * 1024 * 1024)

    adapter_names = [a.name for a in adapters]
    ranked = sorted(adapter_names, key=lambda n: (-int(request_counts.get(n, 0)), n))

    cached: list[str] = []
    used = 0
    decisions: list[AdapterCacheDecision] = []
    for name in ranked:
        count = int(request_counts.get(name, 0))
        should_cache = used + per_adapter_bytes <= budget_bytes and count > 0
        if should_cache:
            cached.append(name)
            used += per_adapter_bytes
        decisions.append(AdapterCacheDecision(name=name, cached=should_cache, estimated_bytes=per_adapter_bytes, use_count=count))
    return cached, decisions
