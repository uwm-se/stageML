"""
Adapter-bank residualization for multi-adapter LoRA inference.

Why this file matters:
    PEFT merge_and_unload is excellent when one adapter is fixed for the whole
    deployed model.  In multi-tenant serving, however, different requests may
    use different adapters.  Repeatedly merging and unmerging the full model is
    not the right abstraction.

This module implements a StageML-style residual adapter bank.  It specializes
LoRA weights per adapter once, stores compact residual weights for each
adapter, and groups mixed-adapter batches at runtime.

The implementation is intentionally plain PyTorch so it is easy to test and
explain in the paper.  It is not meant to beat custom CUDA systems such as
S-LoRA.  Its purpose is to expose the program-transformation idea:

    base weight + adapter-specific stage values -> adapter residual weight
    request activation + adapter id -> grouped residual matmul
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterSpec:
    """One LoRA adapter for one linear projection."""

    name: str
    A: torch.Tensor      # [rank, in_features]
    B: torch.Tensor      # [out_features, rank]
    scaling: float


class DynamicLoRALinear(nn.Module):
    """Reference mixed-adapter LoRA layer.

    This is the runtime form StageML tries to improve.  It keeps A and B as
    separate factors and computes the adapter branch for each adapter group.
    """

    def __init__(self, weight: torch.Tensor, adapters: Sequence[AdapterSpec], bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.adapters = {a.name: a for a in adapters}

    def forward(self, x: torch.Tensor, adapter_ids: Sequence[str]) -> torch.Tensor:
        if x.shape[0] != len(adapter_ids):
            raise ValueError("batch dimension must match number of adapter ids")
        y = F.linear(x, self.weight, self.bias)
        out = y.clone()
        for name in sorted(set(adapter_ids)):
            idx = [i for i, a in enumerate(adapter_ids) if a == name]
            adapter = self.adapters[name]
            xi = x[idx]
            update = adapter.scaling * (xi @ adapter.A.t() @ adapter.B.t())
            out[idx] = out[idx] + update
        return out


class StageMLAdapterBankLinear(nn.Module):
    """StageML residualized multi-adapter linear layer.

    For each adapter, StageML precomputes:

        W_adapter = W + scaling * (B @ A)

    The runtime receives a mixed-adapter batch and groups rows by adapter id.
    Each group runs one residual matmul with that adapter's merged weight.
    """

    def __init__(self, weight: torch.Tensor, adapters: Sequence[AdapterSpec], bias: torch.Tensor | None = None):
        super().__init__()
        self.adapter_names = [a.name for a in adapters]
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        merged = []
        for adapter in adapters:
            merged_weight = weight.detach() + adapter.scaling * (adapter.B.detach() @ adapter.A.detach())
            merged.append(merged_weight)
        self.register_buffer("merged_weights", torch.stack(merged, dim=0).contiguous())
        self.name_to_index = {name: i for i, name in enumerate(self.adapter_names)}

    @classmethod
    def from_lora_factors(cls, weight: torch.Tensor, adapters: Sequence[AdapterSpec], bias: torch.Tensor | None = None):
        return cls(weight, adapters, bias)

    def forward(self, x: torch.Tensor, adapter_ids: Sequence[str]) -> torch.Tensor:
        if x.shape[0] != len(adapter_ids):
            raise ValueError("batch dimension must match number of adapter ids")
        out = torch.empty((x.shape[0], self.merged_weights.shape[1]), device=x.device, dtype=x.dtype)
        for name in sorted(set(adapter_ids)):
            idx = [i for i, a in enumerate(adapter_ids) if a == name]
            weight = self.merged_weights[self.name_to_index[name]]
            out[idx] = F.linear(x[idx], weight, self.bias)
        return out


class OnTheFlyMergeLinear(nn.Module):
    """Slow baseline that merges adapter weights at every request group.

    This models the bad approach for multi-adapter serving: using LoRA merge as
    a runtime operation instead of as a staged deployment/adaptation step.
    """

    def __init__(self, weight: torch.Tensor, adapters: Sequence[AdapterSpec], bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.adapters = {a.name: a for a in adapters}

    def forward(self, x: torch.Tensor, adapter_ids: Sequence[str]) -> torch.Tensor:
        out = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=x.dtype)
        for name in sorted(set(adapter_ids)):
            idx = [i for i, a in enumerate(adapter_ids) if a == name]
            adapter = self.adapters[name]
            merged = self.weight + adapter.scaling * (adapter.B @ adapter.A)
            out[idx] = F.linear(x[idx], merged, self.bias)
        return out


def make_random_adapters(
    *,
    num_adapters: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    scaling: float | None = None,
) -> list[AdapterSpec]:
    """Create random adapter factors for a controlled benchmark."""
    if scaling is None:
        scaling = 16.0 / float(rank)
    adapters = []
    for i in range(num_adapters):
        A = torch.randn(rank, in_features, device=device, dtype=dtype) / (in_features ** 0.5)
        B = torch.randn(out_features, rank, device=device, dtype=dtype) / (rank ** 0.5)
        adapters.append(AdapterSpec(name=f"adapter_{i}", A=A, B=B, scaling=float(scaling)))
    return adapters


def make_adapter_ids(batch: int, num_adapters: int, pattern: str = "round_robin") -> list[str]:
    if pattern == "single":
        return ["adapter_0"] * batch
    if pattern == "round_robin":
        return [f"adapter_{i % num_adapters}" for i in range(batch)]
    if pattern == "clustered":
        ids = []
        group = max(1, batch // num_adapters)
        for i in range(batch):
            ids.append(f"adapter_{min(i // group, num_adapters - 1)}")
        return ids
    raise ValueError(f"unknown adapter id pattern: {pattern}")
