from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEAdapterSpec:
    name: str
    A: torch.Tensor
    B: torch.Tensor
    scaling: float

    def __post_init__(self) -> None:
        if self.A.ndim != 3:
            raise ValueError("A must have shape num_experts by rank by in_features")
        if self.B.ndim != 3:
            raise ValueError("B must have shape num_experts by out_features by rank")
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("A and B must have the same number of experts")
        if self.A.shape[1] != self.B.shape[2]:
            raise ValueError("A rank must match B rank")

    @property
    def num_experts(self) -> int:
        return int(self.A.shape[0])

    @property
    def rank(self) -> int:
        return int(self.A.shape[1])

    @property
    def in_features(self) -> int:
        return int(self.A.shape[2])

    @property
    def out_features(self) -> int:
        return int(self.B.shape[1])


class DynamicMoELoRALayer(nn.Module):
    def __init__(self, expert_weight: torch.Tensor, adapters: Sequence[MoEAdapterSpec], expert_bias: torch.Tensor | None = None):
        super().__init__()
        if expert_weight.ndim != 3:
            raise ValueError("expert_weight must have shape num_experts by out_features by in_features")
        self.register_buffer("expert_weight", expert_weight.detach().clone())
        if expert_bias is not None:
            self.register_buffer("expert_bias", expert_bias.detach().clone())
        else:
            self.expert_bias = None
        self.adapters = {a.name: a for a in adapters}
        self.adapter_names = list(self.adapters.keys())

    def _adapter_name(self, adapter_id: str | int) -> str:
        if isinstance(adapter_id, int):
            return self.adapter_names[adapter_id]
        return str(adapter_id)

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        adapter_ids: Sequence[str | int],
    ) -> torch.Tensor:
        if expert_ids.ndim != 2:
            raise ValueError("expert_ids must have shape tokens by top_k")
        if routing_weights.shape != expert_ids.shape:
            raise ValueError("routing_weights must match expert_ids")
        if x.shape[0] != expert_ids.shape[0] or x.shape[0] != len(adapter_ids):
            raise ValueError("token dimension mismatch")
        num_tokens = x.shape[0]
        out_features = self.expert_weight.shape[1]
        out = torch.zeros((num_tokens, out_features), device=x.device, dtype=x.dtype)
        for t in range(num_tokens):
            name = self._adapter_name(adapter_ids[t])
            adapter = self.adapters[name]
            for k in range(expert_ids.shape[1]):
                e = int(expert_ids[t, k].item())
                gate = routing_weights[t, k].to(dtype=x.dtype)
                base = F.linear(x[t : t + 1], self.expert_weight[e], None)
                if self.expert_bias is not None:
                    base = base + self.expert_bias[e]
                lora = adapter.scaling * (x[t : t + 1] @ adapter.A[e].t() @ adapter.B[e].t())
                out[t : t + 1] = out[t : t + 1] + gate * (base + lora)
        return out


class MaterializedMoELoRALayer(nn.Module):
    def __init__(self, expert_weight: torch.Tensor, adapters: Sequence[MoEAdapterSpec], expert_bias: torch.Tensor | None = None):
        super().__init__()
        if expert_weight.ndim != 3:
            raise ValueError("expert_weight must have shape num_experts by out_features by in_features")
        self.adapter_names = [a.name for a in adapters]
        self.name_to_index = {name: i for i, name in enumerate(self.adapter_names)}
        if expert_bias is not None:
            self.register_buffer("expert_bias", expert_bias.detach().clone())
        else:
            self.expert_bias = None
        merged = []
        for adapter in adapters:
            if adapter.num_experts != expert_weight.shape[0]:
                raise ValueError("adapter and expert weights disagree on expert count")
            delta = adapter.scaling * torch.einsum("eor,eri->eoi", adapter.B.detach(), adapter.A.detach())
            merged.append(expert_weight.detach() + delta)
        self.register_buffer("merged_weight", torch.stack(merged, dim=0).contiguous())

    def _adapter_index(self, adapter_id: str | int) -> int:
        if isinstance(adapter_id, int):
            return int(adapter_id)
        return self.name_to_index[str(adapter_id)]

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        adapter_ids: Sequence[str | int],
    ) -> torch.Tensor:
        if expert_ids.ndim != 2:
            raise ValueError("expert_ids must have shape tokens by top_k")
        if routing_weights.shape != expert_ids.shape:
            raise ValueError("routing_weights must match expert_ids")
        if x.shape[0] != expert_ids.shape[0] or x.shape[0] != len(adapter_ids):
            raise ValueError("token dimension mismatch")
        num_tokens = x.shape[0]
        out_features = self.merged_weight.shape[2]
        out = torch.zeros((num_tokens, out_features), device=x.device, dtype=x.dtype)
        for t in range(num_tokens):
            aidx = self._adapter_index(adapter_ids[t])
            for k in range(expert_ids.shape[1]):
                e = int(expert_ids[t, k].item())
                gate = routing_weights[t, k].to(dtype=x.dtype)
                y = F.linear(x[t : t + 1], self.merged_weight[aidx, e], None)
                if self.expert_bias is not None:
                    y = y + self.expert_bias[e]
                out[t : t + 1] = out[t : t + 1] + gate * y
        return out


class NonMaterializingFusedMoELoRAReference(DynamicMoELoRALayer):
    pass


def make_random_moe_adapters(
    *,
    num_adapters: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    ranks: int | Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    scaling: float | None = None,
) -> list[MoEAdapterSpec]:
    if isinstance(ranks, int):
        rank_values = [int(ranks)] * num_adapters
    else:
        seq = list(ranks)
        rank_values = [int(seq[i % len(seq)]) for i in range(num_adapters)]
    adapters = []
    for i, rank in enumerate(rank_values):
        scale = float(scaling if scaling is not None else 16.0 / float(rank))
        A = torch.randn(num_experts, rank, in_features, device=device, dtype=dtype) / (in_features ** 0.5)
        B = torch.randn(num_experts, out_features, rank, device=device, dtype=dtype) / (rank ** 0.5)
        adapters.append(MoEAdapterSpec(name=f"adapter_{i}", A=A, B=B, scaling=scale))
    return adapters


def make_round_robin_adapter_ids(num_tokens: int, num_adapters: int) -> list[str]:
    return [f"adapter_{i % num_adapters}" for i in range(num_tokens)]


def normalize_routing_weights(expert_ids: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        weights = torch.ones_like(expert_ids, dtype=torch.float32)
    weights = weights.to(torch.float32)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights / denom
