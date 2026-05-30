from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from stageml.moe_stages import MoEStage, join, parse_stage, stage_name


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"


class PlanKind(str, Enum):
    DYNAMIC = "dynamic"
    NON_MATERIALIZING_FUSION = "non_materializing_fusion"
    MATERIALIZED_RESIDUAL = "materialized_residual"
    DISAGGREGATED = "disaggregated"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    precision: Precision | str
    stage: MoEStage

    def with_stage(self, stage: MoEStage | int | str) -> "TensorType":
        return TensorType(self.shape, self.precision, parse_stage(stage))

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return int(n)


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    typ: TensorType | None = None
    name: str | None = None
    value: Any = None

    def stage(self) -> MoEStage:
        if self.typ is not None:
            return self.typ.stage
        if not self.args:
            return MoEStage.BASE
        return join(*(a.stage() for a in self.args))

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "name": self.name,
            "stage": stage_name(self.stage()),
            "shape": list(self.typ.shape) if self.typ is not None else None,
            "precision": str(self.typ.precision) if self.typ is not None else None,
            "args": [a.to_dict() for a in self.args],
        }


def var(name: str, typ: TensorType) -> Expr:
    return Expr(op="var", args=(), typ=typ, name=name)


def const(name: str, value: Any, typ: TensorType) -> Expr:
    return Expr(op="const", args=(), typ=typ, name=name, value=value)


def matmul(left: Expr, right: Expr, name: str | None = None) -> Expr:
    if left.typ is None or right.typ is None:
        typ = None
    else:
        if len(left.typ.shape) != 2 or len(right.typ.shape) != 2:
            raise ValueError("matmul currently expects rank two tensor types")
        if left.typ.shape[1] != right.typ.shape[0]:
            raise ValueError(f"matmul shape mismatch {left.typ.shape} and {right.typ.shape}")
        typ = TensorType((left.typ.shape[0], right.typ.shape[1]), left.typ.precision, join(left.typ.stage, right.typ.stage))
    return Expr(op="matmul", args=(left, right), typ=typ, name=name)


def add(left: Expr, right: Expr, name: str | None = None) -> Expr:
    if left.typ is None or right.typ is None:
        typ = None
    else:
        if left.typ.shape != right.typ.shape:
            raise ValueError(f"add shape mismatch {left.typ.shape} and {right.typ.shape}")
        typ = TensorType(left.typ.shape, left.typ.precision, join(left.typ.stage, right.typ.stage))
    return Expr(op="add", args=(left, right), typ=typ, name=name)


def fold(expr: Expr, name: str | None = None) -> Expr:
    typ = expr.typ.with_stage(MoEStage.BASE) if expr.typ is not None else None
    return Expr(op="fold", args=(expr,), typ=typ, name=name)


def fallback(expr: Expr, reason: str) -> Expr:
    typ = expr.typ.with_stage(MoEStage.TOKEN) if expr.typ is not None else None
    return Expr(op="fallback", args=(expr,), typ=typ, name=reason)


@dataclass(frozen=True)
class MoEAdapterMeta:
    name: str
    rank: int
    num_experts: int
    in_features: int
    out_features: int
    dtype: str = "fp16"
    target_layers: tuple[str, ...] = ()
    request_count: int = 0

    @property
    def factor_numel(self) -> int:
        return self.num_experts * self.rank * (self.in_features + self.out_features)

    @property
    def materialized_numel(self) -> int:
        return self.num_experts * self.in_features * self.out_features


@dataclass(frozen=True)
class ExpertPlan:
    adapter: str
    expert: int | str
    kind: PlanKind
    rank: int
    memory_bytes: int
    estimated_latency_cost: float
    epsilon: float
    reason: str


@dataclass(frozen=True)
class ResidualPlan:
    plans: tuple[ExpertPlan, ...]
    memory_budget_bytes: int
    used_memory_bytes: int
    theta: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for plan in self.plans:
            key = plan.kind.value
            out[key] = out.get(key, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_budget_bytes": self.memory_budget_bytes,
            "used_memory_bytes": self.used_memory_bytes,
            "theta": self.theta,
            "summary": self.by_kind(),
            "metadata": self.metadata,
            "plans": [
                {
                    "adapter": p.adapter,
                    "expert": p.expert,
                    "kind": p.kind.value,
                    "rank": p.rank,
                    "memory_bytes": p.memory_bytes,
                    "estimated_latency_cost": p.estimated_latency_cost,
                    "epsilon": p.epsilon,
                    "reason": p.reason,
                }
                for p in self.plans
            ],
        }


def dtype_bytes(dtype: torch.dtype | str) -> int:
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    mapping = {
        "fp32": 4,
        "float32": 4,
        "fp16": 2,
        "float16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "int8": 1,
        "uint8": 1,
        "int4": 1,
        "int2": 1,
    }
    key = str(dtype).lower()
    if key not in mapping:
        raise ValueError(f"unknown dtype {dtype!r}")
    return mapping[key]
