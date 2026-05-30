from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from stageml.moe_ir import ExpertPlan, MoEAdapterMeta, PlanKind, ResidualPlan, dtype_bytes


@dataclass(frozen=True)
class PlannerConfig:
    memory_budget_mb: float
    theta: float = 0.0
    dtype: str = "fp16"
    hot_request_threshold: int = 100
    disaggregate_request_threshold: int = 500
    rank_specialization_threshold: int = 16
    allow_disaggregated: bool = True
    allow_non_materializing_fusion: bool = True
    allow_materialized: bool = True


@dataclass(frozen=True)
class AdapterQuantInfo:
    epsilon: float
    safe: bool


def materialized_bytes(meta: MoEAdapterMeta, dtype: str) -> int:
    return int(meta.materialized_numel * dtype_bytes(dtype))


def factor_bytes(meta: MoEAdapterMeta, dtype: str) -> int:
    return int(meta.factor_numel * dtype_bytes(dtype))


def estimate_dynamic_cost(meta: MoEAdapterMeta) -> float:
    return float(max(meta.request_count, 1) * meta.rank * meta.num_experts)


def estimate_materialized_cost(meta: MoEAdapterMeta) -> float:
    return float(max(meta.request_count, 1) * meta.num_experts)


def estimate_fused_cost(meta: MoEAdapterMeta) -> float:
    return float(max(meta.request_count, 1) * meta.num_experts * (1.0 + 0.25 * (meta.rank / 16.0)))


def choose_residual_plan(
    adapters: Sequence[MoEAdapterMeta],
    *,
    config: PlannerConfig,
    quant_info: Mapping[str, AdapterQuantInfo] | None = None,
) -> ResidualPlan:
    budget = int(config.memory_budget_mb * 1024 * 1024)
    quant_info = quant_info or {}
    ranked = sorted(
        adapters,
        key=lambda m: (
            -m.request_count,
            m.rank,
            m.name,
        ),
    )
    used = 0
    plans: list[ExpertPlan] = []
    for meta in ranked:
        q = quant_info.get(meta.name, AdapterQuantInfo(epsilon=0.0, safe=True))
        mem_materialized = materialized_bytes(meta, config.dtype)
        if not q.safe or q.epsilon > config.theta:
            plans.append(
                ExpertPlan(
                    adapter=meta.name,
                    expert="all",
                    kind=PlanKind.FALLBACK,
                    rank=meta.rank,
                    memory_bytes=0,
                    estimated_latency_cost=estimate_dynamic_cost(meta),
                    epsilon=float(q.epsilon),
                    reason="quantization error exceeds threshold",
                )
            )
            continue
        if (
            config.allow_materialized
            and meta.request_count >= config.hot_request_threshold
            and used + mem_materialized <= budget
        ):
            used += mem_materialized
            plans.append(
                ExpertPlan(
                    adapter=meta.name,
                    expert="all",
                    kind=PlanKind.MATERIALIZED_RESIDUAL,
                    rank=meta.rank,
                    memory_bytes=mem_materialized,
                    estimated_latency_cost=estimate_materialized_cost(meta),
                    epsilon=float(q.epsilon),
                    reason="hot adapter fits memory budget",
                )
            )
            continue
        if config.allow_non_materializing_fusion and meta.rank <= config.rank_specialization_threshold:
            plans.append(
                ExpertPlan(
                    adapter=meta.name,
                    expert="all",
                    kind=PlanKind.NON_MATERIALIZING_FUSION,
                    rank=meta.rank,
                    memory_bytes=0,
                    estimated_latency_cost=estimate_fused_cost(meta),
                    epsilon=float(q.epsilon),
                    reason="rank is suitable for specialized fused path without materializing full weights",
                )
            )
            continue
        if config.allow_disaggregated and meta.request_count >= config.disaggregate_request_threshold:
            plans.append(
                ExpertPlan(
                    adapter=meta.name,
                    expert="all",
                    kind=PlanKind.DISAGGREGATED,
                    rank=meta.rank,
                    memory_bytes=factor_bytes(meta, config.dtype),
                    estimated_latency_cost=estimate_fused_cost(meta) * 1.25,
                    epsilon=float(q.epsilon),
                    reason="expressible in IR, not yet executable in this artifact",
                )
            )
            continue
        plans.append(
            ExpertPlan(
                adapter=meta.name,
                expert="all",
                kind=PlanKind.DYNAMIC,
                rank=meta.rank,
                memory_bytes=0,
                estimated_latency_cost=estimate_dynamic_cost(meta),
                epsilon=float(q.epsilon),
                reason="cold adapter or no profitable specialization",
            )
        )
    return ResidualPlan(
        plans=tuple(plans),
        memory_budget_bytes=budget,
        used_memory_bytes=used,
        theta=float(config.theta),
        metadata={
            "dtype": config.dtype,
            "memory_budget_mb": config.memory_budget_mb,
            "hot_request_threshold": config.hot_request_threshold,
            "rank_specialization_threshold": config.rank_specialization_threshold,
        },
    )
