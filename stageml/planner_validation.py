from __future__ import annotations

"""Small brute force validation helpers for the StageML residual planner."""

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Iterable, Sequence

from stageml.moe_ir import MoEAdapterMeta, PlanKind
from stageml.residual_planner import (
    PlannerConfig,
    choose_residual_plan,
    estimate_dynamic_cost,
    estimate_materialized_cost,
    materialized_bytes,
)


@dataclass(frozen=True)
class BruteForceValidationRow:
    config: str
    planner_choice: str
    brute_force_choice: str
    match: bool
    planner_cost: float
    brute_force_cost: float
    latency_gap: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _choice_string(names: Iterable[str]) -> str:
    values = sorted(names)
    return "none" if not values else ";".join(values)


def _powerset(seq: Sequence[MoEAdapterMeta]) -> Iterable[tuple[MoEAdapterMeta, ...]]:
    for n in range(len(seq) + 1):
        yield from combinations(seq, n)


def brute_force_materialized_choice(adapters: Sequence[MoEAdapterMeta], config: PlannerConfig) -> tuple[set[str], float]:
    budget = int(config.memory_budget_mb * 1024 * 1024)
    best_choice: set[str] = set()
    best_cost = float("inf")
    for subset in _powerset(adapters):
        names = {m.name for m in subset}
        mem = sum(materialized_bytes(m, config.dtype) for m in subset)
        if mem > budget:
            continue
        cost = 0.0
        for meta in adapters:
            if meta.name in names and meta.request_count >= config.hot_request_threshold:
                cost += estimate_materialized_cost(meta)
            else:
                cost += estimate_dynamic_cost(meta)
        if cost < best_cost or (cost == best_cost and _choice_string(names) < _choice_string(best_choice)):
            best_cost = cost
            best_choice = names
    return best_choice, best_cost


def validate_planner_against_bruteforce(adapters: Sequence[MoEAdapterMeta], config: PlannerConfig, *, config_name: str) -> BruteForceValidationRow:
    planner_config = PlannerConfig(
        memory_budget_mb=config.memory_budget_mb,
        theta=config.theta,
        dtype=config.dtype,
        hot_request_threshold=config.hot_request_threshold,
        disaggregate_request_threshold=config.disaggregate_request_threshold,
        rank_specialization_threshold=config.rank_specialization_threshold,
        allow_disaggregated=False,
        allow_non_materializing_fusion=False,
        allow_materialized=True,
    )
    plan = choose_residual_plan(adapters, config=planner_config)
    planner_selected = {p.adapter for p in plan.plans if p.kind == PlanKind.MATERIALIZED_RESIDUAL}
    planner_cost = sum(float(p.estimated_latency_cost) for p in plan.plans)
    brute_selected, brute_cost = brute_force_materialized_choice(adapters, planner_config)
    return BruteForceValidationRow(
        config=config_name,
        planner_choice=_choice_string(planner_selected),
        brute_force_choice=_choice_string(brute_selected),
        match=planner_selected == brute_selected,
        planner_cost=float(planner_cost),
        brute_force_cost=float(brute_cost),
        latency_gap=float(planner_cost - brute_cost),
    )
