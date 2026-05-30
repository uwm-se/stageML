from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class MoEStage(IntEnum):
    BASE = 0
    ADAPTER = 1
    TENANT = 2
    REQUEST = 3
    ROUTING = 4
    TOKEN = 5


BASE = MoEStage.BASE
ADAPTER = MoEStage.ADAPTER
TENANT = MoEStage.TENANT
REQUEST = MoEStage.REQUEST
ROUTING = MoEStage.ROUTING
TOKEN = MoEStage.TOKEN


_STAGE_NAMES = {
    MoEStage.BASE: "base",
    MoEStage.ADAPTER: "adapter",
    MoEStage.TENANT: "tenant",
    MoEStage.REQUEST: "request",
    MoEStage.ROUTING: "routing",
    MoEStage.TOKEN: "token",
}


_STAGE_ALIASES = {
    "base": MoEStage.BASE,
    "stage0": MoEStage.BASE,
    "deployment": MoEStage.BASE,
    "adapter": MoEStage.ADAPTER,
    "tenant": MoEStage.TENANT,
    "request": MoEStage.REQUEST,
    "routing": MoEStage.ROUTING,
    "token": MoEStage.TOKEN,
    "runtime": MoEStage.TOKEN,
    "stage1": MoEStage.TOKEN,
}


def parse_stage(stage: MoEStage | int | str) -> MoEStage:
    if isinstance(stage, MoEStage):
        return stage
    if isinstance(stage, int):
        return MoEStage(stage)
    key = str(stage).strip().lower().replace("_", " ").replace(" ", "")
    if key not in _STAGE_ALIASES:
        raise ValueError(f"unknown MoE stage {stage!r}")
    return _STAGE_ALIASES[key]


def stage_name(stage: MoEStage | int | str) -> str:
    return _STAGE_NAMES[parse_stage(stage)]


def join(*stages: MoEStage | int | str) -> MoEStage:
    if not stages:
        return MoEStage.BASE
    return MoEStage(max(int(parse_stage(s)) for s in stages))


def earlier_or_equal(left: MoEStage | int | str, right: MoEStage | int | str) -> bool:
    return int(parse_stage(left)) <= int(parse_stage(right))


def later_or_equal(left: MoEStage | int | str, right: MoEStage | int | str) -> bool:
    return int(parse_stage(left)) >= int(parse_stage(right))


def can_fold_at(stage: MoEStage | int | str, cutoff: MoEStage | int | str = MoEStage.ADAPTER) -> bool:
    return earlier_or_equal(stage, cutoff)


@dataclass(frozen=True)
class StageFact:
    name: str
    stage: MoEStage
    reason: str


def summarize_stages(items: Iterable[StageFact]) -> dict[str, int]:
    out = {stage_name(s): 0 for s in MoEStage}
    for item in items:
        out[stage_name(item.stage)] += 1
    return out


def lattice_table() -> list[tuple[str, int]]:
    return [(stage_name(s), int(s)) for s in MoEStage]
