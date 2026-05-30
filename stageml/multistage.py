"""
Multi-stage binding-time model for adapter serving.

The original StageML prototype used two stages:
    stage0: known before inference
    stage1: known only at request time

For multi-adapter serving this is too coarse.  Adapter selection is not as
static as the base model, but it is also not the same as token activations.
This file adds a three-level lattice:

    base     : frozen base model values known at deployment time
    adapter  : values known once a specific adapter is selected
    request  : per-request activations and tokens

This is the research extension that moves StageML beyond single-adapter LoRA
merging.  It lets us talk about specializing a shared model for a bank of
adapters and then selecting a residual computation at request time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class MultiStage(IntEnum):
    """Binding-time lattice for multi-adapter inference."""

    BASE = 0       # frozen base model value, known at deployment time
    ADAPTER = 1    # value known after an adapter is selected or loaded
    REQUEST = 2    # request activation or token-dependent value


def join(*stages: MultiStage) -> MultiStage:
    """Least upper bound of stages.  Higher number means later binding time."""
    if not stages:
        return MultiStage.BASE
    return MultiStage(max(int(s) for s in stages))


def stage_name(stage: MultiStage | int | str) -> str:
    if isinstance(stage, str):
        return stage
    s = MultiStage(stage)
    return {MultiStage.BASE: "base", MultiStage.ADAPTER: "adapter", MultiStage.REQUEST: "request"}[s]


@dataclass(frozen=True)
class StageInfo:
    name: str
    stage: MultiStage
    reason: str


def summarize_stages(items: Iterable[StageInfo]) -> dict[str, int]:
    out = {"base": 0, "adapter": 0, "request": 0}
    for item in items:
        out[stage_name(item.stage)] += 1
    return out


# Short aliases used by paper examples.
BASE = MultiStage.BASE
ADAPTER = MultiStage.ADAPTER
REQUEST = MultiStage.REQUEST
