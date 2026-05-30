from __future__ import annotations

import json
from pathlib import Path

from stageml.moe_ir import ResidualPlan


def plan_to_json(plan: ResidualPlan, *, indent: int = 2) -> str:
    return json.dumps(plan.to_dict(), indent=indent, sort_keys=True)


def write_plan(plan: ResidualPlan, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan_to_json(plan) + "\n", encoding="utf-8")
    return p
