from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageml.moe_ir import MoEAdapterMeta
from stageml.planner_validation import validate_planner_against_bruteforce
from stageml.residual_planner import PlannerConfig


def write(rows, out: str) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": "cost_model_bruteforce_validate", "rows": [r.to_dict() for r in rows]}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    columns = ["config", "planner_choice", "brute_force_choice", "match", "planner_cost", "brute_force_cost", "latency_gap"]
    with out_path.with_suffix(".csv").open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            d = row.to_dict()
            f.write(",".join(str(d[c]) for c in columns) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_outputs/cost_model_bruteforce_validate.json")
    args = ap.parse_args()

    adapters = [
        MoEAdapterMeta("a0", rank=4, num_experts=2, in_features=8, out_features=8, request_count=120),
        MoEAdapterMeta("a1", rank=8, num_experts=2, in_features=8, out_features=8, request_count=90),
        MoEAdapterMeta("a2", rank=16, num_experts=2, in_features=8, out_features=8, request_count=25),
        MoEAdapterMeta("a3", rank=4, num_experts=2, in_features=8, out_features=8, request_count=10),
    ]
    rows = []
    for mb in [0.0, 0.00025, 0.0005, 0.001]:
        cfg = PlannerConfig(memory_budget_mb=mb, dtype="fp16", hot_request_threshold=20, allow_non_materializing_fusion=False, allow_disaggregated=False)
        rows.append(validate_planner_against_bruteforce(adapters, cfg, config_name=f"budget_mb_{mb}"))
    write(rows, args.out)


if __name__ == "__main__":
    main()
