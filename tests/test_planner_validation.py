from stageml.moe_ir import MoEAdapterMeta
from stageml.planner_validation import validate_planner_against_bruteforce
from stageml.residual_planner import PlannerConfig


def test_planner_matches_bruteforce_for_small_materialization_case():
    adapters = [
        MoEAdapterMeta("a", 4, 2, 8, 8, request_count=100),
        MoEAdapterMeta("b", 4, 2, 8, 8, request_count=10),
    ]
    cfg = PlannerConfig(memory_budget_mb=0.001, dtype="fp16", hot_request_threshold=20, allow_non_materializing_fusion=False, allow_disaggregated=False)
    row = validate_planner_against_bruteforce(adapters, cfg, config_name="unit")
    assert row.match
    assert row.latency_gap == 0.0
