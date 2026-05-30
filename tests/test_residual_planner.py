from stageml.moe_ir import MoEAdapterMeta, PlanKind
from stageml.residual_planner import AdapterQuantInfo, PlannerConfig, choose_residual_plan


def test_residual_planner_materializes_hot_adapter_under_budget():
    adapters = [MoEAdapterMeta("hot", 8, 2, 8, 8, request_count=1000)]
    plan = choose_residual_plan(adapters, config=PlannerConfig(memory_budget_mb=1.0, theta=0.1), quant_info={"hot": AdapterQuantInfo(0.0, True)})
    assert plan.plans[0].kind == PlanKind.MATERIALIZED_RESIDUAL


def test_residual_planner_fallback_on_quant_error():
    adapters = [MoEAdapterMeta("bad", 8, 2, 8, 8, request_count=1000)]
    plan = choose_residual_plan(adapters, config=PlannerConfig(memory_budget_mb=1.0, theta=0.1), quant_info={"bad": AdapterQuantInfo(1.0, False)})
    assert plan.plans[0].kind == PlanKind.FALLBACK


def test_residual_planner_uses_fusion_for_small_rank_when_cold():
    adapters = [MoEAdapterMeta("small", 8, 2, 8, 8, request_count=5)]
    plan = choose_residual_plan(adapters, config=PlannerConfig(memory_budget_mb=0.0, theta=0.1), quant_info={"small": AdapterQuantInfo(0.0, True)})
    assert plan.plans[0].kind == PlanKind.NON_MATERIALIZING_FUSION
