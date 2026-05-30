from stageml.moe_ir import ExpertPlan, PlanKind, ResidualPlan
from stageml.moe_mlir_lower import lower_residual_plan_to_mlir


def test_lower_all_residual_plan_kinds_to_mlir_text():
    plans = []
    for i, kind in enumerate(PlanKind):
        plans.append(
            ExpertPlan(
                adapter=f"adapter_{i}",
                expert=i,
                kind=kind,
                rank=8,
                memory_bytes=0,
                estimated_latency_cost=1.0,
                epsilon=0.0,
                reason="test",
            )
        )
    plan = ResidualPlan(tuple(plans), memory_budget_bytes=1024, used_memory_bytes=0, theta=1.0)
    mlir = lower_residual_plan_to_mlir(plan)
    assert "module @stageml_moe_residual_plan" in mlir
    assert "materialized_residual_adapter_2" in mlir or "materialized_residual" in mlir
    assert "non_materializing_fusion" in mlir
    assert "stageml_runtime_fallback" in mlir
    assert "stageml_dynamic_lora" in mlir
