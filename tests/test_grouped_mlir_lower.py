from stageml.moe_ir import ExpertPlan, PlanKind, ResidualPlan
from stageml.moe_mlir_lower import lower_residual_plan_to_mlir


def test_grouped_materialized_lowering_contains_batch_matmul():
    plan = ResidualPlan(
        plans=(
            ExpertPlan(
                adapter="a",
                expert="all",
                kind=PlanKind.MATERIALIZED_RESIDUAL,
                rank=16,
                memory_bytes=1,
                estimated_latency_cost=1.0,
                epsilon=0.0,
                reason="test",
            ),
        ),
        memory_budget_bytes=1024,
        used_memory_bytes=1,
        theta=0.0,
    )
    mlir = lower_residual_plan_to_mlir(plan)
    assert "linalg.batch_matmul" in mlir
    assert "stageml_grouped_materialized_residual" in mlir
