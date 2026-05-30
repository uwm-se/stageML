import pytest

from stageml.disaggregated_execution import DISAGGREGATED_EXECUTION_STATUS, disaggregated_status, require_executable_disaggregated
from stageml.moe_ir import MoEAdapterMeta, PlanKind
from stageml.residual_planner import PlannerConfig, choose_residual_plan


def test_disaggregated_status_is_explicit():
    status = disaggregated_status()
    assert status["status"] == DISAGGREGATED_EXECUTION_STATUS
    with pytest.raises(NotImplementedError):
        require_executable_disaggregated()


def test_planner_marks_disaggregated_as_not_executable():
    adapters = [MoEAdapterMeta("hot", rank=64, num_experts=4, in_features=64, out_features=64, request_count=1000)]
    plan = choose_residual_plan(
        adapters,
        config=PlannerConfig(memory_budget_mb=0.0, hot_request_threshold=1, disaggregate_request_threshold=10, rank_specialization_threshold=1),
    )
    assert plan.plans[0].kind == PlanKind.DISAGGREGATED
    assert "not yet executable" in plan.plans[0].reason
