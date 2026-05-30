from __future__ import annotations

"""Status marker for StageML disaggregated execution plans."""

DISAGGREGATED_EXECUTION_STATUS = "expressible in IR, not yet executable"


def disaggregated_status() -> dict[str, str]:
    return {
        "plan_kind": "disaggregated",
        "status": DISAGGREGATED_EXECUTION_STATUS,
        "claim_boundary": "The IR can represent a disaggregated adapter execution plan, but this artifact does not provide a runnable disaggregated executor.",
    }


def require_executable_disaggregated() -> None:
    raise NotImplementedError(DISAGGREGATED_EXECUTION_STATUS)
