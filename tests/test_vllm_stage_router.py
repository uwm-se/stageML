from __future__ import annotations

import torch

from stageml.vllm_stage_router import StageMLTenantRouter, TenantKernel, route_fused_experts


def fake_vllm(hidden, gate_up, down, topk_weights, topk_ids):
    return torch.zeros((hidden.shape[0], down.shape[1]), device=hidden.device, dtype=hidden.dtype) + 1


def fake_stage(hidden, gate_up, down, topk_weights, topk_ids):
    return torch.zeros((hidden.shape[0], down.shape[1]), device=hidden.device, dtype=hidden.dtype) + 2


def test_router_splits_materialized_and_dynamic_tokens() -> None:
    router = StageMLTenantRouter({
        0: TenantKernel(tenant_id=0, adapter_id=0, backend="test", artifact="x", symbol="s"),
    })
    hidden = torch.randn(4, 3)
    gate_up = torch.randn(2, 4, 3)
    down = torch.randn(2, 3, 2)
    topk_weights = torch.ones(4, 1)
    topk_ids = torch.zeros(4, 1, dtype=torch.long)
    tenant_ids = torch.tensor([0, 1, 0, 1])
    out = route_fused_experts(
        hidden=hidden,
        gate_up=gate_up,
        down=down,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        tenant_ids=tenant_ids,
        router=router,
        vllm_fused_experts=fake_vllm,
        stageml_op=fake_stage,
    )
    assert out.shape == (4, 3)
    assert torch.equal(out[:, 0], torch.tensor([2.0, 1.0, 2.0, 1.0]))


def test_router_rejects_bad_tenant_shape() -> None:
    router = StageMLTenantRouter({})
    hidden = torch.randn(4, 3)
    gate_up = torch.randn(2, 4, 3)
    down = torch.randn(2, 3, 2)
    topk_weights = torch.ones(4, 1)
    topk_ids = torch.zeros(4, 1, dtype=torch.long)
    try:
        route_fused_experts(
            hidden=hidden,
            gate_up=gate_up,
            down=down,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            tenant_ids=torch.tensor([0, 1]),
            router=router,
            vllm_fused_experts=fake_vllm,
            stageml_op=fake_stage,
        )
    except ValueError as exc:
        assert "one tenant id per token" in str(exc)
    else:
        raise AssertionError("expected ValueError")
