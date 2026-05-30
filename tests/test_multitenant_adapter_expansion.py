import torch

from benchmarks.multitenant_bta_system_bench import expand_adapters
from stageml.moe_lora_layers import MoEAdapterSpec


def _adapter(name: str) -> MoEAdapterSpec:
    return MoEAdapterSpec(name, torch.ones(2, 2, 4), torch.ones(2, 3, 2), 1.0)


def test_strict_real_adapter_expansion_uses_independent_inputs():
    adapters, virtual, mode = expand_adapters([_adapter("a"), _adapter("b")], 2, device=torch.device("cpu"), dtype=torch.float32, mode="strict_real")
    assert [a.name for a in adapters] == ["a", "b"]
    assert not virtual
    assert mode == "strict_real"


def test_strict_real_adapter_expansion_rejects_missing_inputs():
    try:
        expand_adapters([_adapter("a")], 2, device=torch.device("cpu"), dtype=torch.float32, mode="strict_real")
    except ValueError as exc:
        assert "strict_real" in str(exc)
    else:
        raise AssertionError("expected strict_real failure")
