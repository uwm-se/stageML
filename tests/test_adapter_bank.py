import torch

from stageml.adapter_bank import DynamicLoRALinear, StageMLAdapterBankLinear, make_adapter_ids, make_random_adapters
from stageml.multistage import BASE, ADAPTER, REQUEST, join


def test_multistage_join():
    assert join(BASE, BASE) == BASE
    assert join(BASE, ADAPTER) == ADAPTER
    assert join(ADAPTER, REQUEST) == REQUEST


def test_adapter_bank_matches_dynamic_lora():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    W = torch.randn(8, 8, device=device, dtype=dtype)
    bias = torch.randn(8, device=device, dtype=dtype)
    adapters = make_random_adapters(num_adapters=3, in_features=8, out_features=8, rank=2, dtype=dtype, device=device)
    x = torch.randn(6, 8, device=device, dtype=dtype)
    adapter_ids = make_adapter_ids(6, 3, "round_robin")
    dyn = DynamicLoRALinear(W, adapters, bias)
    bank = StageMLAdapterBankLinear(W, adapters, bias)
    y_dyn = dyn(x, adapter_ids)
    y_bank = bank(x, adapter_ids)
    assert torch.allclose(y_dyn, y_bank, atol=1e-5, rtol=1e-5)
