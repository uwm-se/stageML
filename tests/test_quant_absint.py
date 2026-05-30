import torch

from stageml.quant_absint import QuantizationConfig, lora_delta, analyze_residualization, safe_to_residualize


def test_quant_error_bound_reports_safe_when_theta_large():
    torch.manual_seed(0)
    W = torch.randn(8, 8)
    A = torch.randn(2, 8) * 0.01
    B = torch.randn(8, 2) * 0.01
    bound = safe_to_residualize(W, A, B, 1.0, theta=100.0, config=QuantizationConfig(bits=4))
    assert bound.safe
    assert bound.epsilon_weight_fro >= 0.0


def test_quant_error_bound_rejects_when_theta_zero_for_nonzero_error():
    torch.manual_seed(1)
    W = torch.randn(8, 8)
    A = torch.randn(2, 8)
    B = torch.randn(8, 2)
    delta = lora_delta(A, B, 1.0)
    bound = analyze_residualization(W, delta, theta=0.0, config=QuantizationConfig(bits=4))
    assert bound.epsilon_weight_fro >= 0.0
    assert bound.safe == (bound.epsilon_weight_fro == 0.0)
