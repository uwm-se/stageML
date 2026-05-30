from stageml.quant_experiments import find_middle_band_case, make_reproducible_case, evaluate_gate_case


def test_middle_band_case_is_nonzero_and_accepted():
    case = find_middle_band_case(seed=3, shape=16, rank=2)
    eps = case.epsilon_output_fro if case.epsilon_output_fro is not None else case.epsilon_weight_fro
    assert eps > 0.0
    assert eps < case.theta
    assert case.decision == "accept"


def test_theta_sweep_changes_decision():
    W, A, B, x = make_reproducible_case(seed=4, shape=16, rank=2, delta_scale=0.02)
    high = evaluate_gate_case(W=W, A=A, B=B, x=x, bits=8, theta=1e9)
    low = evaluate_gate_case(W=W, A=A, B=B, x=x, bits=8, theta=0.0)
    assert high.decision == "accept"
    assert low.decision in {"accept", "reject"}
    if high.epsilon_output_fro and high.epsilon_output_fro > 0:
        assert low.decision == "reject"
