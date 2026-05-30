import torch

from stageml.moe_lora_layers import DynamicMoELoRALayer, MaterializedMoELoRALayer, make_random_moe_adapters, make_round_robin_adapter_ids, normalize_routing_weights


def test_materialized_moe_lora_matches_dynamic():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    num_experts = 4
    in_features = 8
    out_features = 6
    num_adapters = 3
    top_k = 2
    tokens = 7
    expert_weight = torch.randn(num_experts, out_features, in_features, dtype=dtype, device=device)
    adapters = make_random_moe_adapters(
        num_adapters=num_adapters,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        ranks=[2, 3],
        dtype=dtype,
        device=device,
    )
    x = torch.randn(tokens, in_features, dtype=dtype, device=device)
    expert_ids = torch.tensor([[i % num_experts, (i + 1) % num_experts] for i in range(tokens)], dtype=torch.long)
    routing_weights = normalize_routing_weights(expert_ids)
    adapter_ids = make_round_robin_adapter_ids(tokens, num_adapters)
    dyn = DynamicMoELoRALayer(expert_weight, adapters)
    mat = MaterializedMoELoRALayer(expert_weight, adapters)
    y_dyn = dyn(x, expert_ids, routing_weights, adapter_ids)
    y_mat = mat(x, expert_ids, routing_weights, adapter_ids)
    assert torch.allclose(y_dyn, y_mat, atol=1e-5, rtol=1e-5)
