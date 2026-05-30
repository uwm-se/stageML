"""
tests/test_correctness.py
Numerical correctness tests for the StageML specialization pass.

Run with:
    cd stageml
    python -m pytest tests/test_correctness.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from stageml.tracer    import trace_and_annotate
from stageml.evaluator import specialize


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def test_residual_matches_original():
    """The specialized residual must produce the same output as the original model."""
    model = TinyMLP()
    model.eval()

    x = torch.randn(1, 4)
    with torch.no_grad():
        original_out = model(x)

    gm, gamma = trace_and_annotate(model, {"x": "stage1"})
    gm_residual = specialize(gm, gamma)
    with torch.no_grad():
        residual_out = gm_residual(x)

    max_diff = (original_out - residual_out).abs().max().item()
    print(f"Max absolute difference: {max_diff}")
    assert max_diff < 1e-6, f"Residual output differs from original by {max_diff}"


def test_residual_matches_multiple_inputs():
    """Test correctness across multiple random inputs."""
    model = TinyMLP()
    model.eval()

    gm, gamma = trace_and_annotate(model, {"x": "stage1"})
    gm_residual = specialize(gm, gamma)

    for i in range(10):
        x = torch.randn(1, 4)
        with torch.no_grad():
            original_out = model(x)
            residual_out = gm_residual(x)
        max_diff = (original_out - residual_out).abs().max().item()
        assert max_diff < 1e-6, f"Input {i}: diff = {max_diff}"


def test_residual_has_fewer_nodes():
    model = TinyMLP()
    gm, gamma = trace_and_annotate(model, {"x": "stage1"})
    original_count = len(list(gm.graph.nodes))
    gm_residual = specialize(gm, gamma)
    residual_count = len(list(gm_residual.graph.nodes))
    print(f"Original nodes: {original_count}, Residual nodes: {residual_count}")
    assert residual_count <= original_count, "Residual should not have more nodes"
