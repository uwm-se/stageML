import operator

import torch
import torch.nn as nn

from stageml.annotations import stage0
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


class LoraLibStyleLinear(nn.Module):
    def __init__(self, dim=16, rank=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scaling = 1.0 / rank

    def forward(self, x):
        base = x @ self.W.t()
        lora_branch = (x @ self.A.t()) @ self.B.t()
        return base + lora_branch * self.scaling


def test_loralib_style_rewrite_exposes_static_matmul():
    torch.manual_seed(0)
    model = LoraLibStyleLinear().eval()
    x = torch.randn(2, 16)

    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    before_static_compute = sum(1 for n in gm.graph.nodes if annotations.get(n) == stage0 and n.op == "call_function")

    rewritten, rewritten_annotations, stats = optimize_evaluation_order(gm, annotations)
    after_static_compute = sum(1 for n in rewritten.graph.nodes if rewritten_annotations.get(n) == stage0 and n.op == "call_function")

    assert stats.total_rewrites >= 1
    assert after_static_compute > before_static_compute

    residual = specialize(rewritten, rewritten_annotations)
    with torch.no_grad():
        diff = (model(x) - residual(x)).abs().max().item()
    assert diff < 1e-5
