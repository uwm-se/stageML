import torch
import torch.nn as nn

from stageml.real_mlir_lower import lower_to_parseable_mlir
from stageml.tracer import trace_and_annotate


class StaticMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(4, 4))
        self.B = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return x @ (self.A @ self.B)


def test_parseable_mlir_contains_generic_ops_and_stage_attrs():
    model = StaticMerge().eval()
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    text = lower_to_parseable_mlir(gm, annotations)
    assert "builtin.module" in text
    assert "func.func @staged_fn" in text
    assert '"stageml.' in text
    assert "stageml.stage = 0" in text
    assert "func.return" in text
