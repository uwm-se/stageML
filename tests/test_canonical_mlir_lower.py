from pathlib import Path

import torch

from benchmarks.rewrite_ablation_bench import LoraLibStyleLinear
from stageml.canonical_mlir_lower import lower_to_canonical_mlir
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


def test_canonical_mlir_uses_registered_dialects_for_residual_lora():
    model = LoraLibStyleLinear(dim=8, rank=2).eval()
    x = torch.randn(1, 8)
    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    gm, annotations, _ = optimize_evaluation_order(gm, annotations)
    residual = specialize(gm, annotations)
    mlir = lower_to_canonical_mlir(residual, fn_name="test", example_args=(x,))
    assert "linalg.matmul" in mlir
    assert "arith.constant" in mlir
    assert "stageml." not in mlir
    assert "func.func" in mlir
