from stageml.moe_ir import TensorType, Precision, var, matmul, add
from stageml.moe_stages import ADAPTER, BASE, TOKEN


def test_moe_ir_stage_propagation():
    x = var("x", TensorType((4, 8), Precision.FP32, TOKEN))
    w = var("w", TensorType((8, 16), Precision.FP32, BASE))
    a = matmul(x, w)
    assert a.typ.shape == (4, 16)
    assert a.stage() == TOKEN


def test_moe_ir_adapter_static_add():
    w = var("w", TensorType((8, 16), Precision.FP32, BASE))
    d = var("delta", TensorType((8, 16), Precision.FP32, ADAPTER))
    expr = add(w, d)
    assert expr.stage() == ADAPTER
