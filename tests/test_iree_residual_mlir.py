from __future__ import annotations

from pathlib import Path

from stageml.baremetal_backend import IREECompileConfig, build_iree_compile_command
from stageml.iree_residual_mlir import StaticResidualKernelSpec, emit_static_materialized_residual_mlir, write_static_materialized_residual_mlir


def test_static_residual_mlir_uses_static_shapes_and_no_runtime_calls() -> None:
    mlir = emit_static_materialized_residual_mlir(StaticResidualKernelSpec(tokens=4, hidden=8, output=16))
    assert "tensor<4x8xf32>" in mlir
    assert "tensor<8x16xf32>" in mlir
    assert "tensor<4x16xf32>" in mlir
    assert "tensor<?" not in mlir
    assert "func.call" not in mlir
    assert "linalg.matmul" in mlir
    assert "linalg.fill" in mlir


def test_static_residual_mlir_write(tmp_path: Path) -> None:
    path = write_static_materialized_residual_mlir(tmp_path / "residual.mlir", StaticResidualKernelSpec(tokens=2, hidden=3, output=5, dtype="f32"))
    assert path.exists()
    assert "tensor<2x3xf32>" in path.read_text()


def test_iree_target_device_command() -> None:
    cfg = IREECompileConfig(input_mlir="x.mlir", output_dir="out", target_device="cuda", cuda_target="sm_90")
    cmd = build_iree_compile_command(cfg)
    joined = " ".join(cmd)
    assert "--iree-hal-target-device=cuda" in joined
    assert "--iree-hal-target-backends=cuda" not in joined
    assert "--iree-cuda-target=sm_90" in joined
