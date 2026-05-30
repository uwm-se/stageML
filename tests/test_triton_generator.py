from pathlib import Path

import torch

from stageml.triton_generator import (
    generate_residual_matmul_source,
    materialize_residual_weight,
    write_residual_matmul_kernel,
)


def test_generate_triton_source_contains_jit_and_wrapper(tmp_path: Path):
    src = generate_residual_matmul_source()
    assert "@triton.jit" in src
    assert "def residual_matmul" in src
    assert "tl.dot" in src
    p = write_residual_matmul_kernel(tmp_path / "kernel.py")
    assert p.exists()
    assert "residual_matmul" in p.read_text()


def test_materialize_residual_weight_shape_and_values():
    w = torch.zeros(3, 4)
    a = torch.ones(2, 4)
    b = torch.ones(3, 2)
    out = materialize_residual_weight(w, a, b, scaling=0.5)
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.ones(3, 4))
