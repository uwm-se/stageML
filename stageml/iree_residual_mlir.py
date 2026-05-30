from __future__ import annotations

"""IREE-friendly residual MLIR emitters.

The general StageML residual plan is a mixed execution plan. It can contain
external fallback calls, dynamic tensor shapes, and serving/runtime boundaries.
Those are useful for the paper IR, but they are not a good direct input to
IREE's CUDA codegen path.

This module emits the smaller submodule that IREE should compile first: a
single static-shape materialized residual kernel. The kernel corresponds to the
case where StageML has already folded a hot tenant LoRA adapter into an expert
weight W_res, leaving token-time execution as a single matmul.
"""

from dataclasses import dataclass
from pathlib import Path


_VALID_DTYPES = {"f32", "bf16", "f16"}


@dataclass(frozen=True)
class StaticResidualKernelSpec:
    """Shape and type for a materialized residual matmul kernel.

    The emitted function computes Y = X @ W_res.
    * X has shape [tokens, hidden]
    * W_res has shape [hidden, output]
    * Y has shape [tokens, output]
    """

    tokens: int = 16
    hidden: int = 4096
    output: int = 4096
    dtype: str = "f32"
    function_name: str = "stageml_materialized_residual"

    def validate(self) -> None:
        if self.tokens <= 0 or self.hidden <= 0 or self.output <= 0:
            raise ValueError("tokens, hidden, and output must be positive")
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; expected one of {sorted(_VALID_DTYPES)}")
        if not self.function_name.replace("_", "").isalnum():
            raise ValueError("function_name must contain only letters, digits, and underscores")


def emit_static_materialized_residual_mlir(spec: StaticResidualKernelSpec) -> str:
    """Emit a small linalg-on-tensors module accepted by IREE frontends.

    This intentionally avoids dynamic dimensions, external func.call operations,
    HAL/Flow dialect operations, and whole-plan control flow. IREE owns those
    lowerings after this point.
    """
    spec.validate()
    m, k, n, ty = spec.tokens, spec.hidden, spec.output, spec.dtype
    fn = spec.function_name
    return f"""module @stageml_iree_materialized_residual {{
  func.func @{fn}(%x: tensor<{m}x{k}x{ty}>, %w_res: tensor<{k}x{n}x{ty}>) -> tensor<{m}x{n}x{ty}> {{
    %c0 = arith.constant 0.000000e+00 : {ty}
    %empty = tensor.empty() : tensor<{m}x{n}x{ty}>
    %init = linalg.fill ins(%c0 : {ty}) outs(%empty : tensor<{m}x{n}x{ty}>) -> tensor<{m}x{n}x{ty}>
    %y = linalg.matmul ins(%x, %w_res : tensor<{m}x{k}x{ty}>, tensor<{k}x{n}x{ty}>) outs(%init : tensor<{m}x{n}x{ty}>) -> tensor<{m}x{n}x{ty}>
    return %y : tensor<{m}x{n}x{ty}>
  }}
}}
"""


def write_static_materialized_residual_mlir(path: str | Path, spec: StaticResidualKernelSpec) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_static_materialized_residual_mlir(spec))
    return out
