from __future__ import annotations

import argparse

from stageml.iree_residual_mlir import StaticResidualKernelSpec, write_static_materialized_residual_mlir


def main() -> None:
    p = argparse.ArgumentParser(description="Emit an IREE-friendly static StageML residual kernel")
    p.add_argument("--out", required=True, help="Output MLIR path")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--output", type=int, default=4096)
    p.add_argument("--dtype", choices=["f32", "bf16", "f16"], default="f32")
    p.add_argument("--function-name", default="stageml_materialized_residual")
    args = p.parse_args()

    spec = StaticResidualKernelSpec(
        tokens=args.tokens,
        hidden=args.hidden,
        output=args.output,
        dtype=args.dtype,
        function_name=args.function_name,
    )
    path = write_static_materialized_residual_mlir(args.out, spec)
    print(path)


if __name__ == "__main__":
    main()
