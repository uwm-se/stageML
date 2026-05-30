from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch.fx.passes.shape_prop import ShapeProp

from benchmarks.rewrite_ablation_bench import LoraLibStyleLinear
from stageml.canonical_mlir_lower import write_canonical_mlir, verify_canonical_mlir_with_mlir_opt
from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out-dir", default="out/canonical_mlir_demo")
    parser.add_argument("--max-dense-elements", type=int, default=16384)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = LoraLibStyleLinear(dim=args.dim, rank=args.rank).eval()
    x = torch.randn(args.batch, args.dim)

    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    ShapeProp(gm).propagate(x)

    original_path = write_canonical_mlir(
        gm,
        out_dir / "original_lora_canonical.mlir",
        fn_name="original_lora",
        max_dense_elements=args.max_dense_elements,
    )

    rewritten_gm, rewritten_ann, stats = optimize_evaluation_order(gm, annotations)
    ShapeProp(rewritten_gm).propagate(x)
    residual_gm = specialize(rewritten_gm, rewritten_ann)
    ShapeProp(residual_gm).propagate(x)

    residual_path = write_canonical_mlir(
        residual_gm,
        out_dir / "stageml_residual_canonical.mlir",
        fn_name="stageml_residual_lora",
        max_dense_elements=args.max_dense_elements,
    )

    summary_lines = [
        f"rewrite_count={stats.total_rewrites}",
        f"original_mlir={original_path}",
        f"residual_mlir={residual_path}",
    ]

    if shutil.which("mlir-opt"):
        for label, path in [("original", original_path), ("residual", residual_path)]:
            ok, msg = verify_canonical_mlir_with_mlir_opt(path)
            summary_lines.append(f"{label}_verify_ok={ok}")
            (out_dir / f"{label}_verify.txt").write_text(str(ok) + "\n" + msg, encoding="utf-8")
    else:
        summary_lines.append("mlir_opt=not_found")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
