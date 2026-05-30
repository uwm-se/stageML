from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from stageml.evaluator import specialize
from stageml.torch_mlir_backend import write_torch_mlir
from stageml.tracer import trace_and_annotate


class LoRALinear(nn.Module):
    def __init__(self, dim=128, rank=8, alpha=8.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        merged = self.W + (self.B @ self.A) * self.scaling
        return x @ merged.t()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output-type", default="linalg-on-tensors")
    parser.add_argument("--out-dir", default="out/torch_mlir")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = LoRALinear(args.dim, args.rank).eval()
    x = torch.randn(args.batch, args.dim)

    gm, annotations = trace_and_annotate(model, {"x": "stage1"})
    residual = specialize(gm, annotations).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_path = write_torch_mlir(model, x, out_dir / "original_lora.mlir", output_type=args.output_type)
    residual_path = write_torch_mlir(residual, x, out_dir / "stageml_residual_lora.mlir", output_type=args.output_type)

    print(f"Wrote {original_path}")
    print(f"Wrote {residual_path}")


if __name__ == "__main__":
    main()
