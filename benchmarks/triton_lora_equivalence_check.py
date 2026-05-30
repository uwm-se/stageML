from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def pytorch_lora(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    return x @ w + alpha * ((x @ a) @ b)


def run(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.in_features, device=device, dtype=torch.float32)
    w = torch.randn(args.in_features, args.out_features, device=device, dtype=torch.float32)
    a = torch.randn(args.in_features, args.rank, device=device, dtype=torch.float32)
    b = torch.randn(args.rank, args.out_features, device=device, dtype=torch.float32)
    w_res = w + args.alpha * (a @ b)

    # This is an executable equivalence check. It is not formal kernel verification.
    y_dyn = pytorch_lora(x, w, a, b, args.alpha)
    y_res = x @ w_res
    torch.cuda.synchronize() if device == "cuda" else None
    max_err = float((y_dyn - y_res).abs().max().detach().cpu())

    for _ in range(args.warmups):
        pytorch_lora(x, w, a, b, args.alpha)
        x @ w_res
    torch.cuda.synchronize() if device == "cuda" else None

    dyn_times = []
    res_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        pytorch_lora(x, w, a, b, args.alpha)
        torch.cuda.synchronize() if device == "cuda" else None
        dyn_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        x @ w_res
        torch.cuda.synchronize() if device == "cuda" else None
        res_times.append(time.perf_counter() - t0)

    out = {
        "device": device,
        "batch": args.batch,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "rank": args.rank,
        "alpha": args.alpha,
        "max_abs_error": max_err,
        "dynamic_mean_ms": 1000.0 * sum(dyn_times) / len(dyn_times),
        "residual_mean_ms": 1000.0 * sum(res_times) / len(res_times),
        "note": "This is randomized executable equivalence checking, not formal CUDA or Triton verification.",
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--in-features", type=int, default=4096)
    ap.add_argument("--out-features", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="paper_outputs/kernel_equivalence.json")
    args = ap.parse_args()
    result = run(args)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
