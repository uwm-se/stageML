from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from stageml.triton_generator import write_kernel_package


def percentile(values: list[float], pct: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round((pct / 100.0) * (len(xs) - 1)))))
    return float(xs[i])


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("stageml_generated_residual_kernel", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generated module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def time_cuda(fn, warmups: int, repeats: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / (len(times) - 1)) ** 0.5 if len(times) > 1 else 0.0
    return {
        "runs": len(times),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--in-features", type=int, default=256)
    ap.add_argument("--out-features", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out-dir", default="paper_outputs/triton_generated")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        result = {"status": "skipped", "reason": "CUDA is not available"}
        (out_dir / "triton_generated_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    try:
        import triton  # noqa: F401
    except Exception as exc:
        result = {"status": "skipped", "reason": f"Triton import failed: {exc}"}
        (out_dir / "triton_generated_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(0)
    x = torch.randn(args.batch, args.in_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype) / (args.in_features ** 0.5)
    a = torch.randn(args.rank, args.in_features, device="cuda", dtype=dtype) / (args.in_features ** 0.5)
    b = torch.randn(args.out_features, args.rank, device="cuda", dtype=dtype) / (args.rank ** 0.5)
    w_res = w + b @ a

    package = write_kernel_package(out_dir, base_weight=w.detach().cpu(), a=a.detach().cpu(), b=b.detach().cpu())
    mod = load_module(Path(package["kernel_py"]))

    y_torch = x @ w_res.t()
    y_triton = mod.residual_matmul(x, w_res)
    torch.cuda.synchronize()
    max_abs_error = float((y_torch.float() - y_triton.float()).abs().max().detach().cpu())

    torch_times = time_cuda(lambda: x @ w_res.t(), args.warmups, args.repeats)
    triton_times = time_cuda(lambda: mod.residual_matmul(x, w_res), args.warmups, args.repeats)

    result = {
        "status": "ok",
        "batch": args.batch,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "rank": args.rank,
        "dtype": args.dtype,
        "max_abs_error": max_abs_error,
        "torch_matmul": torch_times,
        "generated_triton": triton_times,
        "generated_files": package,
        "claim_boundary": "Generated Triton kernel consumes materialized residual weight; it does not inline full Mixtral tensors into source.",
    }
    (out_dir / "triton_generated_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
