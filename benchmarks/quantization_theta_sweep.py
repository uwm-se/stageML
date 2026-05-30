from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageml.benchmark_env import capture_environment
from stageml.h100_guard import require_h100
from stageml.quant_experiments import evaluate_gate_case, make_reproducible_case, write_quant_cases


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_outputs/quantization_theta_sweep.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shape", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--bits", default="4,6,8")
    ap.add_argument("--thetas", default="0,0.01,0.05,0.1,0.5,1,5,10")
    ap.add_argument("--delta-scale", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--require-h100", action="store_true")
    args = ap.parse_args()

    h100_environment = require_h100() if args.require_h100 else None
    W, A, B, x = make_reproducible_case(seed=args.seed, shape=args.shape, rank=args.rank, delta_scale=args.delta_scale, device=args.device)
    cases = []
    for bits in parse_ints(args.bits):
        for theta in parse_floats(args.thetas):
            cases.append(evaluate_gate_case(W=W, A=A, B=B, x=x, bits=bits, theta=theta, per_channel=True))
    out_json, out_csv = write_quant_cases(cases, args.out)
    payload = {
        "benchmark": "quantization_theta_sweep",
        "purpose": "sweep theta and precision to expose the accept reject boundary of the quantization safety gate",
        "device": args.device,
        "environment": capture_environment(args.device),
        "h100_environment": h100_environment,
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "cases": [c.to_dict() for c in cases],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
