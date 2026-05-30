from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageml.benchmark_env import capture_environment
from stageml.h100_guard import require_h100
from stageml.quant_experiments import find_middle_band_case, write_quant_cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_outputs/quantization_middle_band.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shape", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--theta-multiplier", type=float, default=4.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--require-h100", action="store_true")
    args = ap.parse_args()

    h100_environment = require_h100() if args.require_h100 else None
    case = find_middle_band_case(seed=args.seed, shape=args.shape, rank=args.rank, theta_multiplier=args.theta_multiplier, device=args.device)
    out_json, out_csv = write_quant_cases([case], args.out)
    result = {
        "benchmark": "quantization_middle_band_bench",
        "purpose": "exercise a nonzero epsilon case that is still accepted by the quantization safety gate",
        "case": case.to_dict(),
        "device": args.device,
        "environment": capture_environment(args.device),
        "h100_environment": h100_environment,
        "downstream_validation": "deterministic layer output error at the residualized linear boundary. Run run_perplexity_eval.py separately for the full H100 language modeling guard.",
        "out_json": str(out_json),
        "out_csv": str(out_csv),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
