"""
Publication-oriented StageML policy sweep.

This script sweeps adapter count, request skew, and cache budget to show when a
staged residualization policy is useful.  It is meant to generate a table for a
paper, not to claim replacement of LoRAX/vLLM.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-counts", default="8,16,32")
    p.add_argument("--budgets-mb", default="128,256,512")
    p.add_argument("--hotness", default="0.6,1.2,1.8")
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=120)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default="out/research/policy_impact_sweep.csv")
    args = p.parse_args()

    adapter_counts = [int(x) for x in args.adapter_counts.split(",") if x]
    budgets = [float(x) for x in args.budgets_mb.split(",") if x]
    hotness_values = [float(x) for x in args.hotness.split(",") if x]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out.parent / "policy_sweep_tmp"
    tmp_dir.mkdir(exist_ok=True)

    rows: list[dict[str, str]] = []
    for num_adapters, budget, hotness in itertools.product(adapter_counts, budgets, hotness_values):
        tmp = tmp_dir / f"serving_a{num_adapters}_b{int(budget)}_h{hotness}.csv"
        cmd = [
            sys.executable,
            "benchmarks/end_to_end_adapter_serving_bench.py",
            "--in-features", str(args.in_features),
            "--out-features", str(args.out_features),
            "--rank", str(args.rank),
            "--batch", str(args.batch),
            "--num-adapters", str(num_adapters),
            "--num-batches", str(args.num_batches),
            "--warmup-batches", str(args.warmup_batches),
            "--hotness", str(hotness),
            "--cache-budget-mb", str(budget),
            "--dtype", args.dtype,
            "--out", str(tmp),
        ]
        print("running", " ".join(cmd))
        subprocess.run(cmd, check=True)
        for row in parse_csv(tmp):
            rows.append(row)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
