"""Sweep the StageML adapter-bank benchmark across adapter counts and batches."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-counts", default="1,2,4,8,16")
    parser.add_argument("--batches", default="8,16,32,64")
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--out", default="out/research/multi_adapter_sweep.csv")
    args = parser.parse_args()

    adapter_counts = [int(x) for x in args.adapter_counts.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    tmp_dir = Path("out/research/tmp_multi_adapter")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for num_adapters in adapter_counts:
        for batch in batches:
            tmp = tmp_dir / f"adapters_{num_adapters}_batch_{batch}.csv"
            cmd = [
                sys.executable,
                "benchmarks/multi_adapter_bank_bench.py",
                "--in-features", str(args.dim),
                "--out-features", str(args.dim),
                "--rank", str(args.rank),
                "--batch", str(batch),
                "--num-adapters", str(num_adapters),
                "--dtype", args.dtype,
                "--warmup", str(args.warmup),
                "--iterations", str(args.iterations),
                "--out", str(tmp),
            ]
            print("running", " ".join(cmd))
            subprocess.check_call(cmd)
            with tmp.open(newline="", encoding="utf-8") as f:
                all_rows.extend(list(csv.DictReader(f)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
