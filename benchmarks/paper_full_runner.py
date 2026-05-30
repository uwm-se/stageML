from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, log_path, timeout_sec=None):
    print("RUN", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout_sec)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout_sec} seconds\n")
            code = 124
    print("EXIT", code, "LOG", log_path)
    return code


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(out_dir, commands):
    path = out_dir / "paper_run_manifest.txt"
    path.write_text("\n".join(" ".join(cmd) for cmd in commands), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out-dir", default="out/paper_run")
    parser.add_argument("--include-torch-mlir", action="store_true")
    parser.add_argument("--skip-torch-compile", action="store_true")
    parser.add_argument("--command-timeout-sec", type=int, default=900)
    parser.add_argument("--model-adapter-manifest", default=None, help="Optional model/adapter sweep manifest JSON")
    parser.add_argument("--run-real-model-sweep", action="store_true", help="Run accepted_fusion_h100_bench.py over manifest entries")
    parser.add_argument("--prompts-jsonl", default="/data/stageml_h100_run/data/real_trace.jsonl")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    commands = []

    cmd = [
        py, "benchmarks/gpu_lora_moe_bench.py",
        "--dim", str(args.dim),
        "--rank", str(args.rank),
        "--batch", str(args.batch),
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--out", str(out_dir / "gpu_lora_moe_results.csv"),
    ]
    commands.append(cmd)
    run(cmd, out_dir / "logs" / "gpu_lora_moe.log", args.command_timeout_sec)

    for seq_len in args.seq_lens:
        cmd = [
            py, "benchmarks/llama_scale_lora_block_bench.py",
            "--hidden-size", str(args.dim),
            "--rank", str(args.rank),
            "--batch", str(args.batch),
            "--seq-len", str(seq_len),
            "--dtype", args.dtype,
            "--warmup", str(max(10, args.warmup // 2)),
            "--iterations", str(max(30, args.iterations // 2)),
            "--out", str(out_dir / f"llama_scale_lora_seq{seq_len}.csv"),
        ]
        commands.append(cmd)
        run(cmd, out_dir / "logs" / f"llama_scale_lora_seq{seq_len}.log", args.command_timeout_sec)

    if not args.skip_torch_compile:
        cmd = [
            py, "benchmarks/torch_compile_comparison.py",
            "--dim", str(args.dim),
            "--rank", str(args.rank),
            "--batch", str(args.batch),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--out", str(out_dir / "torch_compile_comparison.csv"),
            "--graph-out", str(out_dir / "torch_compile_exported_graph.txt"),
        ]
        commands.append(cmd)
        run(cmd, out_dir / "logs" / "torch_compile_comparison.log", args.command_timeout_sec)

    cmd = [
        py, "benchmarks/rewrite_ablation_bench.py",
        "--dim", str(args.dim),
        "--rank", str(args.rank),
        "--batch", str(args.batch),
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--out", str(out_dir / "rewrite_ablation.csv"),
        "--artifacts", str(out_dir / "rewrite_ablation_artifacts"),
    ]
    commands.append(cmd)
    run(cmd, out_dir / "logs" / "rewrite_ablation.log", args.command_timeout_sec)

    cmd = [
        py, "benchmarks/lora_baselines_bench.py",
        "--dim", str(args.dim),
        "--rank", str(args.rank),
        "--batch", str(args.batch),
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--out", str(out_dir / "lora_baselines.csv"),
    ]
    commands.append(cmd)
    run(cmd, out_dir / "logs" / "lora_baselines.log", args.command_timeout_sec)

    if args.include_torch_mlir:
        cmd = [
            py, "benchmarks/torch_mlir_demo.py",
            "--dim", "128",
            "--rank", "8",
            "--batch", "1",
            "--output-type", "linalg-on-tensors",
            "--out-dir", str(out_dir / "torch_mlir"),
        ]
        commands.append(cmd)
        run(cmd, out_dir / "logs" / "torch_mlir_demo.log", args.command_timeout_sec)


    if args.run_real_model_sweep:
        if not args.model_adapter_manifest:
            raise SystemExit("--run-real-model-sweep requires --model-adapter-manifest")
        manifest = json.loads(Path(args.model_adapter_manifest).read_text(encoding="utf-8"))
        sweep_dir = out_dir / "real_model_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for model_item in manifest.get("models", []):
            model_repo = model_item.get("repo")
            if not model_repo:
                continue
            for adapter_repo_or_dir in model_item.get("adapter_dirs", model_item.get("adapter_repos", [])):
                adapter_name = str(adapter_repo_or_dir).replace("/", "__")
                run_dir = sweep_dir / model_repo.replace("/", "__") / adapter_name
                cmd = [
                    py, "benchmarks/accepted_fusion_h100_bench.py",
                    "--model", model_repo,
                    "--adapter-dirs", str(adapter_repo_or_dir),
                    "--prompts-jsonl", args.prompts_jsonl,
                    "--out-dir", str(run_dir),
                    "--device", "cuda",
                    "--device-map", "auto",
                    "--memory-budget-mb", "2048",
                    "--repeats", "5",
                    "--bench-dtype", "bf16",
                    "--vllm-csv", "paper_outputs/vllm_lora_baseline.csv",
                ]
                commands.append(cmd)
                run(cmd, out_dir / "logs" / f"real_sweep_{model_repo.replace('/', '__')}_{adapter_name}.log", args.command_timeout_sec)

    write_manifest(out_dir, commands)

    summary_rows = []
    for csv_path in out_dir.glob("*.csv"):
        for row in read_csv_rows(csv_path):
            row = {"source_file": csv_path.name, **row}
            summary_rows.append(row)

    summary_path = out_dir / "combined_summary.csv"
    if summary_rows:
        keys = []
        for row in summary_rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote {summary_path}")

    print(f"Paper run artifacts are under {out_dir}")


if __name__ == "__main__":
    main()
