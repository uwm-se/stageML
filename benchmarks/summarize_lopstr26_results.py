#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, nd=3):
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_runtime_table(data: dict, out: Path, vllm: dict | None = None) -> None:
    dyn = data.get("dynamic", {})
    mat = data.get("materialized", {})
    speedup = dyn.get("p50_ms", 0.0) / mat.get("p50_ms", 1.0) if mat.get("p50_ms", 0.0) else 0.0
    rows = [
        ["StageML dynamic reference", dyn.get("p50_ms", 0.0), dyn.get("p95_ms", 0.0), dyn.get("mean_ms", 0.0), 1.0],
        ["StageML materialized residual", mat.get("p50_ms", 0.0), mat.get("p95_ms", 0.0), mat.get("mean_ms", 0.0), speedup],
    ]
    if vllm:
        rows.append(["vLLM LoRA baseline", vllm.get("p50_ms", 0.0), vllm.get("p95_ms", 0.0), vllm.get("mean_ms", 0.0), 0.0])
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real MoE LoRA benchmark latency}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Plan & P50 ms & P95 ms & Mean ms & Speedup \\",
        r"\midrule",
    ]
    for name, p50, p95, mean, sp in rows:
        latex.append(f"{name} & {fmt(p50)} & {fmt(p95)} & {fmt(mean)} & {fmt(sp)} \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out / "table_runtime.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_correctness_table(data: dict, out: Path) -> None:
    max_err = data.get("max_abs_error_dynamic_vs_materialized", 0.0)
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Correctness check for exact residualization}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Check & Value \\",
        r"\midrule",
        f"Maximum absolute error & {fmt(max_err, 8)} \\",
        f"Tokens evaluated & {data.get('num_tokens', 0)} \\",
        f"Experts used & {data.get('num_experts', 0)} \\",
        f"Top k & {data.get('top_k', 0)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (out / "table_correctness.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_quant_table(data: dict, out: Path) -> None:
    q = data.get("quantization", {})
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantization abstract interpretation decisions}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Adapter & Epsilon & Decision \\",
        r"\midrule",
    ]
    for name, info in sorted(q.items()):
        decision = "accept" if info.get("safe") else "reject"
        latex.append(f"{name} & {fmt(float(info.get('epsilon', 0.0)), 6)} & {decision} \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out / "table_quantization.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_plan_excerpt(data: dict, out: Path) -> None:
    plan_path = data.get("plan_path")
    if plan_path and Path(plan_path).exists():
        plan = read_json(Path(plan_path))
    else:
        plan = {"summary": data.get("plan_summary", {})}
    excerpt = {
        "memory_budget_bytes": plan.get("memory_budget_bytes"),
        "used_memory_bytes": plan.get("used_memory_bytes"),
        "theta": plan.get("theta"),
        "summary": plan.get("summary"),
        "plans": plan.get("plans", [])[:5],
    }
    (out / "residual_plan_excerpt.json").write_text(json.dumps(excerpt, indent=2), encoding="utf-8")


def write_environment(data: dict, out: Path) -> None:
    env = data.get("environment", {})
    lines = [
        "# Hardware and software environment",
        "",
        f"Model under test: {data.get('model')}",
        f"Expert weight path: {data.get('expert_weight_name')}",
        f"Gate path: {data.get('gate_name')}",
        f"GPU: {env.get('gpu_name', 'unknown')}",
        f"NVIDIA SMI: {env.get('nvidia_smi', 'unknown')}",
        f"Torch: {env.get('torch', 'unknown')}",
        f"CUDA runtime: {env.get('cuda_runtime', 'unknown')}",
        f"Python: {env.get('python', 'unknown')}",
    ]
    (out / "environment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--vllm", default=None, help="Optional vLLM baseline JSON or CSV summary")
    ap.add_argument("--out-dir", default="paper_outputs")
    args = ap.parse_args()
    data = read_json(Path(args.results))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vllm = None
    if args.vllm and Path(args.vllm).exists():
        if args.vllm.endswith(".json"):
            vllm = read_json(Path(args.vllm))
        else:
            import csv
            with Path(args.vllm).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                vllm = {k: float(v) if v.replace('.', '', 1).isdigit() else v for k, v in rows[0].items()}

    write_runtime_table(data, out, vllm)
    write_correctness_table(data, out)
    write_quant_table(data, out)
    write_plan_excerpt(data, out)
    write_environment(data, out)

    summary = {
        "model": data.get("model"),
        "expert_weight_name": data.get("expert_weight_name"),
        "gate_name": data.get("gate_name"),
        "num_tokens": data.get("num_tokens"),
        "num_experts": data.get("num_experts"),
        "top_k": data.get("top_k"),
        "adapter_load_modes": data.get("adapter_load_modes"),
        "max_abs_error_dynamic_vs_materialized": data.get("max_abs_error_dynamic_vs_materialized"),
        "plan_summary": data.get("plan_summary"),
        "outputs": {
            "runtime_table": str(out / "table_runtime.tex"),
            "correctness_table": str(out / "table_correctness.tex"),
            "quantization_table": str(out / "table_quantization.tex"),
            "plan_excerpt": str(out / "residual_plan_excerpt.json"),
            "environment": str(out / "environment_summary.md"),
        },
    }
    (out / "result_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
