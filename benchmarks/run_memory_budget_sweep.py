#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from stageml.moe_ir import MoEAdapterMeta
from stageml.residual_planner import AdapterQuantInfo, PlannerConfig, choose_residual_plan


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def meta_from_dict(d: dict) -> MoEAdapterMeta:
    return MoEAdapterMeta(
        name=d["name"],
        rank=int(d["rank"]),
        num_experts=int(d["num_experts"]),
        in_features=int(d["in_features"]),
        out_features=int(d["out_features"]),
        dtype=d.get("dtype", "fp16"),
        request_count=int(d.get("request_count", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute StageML residual plans over memory budgets using one real benchmark result.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--budgets-mb", default="32,64,128,256,512,1024,2048")
    ap.add_argument("--out-dir", default="paper_outputs/memory_sweep")
    args = ap.parse_args()

    data = load_result(Path(args.results))
    metas = [meta_from_dict(d) for d in data.get("adapter_metas", [])]
    if not metas:
        raise SystemExit("result file does not contain adapter_metas. Re-run real_moe_lora_residual_bench.py from the updated package.")
    qraw = data.get("quantization", {})
    qinfo = {name: AdapterQuantInfo(epsilon=float(v.get("epsilon", 0.0)), safe=bool(v.get("safe", True))) for name, v in qraw.items()}
    theta = float(data.get("theta", 1.0))
    budgets = [float(x.strip()) for x in args.budgets_mb.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in budgets:
        plan = choose_residual_plan(metas, config=PlannerConfig(memory_budget_mb=budget, theta=theta), quant_info=qinfo)
        summary = plan.by_kind()
        row = {
            "memory_budget_mb": budget,
            "used_memory_mb": plan.used_memory_bytes / (1024 * 1024),
            "dynamic": summary.get("dynamic", 0),
            "non_materializing_fusion": summary.get("non_materializing_fusion", 0),
            "materialized_residual": summary.get("materialized_residual", 0),
            "disaggregated": summary.get("disaggregated", 0),
            "fallback": summary.get("fallback", 0),
        }
        rows.append(row)
        (out_dir / f"plan_budget_{int(budget)}mb.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    csv_path = out_dir / "memory_budget_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # One chart per script output, no seaborn, no explicit colors.
    plt.figure(figsize=(8, 4.8))
    xs = [r["memory_budget_mb"] for r in rows]
    for key in ["dynamic", "non_materializing_fusion", "materialized_residual", "disaggregated", "fallback"]:
        ys = [r[key] for r in rows]
        plt.plot(xs, ys, marker="o", label=key.replace("_", " "))
    plt.xlabel("Memory budget MB")
    plt.ylabel("Number of adapter plans")
    plt.title("StageML residual plan choice under memory budget")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "memory_budget_sweep.png"
    plt.savefig(fig_path, dpi=180)

    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{StageML residual plan choices under memory budgets}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Budget MB & Used MB & Dynamic & Fusion & Materialized & Disaggregated & Fallback \\",
        r"\midrule",
    ]
    for r in rows:
        latex.append(f"{r['memory_budget_mb']:.0f} & {r['used_memory_mb']:.2f} & {r['dynamic']} & {r['non_materializing_fusion']} & {r['materialized_residual']} & {r['disaggregated']} & {r['fallback']} \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out_dir / "table_memory_sweep.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "figure": str(fig_path), "table": str(out_dir / "table_memory_sweep.tex")}, indent=2))


if __name__ == "__main__":
    main()
