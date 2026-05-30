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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vllm(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    if path.suffix == ".json":
        return load_json(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an honesty checklist for the LOPSTR26 StageML claims.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--vllm", default=None)
    ap.add_argument("--mlir-status", default=None)
    ap.add_argument("--lean-status", default=None)
    ap.add_argument("--out", default="paper_outputs/claim_check.md")
    args = ap.parse_args()

    data = load_json(Path(args.results))
    vllm = load_vllm(Path(args.vllm)) if args.vllm else None
    dyn = data.get("dynamic", {})
    mat = data.get("materialized", {})
    max_err = f(data.get("max_abs_error_dynamic_vs_materialized"))
    speedup = f(dyn.get("p50_ms")) / f(mat.get("p50_ms"), 1.0) if f(mat.get("p50_ms")) > 0 else 0.0
    adapter_modes = data.get("adapter_load_modes", {})
    expert_specific = all(v == "expert_specific" for v in adapter_modes.values()) if adapter_modes else False
    q = data.get("quantization", {})
    has_quant = bool(q)
    any_reject = any(not bool(v.get("safe", True)) for v in q.values()) if q else False

    lines = []
    lines.append("# StageML paper claim check")
    lines.append("")
    lines.append("## Safe claims supported by this run")
    lines.append("")
    lines.append("- The prototype ran on a real Hugging Face model path." if data.get("model") else "- Model path was not recorded.")
    lines.append("- The benchmark used real model hidden states and a discovered MoE gate." if data.get("gate_name") else "- The MoE gate was not recorded.")
    lines.append("- Dynamic and materialized residual MoE LoRA were compared on the same hidden states.")
    lines.append(f"- Maximum absolute error between exact dynamic and materialized residual paths was {max_err:.8g}.")
    lines.append(f"- Materialized residual P50 speedup over dynamic reference was {speedup:.3f} times.")
    if expert_specific:
        lines.append("- All loaded adapters were expert specific according to checkpoint key inspection.")
    else:
        lines.append("- At least one adapter used shared fallback or adapter mode was not recorded. Do not claim expert specific adapter evaluation for all adapters.")
    if has_quant:
        lines.append("- Quantization abstract interpretation produced epsilon values and accept or reject decisions.")
        if any_reject:
            lines.append("- At least one adapter was rejected by the quantization threshold.")
    lines.append("")
    lines.append("## Claims that are not supported by this run unless additional artifacts are present")
    lines.append("")
    if not vllm:
        lines.append("- Do not claim that StageML beats vLLM. No vLLM baseline file was provided.")
    else:
        vp50 = f(vllm.get("p50_ms"))
        lines.append(f"- vLLM baseline was present with P50 {vp50:.3f} ms. Check fairness before claiming superiority.")
    lines.append("- Do not claim full production fused kernel generation unless MLIR and backend lowering are complete.")
    lines.append("- Do not claim perplexity preservation from epsilon alone. Perplexity must be measured on an evaluation dataset.")
    lines.append("- Do not claim CUDA or Triton kernel correctness from the Lean proof. The proof is for the core calculus only.")
    lines.append("- Do not claim that StageML removes token dependent MoE routing. Routing remains dynamic.")
    lines.append("")
    lines.append("## Recommended abstract wording")
    lines.append("")
    lines.append("StageML implements a multistage residualization prototype for MoE LoRA fragments. The prototype separates base, adapter, tenant, request, routing, and token stages, compares dynamic and materialized residual execution on real MoE hidden states, emits residual plans under memory and quantization constraints, and reports numerical error for exact and low precision residualization decisions.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    main()
