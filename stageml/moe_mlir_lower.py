from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from stageml.moe_ir import ExpertPlan, PlanKind, ResidualPlan


@dataclass(frozen=True)
class MLIRLoweringOptions:
    module_name: str = "stageml_moe_residual_plan"
    include_comments: bool = True
    emit_fallback_symbols: bool = True
    emit_grouped_batch_matmul: bool = True


def _sanitize_symbol(text: object) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "unnamed"


def lower_expert_plan_to_mlir(plan: ExpertPlan, *, options: MLIRLoweringOptions | None = None) -> str:
    """Emit a textual MLIR stub for one residual plan node.

    This lowering is intentionally conservative. It produces valid looking MLIR
    module text for the residual plan node boundaries and keeps concrete tensor
    shapes symbolic. It is meant as a compiler artifact and inspection target,
    not as a fully optimized GPU kernel generator.
    """
    options = options or MLIRLoweringOptions()
    adapter = _sanitize_symbol(plan.adapter)
    expert = _sanitize_symbol(plan.expert)
    name = f"{plan.kind.value}_{adapter}_{expert}"
    comment = ""
    if options.include_comments:
        comment = (
            f"  // adapter {plan.adapter} expert {plan.expert} rank {plan.rank} "
            f"epsilon {plan.epsilon} reason {plan.reason}\n"
        )

    if plan.kind == PlanKind.MATERIALIZED_RESIDUAL:
        body = """  func.func @{name}(%x: tensor<?x?xf32>, %w_res: tensor<?x?xf32>) -> tensor<?x?xf32> {{
    %y = linalg.matmul ins(%x, %w_res : tensor<?x?xf32>, tensor<?x?xf32>) outs(%x : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }}""".format(name=name)
    elif plan.kind == PlanKind.NON_MATERIALIZING_FUSION:
        body = """  func.func @{name}(%x: tensor<?x?xf32>, %w: tensor<?x?xf32>, %a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {{
    %base = linalg.matmul ins(%x, %w : tensor<?x?xf32>, tensor<?x?xf32>) outs(%x : tensor<?x?xf32>) -> tensor<?x?xf32>
    %low = linalg.matmul ins(%x, %a : tensor<?x?xf32>, tensor<?x?xf32>) outs(%x : tensor<?x?xf32>) -> tensor<?x?xf32>
    %delta = linalg.matmul ins(%low, %b : tensor<?x?xf32>, tensor<?x?xf32>) outs(%base : tensor<?x?xf32>) -> tensor<?x?xf32>
    %y = arith.addf %base, %delta : tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }}""".format(name=name)
    elif plan.kind == PlanKind.DISAGGREGATED:
        body = """  func.func @{name}(%x: tensor<?x?xf32>, %adapter_server_token: i64) -> tensor<?x?xf32> {{
    // Disaggregated execution boundary. The adapter server call is intentionally represented as an external symbol.
    %y = func.call @stageml_adapter_server(%x, %adapter_server_token) : (tensor<?x?xf32>, i64) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }}""".format(name=name)
    elif plan.kind == PlanKind.FALLBACK:
        body = """  func.func @{name}(%x: tensor<?x?xf32>, %runtime_handle: i64) -> tensor<?x?xf32> {{
    %y = func.call @stageml_runtime_fallback(%x, %runtime_handle) : (tensor<?x?xf32>, i64) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }}""".format(name=name)
    else:
        body = """  func.func @{name}(%x: tensor<?x?xf32>, %runtime_handle: i64) -> tensor<?x?xf32> {{
    %y = func.call @stageml_dynamic_lora(%x, %runtime_handle) : (tensor<?x?xf32>, i64) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }}""".format(name=name)
    return comment + body



def lower_grouped_materialized_residual_to_mlir(
    plan: ResidualPlan,
    *,
    options: MLIRLoweringOptions | None = None,
) -> str:
    """Emit a grouped materialized residual block using linalg.batch_matmul.

    This lowering represents the scalable MoE case: after routing, tokens are
    grouped by selected expert and adapter.  Each group becomes one batch element
    in a 3-D tensor, and all expert pathways are represented by one
    ``linalg.batch_matmul`` over ``[groups, tokens_per_group, hidden]`` and
    ``[groups, hidden, output]``.

    The shape variables are intentionally symbolic because the routing group
    sizes are runtime-dependent.  The artifact is a compiler lowering target and
    inspection artifact, not a complete bufferization pipeline.
    """
    options = options or MLIRLoweringOptions()
    materialized = [p for p in plan.plans if p.kind == PlanKind.MATERIALIZED_RESIDUAL]
    if not materialized:
        return ""
    lines = []
    lines.append("  // Grouped materialized residual lowering for routed MoE experts.")
    lines.append("  // Routing has already compacted tokens into expert/adapter groups.")
    lines.append("  func.func @stageml_grouped_materialized_residual(")
    lines.append("      %grouped_x: tensor<?x?x?xf32>,")
    lines.append("      %grouped_w_residual: tensor<?x?x?xf32>,")
    lines.append("      %empty: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {")
    lines.append("    %y = linalg.batch_matmul ins(%grouped_x, %grouped_w_residual : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>")
    lines.append("    return %y : tensor<?x?x?xf32>")
    lines.append("  }")
    return "\n".join(lines)


def lower_residual_plan_to_mlir(plan: ResidualPlan, *, options: MLIRLoweringOptions | None = None) -> str:
    options = options or MLIRLoweringOptions()
    lines: list[str] = []
    lines.append(f"module @{_sanitize_symbol(options.module_name)} {{")
    if options.emit_fallback_symbols:
        lines.append("  func.func private @stageml_adapter_server(tensor<?x?xf32>, i64) -> tensor<?x?xf32>")
        lines.append("  func.func private @stageml_runtime_fallback(tensor<?x?xf32>, i64) -> tensor<?x?xf32>")
        lines.append("  func.func private @stageml_dynamic_lora(tensor<?x?xf32>, i64) -> tensor<?x?xf32>")
    for p in plan.plans:
        lines.append(lower_expert_plan_to_mlir(p, options=options))
    if options.emit_grouped_batch_matmul:
        grouped = lower_grouped_materialized_residual_to_mlir(plan, options=options)
        if grouped:
            lines.append(grouped)
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_residual_plan_mlir(plan: ResidualPlan, path: str | Path, *, options: MLIRLoweringOptions | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(lower_residual_plan_to_mlir(plan, options=options), encoding="utf-8")
    return p
