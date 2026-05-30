

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
import torch.fx as fx

from stageml.annotations import BindingTime, stage0, stage1
from stageml.tracer       import trace_and_annotate, staging_summary, print_annotated_graph
from stageml.mlir_lower   import lower_to_mlir, print_mlir
from stageml.real_mlir_lower import lower_to_parseable_mlir, write_parseable_mlir
from stageml.evaluator    import specialize
from stageml.rewrite      import optimize_evaluation_order


@dataclass
class StagingReport:
   
    fn_name:      str
    total_ops:    int
    static_ops:   int
    dynamic_ops:  int
    static_pct:   float
    dynamic_ops_list:  list[str] = field(default_factory=list)
    static_ops_list:   list[str] = field(default_factory=list)

    def print(self):
        bar_s = "█" * int(self.static_pct  / 5)
        bar_d = "░" * int((100 - self.static_pct) / 5)
        print(f"""
╔══════════════════════════════════════════════════════════╗
║          StageML Analysis Report: {self.fn_name:<22} ║
╠══════════════════════════════════════════════════════════╣
║  Total ops      : {self.total_ops:<38} ║
║  Stage-0 (fold) : {self.static_ops:<5} ({self.static_pct:>5.1f}%)  {bar_s:<20} ║
║  Stage-1 (keep) : {self.dynamic_ops:<5} ({100-self.static_pct:>5.1f}%)  {bar_d:<20} ║
╠══════════════════════════════════════════════════════════╣
║  Ops eliminated at compile time : {self.static_ops:<24} ║
║  Ops remaining at runtime       : {self.dynamic_ops:<24} ║
╚══════════════════════════════════════════════════════════╝
        """)
        if self.static_ops_list:
            print("  Static ops (will be folded):")
            for op in self.static_ops_list:
                print(f"    ✓ {op}")
        if self.dynamic_ops_list:
            print("  Dynamic ops (kept in residual):")
            for op in self.dynamic_ops_list:
                print(f"    → {op}")
        print()


def compile_model(
    fn:            Callable,
    example_input: Optional[torch.Tensor] = None,
    static_vals:   Optional[dict[str, torch.Tensor]] = None,
    verbose:       bool = True,
    stage_env:     Optional[dict[str, str]] = None,
    enable_rewrite: bool = False,
    mlir_backend:  str = "sketch",
    mlir_output_path: Optional[str] = None,
) -> tuple[Callable, StagingReport]:
   
    static_vals = static_vals or {}

    if stage_env is not None:
        effective_env = stage_env
        fn_name = type(fn).__name__ if not hasattr(fn, "__name__") else fn.__name__
    elif hasattr(fn, "_gamma"):
        effective_env = fn._gamma
        fn_name = fn.__name__
    else:
        raise ValueError(
            "Must provide stage_env=... or decorate the function with @compile_staged"
        )

    gm, annotations = trace_and_annotate(fn, effective_env)

    if enable_rewrite:
        gm, annotations, rewrite_stats = optimize_evaluation_order(gm, annotations)
        if verbose and rewrite_stats.total_rewrites:
            print(f"[StageML Rewrite] Applied {rewrite_stats.total_rewrites} evaluation-order rewrite(s)")

    if example_input is not None and stage_env is not None:
        try:
            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(example_input)
        except Exception:
            pass

    if verbose:
        print_annotated_graph(gm.graph, annotations)

    # Phase 3: lower to MLIR
    if mlir_backend == "parseable":
        mlir_text = lower_to_parseable_mlir(gm, annotations, fn_name=fn_name)
        if mlir_output_path is not None:
            write_parseable_mlir(gm, annotations, mlir_output_path, fn_name=fn_name)
        if verbose:
            print_mlir(mlir_text)
    elif mlir_backend == "sketch":
        mlir_text = lower_to_mlir(gm, annotations)
        if verbose:
            print_mlir(mlir_text)
    else:
        raise ValueError("mlir_backend must be 'sketch' or 'parseable'")

    # Phase 4: specialize
    residual_gm = specialize(gm, annotations, static_vals)

    # Phase 5: build report
    summary      = staging_summary(annotations)
    static_names = [n.name for n, bt in annotations.items() if bt == stage0]
    dynamic_names= [n.name for n, bt in annotations.items() if bt == stage1]

    report = StagingReport(
        fn_name          = fn_name,
        total_ops        = summary["total_ops"],
        static_ops       = summary["static_ops"],
        dynamic_ops      = summary["dynamic_ops"],
        static_pct       = summary["static_pct"],
        static_ops_list  = static_names,
        dynamic_ops_list = dynamic_names,
    )

    if verbose:
        report.print()

    return residual_gm, report
