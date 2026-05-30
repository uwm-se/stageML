"""
stageml/tracer.py
Phase 2 — Stage Propagation

Takes the staging environment Γ from Phase 1 and a torch.fx graph.
Propagates binding-time annotations through every node in the graph.

The propagation rule (the whole analysis in one line):
    stage(node) = S   if ALL operands of node are S
    stage(node) = D   if ANY operand of node is D

This is a forward dataflow analysis over the two-point lattice {S ⊑ D}.
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import torch.fx as fx
from stageml.annotations import BindingTime, stage0, stage1


# ── Custom FX tracer ─────────────────────────────────────────────────────────

class StageMLTracer(fx.Tracer):
    """
    FX tracer that traces *into* primitive modules (nn.Linear, nn.LayerNorm,
    etc.) so their parameters appear as get_attr nodes in the graph.

    get_attr nodes are classified as stage-0 by propagate_stages, because
    trained weights and buffers are always static at deployment time.

    Modules in _LEAF_MODULES are NOT traced into (they are too complex or
    have non-traceable control flow).
    """

    _LEAF_MODULES = (
        # BatchNorm variants: _check_input_dim has data-dependent control flow
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        # Attention / recurrent: complex or non-traceable internal control flow
        nn.MultiheadAttention,
        nn.Transformer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.LSTM,
        nn.GRU,
        nn.LSTMCell,
        nn.GRUCell,
        nn.Embedding,
        nn.EmbeddingBag,
    )

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return isinstance(m, self._LEAF_MODULES)


# ── Stage propagation ─────────────────────────────────────────────────────────

def propagate_stages(
    graph: fx.Graph,
    gamma: dict[str, BindingTime]
) -> dict[fx.Node, BindingTime]:
    """
    Forward dataflow analysis.

    Given:
      graph : the torch.fx computation graph of the function
      gamma : staging environment Γ (parameter name → binding time)

    Returns:
      annotations : dict mapping every fx.Node → BindingTime

    Correctness invariant (proved in proofs/soundness.lean):
      For all nodes n: if annotations[n] == stage0,
      then for all operands op of n: annotations[op] == stage0.
    """
    annotations: dict[fx.Node, BindingTime] = {}

    for node in graph.nodes:

        if node.op == "placeholder":
            # torch.fx lowercases parameter names — build case-insensitive lookup
            gamma_lower = {k.lower(): v for k, v in gamma.items()}
            bt = gamma_lower.get(node.name.lower(), stage1)
            annotations[node] = bt

        elif node.op == "get_attr":
            # Model weights / buffers — always stage-0 (static after training)
            annotations[node] = stage0

        elif node.op in ("call_function", "call_method", "call_module"):
            # Propagation rule: join the stages of all input operands
            # S ⊔ S = S,  S ⊔ D = D,  D ⊔ D = D
            operand_stages = [
                annotations.get(arg, stage1)
                for arg in node.args
                if isinstance(arg, fx.Node)
            ]
            if not operand_stages:
                # No operands → conservative: stage0
                annotations[node] = stage0
            else:
                result = operand_stages[0]
                for s in operand_stages[1:]:
                    result = result.join(s)
                annotations[node] = result

        elif node.op == "output":
            # Output stage is the join of all returned values
            def flatten_args(args):
                for a in args:
                    if isinstance(a, fx.Node):
                        yield a
                    elif isinstance(a, (tuple, list)):
                        yield from flatten_args(a)
            operand_stages = [
                annotations.get(a, stage1) for a in flatten_args(node.args)
            ]
            result = stage0
            for s in operand_stages:
                result = result.join(s)
            annotations[node] = result

        else:
            annotations[node] = stage1  # conservative default

    return annotations


def print_annotated_graph(
    graph: fx.Graph,
    annotations: dict[fx.Node, BindingTime]
) -> None:
    """Pretty-print the annotated graph for debugging."""
    print(f"\n{'─'*60}")
    print(f"{'Node':<30} {'Op':<16} {'Stage'}")
    print(f"{'─'*60}")
    for node in graph.nodes:
        stage = annotations.get(node, stage1)
        marker = "✓ STATIC" if stage == stage0 else "  dynamic"
        print(f"  {node.name:<28} {node.op:<16} {marker}")
    print(f"{'─'*60}\n")


def trace_and_annotate(
    fn: Callable,
    stage_env_or_inputs=None,
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime]]:
    """
    Trace the function/module with torch.fx and annotate every node with its stage.

    Two calling conventions:

      New API — nn.Module + stage_env dict:
          trace_and_annotate(model, {'x': 'stage1'})
        Uses StageMLTracer, which traces into nn.Linear, nn.LayerNorm, etc.
        so that weight/bias parameters appear as get_attr nodes (stage-0).

      Legacy API — @compile_staged decorated function:
          trace_and_annotate(fn, (example_input,))
        Uses standard fx.symbolic_trace via fn._gamma.

    Returns:
        gm          : the traced GraphModule
        annotations : node → BindingTime mapping
    """
    if isinstance(stage_env_or_inputs, dict):
        # Build gamma from the dict (strings or BindingTime values)
        gamma: dict[str, BindingTime] = {}
        for k, v in stage_env_or_inputs.items():
            if isinstance(v, BindingTime):
                gamma[k] = v
            elif isinstance(v, str):
                gamma[k] = stage1 if v.lower() == "stage1" else stage0
            else:
                gamma[k] = stage1

        if isinstance(fn, nn.Module):
            # Use StageMLTracer so that module parameters appear as get_attr nodes
            tracer = StageMLTracer()
            graph  = tracer.trace(fn)
            gm     = fx.GraphModule(fn, graph)
        else:
            # Plain function with a manually-built gamma dict
            gm = fx.symbolic_trace(fn)
        annotations = propagate_stages(gm.graph, gamma)
        return gm, annotations

    # Legacy API: @compile_staged functions
    assert hasattr(fn, "_gamma"), \
        f"{fn.__name__} must be decorated with @compile_staged"
    gm = fx.symbolic_trace(fn)
    annotations = propagate_stages(gm.graph, fn._gamma)
    return gm, annotations


def staging_summary(
    annotations: dict[fx.Node, BindingTime]
) -> dict:
    """
    Compute summary statistics for the staging analysis report.
    Returns a dict with counts and percentages.
    """
    total   = len(annotations)
    static  = sum(1 for v in annotations.values() if v == stage0)
    dynamic = total - static
    return {
        "total_ops":    total,
        "static_ops":   static,
        "dynamic_ops":  dynamic,
        "static_pct":   round(100 * static  / total, 1) if total else 0,
        "dynamic_pct":  round(100 * dynamic / total, 1) if total else 0,
    }
