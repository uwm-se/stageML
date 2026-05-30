"""
stageml/evaluator.py
Phase 4 — Specialization Pass

Takes the stage-annotated FX graph.
Folds all stage-0 nodes at compile time (evaluates them with PyTorch).
Emits a residual graph containing only stage-1 nodes.

Semantic preservation property (proved in proofs/preservation.lean):
    For all stage-1 inputs x:
        eval(original, {static_vals, x}) == eval(residual, x)
"""

from __future__ import annotations
import logging
import torch
import torch.fx as fx
from stageml.annotations import BindingTime, stage0, stage1

logger = logging.getLogger(__name__)


def specialize(
    gm:          fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
    static_vals: dict[str, torch.Tensor] = None,
) -> fx.GraphModule:
    """
    Specialization pass.

    For every node annotated stage0:
      - evaluate it concretely (from gm attributes or user-supplied static_vals)
      - replace it in the graph with a get_attr constant node

    For every node annotated stage1:
      - leave it unchanged in the residual

    Args:
        gm          : traced GraphModule
        annotations : node → BindingTime from Phase 2
        static_vals : optional dict of param_name → tensor for @compile_staged API

    Returns:
        gm : the same GraphModule with stage-0 ops folded in place
    """
    cache: dict[fx.Node, object] = {}
    static_vals = static_vals or {}

    # Seed cache from static_vals for @compile_staged functions (case-insensitive)
    static_lower = {k.lower(): v for k, v in static_vals.items()}
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name.lower() in static_lower:
            cache[node] = static_lower[node.name.lower()]

    # Pass 1: evaluate all stage-0 non-placeholder nodes in topological order
    for node in gm.graph.nodes:
        if annotations.get(node) != stage0:
            continue
        if node.op in ("output", "placeholder") or node in cache:
            continue

        try:
            if node.op == "get_attr":
                obj = gm
                for attr in node.target.split("."):
                    obj = getattr(obj, attr)
                cache[node] = obj

            elif node.op == "call_function":
                args = tuple(
                    cache[a] if isinstance(a, fx.Node) else a
                    for a in node.args
                )
                kwargs = {
                    k: (cache[v] if isinstance(v, fx.Node) else v)
                    for k, v in node.kwargs.items()
                }
                cache[node] = node.target(*args, **kwargs)

            elif node.op == "call_method":
                self_val = (
                    cache[node.args[0]] if isinstance(node.args[0], fx.Node)
                    else node.args[0]
                )
                args = tuple(
                    cache[a] if isinstance(a, fx.Node) else a
                    for a in node.args[1:]
                )
                kwargs = {
                    k: (cache[v] if isinstance(v, fx.Node) else v)
                    for k, v in node.kwargs.items()
                }
                cache[node] = getattr(self_val, node.target)(*args, **kwargs)

            elif node.op == "call_module":
                submod = gm.get_submodule(node.target)
                args = tuple(
                    cache[a] if isinstance(a, fx.Node) else a
                    for a in node.args
                )
                kwargs = {
                    k: (cache[v] if isinstance(v, fx.Node) else v)
                    for k, v in node.kwargs.items()
                }
                cache[node] = submod(*args, **kwargs)

        except Exception as exc:
            logger.debug("Could not evaluate stage-0 node '%s': %s", node.name, exc)

    # Pass 2: rewrite graph — replace each cached stage-0 node with a get_attr constant
    # Collect before iteration to avoid modifying the live node list mid-loop
    nodes_to_fold = [
        n for n in list(gm.graph.nodes)
        if n in cache
        and annotations.get(n) == stage0
        and n.op not in ("output", "placeholder")
    ]

    for node in nodes_to_fold:
        attr_name = f"_stage0_{node.name}"
        val = cache[node]
        if isinstance(val, torch.Tensor):
            gm.register_buffer(attr_name, val.detach())
        else:
            setattr(gm, attr_name, val)

        with gm.graph.inserting_before(node):
            new_node = gm.graph.get_attr(attr_name)

        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

    folded = len(nodes_to_fold)
    kept   = sum(1 for n in gm.graph.nodes if annotations.get(n) == stage1)

    print(f"\n[StageML Specializer]")
    print(f"  Folded {folded} stage-0 ops at compile time")
    print(f"  Kept   {kept} stage-1 ops in residual")

    gm.graph.lint()
    gm.recompile()
    return gm


def validate_preservation(
    original_fn:  callable,
    residual_fn:  callable,
    static_vals:  dict[str, torch.Tensor],
    dynamic_vals: dict[str, torch.Tensor],
    tol:          float = 1e-5,
) -> bool:
    """
    Empirical semantic preservation check.

    Runs both the original and residual on the same inputs
    and asserts numerical equality within tolerance.

    This is the runtime counterpart to the formal proof in
    proofs/preservation.lean.
    """
    import inspect

    all_vals = {**static_vals, **dynamic_vals}

    sig = inspect.signature(original_fn)
    args_orig = [all_vals[p] for p in sig.parameters]
    args_res  = [dynamic_vals[p] for p in sig.parameters if p in dynamic_vals]

    out_orig = original_fn(*args_orig)
    out_res  = residual_fn(*args_res)

    match = torch.allclose(out_orig, out_res, atol=tol)
    status = "PASS" if match else "FAIL"
    print(f"  Semantic preservation check: {status}")
    return match
