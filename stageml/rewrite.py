from __future__ import annotations

import copy
import operator
from dataclasses import dataclass
from typing import Callable

import torch.fx as fx

from stageml.annotations import BindingTime, stage0, stage1
from stageml.tracer import propagate_stages


_MATMUL_TARGETS: set[Callable] = {operator.matmul}
_ADD_TARGETS: set[Callable] = {operator.add}
_MUL_TARGETS: set[Callable] = {operator.mul}

try:
    import torch
    _MATMUL_TARGETS.add(torch.matmul)
    _MATMUL_TARGETS.add(torch.mm)
    _ADD_TARGETS.add(torch.add)
    _MUL_TARGETS.add(torch.mul)
except Exception:
    torch = None


@dataclass(frozen=True)
class RewriteStats:
    lora_assoc_rewrites: int = 0
    lora_merge_rewrites: int = 0
    total_rewrites: int = 0


@dataclass(frozen=True)
class _MatmulParts:
    x: fx.Node
    weight: fx.Node


def _is_call(node: fx.Node, targets: set[Callable]) -> bool:
    return node.op == "call_function" and node.target in targets


def _is_matmul_node(node: fx.Node) -> bool:
    return _is_call(node, _MATMUL_TARGETS)


def _is_add_node(node: fx.Node) -> bool:
    return _is_call(node, _ADD_TARGETS)


def _is_mul_node(node: fx.Node) -> bool:
    return _is_call(node, _MUL_TARGETS)


def _stage_of(annotations: dict[fx.Node, BindingTime], node: fx.Node) -> BindingTime:
    return annotations.get(node, stage1)


def _placeholder_gamma_from_annotations(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> dict[str, BindingTime]:
    gamma: dict[str, BindingTime] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            gamma[node.name] = annotations.get(node, stage1)
    return gamma


def rewrite_static_right_assoc_matmul(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime], RewriteStats]:
    original_gamma = _placeholder_gamma_from_annotations(gm, annotations)
    gm = copy.deepcopy(gm)
    annotations = propagate_stages(gm.graph, original_gamma)
    graph = gm.graph
    rewrites = 0

    for outer in list(graph.nodes):
        if not _is_matmul_node(outer) or len(outer.args) < 2:
            continue

        inner, rhs2 = outer.args[0], outer.args[1]
        if not isinstance(inner, fx.Node) or not isinstance(rhs2, fx.Node):
            continue
        if not _is_matmul_node(inner) or len(inner.args) < 2:
            continue

        x, rhs1 = inner.args[0], inner.args[1]
        if not isinstance(x, fx.Node) or not isinstance(rhs1, fx.Node):
            continue
        if _stage_of(annotations, x) != stage1:
            continue
        if _stage_of(annotations, rhs1) != stage0:
            continue
        if _stage_of(annotations, rhs2) != stage0:
            continue

        with graph.inserting_before(outer):
            combined = graph.call_function(operator.matmul, args=(rhs1, rhs2))
            combined.name = f"stage0_assoc_{rhs1.name}_{rhs2.name}"
            replacement = graph.call_function(operator.matmul, args=(x, combined))
            replacement.name = f"assoc_{outer.name}"

        outer.replace_all_uses_with(replacement)
        rewrites += 1

    if rewrites:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    new_annotations = propagate_stages(gm.graph, original_gamma)
    stats = RewriteStats(lora_assoc_rewrites=rewrites, total_rewrites=rewrites)
    return gm, new_annotations, stats


def _extract_dynamic_matmul_static_weight(
    node: fx.Node,
    annotations: dict[fx.Node, BindingTime],
) -> _MatmulParts | None:
    if not _is_matmul_node(node) or len(node.args) < 2:
        return None
    x, weight = node.args[0], node.args[1]
    if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
        return None
    if _stage_of(annotations, x) != stage1:
        return None
    if _stage_of(annotations, weight) != stage0:
        return None
    return _MatmulParts(x=x, weight=weight)


def _extract_scaled_matmul(
    node: fx.Node,
    annotations: dict[fx.Node, BindingTime],
) -> tuple[_MatmulParts, object] | None:
    parts = _extract_dynamic_matmul_static_weight(node, annotations)
    if parts is not None:
        return parts, 1.0

    if not _is_mul_node(node) or len(node.args) < 2:
        return None

    a, b = node.args[0], node.args[1]
    if isinstance(a, fx.Node):
        parts = _extract_dynamic_matmul_static_weight(a, annotations)
        if parts is not None and not isinstance(b, fx.Node):
            return parts, b
    if isinstance(b, fx.Node):
        parts = _extract_dynamic_matmul_static_weight(b, annotations)
        if parts is not None and not isinstance(a, fx.Node):
            return parts, a
    return None


def rewrite_lora_distributive_merge(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime], RewriteStats]:
    original_gamma = _placeholder_gamma_from_annotations(gm, annotations)
    gm = copy.deepcopy(gm)
    annotations = propagate_stages(gm.graph, original_gamma)
    graph = gm.graph
    rewrites = 0

    for add in list(graph.nodes):
        if not _is_add_node(add) or len(add.args) < 2:
            continue

        lhs, rhs = add.args[0], add.args[1]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            continue

        lhs_parts = _extract_scaled_matmul(lhs, annotations)
        rhs_parts = _extract_scaled_matmul(rhs, annotations)
        if lhs_parts is None or rhs_parts is None:
            continue

        left, left_scale = lhs_parts
        right, right_scale = rhs_parts
        if left.x is not right.x:
            continue

        with graph.inserting_before(add):
            left_weight = left.weight
            if left_scale != 1.0:
                left_weight = graph.call_function(operator.mul, args=(left.weight, left_scale))
                left_weight.name = f"stage0_scaled_{left.weight.name}"

            right_weight = right.weight
            if right_scale != 1.0:
                right_weight = graph.call_function(operator.mul, args=(right.weight, right_scale))
                right_weight.name = f"stage0_scaled_{right.weight.name}"

            merged_weight = graph.call_function(operator.add, args=(left_weight, right_weight))
            merged_weight.name = f"stage0_merged_{left.weight.name}_{right.weight.name}"
            replacement = graph.call_function(operator.matmul, args=(left.x, merged_weight))
            replacement.name = f"merged_{add.name}"

        add.replace_all_uses_with(replacement)
        rewrites += 1

    if rewrites:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    new_annotations = propagate_stages(gm.graph, original_gamma)
    stats = RewriteStats(lora_merge_rewrites=rewrites, total_rewrites=rewrites)
    return gm, new_annotations, stats


def optimize_evaluation_order(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime], RewriteStats]:
    gm1, ann1, assoc_stats = rewrite_static_right_assoc_matmul(gm, annotations)
    gm2, ann2, merge_stats = rewrite_lora_distributive_merge(gm1, ann1)
    total = assoc_stats.total_rewrites + merge_stats.total_rewrites
    stats = RewriteStats(
        lora_assoc_rewrites=assoc_stats.lora_assoc_rewrites,
        lora_merge_rewrites=merge_stats.lora_merge_rewrites,
        total_rewrites=total,
    )
    return gm2, ann2, stats
