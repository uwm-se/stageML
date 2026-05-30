
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from stageml.annotations import BindingTime, stage0, stage1



_CALL_FN_MLIR: dict = {
    F.linear:           "linalg.matmul_plus_bias",
    F.relu:             "arith.maximumf",          # max(x, 0)
    F.softmax:          "stageml.softmax",         # no direct linalg equivalent
    F.dropout:          "stageml.dropout",
    torch.relu:         "arith.maximumf",
    torch.matmul:       "linalg.matmul",
    torch.mm:           "linalg.matmul",
    torch.bmm:          "linalg.batch_matmul",
    torch.add:          "arith.addf",
    torch.mul:          "arith.mulf",
    torch.transpose:    "linalg.transpose",
    torch.sigmoid:      "stageml.sigmoid",
    torch.tanh:         "math.tanh",
}

_MODULE_MLIR: dict = {
    nn.Linear:          "linalg.matmul_plus_bias",
    nn.ReLU:            "arith.maximumf",
    nn.Softmax:         "stageml.softmax",
    nn.LayerNorm:       "stageml.layer_norm",
    nn.MultiheadAttention: "stageml.multi_head_attention",
}

_METHOD_MLIR: dict = {
    "transpose":        "linalg.transpose",
    "reshape":          "memref.reshape",
    "view":             "memref.reshape",
    "contiguous":       "stageml.contiguous",
}


def _mlir_op_name(node: fx.Node, gm: fx.GraphModule) -> str:
    """Map an FX node to a descriptive MLIR op name."""
    if node.op == "get_attr":
        return "arith.constant"
    if node.op == "call_function":
        return _CALL_FN_MLIR.get(
            node.target,
            f"stageml.{getattr(node.target, '__name__', 'op')}",
        )
    if node.op == "call_method":
        return _METHOD_MLIR.get(node.target, f"stageml.{node.target}")
    if node.op == "call_module":
        try:
            submod = gm.get_submodule(node.target)
            for mod_cls, mlir_name in _MODULE_MLIR.items():
                if isinstance(submod, mod_cls):
                    return mlir_name
            return f"stageml.{type(submod).__name__.lower()}"
        except Exception:
            return f"stageml.{node.target}"
    return "stageml.op"


def _mlir_type(node: fx.Node) -> str:
    """
    Return the MLIR tensor type for a node.
    Uses shape metadata from ShapeProp if available; otherwise falls back to *.
    """
    meta = node.meta

    # ShapeProp stores shape in 'tensor_meta'
    if "tensor_meta" in meta:
        tm = meta["tensor_meta"]
        shape = tm.shape if hasattr(tm, "shape") else None
        if shape is not None and len(shape) > 0:
            shape_str = "x".join(str(d) for d in shape)
            return f"tensor<{shape_str}xf32>"

    # torch.compile / dynamo stores in 'example_value'
    if "example_value" in meta:
        ev = meta["example_value"]
        if hasattr(ev, "shape") and ev.shape:
            shape_str = "x".join(str(d) for d in ev.shape)
            return f"tensor<{shape_str}xf32>"

    return "tensor<*xf32>"


def lower_to_mlir(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> str:
    
    lines: list[str] = []
    lines.append("// StageML generated MLIR (improved sketch)")
    lines.append("// Stage-0 = compile-time static  |  Stage-1 = runtime dynamic")
    lines.append("")
    lines.append("module {")
    lines.append("  func.func @staged_fn(")

    # Function signature: one argument per placeholder
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, node in enumerate(placeholders):
        stage  = annotations.get(node, stage1)
        comma  = "," if i < len(placeholders) - 1 else ""
        tensor = _mlir_type(node)
        lines.append(
            f"    %{node.name}: {tensor} {{stageml.stage = {stage.level}}}{comma}"
        )

    # Infer return type from output node
    output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
    ret_type = "tensor<*xf32>"
    if output_nodes:
        out_args = output_nodes[0].args[0]
        if isinstance(out_args, fx.Node):
            ret_type = _mlir_type(out_args)
        elif isinstance(out_args, (list, tuple)) and out_args:
            ret_type = _mlir_type(out_args[0]) if isinstance(out_args[0], fx.Node) else "tensor<*xf32>"

    lines.append(f"  ) -> {ret_type} {{")
    lines.append("")

    # Function body
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        stage    = annotations.get(node, stage1)
        op_name  = _mlir_op_name(node, gm)
        out_type = _mlir_type(node)
        operands = ", ".join(f"%{a.name}" for a in node.args if isinstance(a, fx.Node))
        stage_comment = "STATIC — folded" if stage == stage0 else "dynamic — kept"

        lines.append(
            f"    %{node.name} : {out_type} = {op_name}({operands})"
            f"  // {{stageml.stage = {stage.level}}}  // {stage_comment}"
        )

    # Return statement
    if output_nodes:
        out_args = output_nodes[0].args[0]
        ret_val = f"%{out_args.name}" if isinstance(out_args, fx.Node) else "%result"
    else:
        ret_val = "%result"

    lines.append(f"    return {ret_val} : {ret_type}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def print_mlir(mlir_text: str) -> None:
    print("\n" + "─" * 60)
    print("Generated MLIR:")
    print("─" * 60)
    print(mlir_text)
    print("─" * 60 + "\n")
