from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch.fx as fx

from stageml.annotations import BindingTime, stage0, stage1


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _mlir_type(node: fx.Node) -> str:
    meta = getattr(node, "meta", {})
    if "tensor_meta" in meta:
        tm = meta["tensor_meta"]
        shape = getattr(tm, "shape", None)
        dtype = getattr(tm, "dtype", None)
        elem = "f32"
        if dtype is not None:
            s = str(dtype)
            if "float16" in s:
                elem = "f16"
            elif "bfloat16" in s:
                elem = "bf16"
            elif "float64" in s:
                elem = "f64"
            elif "int64" in s:
                elem = "i64"
            elif "int32" in s:
                elem = "i32"
        if shape is not None:
            dims = "x".join(str(d) for d in shape)
            return f"tensor<{dims}x{elem}>" if dims else f"tensor<{elem}>"
    return "tensor<*xf32>"


def _op_name(node: fx.Node) -> str:
    if node.op == "get_attr":
        return "stageml.constant"
    if node.op == "call_function":
        target = getattr(node.target, "__name__", str(node.target))
        target = target.replace("<built-in function ", "").replace(">", "")
        return f"stageml.{_safe_name(target)}"
    if node.op == "call_method":
        return f"stageml.{_safe_name(str(node.target))}"
    if node.op == "call_module":
        return f"stageml.module_{_safe_name(str(node.target))}"
    return "stageml.unknown"


def lower_to_parseable_mlir(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
    fn_name: str = "staged_fn",
) -> str:
    """
    Emit parseable MLIR using generic custom StageML ops.

    This is intentionally not a fake linalg lowering. It is valid MLIR syntax
    that can be parsed with MLIR tools using --allow-unregistered-dialect, while
    preserving FX node names, operand edges, tensor types, and stage attributes.
    A later backend can lower these custom stageml.* ops into linalg/arith/tensor.
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)

    value_name: dict[fx.Node, str] = {}
    args = []
    for i, node in enumerate(placeholders):
        value_name[node] = f"%arg{i}"
        stage = annotations.get(node, stage1).level
        args.append(f"%arg{i}: {_mlir_type(node)} {{stageml.stage = {stage}, stageml.fx_name = \"{node.name}\"}}")

    ret_node = None
    if output_node is not None:
        out = output_node.args[0]
        if isinstance(out, fx.Node):
            ret_node = out
        elif isinstance(out, (tuple, list)) and out and isinstance(out[0], fx.Node):
            ret_node = out[0]
    ret_type = _mlir_type(ret_node) if ret_node is not None else "tensor<*xf32>"

    lines = [
        "// StageML parseable MLIR",
        "// Parse with: mlir-opt --allow-unregistered-dialect file.mlir",
        "builtin.module {",
        f"  func.func @{_safe_name(fn_name)}({', '.join(args)}) -> {ret_type} {{",
    ]

    temp_idx = 0
    for node in gm.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue

        result = f"%{_safe_name(node.name)}"
        value_name[node] = result
        operands = [value_name[a] for a in node.args if isinstance(a, fx.Node)]
        operand_types = [_mlir_type(a) for a in node.args if isinstance(a, fx.Node)]
        out_type = _mlir_type(node)
        stage = annotations.get(node, stage1).level
        op_name = _op_name(node)
        target = str(node.target).replace('"', '\\"')

        lines.append(
            f"    {result} = \"{op_name}\"({', '.join(operands)}) "
            f"{{stageml.stage = {stage}, stageml.fx_name = \"{node.name}\", stageml.fx_op = \"{node.op}\", stageml.target = \"{target}\"}} "
            f": ({', '.join(operand_types)}) -> {out_type}"
        )
        temp_idx += 1

    if ret_node is not None:
        lines.append(f"    func.return {value_name[ret_node]} : {ret_type}")
    else:
        lines.append("    func.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def write_parseable_mlir(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
    path: str | Path,
    fn_name: str = "staged_fn",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lower_to_parseable_mlir(gm, annotations, fn_name), encoding="utf-8")
    return path


def verify_mlir_with_mlir_opt(path: str | Path) -> tuple[bool, str]:
    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        return False, "mlir-opt not found. Install LLVM/MLIR or use a Colab/Linux image with MLIR tools."
    proc = subprocess.run(
        [mlir_opt, "--allow-unregistered-dialect", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stderr or proc.stdout
