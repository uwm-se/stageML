from __future__ import annotations

import math
import operator
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


def _safe_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if not out:
        out = "v"
    if out[0].isdigit():
        out = "_" + out
    return out


_DTYPE_TO_MLIR = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.uint8: "ui8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
}


_MATMUL_TARGETS = {operator.matmul}
_ADD_TARGETS = {operator.add}
_MUL_TARGETS = {operator.mul}
try:
    _MATMUL_TARGETS.update({torch.matmul, torch.mm})
    _ADD_TARGETS.add(torch.add)
    _MUL_TARGETS.add(torch.mul)
except Exception:
    pass


class CanonicalMLIRError(RuntimeError):
    pass


class _Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._counter = 0

    def fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"%{_safe_name(prefix)}_{self._counter}"

    def emit(self, line: str = "") -> None:
        self.lines.append(line)




def _dtype_to_mlir(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "f32"
    if dtype not in _DTYPE_TO_MLIR:
        raise CanonicalMLIRError(f"Unsupported dtype for canonical MLIR: {dtype}")
    return _DTYPE_TO_MLIR[dtype]


def _tensor_meta(node: fx.Node) -> Any | None:
    return getattr(node, "meta", {}).get("tensor_meta")


def _node_shape(node: fx.Node) -> tuple[int, ...]:
    tm = _tensor_meta(node)
    if tm is None or getattr(tm, "shape", None) is None:
        raise CanonicalMLIRError(
            f"Missing tensor_meta shape for FX node '{node.name}'. "
            "Run torch.fx.passes.shape_prop.ShapeProp before lowering."
        )
    shape = tuple(int(d) for d in tm.shape)
    if any(d < 0 for d in shape):
        raise CanonicalMLIRError(f"Dynamic shapes are not supported yet: {node.name} has {shape}")
    return shape


def _node_dtype(node: fx.Node) -> torch.dtype | None:
    tm = _tensor_meta(node)
    return getattr(tm, "dtype", None) if tm is not None else None


def _tensor_type_from_shape_dtype(shape: tuple[int, ...], dtype: torch.dtype | None) -> str:
    elem = _dtype_to_mlir(dtype)
    if len(shape) == 0:
        return f"tensor<{elem}>"
    return f"tensor<{'x'.join(str(d) for d in shape)}x{elem}>"


def _node_type(node: fx.Node) -> str:
    return _tensor_type_from_shape_dtype(_node_shape(node), _node_dtype(node))


def _elem_type_from_tensor_type(tensor_type: str) -> str:
    inside = tensor_type.removeprefix("tensor<").removesuffix(">")
    return inside.split("x")[-1]


def _rank_from_shape(shape: tuple[int, ...]) -> int:
    return len(shape)


def _identity_affine_map(rank: int) -> str:
    if rank == 0:
        return "affine_map<() -> ()>"
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_map<({dims}) -> ({dims})>"


def _iterator_types(rank: int) -> str:
    return "[" + ", ".join('"parallel"' for _ in range(rank)) + "]"




def _format_float(x: float) -> str:
    if math.isnan(x):
        return "0.000000e+00"
    if math.isinf(x):
        return "0.000000e+00"
    return f"{float(x):.8e}"


def _format_scalar_literal(value: Any, elem_type: str) -> str:
    if elem_type.startswith("f") or elem_type == "bf16":
        return _format_float(float(value))
    if elem_type == "i1":
        return "true" if bool(value) else "false"
    return str(int(value))


def _nested_dense_literal(t: torch.Tensor) -> str:
    cpu = t.detach().cpu()
    if cpu.ndim == 0:
        return _format_float(float(cpu.item())) if cpu.dtype.is_floating_point else str(int(cpu.item()))
    data = cpu.tolist()

    def rec(x: Any) -> str:
        if isinstance(x, list):
            return "[" + ", ".join(rec(v) for v in x) + "]"
        if cpu.dtype.is_floating_point:
            return _format_float(float(x))
        if cpu.dtype == torch.bool:
            return "true" if bool(x) else "false"
        return str(int(x))

    return rec(data)


def _get_attr_value(gm: fx.GraphModule, target: str) -> Any:
    obj: Any = gm
    for part in str(target).split("."):
        obj = getattr(obj, part)
    if isinstance(obj, torch.nn.Parameter):
        obj = obj.detach()
    return obj




def _emit_zero_filled_tensor(em: _Emitter, tensor_type: str, prefix: str) -> str:
    # Use a dense splat constant instead of tensor.empty + linalg.fill.
    # This is more portable across the MLIR versions commonly available in Colab
    # and still uses only registered canonical dialects for the small demo shapes.
    elem_type = _elem_type_from_tensor_type(tensor_type)
    zero = em.fresh(f"{prefix}_zero_tensor")
    literal = _format_scalar_literal(0, elem_type)
    em.emit(f"    {zero} = arith.constant dense<{literal}> : {tensor_type}")
    return zero


def _emit_tensor_constant(
    em: _Emitter,
    gm: fx.GraphModule,
    node: fx.Node,
    max_dense_elements: int,
) -> str:
    value = _get_attr_value(gm, str(node.target))
    result = f"%{_safe_name(node.name)}"

    if not isinstance(value, torch.Tensor):
        raise CanonicalMLIRError(
            f"get_attr node '{node.name}' refers to non-tensor value {type(value)}; "
            "canonical MLIR backend currently supports tensor constants only."
        )

    value = value.detach().cpu().contiguous()
    if value.numel() > max_dense_elements:
        raise CanonicalMLIRError(
            f"Tensor constant '{node.name}' has {value.numel()} elements. "
            f"This backend refuses to dump constants larger than max_dense_elements={max_dense_elements}. "
            "Generate canonical MLIR with a small demo shape, for example dim=8 or dim=128."
        )

    tensor_type = _tensor_type_from_shape_dtype(tuple(value.shape), value.dtype)
    dense = _nested_dense_literal(value)
    em.emit(f"    {result} = arith.constant dense<{dense}> : {tensor_type}")
    return result


def _emit_matmul(em: _Emitter, lhs: str, lhs_type: str, rhs: str, rhs_type: str, out_type: str, name: str) -> str:
    result = f"%{_safe_name(name)}"
    init = _emit_zero_filled_tensor(em, out_type, f"{name}_matmul")
    em.emit(
        f"    {result} = linalg.matmul "
        f"ins({lhs}, {rhs} : {lhs_type}, {rhs_type}) "
        f"outs({init} : {out_type}) -> {out_type}"
    )
    return result


def _emit_transpose(em: _Emitter, src: str, src_type: str, out_type: str, name: str) -> str:
    result = f"%{_safe_name(name)}"
    init = _emit_zero_filled_tensor(em, out_type, f"{name}_transpose")
    em.emit(
        f"    {result} = linalg.transpose "
        f"ins({src} : {src_type}) outs({init} : {out_type}) "
        f"permutation = [1, 0]"
    )
    return result


def _emit_elementwise_tensor_tensor(
    em: _Emitter,
    op_name: str,
    lhs: str,
    lhs_type: str,
    rhs: str,
    rhs_type: str,
    out_type: str,
    result_name: str,
    shape: tuple[int, ...],
) -> str:
    result = f"%{_safe_name(result_name)}"
    elem_type = _elem_type_from_tensor_type(out_type)
    init = _emit_zero_filled_tensor(em, out_type, f"{result_name}_ew")
    rank = _rank_from_shape(shape)
    id_map = _identity_affine_map(rank)
    maps = f"[{id_map}, {id_map}, {id_map}]"
    iterators = _iterator_types(rank)
    arith_op = "arith.addf" if op_name == "add" else "arith.mulf"

    em.emit(
        f"    {result} = linalg.generic "
        f"{{indexing_maps = {maps}, iterator_types = {iterators}}} "
        f"ins({lhs}, {rhs} : {lhs_type}, {rhs_type}) "
        f"outs({init} : {out_type}) {{"
    )
    em.emit(f"    ^bb0(%a: {elem_type}, %b: {elem_type}, %out: {elem_type}):")
    em.emit(f"      %r = {arith_op} %a, %b : {elem_type}")
    em.emit(f"      linalg.yield %r : {elem_type}")
    em.emit(f"    }} -> {out_type}")
    return result


def _emit_elementwise_tensor_scalar(
    em: _Emitter,
    op_name: str,
    tensor_value: str,
    tensor_type: str,
    scalar_value: Any,
    out_type: str,
    result_name: str,
    shape: tuple[int, ...],
) -> str:
    result = f"%{_safe_name(result_name)}"
    elem_type = _elem_type_from_tensor_type(out_type)
    init = _emit_zero_filled_tensor(em, out_type, f"{result_name}_ew")
    rank = _rank_from_shape(shape)
    id_map = _identity_affine_map(rank)
    maps = f"[{id_map}, {id_map}]"
    iterators = _iterator_types(rank)
    arith_op = "arith.addf" if op_name == "add" else "arith.mulf"
    literal = _format_scalar_literal(scalar_value, elem_type)

    em.emit(
        f"    {result} = linalg.generic "
        f"{{indexing_maps = {maps}, iterator_types = {iterators}}} "
        f"ins({tensor_value} : {tensor_type}) "
        f"outs({init} : {out_type}) {{"
    )
    em.emit(f"    ^bb0(%a: {elem_type}, %out: {elem_type}):")
    em.emit(f"      %c = arith.constant {literal} : {elem_type}")
    em.emit(f"      %r = {arith_op} %a, %c : {elem_type}")
    em.emit(f"      linalg.yield %r : {elem_type}")
    em.emit(f"    }} -> {out_type}")
    return result




def lower_to_canonical_mlir(
    gm: fx.GraphModule,
    fn_name: str = "staged_fn",
    example_args: tuple[Any, ...] | None = None,
    max_dense_elements: int = 16384,
) -> str:
    if example_args is not None:
        try:
            ShapeProp(gm).propagate(*example_args)
        except Exception as exc:
            raise CanonicalMLIRError(f"Shape propagation failed before MLIR lowering: {exc}") from exc

    em = _Emitter()
    value_name: dict[fx.Node, str] = {}
    value_type: dict[fx.Node, str] = {}

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
    if output_node is None:
        raise CanonicalMLIRError("FX graph has no output node")

    args_mlir: list[str] = []
    for i, node in enumerate(placeholders):
        arg_name = f"%arg{i}"
        typ = _node_type(node)
        value_name[node] = arg_name
        value_type[node] = typ
        args_mlir.append(f"{arg_name}: {typ}")

    returned = output_node.args[0]
    if isinstance(returned, (tuple, list)):
        if len(returned) != 1 or not isinstance(returned[0], fx.Node):
            raise CanonicalMLIRError("Only a single tensor return is supported")
        returned = returned[0]
    if not isinstance(returned, fx.Node):
        raise CanonicalMLIRError("Only tensor-valued FX returns are supported")

    ret_type = _node_type(returned)

    em.emit("// StageML canonical MLIR")
    em.emit("// Verify with: mlir-opt --verify-each file.mlir -o /dev/null")
    em.emit("builtin.module {")
    em.emit(f"  func.func @{_safe_name(fn_name)}({', '.join(args_mlir)}) -> {ret_type} {{")

    for node in gm.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if len(node.users) == 0:
            continue

        out_type = _node_type(node)
        value_type[node] = out_type

        if node.op == "get_attr":
            value_name[node] = _emit_tensor_constant(em, gm, node, max_dense_elements=max_dense_elements)
            continue

        if node.op == "call_method" and str(node.target) == "t":
            src = node.args[0]
            if not isinstance(src, fx.Node):
                raise CanonicalMLIRError(f"transpose node '{node.name}' has non-node input")
            if _rank_from_shape(_node_shape(src)) != 2:
                raise CanonicalMLIRError("Only rank-2 tensor.t() is supported")
            value_name[node] = _emit_transpose(
                em,
                value_name[src],
                value_type[src],
                out_type,
                node.name,
            )
            continue

        if node.op == "call_function" and node.target in _MATMUL_TARGETS:
            lhs, rhs = node.args[:2]
            if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
                raise CanonicalMLIRError(f"matmul node '{node.name}' expects tensor node operands")
            value_name[node] = _emit_matmul(
                em,
                value_name[lhs],
                value_type[lhs],
                value_name[rhs],
                value_type[rhs],
                out_type,
                node.name,
            )
            continue

        if node.op == "call_function" and node.target in (_ADD_TARGETS | _MUL_TARGETS):
            op_name = "add" if node.target in _ADD_TARGETS else "mul"
            a, b = node.args[:2]
            shape = _node_shape(node)
            if isinstance(a, fx.Node) and isinstance(b, fx.Node):
                value_name[node] = _emit_elementwise_tensor_tensor(
                    em,
                    op_name,
                    value_name[a],
                    value_type[a],
                    value_name[b],
                    value_type[b],
                    out_type,
                    node.name,
                    shape,
                )
                continue
            if isinstance(a, fx.Node) and not isinstance(b, fx.Node):
                value_name[node] = _emit_elementwise_tensor_scalar(
                    em,
                    op_name,
                    value_name[a],
                    value_type[a],
                    b,
                    out_type,
                    node.name,
                    shape,
                )
                continue
            if isinstance(b, fx.Node) and not isinstance(a, fx.Node):
                value_name[node] = _emit_elementwise_tensor_scalar(
                    em,
                    op_name,
                    value_name[b],
                    value_type[b],
                    a,
                    out_type,
                    node.name,
                    shape,
                )
                continue
            raise CanonicalMLIRError(f"elementwise node '{node.name}' has no tensor operand")

        raise CanonicalMLIRError(
            f"Unsupported FX node in canonical MLIR backend: "
            f"name={node.name}, op={node.op}, target={node.target}"
        )

    em.emit(f"    func.return {value_name[returned]} : {ret_type}")
    em.emit("  }")
    em.emit("}")
    return "\n".join(em.lines) + "\n"


def write_canonical_mlir(
    gm: fx.GraphModule,
    path: str | Path,
    fn_name: str = "staged_fn",
    example_args: tuple[Any, ...] | None = None,
    max_dense_elements: int = 16384,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        lower_to_canonical_mlir(
            gm,
            fn_name=fn_name,
            example_args=example_args,
            max_dense_elements=max_dense_elements,
        ),
        encoding="utf-8",
    )
    return path


def verify_canonical_mlir_with_mlir_opt(path: str | Path) -> tuple[bool, str]:
    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        return False, "mlir-opt not found on PATH"
    proc = subprocess.run(
        [mlir_opt, "--verify-each", str(path), "-o", "/dev/null"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stderr or proc.stdout
