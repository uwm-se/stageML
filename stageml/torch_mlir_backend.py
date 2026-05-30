from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _load_torch_mlir() -> Any:
    try:
        import torch_mlir
        return torch_mlir
    except Exception as exc:
        raise RuntimeError(
            "torch-mlir is not installed. Use requirements_torch_mlir_colab.txt on Linux/Colab Python 3.11."
        ) from exc


def _normalize_example_args(example_args: Any) -> tuple[Any, ...]:
    if isinstance(example_args, tuple):
        return example_args
    return (example_args,)


def lower_with_torch_mlir(
    model: nn.Module,
    example_args: Any,
    output_type: str = "linalg-on-tensors",
) -> str:
    torch_mlir = _load_torch_mlir()
    model = model.eval()
    args = _normalize_example_args(example_args)

    with torch.no_grad():
        try:
            from torch_mlir.fx import export_and_import
            module = export_and_import(model, *args, output_type=output_type)
            return str(module)
        except Exception:
            pass

        traced = torch.jit.trace(model, args)
        if hasattr(torch_mlir, "OutputType"):
            normalized = output_type.replace("-", "_").upper()
            enum_value = getattr(torch_mlir.OutputType, normalized, None)
            if enum_value is not None:
                module = torch_mlir.compile(traced, args, output_type=enum_value)
            else:
                module = torch_mlir.compile(traced, args, output_type=output_type)
        else:
            module = torch_mlir.compile(traced, args, output_type=output_type)
        return str(module)


def write_torch_mlir(
    model: nn.Module,
    example_args: Any,
    path: str | Path,
    output_type: str = "linalg-on-tensors",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lower_with_torch_mlir(model, example_args, output_type=output_type)
    path.write_text(text, encoding="utf-8")
    return path
