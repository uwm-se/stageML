"""
Bridge between Hugging Face PEFT LoRA layers and StageML residual layers.

This file is intentionally small and explicit.  It is used for the
end-to-end benchmark where a real PEFT model is loaded, its LoRA layers are
specialized with StageML, and the PEFT LoRA modules are replaced by residual
runtime modules.

The important research point is that the replacement is produced by the
StageML tracing, rewrite, and specialization pipeline.  It is not just a call
to PEFT's merge_and_unload().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

from stageml.evaluator import specialize
from stageml.rewrite import optimize_evaluation_order
from stageml.tracer import trace_and_annotate


@dataclass
class ReplacementStats:
    replaced_layers: int = 0
    skipped_layers: int = 0
    total_rewrites: int = 0
    total_compute_ops_before: int = 0
    total_compute_ops_after: int = 0


class ExtractedLoRALayer(nn.Module):
    """A minimal LoRA layer with the same math as a PEFT LoRA Linear layer.

    It uses the loralib-style evaluation order:

        x @ W.T + scaling * (x @ A.T @ B.T)

    This form is useful because direct binding-time analysis sees the adapter
    branch as dynamic.  StageML's rewrite pass can transform it to expose the
    static product B @ A.
    """

    def __init__(self, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("W", W.detach().clone())
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.scaling = float(scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.W.t() + self.scaling * (x @ self.A.t() @ self.B.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class StageMLResidualLoRALayer(nn.Module):
    """Wrapper around a StageML residual GraphModule.

    Some transformer blocks may call Linear-like modules with extra unused
    arguments.  This wrapper accepts those arguments and forwards only the input
    tensor to the residual graph.
    """

    def __init__(self, residual: nn.Module):
        super().__init__()
        self.residual = residual

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.residual(x)


def count_compute_ops(gm: Any) -> int:
    if not hasattr(gm, "graph"):
        return 0
    return sum(1 for n in gm.graph.nodes if n.op in {"call_function", "call_method", "call_module"})


def is_peft_lora_linear(module: nn.Module, adapter_name: str = "default") -> bool:
    """Return True for PEFT LoRA modules backed by torch.nn.Linear.

    GPT-2 style Conv1D LoRA layers use a different layout and are skipped by
    this bridge.  Qwen, LLaMA-style, Mistral-style, and most modern decoder
    models use Linear projections and are supported.
    """
    if not (hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer")):
        return False
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        return False
    return isinstance(module.base_layer, nn.Linear)


def iter_lora_children(model: nn.Module, adapter_name: str = "default") -> Iterable[tuple[nn.Module, str, nn.Module]]:
    """Yield parent module, child name, child module for supported PEFT LoRA layers."""
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if is_peft_lora_linear(child, adapter_name):
                yield parent, child_name, child


def extract_peft_lora_tensors(module: nn.Module, adapter_name: str = "default") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor | None]:
    """Extract W, A, B, scaling, and optional bias from a PEFT LoRA Linear layer."""
    base = module.base_layer
    W = base.weight.detach()
    bias = base.bias.detach() if getattr(base, "bias", None) is not None else None
    A = module.lora_A[adapter_name].weight.detach()
    B = module.lora_B[adapter_name].weight.detach()
    scaling = float(module.scaling[adapter_name])
    return W, A, B, scaling, bias


def build_stageml_residual_lora_layer(
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    bias: torch.Tensor | None,
    *,
    enable_rewrite: bool = True,
) -> tuple[nn.Module, int, int, int]:
    """Build one StageML residual replacement for a PEFT LoRA Linear layer.

    Returns:
        residual wrapper, rewrite_count, compute_ops_before, compute_ops_after
    """
    src = ExtractedLoRALayer(W, A, B, scaling, bias).to(device=W.device, dtype=W.dtype).eval()
    gm, annotations = trace_and_annotate(src, {"x": "stage1"})
    compute_ops_before = count_compute_ops(gm)
    rewrite_count = 0
    if enable_rewrite:
        gm, annotations, stats = optimize_evaluation_order(gm, annotations)
        rewrite_count = int(stats.total_rewrites)
    residual = specialize(gm, annotations)
    residual = residual.to(device=W.device, dtype=W.dtype).eval()
    compute_ops_after = count_compute_ops(residual)
    return StageMLResidualLoRALayer(residual).eval(), rewrite_count, compute_ops_before, compute_ops_after


def replace_lora_layers_with_stageml(
    model: nn.Module,
    *,
    adapter_name: str = "default",
    enable_rewrite: bool = True,
    max_layers: int | None = None,
) -> ReplacementStats:
    """Replace supported PEFT LoRA Linear layers in-place with StageML residual layers.

    This is the end-to-end bridge used by the full-model benchmark.
    It preserves the surrounding Hugging Face model and only swaps the PEFT LoRA
    projection modules for StageML residual modules.
    """
    stats = ReplacementStats()
    candidates = list(iter_lora_children(model, adapter_name=adapter_name))

    for parent, child_name, child in candidates:
        if max_layers is not None and stats.replaced_layers >= max_layers:
            stats.skipped_layers += 1
            continue
        try:
            W, A, B, scaling, bias = extract_peft_lora_tensors(child, adapter_name=adapter_name)
            replacement, rewrites, before_ops, after_ops = build_stageml_residual_lora_layer(
                W, A, B, scaling, bias, enable_rewrite=enable_rewrite
            )
            setattr(parent, child_name, replacement)
            stats.replaced_layers += 1
            stats.total_rewrites += rewrites
            stats.total_compute_ops_before += before_ops
            stats.total_compute_ops_after += after_ops
        except Exception:
            stats.skipped_layers += 1
            continue

    return stats
