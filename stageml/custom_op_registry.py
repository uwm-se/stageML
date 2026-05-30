from __future__ import annotations

"""Optional PyTorch custom-op registration for StageML residual kernels.

The production target is a C++/CUDA op that launches a compiled IREE/Triton/PTX
kernel. This file provides the Python-side registration and a correctness
fallback so the rest of the system can be developed before the binary op exists.
When ``STAGEML_CUSTOM_OP_LIB`` points to a compiled extension, the extension is
loaded first and may override/provide the actual CUDA implementation.
"""

import os
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_REGISTERED = False


def _fallback_moe(hidden: Any, gate_up: Any, down: Any, topk_weights: Any, topk_ids: Any) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the StageML custom op fallback")
    outputs = []
    for token in range(hidden.shape[0]):
        y = torch.zeros((down.shape[1],), device=hidden.device, dtype=hidden.dtype)
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot].item())
            weight = topk_weights[token, slot].to(dtype=hidden.dtype)
            gu = torch.matmul(hidden[token], gate_up[expert].transpose(0, 1))
            gate, up = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            y = y + weight * torch.matmul(act, down[expert].transpose(0, 1))
        outputs.append(y)
    return torch.stack(outputs, dim=0)


def load_binary_custom_op() -> bool:
    """Load a compiled C++/CUDA custom op library if configured."""
    if torch is None:
        return False
    lib = os.environ.get("STAGEML_CUSTOM_OP_LIB")
    if not lib:
        return False
    torch.ops.load_library(lib)
    return True


def register_python_fallback_custom_op() -> bool:
    """Register ``stageml::residual_moe`` as a Python custom op fallback.

    This is not the final performance path. It is useful for development and
    vLLM-router integration tests because callers can use one operator name
    before the C++/CUDA implementation exists.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    if torch is None or not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
        return False

    try:
        @torch.library.custom_op("stageml::residual_moe", mutates_args=())
        def residual_moe(hidden, gate_up, down, topk_weights, topk_ids):  # type: ignore[no-untyped-def]
            return _fallback_moe(hidden, gate_up, down, topk_weights, topk_ids)

        @torch.library.register_fake("stageml::residual_moe")
        def _fake(hidden, gate_up, down, topk_weights, topk_ids):  # type: ignore[no-untyped-def]
            return hidden.new_empty((hidden.shape[0], down.shape[1]))
    except Exception:
        return False
    _REGISTERED = True
    return True


def ensure_stageml_custom_op(*, allow_python_fallback: bool = True) -> bool:
    """Ensure a callable ``torch.ops.stageml.residual_moe`` exists."""
    if torch is None:
        return False
    try:
        load_binary_custom_op()
    except Exception:
        if not allow_python_fallback:
            raise
    try:
        getattr(torch.ops.stageml, "residual_moe")
        return True
    except Exception:
        pass
    if allow_python_fallback:
        return register_python_fallback_custom_op()
    return False
