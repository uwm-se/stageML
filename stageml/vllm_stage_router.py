from __future__ import annotations

"""StageML router for a vLLM fused-MoE interception point.

The intended production integration is a small patch inside vLLM's fused MoE
layer. That patch should call ``route_fused_experts`` with per-token tenant ids.
Tokens belonging to tenants with a compiled StageML residual kernel are routed
to the StageML op. Other tokens stay on vLLM's standard ``fused_experts`` path.

This module is import-safe outside vLLM. It lets the repo test the routing logic
without forking vLLM.
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable

import torch

from stageml.baremetal_backend import KernelManifest, read_kernel_manifest
from stageml.custom_op_registry import ensure_stageml_custom_op


@dataclass(frozen=True)
class TenantKernel:
    tenant_id: int
    adapter_id: int
    backend: str
    artifact: str
    symbol: str
    enabled: bool = True


class StageMLTenantRouter:
    def __init__(self, kernels: dict[int, TenantKernel]):
        self.kernels = dict(kernels)

    @classmethod
    def from_manifest(cls, path: str | Path) -> "StageMLTenantRouter":
        manifest: KernelManifest = read_kernel_manifest(path)
        kernels = {
            entry.tenant_id: TenantKernel(
                tenant_id=entry.tenant_id,
                adapter_id=entry.adapter_id,
                backend=entry.backend,
                artifact=entry.artifact,
                symbol=entry.symbol,
                enabled=entry.enabled,
            )
            for entry in manifest.entries
            if entry.enabled
        }
        return cls(kernels)

    def has_kernel(self, tenant_id: int) -> bool:
        entry = self.kernels.get(int(tenant_id))
        return entry is not None and entry.enabled

    def materialized_mask(self, tenant_ids: torch.Tensor) -> torch.Tensor:
        flat = tenant_ids.detach().cpu().tolist()
        values = [self.has_kernel(int(t)) for t in flat]
        return torch.tensor(values, device=tenant_ids.device, dtype=torch.bool)


def resolve_vllm_fused_experts() -> tuple[Callable[..., Any] | None, str]:
    candidates = [
        ("vllm.model_executor.layers.fused_moe.fused_moe", "fused_experts"),
        ("vllm.model_executor.layers.fused_moe", "fused_experts"),
    ]
    errors: list[str] = []
    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr)
            if callable(fn):
                return fn, f"{module_name}.{attr}"
            errors.append(f"{module_name}.{attr} exists but is not callable")
        except Exception as exc:
            errors.append(f"{module_name}.{attr}: {exc}")
    return None, "; ".join(errors)


def call_vllm_fused_experts(fn: Callable[..., Any], hidden: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    topk_ids = topk_ids.to(dtype=torch.long).contiguous()
    topk_weights = topk_weights.contiguous()
    attempts = [
        lambda: fn(
            hidden_states=hidden,
            w1=gate_up,
            w2=down,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
        ),
        lambda: fn(hidden, gate_up, down, topk_weights, topk_ids, False),
        lambda: fn(hidden, gate_up, down, topk_weights, topk_ids),
    ]
    last: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not call vLLM fused_experts: {last}")


def call_stageml_residual_op(hidden: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    if not ensure_stageml_custom_op(allow_python_fallback=True):
        raise RuntimeError("StageML residual custom op is not available")
    return torch.ops.stageml.residual_moe(hidden, gate_up, down, topk_weights, topk_ids)


def route_fused_experts(
    *,
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tenant_ids: torch.Tensor,
    router: StageMLTenantRouter,
    vllm_fused_experts: Callable[..., Any] | None = None,
    stageml_op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Route tokens between StageML residual kernels and vLLM fused experts.

    ``tenant_ids`` is per token. A full vLLM server patch should construct it
    from request metadata after batching.
    """
    if hidden.shape[0] != tenant_ids.numel():
        raise ValueError("tenant_ids must contain one tenant id per token")
    if vllm_fused_experts is None:
        vllm_fused_experts, source = resolve_vllm_fused_experts()
        if vllm_fused_experts is None:
            raise RuntimeError(f"vLLM fused experts could not be resolved: {source}")
    stageml_op = stageml_op or call_stageml_residual_op
    mask = router.materialized_mask(tenant_ids)
    if bool(mask.all()):
        return stageml_op(hidden, gate_up, down, topk_weights, topk_ids)
    if not bool(mask.any()):
        return call_vllm_fused_experts(vllm_fused_experts, hidden, gate_up, down, topk_weights, topk_ids)

    out = torch.empty((hidden.shape[0], down.shape[1]), device=hidden.device, dtype=hidden.dtype)
    for selected, fn in [(mask, stageml_op), (~mask, None)]:
        idx = torch.nonzero(selected, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        sub_hidden = hidden.index_select(0, idx).contiguous()
        sub_weights = topk_weights.index_select(0, idx).contiguous()
        sub_ids = topk_ids.index_select(0, idx).contiguous()
        if fn is None:
            sub_out = call_vllm_fused_experts(vllm_fused_experts, sub_hidden, gate_up, down, sub_weights, sub_ids)
        else:
            sub_out = fn(sub_hidden, gate_up, down, sub_weights, sub_ids)
        out.index_copy_(0, idx, sub_out)
    return out
