from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int = 4
    symmetric: bool = True
    per_channel: bool = False
    channel_dim: int = 0
    eps: float = 1e-12


@dataclass(frozen=True)
class QuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    config: QuantizationConfig


@dataclass(frozen=True)
class QuantizationErrorBound:
    epsilon_weight_fro: float
    epsilon_output_fro: float | None
    theta: float
    safe: bool
    bits: int
    per_channel: bool


def _integer_range(config: QuantizationConfig) -> tuple[int, int]:
    if config.symmetric:
        qmax = (2 ** (config.bits - 1)) - 1
        return -qmax, qmax
    return 0, (2 ** config.bits) - 1


def _reduce_dims(x: torch.Tensor, channel_dim: int) -> tuple[int, ...]:
    return tuple(i for i in range(x.ndim) if i != channel_dim)


def quantize(x: torch.Tensor, config: QuantizationConfig = QuantizationConfig()) -> QuantizedTensor:
    if not torch.is_floating_point(x):
        x = x.float()
    qmin, qmax = _integer_range(config)
    if config.per_channel:
        dims = _reduce_dims(x, config.channel_dim)
        max_abs = x.abs().amax(dim=dims, keepdim=True).clamp_min(config.eps)
    else:
        max_abs = x.abs().max().clamp_min(config.eps)
    if config.symmetric:
        scale = max_abs / float(qmax)
        zero_point = torch.zeros_like(scale)
        q = torch.round(x / scale).clamp(qmin, qmax).to(torch.int32)
    else:
        if config.per_channel:
            dims = _reduce_dims(x, config.channel_dim)
            xmin = x.amin(dim=dims, keepdim=True)
            xmax = x.amax(dim=dims, keepdim=True)
        else:
            xmin = x.min()
            xmax = x.max()
        scale = ((xmax - xmin).clamp_min(config.eps)) / float(qmax - qmin)
        zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        q = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.int32)
    return QuantizedTensor(q=q, scale=scale.detach(), zero_point=zero_point.detach(), config=config)


def dequantize(qt: QuantizedTensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    y = (qt.q.to(torch.float32) - qt.zero_point.to(torch.float32)) * qt.scale.to(torch.float32)
    if dtype is not None:
        y = y.to(dtype)
    return y


def quant_dequant(x: torch.Tensor, config: QuantizationConfig = QuantizationConfig()) -> torch.Tensor:
    return dequantize(quantize(x, config), dtype=torch.float32)


def lora_delta(A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    return float(scaling) * (B.to(torch.float32) @ A.to(torch.float32))


def residualization_error(
    W: torch.Tensor,
    delta: torch.Tensor,
    *,
    config: QuantizationConfig = QuantizationConfig(),
) -> torch.Tensor:
    W32 = W.to(torch.float32)
    DQ_W = quant_dequant(W32, config)
    DQ_residual = quant_dequant(W32 + delta.to(torch.float32), config)
    dynamic_equivalent = DQ_W + delta.to(torch.float32)
    return DQ_residual - dynamic_equivalent


def analyze_residualization(
    W: torch.Tensor,
    delta: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
    theta: float,
    config: QuantizationConfig = QuantizationConfig(),
) -> QuantizationErrorBound:
    err = residualization_error(W, delta, config=config)
    eps_w = float(torch.linalg.vector_norm(err).item())
    eps_out: float | None = None
    if x is not None:
        out_err = x.to(torch.float32) @ err.t().contiguous()
        eps_out = float(torch.linalg.vector_norm(out_err).item())
    compare_value = eps_out if eps_out is not None else eps_w
    return QuantizationErrorBound(
        epsilon_weight_fro=eps_w,
        epsilon_output_fro=eps_out,
        theta=float(theta),
        safe=bool(compare_value <= theta),
        bits=int(config.bits),
        per_channel=bool(config.per_channel),
    )


def safe_to_residualize(
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    *,
    theta: float,
    x: torch.Tensor | None = None,
    config: QuantizationConfig = QuantizationConfig(),
) -> QuantizationErrorBound:
    delta = lora_delta(A, B, scaling)
    return analyze_residualization(W, delta, x=x, theta=theta, config=config)
