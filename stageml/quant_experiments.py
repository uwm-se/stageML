from __future__ import annotations

"""Reusable quantization gate experiments for StageML artifacts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from stageml.quant_absint import QuantizationConfig, lora_delta, safe_to_residualize


@dataclass(frozen=True)
class QuantGateCase:
    bits: int
    per_channel: bool
    theta: float
    epsilon_weight_fro: float
    epsilon_output_fro: float | None
    decision: str
    max_output_abs_error: float
    relative_output_error: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_reproducible_case(*, seed: int, shape: int, rank: int, delta_scale: float, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Use a CPU generator for deterministic values across CPU and CUDA runs,
    # then move the tensors to the requested device.  This keeps the middle
    # band and theta sweep reproducible while allowing H100 execution.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(shape, shape, generator=g, dtype=torch.float32) * 0.05
    A = torch.randn(rank, shape, generator=g, dtype=torch.float32) * delta_scale
    B = torch.randn(shape, rank, generator=g, dtype=torch.float32) * delta_scale
    x = torch.randn(min(32, shape), shape, generator=g, dtype=torch.float32) * 0.05
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    return W.to(dev), A.to(dev), B.to(dev), x.to(dev)


def evaluate_gate_case(
    *,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
    bits: int,
    theta: float,
    per_channel: bool = True,
    scaling: float = 1.0,
) -> QuantGateCase:
    cfg = QuantizationConfig(bits=bits, per_channel=per_channel, channel_dim=0)
    bound = safe_to_residualize(W, A, B, scaling, theta=theta, x=x, config=cfg)
    delta = lora_delta(A, B, scaling)
    # Downstream layer validation at the same boundary as the abstract gate.
    # This is a deterministic proxy used by unit tests and smoke artifacts.  A
    # full language-model perplexity guard can be run separately in H100 mode.
    from stageml.quant_absint import quant_dequant

    dynamic = x @ (quant_dequant(W, cfg) + delta).t()
    residual = x @ quant_dequant(W + delta, cfg).t()
    diff = (residual - dynamic).float()
    denom = float(torch.linalg.vector_norm(dynamic.float()).item()) + 1e-12
    max_abs = float(diff.abs().max().item())
    rel = float(torch.linalg.vector_norm(diff).item() / denom)
    return QuantGateCase(
        bits=bits,
        per_channel=per_channel,
        theta=float(theta),
        epsilon_weight_fro=float(bound.epsilon_weight_fro),
        epsilon_output_fro=bound.epsilon_output_fro,
        decision="accept" if bound.safe else "reject",
        max_output_abs_error=max_abs,
        relative_output_error=rel,
    )


def find_middle_band_case(
    *,
    seed: int = 7,
    shape: int = 32,
    rank: int = 4,
    bits_candidates: Iterable[int] = (8, 6, 5, 4),
    theta_multiplier: float = 4.0,
    device: str = "cpu",
) -> QuantGateCase:
    W, A, B, x = make_reproducible_case(seed=seed, shape=shape, rank=rank, delta_scale=0.02, device=device)
    best: QuantGateCase | None = None
    for bits in bits_candidates:
        probe = evaluate_gate_case(W=W, A=A, B=B, x=x, bits=bits, theta=1e30, per_channel=True)
        eps = float(probe.epsilon_output_fro if probe.epsilon_output_fro is not None else probe.epsilon_weight_fro)
        if eps > 0.0:
            theta = max(eps * float(theta_multiplier), eps + 1e-9)
            case = evaluate_gate_case(W=W, A=A, B=B, x=x, bits=bits, theta=theta, per_channel=True)
            if 0.0 < eps < theta and case.decision == "accept":
                return case
            best = case
    if best is not None:
        return best
    raise RuntimeError("could not construct a nonzero quantization middle band case")


def write_quant_cases(cases: list[QuantGateCase], out_json: str | Path) -> tuple[Path, Path]:
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": [c.to_dict() for c in cases]}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = out_path.with_suffix(".csv")
    columns = [
        "bits",
        "per_channel",
        "theta",
        "epsilon_weight_fro",
        "epsilon_output_fro",
        "decision",
        "max_output_abs_error",
        "relative_output_error",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for case in cases:
            row = case.to_dict()
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")
    return out_path, csv_path
