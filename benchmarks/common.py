from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_latency_ms(fn, x, *, warmup: int = 50, iterations: int = 200) -> float:
    device = x.device if isinstance(x, torch.Tensor) else get_device()
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            fn(x)
        synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def count_compute_ops(gm) -> int:
    return sum(1 for n in gm.graph.nodes if n.op in {"call_function", "call_method", "call_module"})


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
