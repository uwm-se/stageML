from __future__ import annotations

"""Shared benchmark statistics utilities for StageML artifacts.

The project uses these helpers so every latency benchmark reports the same
measurement fields. The goal is to avoid tables that only show a single P50
number without run count or variance.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


STAT_FIELDS = [
    "runs",
    "warmups",
    "mean_ms",
    "std_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(float(x) for x in values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(values[lo])
    weight = rank - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def summarize_ms(times_ms: Iterable[float], *, warmups: int = 0) -> dict[str, Any]:
    values = [float(x) for x in times_ms]
    if not values:
        return {
            "runs": 0,
            "warmups": int(warmups),
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "samples_ms": [],
        }
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "runs": len(values),
        "warmups": int(warmups),
        "mean_ms": float(mean),
        "std_ms": float(std),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "samples_ms": values,
    }


def time_ms(fn: Callable[[], Any], *, warmups: int, repeats: int, use_cuda_events: bool = True) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    with_no_grad = getattr(torch, "no_grad", None) if torch is not None else None
    ctx = with_no_grad() if with_no_grad is not None else _NullContext()
    with ctx:
        for _ in range(warmups):
            fn()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(repeats):
            if (
                use_cuda_events
                and torch is not None
                and torch.cuda.is_available()
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1000.0)
    return summarize_ms(samples, warmups=warmups)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False


def speedup(baseline: dict[str, Any], optimized: dict[str, Any], metric: str = "p50_ms") -> float:
    try:
        return float(baseline[metric]) / float(optimized[metric])
    except Exception:
        return float("nan")


def attach_speedups(result: dict[str, Any], baseline_key: str, system_keys: list[str]) -> None:
    base = result.get(baseline_key, {})
    if not isinstance(base, dict):
        return
    for key in system_keys:
        block = result.get(key, {})
        if not isinstance(block, dict):
            continue
        block["speedup_over_dynamic_p50"] = speedup(base, block, "p50_ms")
        block["speedup_over_dynamic_mean"] = speedup(base, block, "mean_ms")


def write_json_csv(
    result: dict[str, Any],
    out_json: str | Path,
    *,
    systems: list[str],
    extra_columns: list[str] | None = None,
) -> tuple[Path, Path]:
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = out_path.with_suffix(".csv")
    columns = [
        "system",
        "status",
        "runs",
        "warmups",
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "speedup_over_dynamic_p50",
        "speedup_over_dynamic_mean",
    ]
    if extra_columns:
        columns.extend(extra_columns)
    rows: list[dict[str, Any]] = []
    for system in systems:
        block = result.get(system, {})
        if not isinstance(block, dict):
            block = {"status": "missing"}
        row = {"system": system, "status": block.get("status", "ok")}
        for col in columns:
            if col in row:
                continue
            row[col] = block.get(col, "")
        rows.append(row)
    with csv_path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(col, "")) for col in columns) + "\n")
    return out_path, csv_path


def _csv_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    text = str(value)
    if any(ch in text for ch in [",", "\n", '"']):
        text = '"' + text.replace('"', '""') + '"'
    return text
