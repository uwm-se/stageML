from __future__ import annotations

"""Environment capture helpers for benchmark validity checks.

A single benchmark table is only interpretable when every measured row is
produced in the same Python process and under the same CUDA, torch and GPU
configuration.  These helpers make that assumption explicit in each result row.
"""

import os
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Any, Mapping

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class BenchmarkEnvironment:
    python: str
    executable: str
    platform: str
    torch_version: str
    torch_cuda_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str
    cuda_capability: str
    process_id: int

    def comparable(self) -> dict[str, Any]:
        data = asdict(self)
        # Process id is recorded for provenance but should not be used when
        # comparing serialized files across separate runs.  Inside one table it
        # must still be identical because all rows are produced in one process.
        return data


def capture_environment(device: str | None = None) -> dict[str, Any]:
    if torch is None:
        torch_version = "unavailable"
        torch_cuda_version = "unavailable"
        cuda_available = False
        cuda_device_count = 0
        cuda_device_name = "none"
        cuda_capability = "none"
    else:
        torch_version = str(getattr(torch, "__version__", "unknown"))
        torch_cuda_version = str(getattr(torch.version, "cuda", None))
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available:
            index = 0
            if device and str(device).startswith("cuda:"):
                try:
                    index = int(str(device).split(":", 1)[1])
                except Exception:
                    index = 0
            cuda_device_name = str(torch.cuda.get_device_name(index))
            try:
                major, minor = torch.cuda.get_device_capability(index)
                cuda_capability = f"sm_{major}{minor}"
            except Exception:
                cuda_capability = "unknown"
        else:
            cuda_device_name = "none"
            cuda_capability = "none"
    env = BenchmarkEnvironment(
        python=sys.version.split()[0],
        executable=sys.executable,
        platform=platform.platform(),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        cuda_capability=cuda_capability,
        process_id=os.getpid(),
    )
    return env.comparable()


def attach_environment(block: dict[str, Any], env: Mapping[str, Any]) -> dict[str, Any]:
    block["environment"] = dict(env)
    return block


def assert_same_environment(result: Mapping[str, Any], system_keys: list[str]) -> None:
    """Fail if any successful row in a table was not produced in one environment."""
    reference: dict[str, Any] | None = None
    reference_key: str | None = None
    for key in system_keys:
        block = result.get(key)
        if not isinstance(block, Mapping):
            continue
        if block.get("status", "ok") not in {"ok", None}:
            continue
        env = block.get("environment")
        if not isinstance(env, Mapping):
            raise RuntimeError(f"benchmark row {key} has no environment metadata")
        env_dict = dict(env)
        if reference is None:
            reference = env_dict
            reference_key = key
            continue
        if env_dict != reference:
            raise RuntimeError(
                "confounded benchmark table: "
                f"row {key} environment differs from row {reference_key}. "
                f"{key}={env_dict}; {reference_key}={reference}"
            )
