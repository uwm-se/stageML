from __future__ import annotations

"""Runtime checks for H100-only paper benchmark runs.

The normal artifact has smoke tests that can run on CPU.  Paper numbers should
come from the H100 suite.  These helpers make that boundary explicit and fail
before a benchmark starts if the current process is not using CUDA on an H100
class device.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class H100Environment:
    cuda_available: bool
    device_index: int
    device_name: str
    capability: str
    torch_version: str
    torch_cuda_version: str
    total_memory_gb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_h100_environment(device_index: int = 0) -> H100Environment:
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return H100Environment(
            cuda_available=False,
            device_index=device_index,
            device_name="none",
            capability="none",
            torch_version=str(torch.__version__),
            torch_cuda_version=str(torch.version.cuda),
            total_memory_gb=0.0,
        )
    name = str(torch.cuda.get_device_name(device_index))
    major, minor = torch.cuda.get_device_capability(device_index)
    props = torch.cuda.get_device_properties(device_index)
    return H100Environment(
        cuda_available=True,
        device_index=device_index,
        device_name=name,
        capability=f"sm_{major}{minor}",
        torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda),
        total_memory_gb=float(props.total_memory) / (1024.0 ** 3),
    )


def require_h100(device_index: int = 0) -> dict[str, Any]:
    env = current_h100_environment(device_index)
    if not env.cuda_available:
        raise RuntimeError("H100 benchmark requested but CUDA is not available")
    name_ok = "H100" in env.device_name.upper()
    capability_ok = env.capability == "sm_90"
    if not (name_ok or capability_ok):
        raise RuntimeError(
            "H100 benchmark requested but current device is not H100. "
            f"device={env.device_name}, capability={env.capability}"
        )
    return env.to_dict()
