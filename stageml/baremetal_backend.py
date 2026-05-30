from __future__ import annotations

"""Bare metal backend hooks for StageML residual plans.

This module is intentionally narrow. It does not pretend that StageML already
owns a production CUDA backend. It provides the missing engineering bridge:

* take an MLIR residual plan file,
* invoke IREE's compiler when it is installed,
* dump generated executable artifacts for inspection,
* write a manifest that a PyTorch custom op or a vLLM router can consume.

The functions are safe to import on machines without IREE or CUDA. In that case
compile calls return a structured ``skipped`` result that records the exact
command that should be run on the H100 machine.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import os
import shutil
import subprocess
from typing import Any


@dataclass(frozen=True)
class IREECompileConfig:
    input_mlir: str
    output_dir: str
    module_name: str = "stageml_residual_plan"
    iree_compile: str = "iree-compile"
    target_backend: str = "cuda"
    target_device: str | None = None
    cuda_target: str = "sm_90"
    dump_executables: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def input_path(self) -> Path:
        return Path(self.input_mlir)

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def vmfb_path(self) -> Path:
        return self.out_dir / f"{self.module_name}.vmfb"

    @property
    def executable_dump_dir(self) -> Path:
        return self.out_dir / "iree_executables"


def build_iree_compile_command(config: IREECompileConfig) -> list[str]:
    """Build the IREE compile command used by StageML.

    IREE's exact GPU flags are version dependent. We keep them explicit and
    visible so the artifact records what was attempted.
    """
    cmd = [
        config.iree_compile,
        str(config.input_path),
        "-o",
        str(config.vmfb_path),
    ]
    if config.target_device:
        cmd.append(f"--iree-hal-target-device={config.target_device}")
    else:
        cmd.append(f"--iree-hal-target-backends={config.target_backend}")
    effective_target = config.target_device or config.target_backend
    if effective_target == "cuda" and config.cuda_target:
        cmd.append(f"--iree-cuda-target={config.cuda_target}")
    if config.dump_executables:
        cmd.append(f"--iree-hal-dump-executable-files-to={config.executable_dump_dir}")
    cmd.extend(config.extra_args)
    return cmd


@dataclass
class IREECompileResult:
    status: str
    command: list[str]
    output_dir: str
    vmfb: str | None
    executable_artifacts: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _collect_artifacts(config: IREECompileConfig) -> list[str]:
    artifacts: list[str] = []
    if config.vmfb_path.exists():
        artifacts.append(str(config.vmfb_path))
    if config.executable_dump_dir.exists():
        for path in sorted(config.executable_dump_dir.rglob("*")):
            if path.is_file():
                artifacts.append(str(path))
    return artifacts


def compile_mlir_with_iree(config: IREECompileConfig, *, dry_run: bool = False) -> IREECompileResult:
    """Compile StageML MLIR with IREE if the compiler is installed.

    The result is deliberately structured for JSON output. A skipped result is
    not an error. It means the repo patch is installed but the machine does not
    have IREE's compiler on PATH.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_iree_compile_command(config)
    if not config.input_path.exists():
        return IREECompileResult(
            status="error",
            command=cmd,
            output_dir=str(config.out_dir),
            vmfb=None,
            executable_artifacts=[],
            reason=f"input MLIR file not found: {config.input_path}",
        )
    if dry_run:
        return IREECompileResult(
            status="dry_run",
            command=cmd,
            output_dir=str(config.out_dir),
            vmfb=str(config.vmfb_path),
            executable_artifacts=[],
            reason="dry run requested",
        )
    if shutil.which(config.iree_compile) is None:
        return IREECompileResult(
            status="skipped",
            command=cmd,
            output_dir=str(config.out_dir),
            vmfb=str(config.vmfb_path),
            executable_artifacts=[],
            reason=f"{config.iree_compile} was not found on PATH",
        )
    proc = subprocess.run(cmd, text=True, capture_output=True)
    artifacts = _collect_artifacts(config)
    return IREECompileResult(
        status="ok" if proc.returncode == 0 else "error",
        command=cmd,
        output_dir=str(config.out_dir),
        vmfb=str(config.vmfb_path) if config.vmfb_path.exists() else None,
        executable_artifacts=artifacts,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        reason="" if proc.returncode == 0 else "iree-compile failed",
    )


@dataclass(frozen=True)
class KernelManifestEntry:
    tenant_id: int
    adapter_id: int
    backend: str
    artifact: str
    symbol: str = "stageml_residual_moe"
    enabled: bool = True


@dataclass(frozen=True)
class KernelManifest:
    version: int
    model: str
    entries: tuple[KernelManifestEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model": self.model,
            "entries": [asdict(e) for e in self.entries],
        }


def write_kernel_manifest(path: str | Path, manifest: KernelManifest) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return out


def read_kernel_manifest(path: str | Path) -> KernelManifest:
    data = json.loads(Path(path).read_text())
    entries = tuple(KernelManifestEntry(**entry) for entry in data.get("entries", []))
    return KernelManifest(version=int(data.get("version", 1)), model=str(data.get("model", "unknown")), entries=entries)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
