from __future__ import annotations

from pathlib import Path

from stageml.baremetal_backend import (
    IREECompileConfig,
    KernelManifest,
    KernelManifestEntry,
    build_iree_compile_command,
    compile_mlir_with_iree,
    read_kernel_manifest,
    write_kernel_manifest,
)


def test_iree_command_contains_cuda_target_and_dump_dir(tmp_path: Path) -> None:
    mlir = tmp_path / "plan.mlir"
    mlir.write_text("module {}\n")
    cfg = IREECompileConfig(input_mlir=str(mlir), output_dir=str(tmp_path / "out"), cuda_target="sm_90")
    cmd = build_iree_compile_command(cfg)
    joined = " ".join(cmd)
    assert "iree-compile" in cmd[0]
    assert "--iree-hal-target-backends=cuda" in joined
    assert "--iree-cuda-target=sm_90" in joined
    assert "--iree-hal-dump-executable-files-to=" in joined


def test_iree_dry_run_records_command(tmp_path: Path) -> None:
    mlir = tmp_path / "plan.mlir"
    mlir.write_text("module {}\n")
    cfg = IREECompileConfig(input_mlir=str(mlir), output_dir=str(tmp_path / "out"))
    result = compile_mlir_with_iree(cfg, dry_run=True)
    assert result.status == "dry_run"
    assert result.vmfb and result.vmfb.endswith("stageml_residual_plan.vmfb")
    assert result.command[0] == "iree-compile"


def test_kernel_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = KernelManifest(
        version=1,
        model="mixtral",
        entries=(KernelManifestEntry(tenant_id=7, adapter_id=2, backend="iree_cuda", artifact="x.vmfb"),),
    )
    path = write_kernel_manifest(tmp_path / "manifest.json", manifest)
    loaded = read_kernel_manifest(path)
    assert loaded.model == "mixtral"
    assert loaded.entries[0].tenant_id == 7
    assert loaded.entries[0].backend == "iree_cuda"
