from pathlib import Path

from stageml.sm90_native_backend import NativeSM90Config, build_nvcc_commands, compile_native_sm90, emit_native_sm90_cuda


def test_emit_native_sm90_cuda_contains_static_shape(tmp_path: Path):
    cfg = NativeSM90Config(output_dir=str(tmp_path), tokens=2, hidden=3, output=4)
    path = emit_native_sm90_cuda(cfg)
    text = path.read_text()
    assert "stageml_materialized_residual_sm90" in text
    assert "row >= 2" in text
    assert "col >= 4" in text
    assert "kk < 3" in text


def test_build_nvcc_commands_targets_sm90(tmp_path: Path):
    cfg = NativeSM90Config(output_dir=str(tmp_path), tokens=2, hidden=3, output=4)
    emit_native_sm90_cuda(cfg)
    ptx_cmd, cubin_cmd = build_nvcc_commands(cfg)
    assert "-arch=sm_90" in ptx_cmd
    assert "-arch=sm_90" in cubin_cmd
    assert "-ptx" in ptx_cmd
    assert "-cubin" in cubin_cmd


def test_compile_native_sm90_dry_run(tmp_path: Path):
    cfg = NativeSM90Config(output_dir=str(tmp_path), tokens=2, hidden=3, output=4)
    result = compile_native_sm90(cfg, dry_run=True)
    assert result.status == "dry_run"
    assert result.cu is not None
    assert result.ptx is not None
    assert result.cubin is not None
    assert Path(result.cu).exists()
