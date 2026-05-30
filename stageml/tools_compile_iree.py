from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageml.baremetal_backend import IREECompileConfig, compile_mlir_with_iree


def main() -> None:
    p = argparse.ArgumentParser(description="Compile StageML residual MLIR with IREE")
    p.add_argument("--input-mlir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--module-name", default="stageml_residual_plan")
    p.add_argument("--iree-compile", default="iree-compile")
    p.add_argument("--cuda-target", default="sm_90")
    p.add_argument("--target-device", default=None, help="Use --iree-hal-target-device instead of --iree-hal-target-backends")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra-arg", action="append", default=[])
    args = p.parse_args()
    cfg = IREECompileConfig(
        input_mlir=args.input_mlir,
        output_dir=args.out_dir,
        module_name=args.module_name,
        iree_compile=args.iree_compile,
        target_device=args.target_device,
        cuda_target=args.cuda_target,
        extra_args=tuple(args.extra_arg),
    )
    result = compile_mlir_with_iree(cfg, dry_run=args.dry_run)
    out = Path(args.out_dir) / "iree_compile_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
