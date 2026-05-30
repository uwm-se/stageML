from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageml.sm90_native_backend import NativeSM90Config, compile_native_sm90, write_native_sm90_result


def main() -> int:
    p = argparse.ArgumentParser(description="Compile an extracted StageML residual kernel to native sm_90 CUDA artifacts")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--output", type=int, default=4096)
    p.add_argument("--dtype", default="f32")
    p.add_argument("--arch", default="sm_90")
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = NativeSM90Config(
        output_dir=args.out_dir,
        tokens=args.tokens,
        hidden=args.hidden,
        output=args.output,
        dtype=args.dtype,
        arch=args.arch,
        nvcc=args.nvcc,
    )
    result = compile_native_sm90(cfg, dry_run=args.dry_run)
    out = Path(args.out_dir) / "native_sm90_compile_result.json"
    write_native_sm90_result(out, result)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0 if result.status in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
