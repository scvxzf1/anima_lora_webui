"""Build a self-contained Krea-2 NF4 v2 checkpoint without requantizing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.quantize import (  # noqa: E402
    inspect_nf4_checkpoint,
    upgrade_nf4_checkpoint_to_self_contained,
)


DEFAULT_BF16 = ROOT / "models/diffusion_models/krea2_raw_bf16.safetensors"
DEFAULT_NF4 = ROOT / "models/diffusion_models/krea2_raw_nf4.safetensors"
DEFAULT_OUT = (
    ROOT / "models/diffusion_models/krea2_raw_nf4_self_contained.safetensors"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, default=DEFAULT_BF16)
    parser.add_argument("--nf4", type=Path, default=DEFAULT_NF4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output atomically; source files are never modified.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = upgrade_nf4_checkpoint_to_self_contained(
        args.bf16,
        args.nf4,
        args.out,
        overwrite=args.overwrite,
    )
    info = inspect_nf4_checkpoint(args.out)
    if not info.self_contained:
        raise RuntimeError(f"Output was not recognized as self-contained NF4: {args.out}")
    print(
        "Built Krea-2 NF4 v{version}: {linears} Linear4bit + "
        "{model_tensors} model tensors, {size:.2f}GB -> {path}".format(
            version=info.version,
            linears=result["linear4bit_count"],
            model_tensors=result["model_tensor_count"],
            size=result["bytes"] / 1e9,
            path=args.out,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
