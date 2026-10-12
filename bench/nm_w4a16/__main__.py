"""Default execution only emits a torch-free analytic plan."""

import argparse
import json

from library.runtime.nm_w4a16.config import NMConfig
from library.runtime.nm_w4a16.cost import comparison_report


def main():
    parser = argparse.ArgumentParser(description="Experimental 2:4 W4A16 versus dense BF16")
    parser.add_argument("mode", nargs="?", choices=("plan", "probe"), default="plan")
    parser.add_argument("--tokens", type=int, default=4032)
    parser.add_argument("--in-features", type=int, default=6144)
    parser.add_argument("--out-features", type=int, default=16384)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mask-mode", choices=("forward", "bidirectional"), default="bidirectional")
    parser.add_argument("--backend", choices=("reference", "cusparselt"), default="reference")
    parser.add_argument("--cache-sparse", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize a bounded GPU probe")
    parser.add_argument("--device", help="Explicit CUDA device, e.g. cuda:1; there is no default GPU")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        config = NMConfig(args.group_size, args.mask_mode, args.backend, args.cache_sparse)
        config.validate_shape(args.out_features, args.in_features)
        report = comparison_report(args.tokens, args.in_features, args.out_features, group_size=args.group_size)
        if args.warmup < 0 or args.iterations <= 0 or args.rank <= 0:
            raise ValueError("warmup must be nonnegative; iterations and rank must be positive")
        if args.mode == "plan":
            if args.execute:
                raise ValueError("--execute is only valid with probe")
            report["configuration"] = vars(args)
        else:
            if not args.execute or args.device is None:
                raise ValueError("probe requires BOTH --execute and an explicit --device cuda:<index>")
            from .probe import run_probe

            report = run_probe(args, config)
    except (ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
