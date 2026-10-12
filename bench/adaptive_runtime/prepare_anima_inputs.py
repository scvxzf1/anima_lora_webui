"""Package existing Anima latent/text caches without rebuilding either cache."""

import argparse
import json
from pathlib import Path

from bench.adaptive_runtime.anima_inputs import prepare_anima_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--latent-key")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_anima_inputs(args.latent_cache, args.text_cache, args.output,
                                          seed=args.seed, latent_key=args.latent_key), indent=2))


if __name__ == "__main__":
    main()
