#!/usr/bin/env python3
"""Create or verify the experimental fixed-policy Anima mixed NF4 artifact."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from library.anima.mixed_nf4 import convert_checkpoint, verify_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.verify_only:
        result = verify_checkpoint(args.output, args.source)
    else:
        if not args.source:
            p.error('--source is required for conversion')
        result = convert_checkpoint(args.source, args.output, device=args.device,
            progress=lambda n, total, k: print(f'[{n}/{total}] {k}', file=sys.stderr, flush=True))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
