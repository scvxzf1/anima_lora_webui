"""Check whether real encoder initialization changes Python caption RNG state."""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.experiments.preprocess_batch_probe import load_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.75)
    model, encode = load_workload(
        SimpleNamespace(kind="anima_text", weights=args.weights)
    )
    rows = []
    for batch in (6, 3, 6, 3):
        random.seed(20260906)
        before = random.getstate()
        outputs = encode(["A detailed illustration."] * batch)
        rows.append(dict(batch=batch, python_rng_changed=before != random.getstate()))
        del outputs
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows))
    del model, encode


if __name__ == "__main__":
    main()
