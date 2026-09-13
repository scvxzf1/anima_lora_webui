"""Bounded real-model compile probe; does not alter production capability gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error("steps must be between 1 and 10")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import toml
    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from library.runtime import harness
    from library.training import compat_matrix

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = toml.load(ROOT / "output/runs/z-image-100step-20260905/configs/formal.toml")
    config.update(
        torch_compile=not args.eager,
        max_train_steps=args.steps,
        max_train_epochs=None,
        sample_at_first=False,
        sample_every_n_steps=0,
        sample_every_n_epochs=0,
        save_every_n_steps=0,
        output_dir=str(output / "ckpt"),
        logging_dir=str(output / "logs"),
        output_name="zimage-compile-probe",
        memory_probe_max_steps=args.steps,
    )
    config_path = output / "config.toml"
    config_path.write_text(toml.dumps(config), encoding="utf-8")
    stats = {"compiled_graphs": 0, "compiled_layers": 0, "eager": args.eager}
    original_error = compat_matrix._CompatBuilder.error

    def probe_error(self, code, key, message):
        if code == "z_image_torch_compile":
            print("PROBE: bypassing only unvalidated Z-Image compile gate", flush=True)
            return
        return original_error(self, code, key, message)

    inductor = lookup_backend("inductor")

    def counted_inductor(graph, inputs):
        stats["compiled_graphs"] += 1
        print(f"PROBE: Inductor graph {stats['compiled_graphs']}", flush=True)
        return inductor(graph, inputs)

    def compile_resident(model, network, **kwargs):
        del network, kwargs
        if model.__class__.__name__ != "ZImageTransformer2DModel":
            raise TypeError(f"unexpected probe model: {type(model)}")
        resident = len(model.layers) - int(getattr(model, "blocks_to_swap", 0))
        for layer in model.layers[:resident]:
            layer.forward = torch.compile(
                layer.forward, backend=counted_inductor, dynamic=False,
            )
        stats["compiled_layers"] = resident
        print(f"PROBE: compiled {resident}/{len(model.layers)} resident layers", flush=True)

    compat_matrix._CompatBuilder.error = probe_error
    harness.compile_blocks_for_training = compile_resident
    sys.argv = [str(ROOT / "train.py"), "--config_file", str(config_path)]
    start = time.monotonic()
    try:
        runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
        if not args.eager and not stats["compiled_graphs"]:
            raise RuntimeError("compile probe executed no Inductor graphs")
        stats["status"] = "ok"
    except BaseException as exc:
        stats.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stats["wall_seconds"] = time.monotonic() - start
        if torch.cuda.is_initialized():
            stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            stats["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "probe.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
