"""Real cache parity smoke in a NEW output directory; source data is read-only."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file


def prepare(source, output):
    images = sorted(source.glob("*.png"))[:4]
    if not images:
        raise ValueError("source needs at least one PNG")
    fixtures = output / "inputs"
    fixtures.mkdir()
    for size in (512, 1024):
        for index in range(32):
            original = images[index % len(images)]
            target = fixtures / f"{size}-{index:03}.png"
            with Image.open(original) as image:
                image.convert("RGB").resize((size, size)).save(target)
            caption = original.with_suffix(".txt")
            target.with_suffix(".txt").write_text(
                caption.read_text() if caption.exists() else "A detailed illustration.",
                encoding="utf-8",
            )
    return fixtures


def compare(fixed, auto, suffix):
    files = sorted(fixed.glob(f"*{suffix}"))
    assert len(files) == len(list(auto.glob(f"*{suffix}"))) == 64
    maximum = 0.0
    exact = 0
    for path in files:
        other = auto / path.name
        if suffix == ".npz":
            with np.load(path) as handle:
                left = {key: torch.from_numpy(handle[key]) for key in handle.files}
            with np.load(other) as handle:
                right = {key: torch.from_numpy(handle[key]) for key in handle.files}
        else:
            left, right = load_file(path), load_file(other)
            with (
                safe_open(path, framework="pt") as a,
                safe_open(other, framework="pt") as b,
            ):
                assert a.metadata() == b.metadata()
        assert left.keys() == right.keys()
        identical = True
        for key, value in left.items():
            actual = right[key]
            assert value.shape == actual.shape and value.dtype == actual.dtype
            assert torch.isfinite(actual).all()
            if value.is_floating_point():
                relative = float(
                    (actual.float() - value.float()).norm()
                    / value.float().norm().clamp_min(1e-8)
                )
                maximum = max(maximum, relative)
                assert relative < 0.03, (path.name, key, relative)
            else:
                assert torch.equal(value, actual), (path.name, key)
            identical = identical and torch.equal(value, actual)
        exact += identical
    return dict(files=len(files), exact_files=exact, max_relative_l2=maximum)


def run_pair(output, family, run, suffix):
    fixed, auto = output / f"{family}-fixed", output / f"{family}-auto"
    fixed.mkdir()
    auto.mkdir()
    timings = {}
    for name, directory, batch in (("fixed", fixed, 2), ("auto", auto, "auto")):
        start = time.perf_counter()
        stats = run(directory, batch)
        timings[name] = time.perf_counter() - start
        assert stats.written == 64
    before = {path.name: path.stat().st_mtime_ns for path in auto.glob(f"*{suffix}")}
    skipped = run(auto, "auto")
    assert skipped.written == 0 and skipped.skipped == 64
    assert before == {
        path.name: path.stat().st_mtime_ns for path in auto.glob(f"*{suffix}")
    }
    return dict(
        family=family, **compare(fixed, auto, suffix), seconds=timings, reused=64
    )


def vae_smoke(args, inputs):
    from library.models.qwen_vae import load_vae
    from library.preprocess.latents import cache_latents

    model = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=64,
        disable_cache=True,
    )
    model.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    result = run_pair(
        args.output,
        "vae",
        lambda directory, batch: cache_latents(
            inputs,
            model,
            cache_dir=directory,
            batch_size=batch,
            io_workers=2,
        ),
        ".npz",
    )
    model.to("cpu")
    return result


def text_smoke(args, inputs):
    from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer
    from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
    from library.preprocess.text import cache_text_embeddings

    model, tokenizer = load_qwen3_text_encoder(
        args.text, dtype=torch.bfloat16, device="cuda"
    )
    tokenizer = AnimaTokenizeStrategy(
        qwen3_tokenizer=tokenizer, t5_tokenizer=load_t5_tokenizer(None)
    )

    def run(directory, batch):
        import random

        random.seed(20260906)
        return cache_text_embeddings(
            inputs,
            tokenizer,
            AnimaTextEncodingStrategy(),
            model,
            device=torch.device("cuda"),
            cache_dir=directory,
            batch_size=batch,
            caption_shuffle_variants=3,
            min_pixels=0,
        )

    result = run_pair(args.output, "anima-text", run, ".safetensors")
    model.to("cpu")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.75)
    inputs = prepare(args.source_dir, args.output)
    results = []
    for run in (vae_smoke, text_smoke):
        results.append(run(args, inputs))
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(results[-1]), flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
