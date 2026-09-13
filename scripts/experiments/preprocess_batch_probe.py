"""Bounded, read-only-model GPU sweep for preprocessing batch selection.

Run as ``python -m scripts.experiments.preprocess_batch_probe --help``.
Only the explicitly supplied JSONL output is written; dataset caches are untouched.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch


def load_workload(args):
    if args.kind == "vae":
        from library.models.qwen_vae import load_vae

        model = (
            load_vae(
                args.weights,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=64,
                disable_cache=True,
            )
            .to("cuda", dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
        )

        def encode(items):
            images = torch.stack(items).to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                return (model.encode_pixels_to_latents(images).float().cpu(),)

        return model, encode
    if args.kind == "anima_text":
        from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer
        from library.anima.strategy import (
            AnimaTokenizeStrategy,
            AnimaTextEncodingStrategy,
        )

        model, tokenizer = load_qwen3_text_encoder(
            args.weights,
            dtype=torch.bfloat16,
            device="cuda",
        )
        tokenizer = AnimaTokenizeStrategy(
            qwen3_tokenizer=tokenizer,
            t5_tokenizer=load_t5_tokenizer(None),
        )
        strategy = AnimaTextEncodingStrategy()

        def encode(items):
            with torch.no_grad():
                outputs = strategy.encode_tokens(
                    tokenizer, [model], tokenizer.tokenize(items)
                )
            return tuple(value.detach().cpu() for value in outputs)

        return model, encode
    from library.models.krea2_raw.strategy import (
        Krea2TextEncodingStrategy,
        Krea2TokenizeStrategy,
        load_krea2_text_encoder,
    )

    model, _ = load_krea2_text_encoder(
        args.weights, device="cuda", dtype=torch.bfloat16
    )
    tokenizer = Krea2TokenizeStrategy()
    strategy = Krea2TextEncodingStrategy()

    def encode(items):
        with torch.no_grad():
            outputs = strategy.encode_tokens(
                tokenizer, [model], tokenizer.tokenize(items)
            )
        return tuple(value.detach().cpu() for value in outputs)

    return model, encode


def sample_items(args, size):
    if args.kind == "vae":
        from PIL import Image
        import numpy as np
        from library.datasets.image_utils import IMAGE_TRANSFORMS

        if args.image:
            with Image.open(args.image) as image:
                pixels = np.array(image.convert("RGB").resize((size, size)))
            return [IMAGE_TRANSFORMS(pixels)] * args.items
        generator = torch.Generator().manual_seed(20260906)
        return [torch.rand((3, size, size), generator=generator) * 2 - 1] * args.items
    caption = "A detailed illustration of a city street with trees, windows and people."
    return [(caption + " ") * size] * args.items


def attempt(encode, items):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    start = time.perf_counter()
    outputs = encode(items)
    torch.cuda.synchronize()
    return outputs, {
        "seconds": time.perf_counter() - start,
        "base_bytes": base,
        "peak_bytes": torch.cuda.max_memory_allocated(),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        "free_before_bytes": free,
        "total_bytes": total,
    }


def sweep(args, encode, emit):
    for size in args.sizes:
        items = sample_items(args, size)
        reference = None
        for batch in args.batches:
            failed = False
            for repeat in range(args.repeats):
                row = {"size": size, "batch": batch, "repeat": repeat, "mode": "fixed"}
                try:
                    outputs, measured = attempt(encode, items[:batch])
                    row.update(
                        measured, status="ok", throughput=batch / measured["seconds"]
                    )
                    current = outputs[0][0].float()
                    if reference is None:
                        reference = current.clone()
                    row["relative_l2"] = float(
                        (current - reference).norm() / reference.norm().clamp_min(1e-8)
                    )
                    del outputs, current
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status="oom", error=str(exc).splitlines()[0])
                    failed = True
                emit(row)
                if failed:
                    gc.collect()
                    torch.cuda.empty_cache()
                    # The exception scope has ended, so traceback tensors are gone.
                    outputs, recovery = attempt(encode, items[:1])
                    del outputs
                    emit(
                        dict(
                            size=size, batch=1, mode="recovery", status="ok", **recovery
                        )
                    )
                    break
            if failed:
                break


def ablate(args, encode, emit):
    from library.preprocess.adaptive_batch import AutoBatcher

    for size in args.sizes:
        items = sample_items(args, size)
        for repeat in range(args.repeats):
            # Reverse run order on alternate repeats to expose thermal/order bias.
            modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
            for mode in modes:
                gc.collect()
                torch.cuda.empty_cache()
                warmup, _ = attempt(encode, items[:1])
                del warmup
                events = []

                def record(event):
                    events.append(event)
                    emit(
                        dict(event, size=size, repeat=repeat, mode=mode, event="batch")
                    )

                runner = AutoBatcher(
                    "cuda",
                    label=f"{args.kind}/{size}",
                    max_batch=args.max_batch,
                    on_event=record,
                    predict=mode not in {"no_prediction", "no_backoff"},
                    throughput=mode != "no_throughput",
                    retry_oom=mode != "no_backoff",
                )
                start = time.perf_counter()
                completed = 0
                peak = 0
                status = "ok"
                try:
                    if mode.startswith("fixed"):
                        batch = int(mode.removeprefix("fixed"))
                        for offset in range(0, len(items), batch):
                            outputs, measured = attempt(
                                encode, items[offset : offset + batch]
                            )
                            completed += len(outputs[0])
                            peak = max(peak, measured["peak_bytes"])
                            del outputs
                    else:
                        for _, outputs in runner.iter_encoded(items, encode):
                            completed += len(outputs[0])
                            del outputs
                        peak = max((e.get("peak_bytes", 0) for e in events), default=0)
                except torch.cuda.OutOfMemoryError:
                    status = "oom"
                if events:
                    peak = max(e.get("peak_bytes", 0) for e in events)
                seconds = time.perf_counter() - start
                emit(
                    dict(
                        event="summary",
                        size=size,
                        repeat=repeat,
                        mode=mode,
                        status=status,
                        completed=completed,
                        seconds=seconds,
                        throughput=completed / seconds,
                        peak_bytes=peak,
                        final_batch=runner.batch_size
                        if not mode.startswith("fixed")
                        else batch,
                        oom_count=sum(e["status"] == "oom" for e in events)
                        + (status == "oom"),
                    )
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=["vae", "krea_text", "anima_text"], required=True
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "fixed1",
            "fixed2",
            "fixed4",
            "fixed16",
            "auto",
            "no_prediction",
            "no_throughput",
            "no_backoff",
        ],
    )
    args = parser.parse_args()
    if min(*args.batches, *args.sizes, args.repeats) < 1 or args.items < max(
        args.batches
    ):
        parser.error("positive sizes/repeats required; items must cover every batch")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:

        def emit(row):
            row.update(kind=args.kind)
            line = json.dumps(row)
            output.write(line + "\n")
            output.flush()
            if row.get("event") != "batch":
                print(line, flush=True)

        emit(
            {
                "event": "environment",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "memory_fraction": args.memory_fraction,
            }
        )
        try:
            model, encode = load_workload(args)
            (ablate if args.ablate else sweep)(args, encode, emit)
            del encode, model
        except torch.cuda.OutOfMemoryError as exc:
            emit({"event": "fatal_oom", "error": str(exc).splitlines()[0]})
            raise


if __name__ == "__main__":
    main()
