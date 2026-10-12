"""Diagnose frozen ctx/pool lifetime through the actual preview CPU parking."""

import argparse
import gc
from pathlib import Path
import weakref
import torch
from .model import build_model, drain
from .inputs import load_input
from .runner import DEFAULT_MODEL, step
from .manifest import ROOT, GPU_UUID, validate_device, write_json
from library.runtime.block_swap_payload import block_swap_payload_residency


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["bf16", "w8a16", "w8a8"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--swap", type=int, default=12)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    validate_device(GPU_UUID)
    model, network, initial, meta = build_model(
        Path(DEFAULT_MODEL), args.mode, args.swap, 0
    )
    tensors, inputs = load_input(
        ROOT / "output/ab-anima-20261002/cache",
        ROOT / "output/ab-anima-20261002/images-1024",
        0,
        0,
    )
    batch = {k: v.to("cuda") for k, v in tensors.items()}
    optimizer = torch.optim.AdamW(
        [p for p in network.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01
    )
    refs = []
    original = None
    if args.mode != "bf16":
        from library.runtime.convrot import fused

        cls = fused._FusedW8A8Fn if args.mode == "w8a8" else fused._FusedW8A16Fn
        original = cls.backward

        def capture(ctx, dy):
            result = original(ctx, dy)
            refs.extend(
                weakref.ref(t)
                for t in [ctx.w_q, ctx.w_scale, ctx.hadamard]
                if t is not None
            )
            return result

        cls.backward = staticmethod(capture)

    def point(name):
        gc.collect()
        torch.cuda.synchronize()
        payloads = {}
        for ref in refs:
            t = ref()
            if t is not None and t.device.type == "cuda":
                payloads[
                    (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                ] = t.untyped_storage().nbytes()
        return {
            "phase": name,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "live_backward_ctx_cuda_storage_bytes": sum(payloads.values()),
            "unique_live_ctx_cuda_storage_count": len(payloads),
            "managed_payload_residency": block_swap_payload_residency(model),
        }

    try:
        output, loss, finite = step(model, network, optimizer, batch, args.swap)
        drain(model)
        rows = [point("after_backward_output_graph_live")]
        model.to("cpu")
        network.to("cpu")
        rows.append(point("preview_cpu_park_output_graph_live"))
        del output
        rows.append(point("preview_cpu_park_output_graph_deleted"))
        write_json(
            args.out,
            {
                "status": "PASS" if all(finite.tolist()) else "FAIL",
                "mode": args.mode,
                "swap": args.swap,
                "inputs": inputs,
                "model": meta,
                "points": rows,
                "scope": "one actual checkpointed step; frozenctx weakrefs; no VAE loaded; diagnostic, not throughput",
            },
        )
        print(rows, flush=True)
    finally:
        if original is not None:
            cls.backward = staticmethod(original)


if __name__ == "__main__":
    main()
