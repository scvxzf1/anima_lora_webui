"""Deterministic four-update replay with/without real preview CPU/VAE lifecycle."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from .inputs import load_input
from .manifest import GPU_UUID, ROOT, validate_device, write_json
from .model import build_model, drain
from .numerics import tensor_hash
from .runner import DEFAULT_MODEL, step


def fingerprints(network, optimizer):
    params = {name: param for name, param in network.named_parameters() if param.requires_grad}
    def digest(value):
        work = value.reshape(1) if value.ndim == 0 else value
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(work)}
    return {
        "parameters": {name: digest(param) for name, param in params.items()},
        "gradients": {name: digest(param.grad) if param.grad is not None else None
                      for name, param in params.items()},
        "optimizer": {name: {key: digest(value) if isinstance(value, torch.Tensor) else value
                             for key, value in optimizer.state[param].items()}
                      for name, param in params.items()},
        "cpu_rng": tensor_hash(torch.get_rng_state()),
        "cuda_rng": [tensor_hash(value) for value in torch.cuda.get_rng_state_all()],
    }


def preview(model, network, vae, batch, directory):
    from library.training.train_bootstrap import decode_deferred_samples_safely
    from library.training.preview_lifecycle import preview_lifecycle
    from library.runtime.block_swap_payload import block_swap_payload_residency

    accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cuda:0"), trackers=[],
                                  unwrap_model=lambda model: model)
    args = SimpleNamespace(output_dir=str(directory), model_family="anima",
                           sample_prompts="fixed replay", sample_every_n_steps=2)
    staging = directory / "sample" / "latents"
    staging.mkdir(parents=True)
    # Exercise inference hooks and RNG isolation, then the actual shared decode.
    with preview_lifecycle(model, network):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["x"], batch["timesteps"], batch["context"],
                        padding_mask=batch["padding_mask"])
            if not torch.isfinite(out).all():
                raise RuntimeError("Preview output nonfinite")
            torch.rand(3, device="cuda")
            torch.rand(3)
            torch.save({"latents": out.detach().cpu(), "prompt": "fixed replay", "enum": 0},
                       staging / "replay.pt")
        del out
    drain(model)
    residency = []
    from library.anima import training
    original = training.decode_pending_samples
    def decode(*args):
        payload = block_swap_payload_residency(model)
        residency.append(payload)
        if any(n for device, n in payload["bytes_by_device"].items()
               if device.startswith("cuda")):
            raise RuntimeError("Managed CUDA payload survived CPU parking")
        return original(*args)
    training.decode_pending_samples = decode
    try:
        decode_deferred_samples_safely(
            accelerator, args, SimpleNamespace(unet=model, network=network), vae)
    finally:
        training.decode_pending_samples = original
    images = list((directory / "sample").glob("*.png"))
    if len(images) != 1 or staging.exists():
        raise RuntimeError("Actual VAE preview did not decode successfully")
    return residency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bf16", "w8a16", "w8a8"], required=True)
    parser.add_argument("--swap", type=int, required=True)
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    validate_device(GPU_UUID)
    from library.models.qwen_vae import load_vae
    vae = load_vae("/home/scv/pac/models/vae/anima/anima_vae.safetensors",
                   device="cpu", dtype=torch.bfloat16, eval=True)
    model, network, initial, metadata = build_model(
        Path(DEFAULT_MODEL), args.mode, args.swap, 0, flash_deterministic=True)
    params = [param for param in network.parameters() if param.requires_grad]
    identities = [id(param) for param in params]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=.01)
    torch.manual_seed(1234)
    history, residency = [], []
    for index in range(4):
        batch, inputs = load_input(ROOT / "output/ab-anima-20261002/cache",
                                   ROOT / "output/ab-anima-20261002/images-1024",
                                   index % 2, 17 + index, [.1, .5, .9][index % 3])
        batch = {key: value.to("cuda") for key, value in batch.items()}
        output, loss, finite = step(model, network, optimizer, batch, args.swap)
        del output  # Completed diagnostic graph must not outlive the step.
        drain(model)
        if not all(finite.tolist()):
            raise RuntimeError("Nonfinite replay update")
        history.append({"step": index + 1, "loss": float(loss), "input_hash": inputs["input_hash"]})
        if index == 1 and args.cycle:
            residency = preview(model, network, vae, batch, args.out_dir)
            if identities != [id(param) for param in network.parameters() if param.requires_grad]:
                raise RuntimeError("Preview changed Parameter identity")
    write_json(args.out_dir / "result.json", {
        "status": "PASS", "mode": args.mode, "swap": args.swap, "cycle": args.cycle,
        "model": metadata, "steps": history, "effective_updates": 4,
        "optimizer_step_values": sorted({int(state["step"]) for state in optimizer.state.values()}),
        "preview_park_residency": residency, "fingerprints": fingerprints(network, optimizer),
        "scope": "deterministic fixed replay; one inference/VAE lifecycle at step2; not production dataloader resume or long-run quality",
    })


if __name__ == "__main__":
    main()
