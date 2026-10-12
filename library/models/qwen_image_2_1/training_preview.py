"""Live-weight preview generation with frozen startup image/text conditions."""

import time
from pathlib import Path

import torch

from library.models.qwen_image_2_1.preview_conditions import condition_key, reference_key
from library.models.qwen_image_2_1.sampling import generate_preview
from library.models.qwen_image_2_1.preview_memory import preview_state_scope, preview_swap_scope
from library.runtime.accelerator import prepare_dtype
from library.training.preview_spec import preview_references
from library.training.sample_preview_common import failed_on_any_process, should_sample


def _sample_local(accelerator, args, dit, network, prompts, outputs, epoch, steps):
    dtype, _ = prepare_dtype(args)
    latent_dir = Path(args.output_dir) / "sample" / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    with preview_state_scope(dit, network, accelerator.device):
        with preview_swap_scope(dit, args), torch.no_grad(), accelerator.autocast():
            for prompt in prompts:
                seed = int(prompt.get("seed", (getattr(args, "seed", 0) or 0) + prompt["enum"]))
                reference = outputs[reference_key(prompt)] if preview_references(prompt) else None
                latent = generate_preview(dit, prompt, outputs[condition_key(prompt)],
                                          outputs.get(condition_key(prompt, negative=True)), reference,
                                          device=accelerator.device, dtype=dtype, seed=seed,
                                          use_kv_cache=prompt.get("use_kv_cache", False))
                step_label = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                prefix = (getattr(args, "output_name", "") or "")
                stem = f"{prefix}_{step_label}_{prompt['enum']:02d}_{time.strftime('%Y%m%d%H%M%S')}_{seed}"
                torch.save({"latents": latent.cpu(), "model_family": "qwen_image_2_1",
                            "metadata": {"width": 512, "height": 512, "sample_steps": 28,
                                         "guidance_scale": prompt.get("scale", 1.0), **prompt,
                                         "sample_sampler": "euler", "seed": seed, "step": steps}},
                           latent_dir / f"{stem}.pt")
                del latent


def sample_images(accelerator, args, epoch, steps, dit, vae, text_encoder, tokenize_strategy,
                  text_encoding_strategy, sample_prompts_te_outputs=None, prompt_replacement=None,
                  network=None, sample_prompts_snapshot=None):
    if not should_sample(args, epoch, steps):
        return
    processes = max(int(getattr(accelerator, "num_processes", 1)), 1)
    rank = int(getattr(accelerator, "process_index", 0))
    error = None
    dit = accelerator.unwrap_model(dit)
    network = accelerator.unwrap_model(network) if network is not None else None
    try:
        if sample_prompts_snapshot is None or sample_prompts_te_outputs is None:
            raise ValueError("Qwen preview conditions were not cached before DiT loading")
        _sample_local(accelerator, args, dit, network, sample_prompts_snapshot[rank::processes],
                      sample_prompts_te_outputs, epoch, steps)
    except Exception as exc:
        error = exc
    if failed_on_any_process(accelerator, error is not None, processes):
        if error:
            raise error
        raise RuntimeError("Qwen preview failed on another process")
    from library.anima.training import decode_samples_for_live_preview

    accelerator.wait_for_everyone()
    decode_error = None
    try:
        decode_samples_for_live_preview(accelerator, args, vae, dit=dit, network=network)
    except Exception as exc:
        decode_error = exc
    if failed_on_any_process(accelerator, decode_error is not None, processes):
        if decode_error is not None:
            raise decode_error
        raise RuntimeError("Qwen preview decode failed on another process")
    accelerator.wait_for_everyone()
