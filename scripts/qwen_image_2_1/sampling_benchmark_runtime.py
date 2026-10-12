"""Measure configured Qwen preview policy without changing historical A/B runs."""

import time

import torch

from library.models.qwen_image_2_1.preview_conditions import condition_key, reference_key
from library.models.qwen_image_2_1.preview_memory import preview_state_scope, preview_swap_scope
from library.training.preview_spec import preview_references
from scripts.qwen_image_2_1.sampling_prepare_ab_runtime import DEVICE, DTYPE, sync


def sample_variant(dit, network, function, prompts, conditions, steps, label, args, recorder, counter):
    config = args.preview_memory_config
    original_count = dit.blocks_to_swap
    effective_count = (original_count if config.get('qwen_sample_swap_mode', 'inherit') == 'inherit'
                       else config.get('qwen_sample_blocks_to_swap', 0))
    counter.reset()
    sync()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    images = []
    restore_started = None
    setup_s = restore_s = 0.0
    try:
        with preview_state_scope(dit, network, DEVICE):
            counter.phase = 'inference_setup'
            with preview_swap_scope(dit, config):
                sync()
                setup_s = time.perf_counter() - started
                try:
                    for index, original in enumerate(prompts):
                        prompt = dict(original, sample_steps=steps)
                        seed = int(prompt.get('seed', args.runtime_seed + prompt['enum']))
                        reference = conditions[reference_key(prompt)] if preview_references(prompt) else None
                        counter.phase = 'inference_generation'
                        sync()
                        image_started = time.perf_counter()
                        with torch.no_grad(), torch.autocast('cuda', dtype=DTYPE):
                            latent = function(dit, prompt, conditions[condition_key(prompt)],
                                              conditions.get(condition_key(prompt, negative=True)), reference,
                                              device=DEVICE, dtype=DTYPE, seed=seed, use_kv_cache=False)
                        sync()
                        elapsed = time.perf_counter() - image_started
                        if not bool(torch.isfinite(latent).all()):
                            raise RuntimeError(f'{label} image {index} produced nonfinite latent')
                        path = args.output / f'{label}_image{index}.pt'
                        metadata = dict(prompt, seed=seed, sample_sampler='euler', variant=label,
                                        source_checkpoint=str(args.checkpoint),
                                        effective_blocks_to_swap=effective_count)
                        torch.save({'latents': latent.detach().cpu(), 'model_family': 'qwen_image_2_1',
                                    'metadata': metadata}, path)
                        images.append({'path': path.name, 'generation_s': elapsed, 'prompt': metadata})
                        recorder.event('image_complete', label=label, index=index, **images[-1],
                                       prepare_calls=dict(counter.counts))
                        del latent
                finally:
                    counter.phase = 'restore_training'
                    restore_started = time.perf_counter()
            sync()
            restore_s = time.perf_counter() - restore_started
        expected = {}
        if effective_count:
            expected['inference_setup'] = 1
        if original_count:
            expected['restore_training'] = 1
        if counter.counts != expected or dit.blocks_to_swap != original_count:
            raise RuntimeError(f'Preview swap restoration/count mismatch: {counter.counts}; expected {expected}')
    finally:
        counter.phase = 'idle'
        recorder.event('variant_complete', label=label, setup_s=setup_s, restore_s=restore_s,
                       total_wall_s=time.perf_counter() - started, images=images,
                       training_blocks_to_swap=original_count, effective_blocks_to_swap=effective_count,
                       prepare_calls=dict(counter.counts),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(DEVICE))
    return images
