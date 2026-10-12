"""Runtime helpers for the isolated prepare A/B GPU experiment."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

DEVICE = torch.device('cuda:0')
DTYPE = torch.bfloat16


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def sync():
    torch.cuda.synchronize(DEVICE)


def cache_conditions(args, config, recorder):
    from library import train_util
    from library.models.qwen_image_2_1.preview_conditions import cache_preview_text, cache_preview_references
    from library.models.qwen_image_2_1.strategy import QwenImage21TokenizeStrategy, QwenImage21TextEncodingStrategy
    from library.models.qwen_image_2_1.text_encoder_runtime import text_encoder_for_cache
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_vae
    prompts = train_util.load_prompts(config['sample_prompts'])
    if len(prompts) != 2:
        raise ValueError(f'Exactly two original sample prompts required, got {len(prompts)}')
    short = dict(prompts[0], enum=2, guidance_scale=3.0, negative_prompt='', sample_steps=args.warmup_steps)
    snapshot_file = args.output / 'prompts.json'
    snapshot_file.write_text(json.dumps([*prompts, short], ensure_ascii=False, indent=2))
    runtime_args = SimpleNamespace(**(config | {'sample_prompts': str(snapshot_file), 'output_dir': str(args.output)}))
    trainer = SimpleNamespace()
    tokenizer = QwenImage21TokenizeStrategy(config['qwen3'])
    started = time.perf_counter()
    with text_encoder_for_cache(config['qwen3'], dtype=DTYPE, device='cuda:0',
                                offload='on', cache_policy='auto') as model:
        cache_preview_text(trainer, runtime_args, model, tokenizer, QwenImage21TextEncodingStrategy())
    del model, tokenizer
    clean()
    recorder.event('text_conditions_cached', wall_s=time.perf_counter() - started)
    vae = load_qwen_image_2_1_vae(config['vae'], dtype=DTYPE, device='cpu')
    accelerator = SimpleNamespace(device=DEVICE, num_processes=1)
    cache_preview_references(trainer, runtime_args, accelerator, vae)
    del vae
    clean()
    torch.save({'prompts': trainer.sample_prompts_snapshot,
                'conditions': trainer.sample_prompts_te_outputs}, args.output / 'conditions.pt')
    recorder.event('conditions_saved', path='conditions.pt')
    return trainer.sample_prompts_snapshot, trainer.sample_prompts_te_outputs


def build_model(config, args, recorder):
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
    from library.models.qwen_image_2_1.attention_backend import prepare_qwen_image_2_1_attention
    from library.models.qwen_image_2_1.block_swap import enable_qwen_image_2_1_block_swap
    from library.models.qwen_image_2_1.compile import compile_qwen_image_2_1_blocks
    from library.models.qwen_image_2_1.lora_targets import qwen_image_2_1_target_kwargs
    from networks.lora_anima import create_network
    started = time.perf_counter()
    dit = load_qwen_image_2_1_transformer(config['pretrained_model_name_or_path'], dtype=DTYPE, device='cpu')
    dit.requires_grad_(False)
    base_parameters = tuple(dit.parameters())
    prepare_qwen_image_2_1_attention(dit, 'torch', dtype=DTYPE)
    dit.enable_gradient_checkpointing()
    swap_count = int(config.get('blocks_to_swap', 24))
    if not 0 <= swap_count <= len(dit.transformer_blocks) - 2:
        raise ValueError(f'Configured blocks_to_swap must be in 0..{len(dit.transformer_blocks) - 2}')
    enable_qwen_image_2_1_block_swap(
        dit, swap_count, DEVICE, transfer_dtype=config.get('block_swap_transfer_dtype', 'bf16'),
        restore_mode=config.get('block_swap_restore_mode', 'slab'),
    )
    targets = {key: value for key, value in qwen_image_2_1_target_kwargs().items() if value is not None}
    network = create_network(1.0, 16, 16.0, None, [], dit, model_family='qwen_image_2_1', **targets)
    network.apply_to([], dit, False, True)
    info = network.load_weights(str(args.checkpoint))
    if info is None or info.missing_keys or info.unexpected_keys:
        raise RuntimeError('Adapter checkpoint strict load failed: ' + str({
            'missing': len(info.missing_keys) if info is not None else None,
            'unexpected': len(info.unexpected_keys) if info is not None else None,
        }))
    network.enable_gradient_checkpointing()
    network.to(device=DEVICE, dtype=DTYPE)
    dit.move_to_device_except_swap_blocks(DEVICE)
    dit.prepare_block_swap_before_forward()
    count = compile_qwen_image_2_1_blocks(dit, backend=config.get('dynamo_backend', 'inductor'),
                mode=config.get('compile_inductor_mode'), dynamic_seq=config.get('compile_dynamic_seq', True), scope='all')
    if count != len(dit.transformer_blocks):
        raise RuntimeError('Not all Qwen blocks compiled')
    sync()
    recorder.event('model_ready', wall_s=time.perf_counter() - started, compiled_blocks=count,
                   blocks_to_swap=swap_count,
                   block_swap_transfer_dtype=config.get('block_swap_transfer_dtype', 'bf16'),
                   block_swap_restore_mode=config.get('block_swap_restore_mode', 'slab'),
                   adapter_load=str(info), base_frozen=all(not p.requires_grad for p in base_parameters))
    return dit, network, base_parameters


class PrepareCounter:
    def __init__(self, dit):
        self.phase = 'idle'
        self.counts = {}
        adapter = dit._qwen_image_2_1_block_swap_adapter
        original = adapter.prepare_block_swap_before_forward

        def counted(*args, **kwargs):
            self.counts[self.phase] = self.counts.get(self.phase, 0) + 1
            return original(*args, **kwargs)

        adapter.prepare_block_swap_before_forward = counted
        dit.prepare_block_swap_before_forward = counted

    def reset(self):
        self.counts = {}


def sample_variant(dit, network, function, prompts, conditions, steps, label, args, recorder, counter):
    from library.models.qwen_image_2_1.preview_conditions import condition_key, reference_key
    from library.training.preview_spec import preview_references
    counter.reset()
    sync()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    images = []
    dit.eval()
    network.eval()
    setup_s = restore_s = 0.0
    try:
        counter.phase = 'inference_setup'
        dit.switch_block_swap_for_inference()
        sync()
        setup_s = time.perf_counter() - started
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
                                  device=DEVICE, dtype=DTYPE, seed=seed)
            sync()
            elapsed = time.perf_counter() - image_started
            if not bool(torch.isfinite(latent).all()):
                raise RuntimeError(f'{label} image {index} produced nonfinite latent')
            path = args.output / f'{label}_image{index}.pt'
            metadata = dict(prompt, seed=seed, sample_sampler='euler', variant=label, source_checkpoint=str(args.checkpoint))
            torch.save({'latents': latent.detach().cpu(), 'model_family': 'qwen_image_2_1',
                        'metadata': metadata}, path)
            images.append({'path': path.name, 'generation_s': elapsed, 'prompt': metadata})
            recorder.event('image_complete', label=label, index=index, **images[-1], prepare_calls=dict(counter.counts))
            del latent
    finally:
        counter.phase = 'restore_training'
        restore_started = time.perf_counter()
        dit.train()
        network.train()
        dit.switch_block_swap_for_training()
        sync()
        restore_s = time.perf_counter() - restore_started
        counter.phase = 'idle'
        if len(images) == len(prompts):
            predictions = sum(steps * (1 + int(
                float(p.get('guidance_scale', p.get('scale', 1.0))) > 1
                and p.get('negative_prompt') is not None
                and condition_key(p, negative=True) in conditions
            )) for p in prompts)
            expected_generation = predictions if function.__module__ == 'qwen_sampling_prepare_original' else 0
            expected_counts = {'inference_setup': 1, 'restore_training': 1}
            if expected_generation:
                expected_counts['inference_generation'] = expected_generation
            if counter.counts != expected_counts:
                raise RuntimeError(f'{label}: unexpected prepare counts {counter.counts}, expected {expected_counts}')
        recorder.event('variant_complete', label=label, setup_s=setup_s, restore_s=restore_s,
                       total_wall_s=time.perf_counter() - started, images=images,
                       prepare_calls=dict(counter.counts),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(DEVICE))
    return images


def compare(first, second, output, recorder, label):
    rows = []
    for left, right in zip(first, second, strict=True):
        a = torch.load(output / left['path'], map_location='cpu', weights_only=True)['latents'].float()
        b = torch.load(output / right['path'], map_location='cpu', weights_only=True)['latents'].float()
        difference = a - b
        rows.append({'left': left['path'], 'right': right['path'], 'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                     'torch_equal': torch.equal(a, b), 'max_abs': difference.abs().max().item(),
                     'relative_l2': (difference.norm() / a.norm().clamp_min(1e-12)).item()})
    recorder.event('latent_comparison', label=label, images=rows)
    if any(not row['finite'] or not row['torch_equal'] for row in rows):
        raise RuntimeError(f'{label}: A/B latents are not exactly equal; inspect recorded errors before acceptance')


def verify_exception_restore(dit, network, prompts, conditions, args, recorder, counter):
    from library.models.qwen_image_2_1.family import forward_for_loss
    from library.models.qwen_image_2_1.sampling import generate_preview
    calls = 0

    def failing_forward(*values, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('intentional second-prompt failure')
        return forward_for_loss(*values, **kwargs)

    def failing_sample(*values, **kwargs):
        return generate_preview(*values, **kwargs, forward=failing_forward)

    try:
        sample_variant(dit, network, failing_sample, prompts[:2], conditions, 2,
                       'exception_probe_B', args, recorder, counter)
    except RuntimeError as error:
        if str(error) != 'intentional second-prompt failure':
            raise
    else:
        raise RuntimeError('Expected second-prompt exception did not occur')
    if dit.offloader.forward_only or not dit.training or not network.training or dit.offloader.futures:
        raise RuntimeError('Exception failed to restore and drain the training layout')
    recorder.event('exception_restore_verified', forward_calls=calls,
                   prepare_calls=dict(counter.counts), pending_transfers=0)


def restored_training_step(dit, network, base_parameters, prompts, conditions, args, recorder, counter):
    from library.models.qwen_image_2_1.family import forward_for_loss
    from library.models.qwen_image_2_1.preview_conditions import condition_key, reference_key
    prompt = next((p for p in prompts[:2] if reference_key(p) in conditions), None)
    if prompt is None:
        raise ValueError('Restored training check requires a real edit reference')
    references = tuple(t.to(device=DEVICE, dtype=DTYPE) for t in conditions[reference_key(prompt)])
    clean_latent = references[0][:, :, :8, :8].contiguous()
    positive = conditions[condition_key(prompt)]
    hidden, mask, slot = (t.to(device=DEVICE, dtype=DTYPE if i == 0 else torch.bool) for i, t in enumerate(positive))
    parameters = [p for p in network.parameters() if p.requires_grad]
    if not parameters or any(p.requires_grad or p.grad is not None for p in base_parameters):
        raise RuntimeError('Adapter/base training ownership invalid')
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    before = [p.detach().float().cpu().clone() for p in parameters]
    dit.train()
    network.train()
    dit.enable_gradient_checkpointing()
    counter.phase = 'restored_training_step'
    dit.prepare_block_swap_before_forward()
    torch.manual_seed(1234)
    noise = torch.randn_like(clean_latent)
    sigma = torch.full((1,), 0.5, device=DEVICE)
    noisy = (0.5 * clean_latent + 0.5 * noise).requires_grad_(True)
    sync()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=DTYPE):
        prediction = forward_for_loss(dit, noisy, hidden, mask, sigma,
                                       reference_latents=references, image_slot_mask=slot)
        loss = (prediction.float() - (noise - clean_latent).float()).square().mean()
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero = sum(int(torch.count_nonzero(g).item()) for g in gradients)
    if not bool(torch.isfinite(loss)) or not finite or nonzero == 0:
        raise RuntimeError('Restored training loss/adapter gradients failed')
    if any(p.requires_grad or p.grad is not None for p in base_parameters):
        raise RuntimeError('Frozen base unexpectedly trained')
    optimizer.step()
    sync()
    update = sum(float((p.detach().float().cpu() - old).square().sum()) for p, old in zip(parameters, before)) ** 0.5
    if not update > 0:
        raise RuntimeError('Restored AdamW step did not change adapter parameters')
    torch.save({key: value.detach().cpu() for key, value in network.state_dict().items()},
               args.output / 'restored_training_adapter.pt')
    recorder.event('restored_training_step', loss=loss.item(), gradients_finite=finite,
                   gradient_nonzero_elements=nonzero, adapter_update_l2=update,
                   base_frozen=True, full_checkpoint=bool(dit.gradient_checkpointing),
                   wall_s=time.perf_counter() - started, target_latent_shape=list(clean_latent.shape))
    counter.phase = 'idle'
    del optimizer, gradients, before, references, clean_latent, hidden, mask, slot, noise, prediction, noisy, loss


def decode_results(config, args, recorder, results):
    from PIL import Image, ImageDraw, PngImagePlugin
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_vae
    from library.models.qwen_image_2_1.preview_decode import decode_latent
    vae = load_qwen_image_2_1_vae(config['vae'], dtype=DTYPE, device=DEVICE)
    decoded = {}
    for label, rows in results.items():
        images = []
        for row in rows:
            record = torch.load(args.output / row['path'], map_location='cpu', weights_only=True)
            with torch.no_grad():
                image = decode_latent(vae, record['latents'])
            info = PngImagePlugin.PngInfo()
            info.add_text('qwen_preview', json.dumps(record['metadata'], ensure_ascii=False))
            image.save((args.output / row['path']).with_suffix('.png'), pnginfo=info)
            images.append(image)
        decoded[label] = images
    labels = list(decoded)
    for index in range(2):
        width = sum(decoded[label][index].width for label in labels)
        height = max(decoded[label][index].height for label in labels)
        canvas = Image.new('RGB', (width, height + 28), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label in labels:
            canvas.paste(decoded[label][index], (x, 28))
            draw.text((x + 8, 8), label, fill='black')
            x += decoded[label][index].width
        canvas.save(args.output / f'comparison_image{index}.png')
    del vae
    clean()
    recorder.event('decoded_pngs_saved', variants=labels)


def run(args, config, recorder, baseline):
    from library.models.qwen_image_2_1.sampling import generate_preview
    args.runtime_seed = int(config.get('seed', 0) or 0)
    prompts, conditions = cache_conditions(args, config, recorder)
    dit, network, base_parameters = build_model(config, args, recorder)
    counter = PrepareCounter(dit)
    functions = {'A': baseline, 'B': generate_preview}
    results = {}
    for variant in ('A', 'B'):
        sample_variant(dit, network, functions[variant], prompts[:2], conditions,
                       args.warmup_steps, f'warmup_{variant}', args, recorder, counter)
    cfg_results = {}
    for variant in ('A', 'B'):
        cfg_results[variant] = sample_variant(dit, network, functions[variant], prompts[2:], conditions,
                       args.warmup_steps, f'cfg3_short_{variant}', args, recorder, counter)
    compare(cfg_results['A'], cfg_results['B'], args.output, recorder, 'CFG3_A_vs_B')
    warm_baselines = []
    for index in range(args.baseline_repeats):
        label = f'hot_baseline_A{index}'
        warm_baselines.append(sample_variant(dit, network, baseline, prompts[:2], conditions,
                              args.steps, label, args, recorder, counter))
    for index in range(1, len(warm_baselines)):
        compare(warm_baselines[0], warm_baselines[index], args.output, recorder, 'A_repeat_stability')
    for index, variant in enumerate(args.order):
        label = f'actual_{index}_{variant}'
        results[label] = sample_variant(dit, network, functions[variant], prompts[:2], conditions,
                                       args.steps, label, args, recorder, counter)
    reference_a = next(rows for label, rows in results.items() if label.endswith('_A'))
    for label, rows in results.items():
        compare(reference_a, rows, args.output, recorder, f'A_vs_{label}')
        if label.endswith('_A'):
            compare(warm_baselines[0], rows, args.output, recorder, f'A_repeat_vs_{label}')
    verify_exception_restore(dit, network, prompts, conditions, args, recorder, counter)
    restored_training_step(dit, network, base_parameters, prompts, conditions, args, recorder, counter)
    dit.offloader.flush_profile_events(blocking=True)
    del dit, network, base_parameters, counter
    clean()
    if not args.skip_decode:
        decode_results(config, args, recorder, results)
