"""Identical outer-forward instrumentation and frozen-fixture KV acceptance."""
from __future__ import annotations

from collections import Counter
from functools import wraps
import json
from pathlib import Path
import time

import torch
from scripts.qwen_image_2_1.sampling_prepare_ab_runtime import (
    DEVICE, PrepareCounter, build_model, clean, decode_results, restored_training_step,
    sample_variant, sync,
)


def compile_snapshot():
    from torch._dynamo.utils import counters
    return {str(group): {str(key): int(value) for key, value in rows.items()}
            for group, rows in counters.items()}


def compile_delta(before, after):
    return {group: {key: value - before.get(group, {}).get(key, 0)
                    for key, value in values.items()
                    if value != before.get(group, {}).get(key, 0)}
            for group, values in after.items()}


class ForwardTrace:
    """Patch only model.forward; official capability and compiled blocks survive.

    Forward timing excludes CPU copies. Batch wall time includes identical
    synchronization/CPU-trace overhead for A/B/C; it is not production latency.
    Cache references retain objects, never separate K/V tensor references.
    """

    def __init__(self, dit):
        self.dit = dit
        self.original = dit.forward
        self.rows = []
        self.predictions = []
        self.caches = {}
        self.cache_records = []
        self.prompt = None
        self.calls = 0
        self.fail_at = None
        self.expected = Counter()
        self.expected_caches = 0
        self.branches = 1
        dit.forward = self.forward

    def forward(self, *values, **kwargs):
        self.calls += 1
        mode = kwargs.get('kv_cache_mode') or 'no_cache'
        cache = kwargs.get('kv_cache')
        if cache is not None:
            self.caches[id(cache)] = cache
        if self.fail_at == self.calls:
            if mode != 'cached' or cache is None:
                raise RuntimeError('Exception probe did not traverse real cached KV')
            raise RuntimeError('intentional second-prompt cached failure')
        sync()
        started = time.perf_counter()
        output = self.original(*values, **kwargs)
        sync()
        elapsed = time.perf_counter() - started
        target_tokens = (int(self.prompt['height']) // 16) * (int(self.prompt['width']) // 16)
        trace_started = time.perf_counter()
        prediction = output[0][:, -target_tokens:].detach().cpu().clone()
        self.predictions.append(prediction)
        self.rows.append({'prompt_enum': self.prompt['enum'], 'call': self.calls,
                          'prompt_prediction': len([r for r in self.rows if r['prompt_enum'] == self.prompt['enum']]),
                          'branch': (sum(r['prompt_enum'] == self.prompt['enum'] for r in self.rows) % self.branches),
                          'is_first_branch_forward': sum(r['prompt_enum'] == self.prompt['enum'] for r in self.rows) < self.branches,
                          'mode': mode, 'forward_s': elapsed, 'target_shape': list(prediction.shape),
                          'cache_id': id(cache) if cache is not None else None,
                          'trace_copy_s': time.perf_counter() - trace_started})
        if mode == 'extract':
            layers = []
            for index, layer in enumerate(cache.layer_caches):
                tensors = [layer.k, layer.v]
                if any(t is None or t.device != DEVICE for t in tensors):
                    raise RuntimeError('Extracted KV must reside entirely on cuda:0')
                layers.append({'layer': index, 'prefix_tokens': layer.k.shape[1],
                               'k_shape': list(layer.k.shape), 'v_shape': list(layer.v.shape),
                               'bytes': sum(t.numel() * t.element_size() for t in tensors),
                               'devices': [str(t.device) for t in tensors]})
            self.cache_records.append({'prompt_enum': self.prompt['enum'], 'cache_id': id(cache),
                                       'branch': sum(r['prompt_enum'] == self.prompt['enum'] for r in self.cache_records),
                                       'layers': layers, 'bytes': sum(r['bytes'] for r in layers)})
        return output

    def wrap(self, function):
        @wraps(function)
        def sample(dit, prompt, *values, **kwargs):
            self.prompt = prompt
            self.branches = 1 + int(float(prompt.get('guidance_scale', prompt.get('scale', 1))) > 1
                                    and prompt.get('negative_prompt') is not None and values[1] is not None)
            self.expected_caches += self.branches
            self.expected['extract'] += self.branches
            self.expected['cached'] += self.branches * (int(prompt['sample_steps']) - 1)
            return function(dit, prompt, *values, **kwargs)
        return sample

    def report(self, output, label, recorder, variant):
        cleared = all(layer.k is None and layer.v is None
                      for cache in self.caches.values() for layer in cache.layer_caches)
        if not cleared:
            raise RuntimeError('Sampler retained populated KV after return/exception')
        counts = dict(Counter(r['mode'] for r in self.rows))
        if variant == 'C' and not self.caches:
            raise RuntimeError('C silently fell back to uncached sampling')
        if variant != 'C' and self.caches:
            raise RuntimeError('Frozen A/B unexpectedly created KV')
        if self.fail_at is None:
            expected = +self.expected if variant == 'C' else Counter(no_cache=sum(self.expected.values()))
            if Counter(counts) != expected:
                raise RuntimeError(f'{label}: forward mode counts {counts} differ from {dict(expected)}')
            if variant == 'C' and (len(self.caches) != self.expected_caches
                                   or len(self.cache_records) != self.expected_caches):
                raise RuntimeError('KV cache identity reused across prompt/CFG branches')
        if self.fail_at is not None and (self.calls != 4 or counts != {'extract': 2, 'cached': 1}
                                         or len(self.caches) != 2):
            raise RuntimeError('Exception probe did not extract two distinct prompt caches before failing')
        first = [r['forward_s'] for r in self.rows if r['is_first_branch_forward']]
        later = [r['forward_s'] for r in self.rows if not r['is_first_branch_forward']]
        torch.save({'predictions': self.predictions, 'rows': self.rows}, output / f'{label}_predictions.pt')
        recorder.event('forward_trace', label=label, counts=counts, forwards=self.rows,
                       first_forward_mean_s=sum(first) / len(first) if first else None,
                       subsequent_mean_s=sum(later) / len(later) if later else None,
                       cache_records=self.cache_records, cache_count=len(self.caches), caches_cleared=cleared,
                       trace_copy_s=sum(r['trace_copy_s'] for r in self.rows))
        self.caches.clear()
        self.predictions.clear()

    def close(self):
        self.dit.forward = self.original


def measured_sample(dit, network, function, prompts, conditions, steps, label, variant,
                    args, recorder, counter, fail_at=None):
    trace = ForwardTrace(dit)
    trace.fail_at = fail_at
    before = compile_snapshot()
    try:
        return sample_variant(dit, network, trace.wrap(function), prompts, conditions,
                              steps, label, args, recorder, counter)
    finally:
        trace.close()
        trace.report(args.output, label, recorder, variant)
        after = compile_snapshot()
        recorder.event('compile_counters', label=label, phase=('warmup' if label.startswith('warmup') else
                              'formal' if label.startswith('actual_') else 'probe'),
                       before=before, after=after, delta=compile_delta(before, after))


def tensor_metrics(a, b):
    same_shape = a.shape == b.shape
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    row = {'finite': finite, 'same_shape': same_shape, 'left_shape': list(a.shape),
           'right_shape': list(b.shape), 'torch_equal': torch.equal(a, b)}
    if same_shape and finite:
        difference = a.float() - b.float()
        row.update(max_abs=difference.abs().max().item(),
                   relative_l2=(difference.norm() / a.float().norm().clamp_min(1e-12)).item())
    else:
        row.update(max_abs=None, relative_l2=None)
    row['provisional_gate_pass'] = finite and same_shape and row['max_abs'] <= 0.0625 and row['relative_l2'] <= 1e-3
    return row


def compare_results(left, right, output, recorder, label, exact=False):
    rows = []
    for a, b in zip(left, right, strict=True):
        first = torch.load(output / a['path'], map_location='cpu', weights_only=True)['latents']
        second = torch.load(output / b['path'], map_location='cpu', weights_only=True)['latents']
        rows.append({'left': a['path'], 'right': b['path'], **tensor_metrics(first, second)})
    recorder.event('latent_comparison', label=label, images=rows)
    if exact and any(not r['torch_equal'] or not r['finite'] or not r['same_shape'] for r in rows):
        raise RuntimeError(f'{label}: exact fixture/repeat guard failed')
    return all(r['provisional_gate_pass'] for r in rows)


def compare_predictions(left, right, output, recorder, exact=False):
    a = torch.load(output / f'{left}_predictions.pt', map_location='cpu', weights_only=True)
    b = torch.load(output / f'{right}_predictions.pt', map_location='cpu', weights_only=True)
    rows = [{'index': index, 'prompt_enum': x['prompt_enum'], 'mode_left': x['mode'],
             'mode_right': y['mode'], **tensor_metrics(p, q)}
            for index, (p, q, x, y) in enumerate(zip(a['predictions'], b['predictions'], a['rows'], b['rows'], strict=True))]
    recorder.event('prediction_comparison', left=left, right=right,
                   trajectory='independently_evolving_sampler_trajectories', predictions=rows)
    if exact and any(not r['torch_equal'] or not r['finite'] for r in rows):
        raise RuntimeError(f'{left} vs {right}: prediction exact guard failed')
    return all(r['provisional_gate_pass'] for r in rows)


def exception_probe(dit, network, functions, prompts, conditions, args, recorder, counter):
    label = 'exception_probe_C'
    try:
        measured_sample(dit, network, functions['C'], prompts[:2], conditions, 2,
                        label, 'C', args, recorder, counter, fail_at=4)
    except RuntimeError as error:
        if str(error) != 'intentional second-prompt cached failure':
            raise
    else:
        raise RuntimeError('Expected actual KV exception did not occur')
    if dit.offloader.forward_only or not dit.training or not network.training or dit.offloader.futures:
        raise RuntimeError('Exception failed to drain transfers and restore training')
    recorder.event('exception_restore_verified', label=label, fail_at_forward=4,
                   cache_path='second_prompt_second_prediction_cached', pending_transfers=0,
                   dit_training=dit.training, network_training=network.training,
                   prepare_calls=dict(counter.counts))


def pixel_metrics(results, output, recorder):
    from PIL import Image, ImageDraw
    import numpy as np
    reference = next(label for label in results if label.endswith('_A'))
    labels = list(results)
    rows = []
    for index in range(2):
        images = [Image.open((output / results[label][index]['path']).with_suffix('.png')).convert('RGB') for label in labels]
        a = np.asarray(images[labels.index(reference)], dtype=np.float64)
        for label, image in zip(labels, images, strict=True):
            b = np.asarray(image, dtype=np.float64)
            difference = np.abs(a - b)
            rows.append({'prompt_index': index, 'left': reference, 'right': label,
                         'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()),
                         'rmse': float(np.sqrt(np.mean(difference ** 2))),
                         'differing_fraction': float(np.mean(np.any(a != b, axis=-1)))})
        canvas = Image.new('RGB', (sum(i.width for i in images), max(i.height for i in images) + 28), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, image in zip(labels, images, strict=True):
            canvas.paste(image, (x, 28))
            draw.text((x + 8, 8), label, fill='black')
            x += image.width
        canvas.save(output / f'abc_comparison_image{index}.png')
        for image in images:
            image.close()
    recorder.event('pixel_comparison', units='8bit_RGB', images=rows)


def load_conditions(args, recorder):
    data = torch.load(args.conditions, map_location='cpu', weights_only=True)
    prompts, conditions = data['prompts'], data['conditions']
    if len(prompts) != 3 or [p['enum'] for p in prompts] != [0, 1, 2] or prompts[2]['guidance_scale'] != 3:
        raise ValueError('Frozen conditions require original two prompts and enum2 CFG3')
    if any(p.get('guidance_scale', 1) != 1 for p in prompts[:2]):
        raise ValueError('Exception probe requires original CFG1 prompts')
    (args.output / 'prompts.json').write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    from scripts.qwen_image_2_1.sampling_prepare_ab import digest
    import shutil
    reference_hashes = {}
    for index, source in enumerate(dict.fromkeys(p['reference_image'] for p in prompts)):
        path = Path(source)
        target = args.output / 'snapshot' / f'reference_{index}.png'
        shutil.copy2(path, target)
        reference_hashes[source] = {'sha256': digest(path), 'snapshot': str(target.relative_to(args.output)),
                                    'snapshot_sha256': digest(target)}
    recorder.summary['reference_manifest'] = reference_hashes
    recorder.event('frozen_conditions_loaded', path=str(args.conditions), prompts=prompts,
                   condition_keys=list(conditions), map_location='cpu', reference_hashes=reference_hashes)
    return prompts, conditions


def compare_batch(results, args, recorder):
    baseline = next(label for label in results if label.endswith('_A'))
    c_labels = [label for label in results if label.endswith('_C')]
    accepted = True
    for label, rows in results.items():
        exact = not label.endswith('_C')
        passed = compare_results(results[baseline], rows, args.output, recorder,
                                 f'{baseline}_vs_{label}', exact=exact)
        predictions_passed = compare_predictions(baseline, label, args.output, recorder, exact=exact)
        accepted = passed and predictions_passed and accepted
    for label in c_labels[1:]:
        compare_results(results[c_labels[0]], results[label], args.output, recorder, 'C_repeat_stability', exact=True)
        compare_predictions(c_labels[0], label, args.output, recorder, exact=True)
    return accepted


def run(args, config, recorder, functions):
    from library.models.qwen_image_2_1.sampling import generate_preview
    args.runtime_seed = int(config.get('seed', 0) or 0)
    def cached_sample(*values, **kwargs):
        return generate_preview(*values, **kwargs, use_kv_cache=True)
    functions['C'] = cached_sample
    prompts, conditions = load_conditions(args, recorder)
    recorder.event('measurement_contract', prediction_copy='CPU BF16 after synchronized forward',
                   batch_wall_includes='setup+sample+restore, tracing copies/sync, old recorder/latent saves',
                   forward_wall_excludes='CPU trace copies; cache extraction stays inside first forward')
    dit, network, base_parameters = build_model(config, args, recorder)
    if len(dit.transformer_blocks) != 32:
        raise RuntimeError('Fixture requires all 32 compiled Qwen blocks')
    counter = PrepareCounter(dit)
    for variant in 'ABC':
        measured_sample(dit, network, functions[variant], prompts[:2], conditions, args.warmup_steps,
                        f'warmup_{variant}', variant, args, recorder, counter)
    cfg = {}
    for variant in 'ABC':
        label = f'cfg3_short_{variant}'
        cfg[label] = measured_sample(dit, network, functions[variant], prompts[2:], conditions, 2,
                                     label, variant, args, recorder, counter)
    accepted = compare_batch(cfg, args, recorder)
    results = {}
    for index, variant in enumerate(args.order):
        label = f'actual_{index}_{variant}'
        results[label] = measured_sample(dit, network, functions[variant], prompts[:2], conditions,
                                         args.steps, label, variant, args, recorder, counter)
    accepted = compare_batch(results, args, recorder) and accepted
    exception_probe(dit, network, functions, prompts, conditions, args, recorder, counter)
    before = compile_snapshot()
    restored_training_step(dit, network, base_parameters, prompts, conditions, args, recorder, counter)
    after = compile_snapshot()
    recorder.event('compile_counters', label='restored_training', phase='restored_training',
                   before=before, after=after, delta=compile_delta(before, after))
    dit.offloader.flush_profile_events(blocking=True)
    del dit, network, base_parameters, counter
    clean()
    if not args.skip_decode:
        decode_results(config, args, recorder, results)
        pixel_metrics(results, args.output, recorder)
    recorder.summary['provisional_numerical_gate_pass'] = accepted
    return accepted
