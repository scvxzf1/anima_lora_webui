"""Opt-in CMP 90HX real-weight KV oracle: compiled, then same-model eager.

Run only after the primary ABC experiment has released the GPU. This does not
train, decode, change weights, or widen the provisional cached/uncached gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qwen_image_2_1.sampling_prepare_ab import Recorder, digest, gpu_preflight, load_inputs

GPU_UUID = 'GPU-806debd9-d254-da57-d515-ccae15b4ac6a'
FIXTURE = ROOT / 'output/runs/qwen-sampling-prepare-ab-20261002-r4'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'output'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--conditions', type=Path, default=FIXTURE / 'conditions.pt')
    result.add_argument('--baseline-sampling', type=Path, default=FIXTURE / 'baseline_sampling.py')
    result.add_argument('--gpu-uuid', default=GPU_UUID)
    result.add_argument('--min-free-mib', type=int, default=9000)
    return result


def archive_inputs(args, recorder):
    original = json.loads((FIXTURE / 'summary.json').read_text())
    if (args.config.resolve() != (ROOT / original['cli']['config']).resolve()
            or digest(args.config) != original['config_sha256']):
        raise ValueError('Config path/hash differs from frozen r4 fixture')
    checkpoint = original['checkpoint']
    if (str(args.checkpoint.resolve()) != checkpoint['path']
            or digest(args.checkpoint) != checkpoint['sha256']):
        raise ValueError('Checkpoint path/hash differs from frozen r4 fixture')
    for name, expected in {'conditions': FIXTURE / 'conditions.pt',
                           'baseline_sampling': FIXTURE / 'baseline_sampling.py'}.items():
        if getattr(args, name).resolve() != expected.resolve():
            raise ValueError(f'{name} must use the frozen r4 fixture path')
    if digest(args.baseline_sampling) not in original['source_sha256'].values():
        raise ValueError('Frozen A source is absent from r4 source hashes')
    folder = args.output / 'snapshot'
    folder.mkdir(exist_ok=False)
    sources = {Path(path) for path in recorder.summary['source_sha256']}
    sources.update({Path(__file__), Path(__file__).with_name('kv_numerics.py'),
                    args.conditions, args.config, FIXTURE / 'summary.json'})
    import diffusers.models.transformers.transformer_qwenimage21 as official
    import diffusers.schedulers.scheduling_flow_match_euler_discrete as scheduler_source
    sources.add(Path(official.__file__))
    sources.add(Path(scheduler_source.__file__))
    manifest = {}
    for index, source in enumerate(sorted(sources)):
        destination = folder / f'{index:03d}_{source.name}'
        before = digest(source)
        shutil.copy2(source, destination)
        after = digest(destination)
        if before != after or digest(source) != before:
            raise RuntimeError(f'Source changed while snapshotting: {source}')
        manifest[str(source.resolve())] = {'sha256': before, 'snapshot': destination.name,
                                         'snapshot_sha256': after}
    recorder.summary['frozen_manifest'] = manifest
    recorder.summary['numerical_contract'] = {
        'helper_native': 'exact output and every layer K/V; finite required',
        'uncached': 'diagnostic only; provisional max_abs=0.0625 AND relative_l2=1e-3 unchanged',
        'steps': 2, 'prompt_enum': 2, 'cfg': 3, 'width': 384, 'height': 512, 'seed': 42,
        'second_latent': 'common compiled uncached first CFG prediction, official scheduler',
        'compiled': 'all 32 Inductor inner forwards installed; no comparison-time disable',
        'eager': 'same model/weights; unwrap only _torchdynamo_orig_callable after compiled oracle',
    }
    recorder.event('inputs_archived', manifest=manifest)


def tensor_layout(value):
    return {'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': str(value.dtype),
            'device': str(value.device), 'contiguous': value.is_contiguous()}


def tensor_metrics(left, right):
    import torch
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    same_shape = left.shape == right.shape
    result = {'exact': torch.equal(left, right), 'finite': finite, 'same_shape': same_shape,
              'left_layout': tensor_layout(left), 'right_layout': tensor_layout(right)}
    if finite and same_shape:
        delta = left.float() - right.float()
        result.update(max_abs=delta.abs().max().item(),
                      relative_l2=(delta.norm() / right.float().norm().clamp_min(1e-12)).item())
    else:
        result.update(max_abs=None, relative_l2=None)
    result['provisional_gate_pass'] = bool(finite and same_shape and result['max_abs'] <= 0.0625
                                          and result['relative_l2'] <= 1e-3)
    return result


def official_layout(latent, text, mask, sigma, references, slots):
    """Independent official input construction; never invokes family/helper."""
    import torch
    refs = (references,) if isinstance(references, torch.Tensor) else tuple(references or ())
    batch, _, height, width = latent.shape
    target = latent.flatten(2).transpose(1, 2)
    packed = torch.cat([*(r.flatten(2).transpose(1, 2) for r in refs), target], dim=1) if refs else target
    shapes = [[(1, r.shape[-2], r.shape[-1]) for r in (*refs, latent)] for _ in range(batch)]
    image_slots = slots.bool() if refs else torch.zeros_like(mask, dtype=torch.bool)
    image_mask = torch.cat([image_slots, torch.ones(batch, height * width // 4,
                                                   device=mask.device, dtype=torch.bool)], dim=1)
    return dict(hidden_states=packed, encoder_hidden_states=text, encoder_hidden_states_mask=mask.bool(),
                timestep=sigma.to(latent.dtype), img_shapes=shapes, img_mask=image_mask, return_dict=False)


class NativeKV:
    """Independent official extract/cached caller, one explicit branch cache."""

    def __init__(self, model):
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
        self.model = model
        self.cache = QwenImage21KVCache(len(model.transformer_blocks))
        self.extracted = False

    def __call__(self, latent, text, mask, sigma, *, reference_latents=None, image_slot_mask=None):
        values = official_layout(latent, text, mask, sigma, reference_latents, image_slot_mask)
        output = self.model(**values, kv_cache=self.cache,
                            kv_cache_mode='cached' if self.extracted else 'extract')[0]
        self.extracted = True
        count = latent.shape[-2] * latent.shape[-1]
        return output[:, -count:].transpose(1, 2).reshape_as(latent)

    def close(self):
        for layer in self.cache.layer_caches:
            layer.k = layer.v = None
        self.extracted = False


def cache_comparison(helper, native):
    import torch
    layers = []
    for index, (left, right) in enumerate(zip(helper.layer_caches, native.layer_caches, strict=True)):
        if any(value is None for value in (left.k, left.v, right.k, right.v)):
            raise RuntimeError('All 32 layers must have populated K/V')
        k, v = tensor_metrics(left.k, right.k), tensor_metrics(left.v, right.v)
        layers.append({'layer': index, 'k': k, 'v': v,
                       'helper_bytes': sum(t.numel() * t.element_size() for t in (left.k, left.v)),
                       'native_bytes': sum(t.numel() * t.element_size() for t in (right.k, right.v)),
                       'all_cuda0': all(t.device == torch.device('cuda:0')
                                        for t in (left.k, left.v, right.k, right.v))})
    return {'layers': layers, 'helper_bytes': sum(r['helper_bytes'] for r in layers),
            'native_bytes': sum(r['native_bytes'] for r in layers),
            'all_exact': all(r['k']['exact'] and r['v']['exact'] and r['k']['finite']
                             and r['v']['finite'] and r['all_cuda0'] for r in layers)}


def load_fixture(args, recorder):
    import torch
    from library.models.qwen_image_2_1.preview_conditions import condition_key, reference_key
    data = torch.load(args.conditions, map_location='cpu', weights_only=True)
    if len(data['prompts']) != 3 or [p['enum'] for p in data['prompts']] != [0, 1, 2]:
        raise ValueError('Frozen fixture requires prompts enum 0/1/2')
    prompt = data['prompts'][2]
    required = {'enum': 2, 'width': 384, 'height': 512, 'seed': 42,
                'sample_steps': 2, 'guidance_scale': 3.0, 'negative_prompt': ''}
    if any(prompt.get(key) != value for key, value in required.items()):
        raise ValueError('Frozen enum2 CFG3 prompt changed')
    conditions = data['conditions']
    branches = [tuple(t.to('cuda:0', dtype=torch.bfloat16 if i == 0 else torch.bool)
                      for i, t in enumerate(conditions[condition_key(prompt, negative=negative)]))
                for negative in (False, True)]
    raw_refs = conditions[reference_key(prompt)]
    refs = (raw_refs,) if isinstance(raw_refs, torch.Tensor) else tuple(raw_refs)
    refs = tuple(t.to('cuda:0', dtype=torch.bfloat16) for t in refs)
    recorder.event('conditions_loaded', prompt=prompt, conditions_sha256=digest(args.conditions),
                   branches=[list(map(tensor_layout, values)) for values in branches],
                   references=list(map(tensor_layout, refs)))
    return prompt, branches, refs


def scheduler_fixture(recorder):
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    generator = torch.Generator(device='cuda:0').manual_seed(42)
    latent = torch.randn((1, 64, 32, 24), generator=generator, device='cuda:0', dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    mu = 0.5 + (32 * 24 - 256) * (1.15 - 0.5) / (4096 - 256)
    input_sigmas = np.linspace(1.0, 1 / 2, 2)
    scheduler.set_timesteps(2, device='cuda:0', sigmas=input_sigmas, mu=mu)
    sigmas = [(t / 1000).expand(1) for t in scheduler.timesteps]
    recorder.event('scheduler_fixture', mu=mu, input_sigmas=input_sigmas.tolist(),
                   timesteps=scheduler.timesteps.cpu().tolist(), sigmas=scheduler.sigmas.cpu().tolist(),
                   latent=tensor_layout(latent), sigma_layouts=list(map(tensor_layout, sigmas)))
    return latent, sigmas, scheduler


def prefix_invariance(native, latent, sigma, values, refs, recorder, mode):
    """Re-extract the positive prefix at a new target/sigma, without a fifth cache."""
    import torch
    initial = [(layer.k.detach().cpu().clone(), layer.v.detach().cpu().clone())
               for layer in native.cache.layer_caches]
    native.close()
    changed = -latent
    changed_sigma = torch.full_like(sigma, 0.37)
    native(changed, values[0], values[1], changed_sigma,
           reference_latents=refs, image_slot_mask=values[2])
    rows = [{'layer': index, 'k': tensor_metrics(layer.k.cpu(), k),
             'v': tensor_metrics(layer.v.cpu(), v)}
            for index, (layer, (k, v)) in enumerate(zip(native.cache.layer_caches, initial, strict=True))]
    recorder.event('prefix_invariance', mode=mode, changed_target=tensor_layout(changed),
                   changed_sigma=changed_sigma.cpu().tolist(), layers=rows,
                   all_exact=all(row['k']['exact'] and row['v']['exact'] for row in rows), gate='diagnostic_only')


def oracle_batch(dit, network, branches, refs, latent, sigmas, scheduler, mode, args, recorder, second=None):
    import torch
    from library.models.qwen_image_2_1.preview_kv import PreviewKVForward, supports_preview_kv
    from library.models.qwen_image_2_1.family import forward_for_loss
    helpers = [PreviewKVForward(dit) for _ in branches]
    natives = [NativeKV(dit) for _ in branches]
    caches = [h.cache for h in helpers] + [n.cache for n in natives]
    predictions, passed = {}, True
    dit.eval()
    network.eval()
    try:
        dit.switch_block_swap_for_inference()
        if not supports_preview_kv(dit, forward_for_loss) or len({id(c) for c in caches}) != 4:
            raise RuntimeError('Expected supported helper and four isolated CFG caches')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for step, sigma in enumerate(sigmas):
                current = latent if step == 0 else second
                uncached = []
                for branch, values in zip(('positive', 'negative'), branches, strict=True):
                    index = int(branch == 'negative')
                    kwargs = {'reference_latents': refs, 'image_slot_mask': values[2]}
                    h = helpers[index](current, values[0], values[1], sigma, **kwargs)
                    n = natives[index](current, values[0], values[1], sigma, **kwargs)
                    u = forward_for_loss(dit, current, values[0], values[1], sigma, **kwargs)
                    torch.cuda.synchronize()
                    exact = tensor_metrics(h, n)
                    prefix = cache_comparison(helpers[index].cache, natives[index].cache)
                    passed = passed and exact['exact'] and exact['finite'] and prefix['all_exact']
                    packed = official_layout(current, values[0], values[1], sigma, refs, values[2])
                    recorder.event('branch_comparison', mode=mode, step=step, branch=branch,
                                   cache_mode='extract' if step == 0 else 'cached',
                                   helper_native=exact, helper_uncached=tensor_metrics(h, u),
                                   native_uncached=tensor_metrics(n, u), prefix=prefix,
                                   inputs={key: tensor_layout(value) for key, value in packed.items()
                                           if isinstance(value, torch.Tensor)},
                                   latent=tensor_layout(current), sigma=tensor_layout(sigma))
                    predictions[f'{step}_{branch}'] = {key: value.detach().cpu().clone()
                                                        for key, value in {'helper': h, 'native': n, 'uncached': u}.items()}
                    uncached.append(u)
                if step == 0 and second is None:
                    cfg = uncached[1] + 3 * (uncached[0] - uncached[1])
                    second = scheduler.step(cfg, scheduler.timesteps[0], latent, return_dict=False)[0].to(torch.bfloat16)
                    recorder.event('common_second_latent', mode=mode, layout=tensor_layout(second),
                                   finite=bool(torch.isfinite(second).all()))
            total = sum(t.numel() * t.element_size() for c in caches for layer in c.layer_caches
                        for t in (layer.k, layer.v))
            recorder.event('cache_footprint', mode=mode, caches=4, total_bytes=total, total_gib=total / 2**30)
            prefix_invariance(natives[0], second, sigmas[1], branches[0], refs, recorder, mode)
        torch.save(predictions, args.output / f'{mode}_predictions.pt')
    finally:
        for helper in helpers:
            helper.close()
        for native in natives:
            native.close()
        cleared = all(layer.k is None and layer.v is None for c in caches for layer in c.layer_caches)
        dit.train()
        network.train()
        dit.switch_block_swap_for_training()
        torch.cuda.synchronize()
        restored = not dit.offloader.forward_only and not dit.offloader.futures and dit.training and network.training
        recorder.event('lifetime_restored', mode=mode, caches_cleared=cleared, restored=restored,
                       pending_transfers=len(dit.offloader.futures))
        if not cleared or not restored:
            raise RuntimeError('KV lifetime or block-swap training layout not restored')
    return second, predictions, passed


def run(args, config, recorder):
    import torch
    import diffusers
    from torch._dynamo.utils import counters
    from scripts.qwen_image_2_1.sampling_prepare_ab_runtime import build_model
    if config.get('dynamo_backend', 'inductor') != 'inductor':
        raise ValueError('Real GPU oracle requires Inductor, not a substitute compile backend')
    _, branches, refs = load_fixture(args, recorder)
    first, sigmas, scheduler = scheduler_fixture(recorder)
    dit, network, _ = build_model(config, args, recorder)
    adapter = dit._qwen_image_2_1_block_swap_adapter
    compiled = list(adapter.inner_forwards)
    if len(compiled) != 32 or any(not hasattr(f, '_torchdynamo_orig_callable') for f in compiled):
        raise RuntimeError('Expected all 32 genuinely compiled block inner forwards')
    recorder.event('runtime_contract', torch_version=torch.__version__, diffusers_version=diffusers.__version__,
                   model_id=id(dit), compiled_blocks=len(compiled), backend='inductor',
                   blocks_to_swap=24, restore_mode='slab', dtype='torch.bfloat16',
                   suppress_errors=bool(torch._dynamo.config.suppress_errors))
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError('Oracle forbids silent Dynamo compile error fallback')
    second, compiled_predictions, compiled_ok = oracle_batch(
        dit, network, branches, refs, first, sigmas, scheduler, 'compiled_inductor', args, recorder)
    recorder.event('compile_counters', mode='compiled_inductor',
                   counters={str(group): {str(key): int(value) for key, value in rows.items()}
                             for group, rows in counters.items()})
    torch.save({'first_latent': first.cpu(), 'second_latent': second.cpu(),
                'sigmas': [s.cpu() for s in sigmas]}, args.output / 'common_inputs.pt')
    recorder.summary['common_inputs_sha256'] = digest(args.output / 'common_inputs.pt')
    try:
        adapter.inner_forwards[:] = [f._torchdynamo_orig_callable for f in compiled]
        recorder.event('explicit_eager_transition', unwrapped_blocks=len(compiled), model_id=id(dit),
                       weights_changed=False, compiled_comparison_complete=True)
        _, eager_predictions, eager_ok = oracle_batch(
            dit, network, branches, refs, first, sigmas, scheduler, 'eager_same_model', args, recorder, second)
    finally:
        adapter.inner_forwards[:] = compiled
    for name in compiled_predictions:
        recorder.event('compile_eager_comparison', prediction=name,
                       paths={key: tensor_metrics(compiled_predictions[name][key], eager_predictions[name][key])
                              for key in ('helper', 'native', 'uncached')}, gate='diagnostic_only')
    recorder.summary['helper_native_exact_pass'] = compiled_ok and eager_ok
    return compiled_ok and eager_ok


def main():
    args = parser().parse_args()
    if args.gpu_uuid != GPU_UUID:
        raise ValueError('Oracle is bound to the accepted CMP 90HX UUID')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(args.output)
    try:
        config = load_inputs(args, recorder)
        archive_inputs(args, recorder)
        recorder.event('gpu_preflight', **gpu_preflight(args))
        passed = run(args, config, recorder)
        recorder.summary['status'] = 'ok' if passed else 'helper_native_mismatch'
        recorder.event('run_complete')
        if not passed:
            raise SystemExit(1)
    except BaseException as error:
        if isinstance(error, SystemExit) and recorder.summary['status'] == 'helper_native_mismatch':
            raise
        recorder.summary['status'] = 'error'
        recorder.event('error', error_type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
