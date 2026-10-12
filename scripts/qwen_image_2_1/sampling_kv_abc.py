"""Opt-in frozen A/B versus invocation-local KV C acceptance (CMP 90HX only)."""
from __future__ import annotations

import argparse
import importlib.util
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
B_SOURCE = ROOT / 'output/runs/qwen-sampling-kv-abc-20261002-sources/b_sampling.py'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'output'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--baseline-sampling', type=Path, default=FIXTURE / 'baseline_sampling.py')
    result.add_argument('--b-sampling', type=Path, default=B_SOURCE)
    result.add_argument('--conditions', type=Path, default=FIXTURE / 'conditions.pt')
    result.add_argument('--gpu-uuid', default=GPU_UUID)
    result.add_argument('--steps', type=int, default=28)
    result.add_argument('--warmup-steps', type=int, default=2)
    result.add_argument('--order', nargs='+', choices=['A', 'B', 'C'], default=list('ABCCBA'))
    result.add_argument('--min-free-mib', type=int, default=9000)
    result.add_argument('--skip-decode', action='store_true')
    return result


def import_sampler(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_preview


def archive_inputs(args, recorder):
    original = json.loads((FIXTURE / 'summary.json').read_text())
    expected_config = (ROOT / original['cli']['config']).resolve()
    if args.config.resolve() != expected_config or digest(args.config) != original['config_sha256']:
        raise ValueError('Config path/hash differs from frozen r4 fixture')
    expected_checkpoint = original['checkpoint']
    if (str(args.checkpoint.resolve()) != expected_checkpoint['path']
            or digest(args.checkpoint) != expected_checkpoint['sha256']):
        raise ValueError('Checkpoint path/hash differs from frozen r4 fixture')
    required = {'baseline_sampling': FIXTURE / 'baseline_sampling.py',
                'b_sampling': B_SOURCE, 'conditions': FIXTURE / 'conditions.pt'}
    for name, expected in required.items():
        actual = getattr(args, name)
        if actual.resolve() != expected.resolve():
            raise ValueError(f'{name} must use the frozen fixture path')
    source_hashes = original['source_sha256']
    if digest(args.baseline_sampling) not in source_hashes.values():
        raise ValueError('Frozen A source is absent from r4 source hashes')
    sampling_key = str(ROOT / 'library/models/qwen_image_2_1/sampling.py')
    if digest(args.b_sampling) != source_hashes[sampling_key]:
        raise ValueError('Frozen B source differs from accepted r4 production sampler')
    folder = args.output / 'snapshot'
    folder.mkdir(exist_ok=False)
    files = {'a_sampling.py': args.baseline_sampling, 'b_sampling.py': args.b_sampling,
             'c_sampling.py': ROOT / 'library/models/qwen_image_2_1/sampling.py',
             'c_preview_kv.py': ROOT / 'library/models/qwen_image_2_1/preview_kv.py',
             'conditions.pt': args.conditions, 'config.runtime.toml': args.config,
             'fixture_summary.json': FIXTURE / 'summary.json',
             'sampling_kv_abc.py': Path(__file__),
             'sampling_kv_abc_runtime.py': Path(__file__).with_name('sampling_kv_abc_runtime.py')}
    manifest = {}
    for name, source in files.items():
        shutil.copy2(source, folder / name)
        manifest[name] = {'source': str(source.resolve()), 'sha256': digest(source),
                          'snapshot_sha256': digest(folder / name)}
    recorder.summary['frozen_manifest'] = manifest
    recorder.summary['numerical_gate'] = {'max_abs': 0.0625, 'relative_l2': 1e-3,
                                         'provisional': True, 'both_required': True}
    recorder.event('frozen_inputs_archived', manifest=manifest)


def main():
    args = parser().parse_args()
    if args.gpu_uuid != GPU_UUID or min(args.steps, args.warmup_steps) < 1 or not set('ABC') <= set(args.order):
        raise ValueError('Bound CMP90HX UUID, positive steps and all A/B/C required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(args.output)
    try:
        archive_inputs(args, recorder)
        recorder.event('gpu_preflight', **gpu_preflight(args))
        config = load_inputs(args, recorder)
        from scripts.qwen_image_2_1.sampling_kv_abc_runtime import run
        passed = run(args, config, recorder, {
            'A': import_sampler(args.baseline_sampling, 'qwen_sampling_prepare_original'),
            'B': import_sampler(args.b_sampling, 'qwen_sampling_kv_frozen_b')})
        recorder.summary['status'] = 'ok' if passed else 'numerical_review_required'
        recorder.event('run_complete')
    except BaseException as error:
        recorder.summary['status'] = 'error'
        recorder.event('error', error_type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
