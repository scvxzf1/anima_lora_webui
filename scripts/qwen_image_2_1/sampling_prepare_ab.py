"""Opt-in real-checkpoint sampling prepare A/B; never touches the source run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class Recorder:
    def __init__(self, output):
        self.output = output
        self.summary = {'status': 'running', 'events': []}

    def event(self, event_name, **values):
        row = {'event': event_name, 'wall_time': time.time(), **values}
        with (self.output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.summary['events'].append(row)
        self.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def flush(self):
        temporary = self.output / 'summary.json.tmp'
        temporary.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(self.output / 'summary.json')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--baseline-sampling', type=Path, required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--steps', type=int, default=28)
    result.add_argument('--warmup-steps', type=int, default=2)
    result.add_argument('--order', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    result.add_argument('--baseline-repeats', type=int, default=2,
                        help='Independent hot A repetitions before ordered A/B')
    result.add_argument('--min-free-mib', type=int, default=9000)
    result.add_argument('--skip-decode', action='store_true')
    return result


def gpu_preflight(args):
    if not args.gpu_uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must equal the requested GPU UUID before startup')
    command = ['nvidia-smi', '-i', args.gpu_uuid]
    row = subprocess.run(command + ['--query-gpu=uuid,name,memory.total,memory.free',
                         '--format=csv,noheader,nounits'], check=True, capture_output=True,
                         text=True).stdout.strip().split(',')
    if len(row) != 4 or row[0].strip() != args.gpu_uuid or 'CMP 90HX' not in row[1]:
        raise RuntimeError(f'Requested UUID is not CMP 90HX: {row}')
    if int(row[3]) < args.min_free_mib:
        raise RuntimeError(f'Insufficient free GPU memory: {row[3]} MiB')
    pids = subprocess.run(command + ['--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True).stdout.strip()
    if pids and pids != 'No running processes found':
        raise RuntimeError(f'Requested GPU has existing compute processes: {pids}')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, 'uuid', ''))
    if not actual_uuid.startswith('GPU-'):
        actual_uuid = 'GPU-' + actual_uuid
    if 'CMP 90HX' not in properties.name or actual_uuid != args.gpu_uuid:
        raise RuntimeError('CUDA device 0 mapping does not match queried UUID')
    return {'uuid': row[0].strip(), 'name': row[1].strip(), 'total_mib': int(row[2]),
            'cuda_total_mib': properties.total_memory / 2**20,
            'free_mib': int(row[3]), 'cuda_device': 'cuda:0', 'compute_pids_before': []}


def load_inputs(args, recorder):
    import tomllib
    config = tomllib.loads(args.config.read_text())
    expected = {'model_family': 'qwen_image_2_1', 'network_module': 'networks.lora_anima',
                'network_dim': 16, 'network_alpha': 16, 'blocks_to_swap': 24,
                'mixed_precision': 'bf16', 'base_compute': 'bf16', 'attn_mode': 'torch',
                'gradient_checkpointing': True, 'torch_compile': True,
                'block_swap_transfer_dtype': 'bf16', 'block_swap_restore_mode': 'slab'}
    for key, value in expected.items():
        if key in ('block_swap_transfer_dtype', 'block_swap_restore_mode') and getattr(args, 'allow_configured_blocks', False):
            from library.runtime.block_swap_config import (
                normalize_block_swap_transfer_dtype, normalize_block_swap_restore_mode,
            )
            normalize = (normalize_block_swap_transfer_dtype if key == 'block_swap_transfer_dtype'
                         else normalize_block_swap_restore_mode)
            config[key] = normalize(config.get(key, value))
            continue
        if key == 'blocks_to_swap' and getattr(args, 'allow_configured_blocks', False):
            configured = config.get(key, 0)
            if not isinstance(configured, int) or isinstance(configured, bool) or not 0 <= configured <= 30:
                raise ValueError(f'Configured blocks_to_swap must be an integer in 0..30, got {configured!r}')
            config[key] = configured
            continue
        if config.get(key) != value:
            raise ValueError(f'Runtime contract requires {key}={value!r}, got {config.get(key)!r}')
    for key in ('qwen_fused_projections', 'selective_checkpoint'):
        if config.get(key, 'off') != 'off':
            raise ValueError(f'{key} must remain off')
    if config.get('network_args') or config.get('cpu_offload_checkpointing', False):
        raise ValueError('Only plain LoRA/full checkpoint runtime is supported')
    if config.get('compile_block_scope', 'all') != 'all':
        raise ValueError('compile_block_scope must be all')
    for key in ('pretrained_model_name_or_path', 'qwen3', 'vae', 'sample_prompts'):
        path = Path(os.path.expandvars(config[key])).expanduser()
        config[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    if '400' not in args.checkpoint.stem:
        raise ValueError('Checkpoint filename must identify step 400')
    sources = [Path(__file__), Path(__file__).with_name('sampling_prepare_ab_runtime.py'),
               args.baseline_sampling, *sorted((ROOT / 'library/models/qwen_image_2_1').glob('*.py')),
               ROOT / 'library/training/bootstrap.py', ROOT / 'library/runtime/offloading.py',
               *sorted((ROOT / 'networks/lora_anima').glob('*.py'))]
    manifest = {str(path): digest(path) for path in sources}
    recorder.summary.update(config=config, source_sha256=manifest,
                            checkpoint={'path': str(args.checkpoint.resolve()),
                                        'sha256': digest(args.checkpoint),
                                        'bytes': args.checkpoint.stat().st_size},
                            config_sha256=digest(args.config), cli=vars(args) | {
                                key: str(value) for key, value in vars(args).items() if isinstance(value, Path)})
    recorder.event('inputs_validated', sample_prompts_sha256=digest(config['sample_prompts']))
    return config


def baseline_function(path):
    spec = importlib.util.spec_from_file_location('qwen_sampling_prepare_original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_preview


def main():
    args = parser().parse_args()
    if min(args.steps, args.warmup_steps, args.baseline_repeats) < 1 or not {'A', 'B'} <= set(args.order):
        raise ValueError('Positive steps/repeats and both A/B in order required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(args.output)
    try:
        recorder.event('gpu_preflight', **gpu_preflight(args))
        config = load_inputs(args, recorder)
        from scripts.qwen_image_2_1.sampling_prepare_ab_runtime import run
        run(args, config, recorder, baseline_function(args.baseline_sampling))
        recorder.summary['status'] = 'ok'
        recorder.event('run_complete')
    except BaseException as error:
        recorder.summary['status'] = 'error'
        recorder.event('error', error_type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
