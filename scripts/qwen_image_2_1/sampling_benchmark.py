"""Isolated 1024-square, two-prompt Qwen edit LoRA sampling benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.qwen_image_2_1.sampling_prepare_ab import Recorder, digest, gpu_preflight, load_inputs  # noqa: E402


class Telemetry:
    """Bounded UUID-specific polling; telemetry failures never hide the run error."""

    fields = ('timestamp', 'utilization.gpu', 'utilization.memory', 'memory.used',
              'power.draw', 'temperature.gpu', 'clocks.sm', 'clocks.mem')

    def __init__(self, args):
        self.args = args
        self.phase = 'preflight'
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.poll, name='gpu-telemetry', daemon=True)
        self.errors = []

    def poll(self):
        with (self.args.output / 'gpu_telemetry.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('wall_time', 'phase', *self.fields))
            while not self.stop_event.is_set():
                started = time.monotonic()
                phase = self.phase
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '-i', self.args.gpu_uuid,
                         '--query-gpu=' + ','.join(self.fields), '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True, timeout=5)
                    values = next(csv.reader([result.stdout.strip()]))
                    if len(values) != len(self.fields):
                        raise ValueError('Unexpected telemetry column count')
                    writer.writerow((time.time(), phase, *(value.strip() for value in values)))
                    stream.flush()
                except Exception as error:
                    self.errors.append({'wall_time': time.time(), 'phase': phase, 'error': str(error)})
                self.stop_event.wait(max(0, 1 - (time.monotonic() - started)))

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=7)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'output'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--steps', type=int, default=28)
    result.add_argument('--warmup-steps', type=int, default=2)
    result.add_argument('--min-free-mib', type=int, default=9000)
    return result


def snapshot_prompts(config, args, recorder):
    from library import train_util
    from library.training.preview_spec import preview_references, reference_path, read_reference
    source = Path(config['sample_prompts'])
    prompts = train_util.load_prompts(str(source))
    if len(prompts) != 2:
        raise ValueError('Exactly two original edit prompts required')
    references = []
    for index, prompt in enumerate(prompts):
        paths = preview_references(prompt)
        if not paths:
            raise ValueError(f'Prompt {index} is not an edit prompt')
        for value in paths:
            path = reference_path(value)
            image = read_reference(path)
            references.append({'prompt_index': index, 'path': str(path),
                               'sha256': digest(path), 'dimensions': list(image.size)})
            image.close()
        prompt.update(width=1024, height=1024, use_kv_cache=False)
    target = args.output / 'benchmark_prompts_1024.json'
    target.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    recorder.summary.update(prompt_source={'path': str(source), 'sha256': digest(source)},
                            references=references, benchmark_prompts_sha256=digest(target))
    recorder.event('benchmark_prompts_saved', path=str(target), prompts=prompts)
    return config | {'sample_prompts': str(target)}


def compile_counters():
    from torch._dynamo.utils import counters
    return {str(group): {str(key): int(value) for key, value in values.items()}
            for group, values in counters.items()}


def release_model(dit):
    """Break offloader hook/worker references before loading the decode VAE."""
    offloader = dit.offloader
    offloader.flush_profile_events(blocking=True)
    offloader.thread_pool.shutdown(wait=True)
    for handle in getattr(offloader, 'remove_handles', []):
        handle.remove()
    offloader.remove_handles = []
    stop_poller = getattr(offloader, '_stop_profile_poller', None)
    if stop_poller is not None:
        stop_poller()


def decode(config, args, recorder, rows):
    import torch
    from PIL import PngImagePlugin
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_vae
    from library.models.qwen_image_2_1.preview_decode import decode_latent
    from library.models.qwen_image_2_1.preview_decode_policy import decode_with_policy
    from library.training.preview_decode_config import decode_policy
    from scripts.qwen_image_2_1.sampling_prepare_ab_runtime import DEVICE, DTYPE, clean, sync
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    vae = load_qwen_image_2_1_vae(config['vae'], dtype=DTYPE, device='cpu')
    sync()
    recorder.event('decode_vae_loaded', wall_s=time.perf_counter() - started)
    try:
        for index, row in enumerate(rows):
            record = torch.load(args.output / row['path'], map_location='cpu', weights_only=True)
            sync()
            image_started = time.perf_counter()
            with torch.no_grad():
                image = decode_with_policy(vae, record['latents'], DEVICE,
                                           decode_policy(config), decode_latent)
            sync()
            decode_s = time.perf_counter() - image_started
            info = PngImagePlugin.PngInfo()
            info.add_text('qwen_preview', json.dumps(record['metadata'], ensure_ascii=False))
            path = (args.output / row['path']).with_suffix('.png')
            image.save(path, pnginfo=info)
            recorder.event('image_decoded', index=index, path=path.name, decode_s=decode_s,
                           decode_policy=decode_policy(config),
                           timing_includes_device_transfers=True,
                           dimensions=list(image.size))
            image.close()
    finally:
        recorder.event('decode_complete', wall_s=time.perf_counter() - started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(DEVICE))
        del vae
        clean()


def run(args, config, recorder, telemetry):
    from library.models.qwen_image_2_1.sampling import generate_preview
    from scripts.qwen_image_2_1.sampling_prepare_ab_runtime import (
        cache_conditions, build_model, PrepareCounter, clean, sync)
    from scripts.qwen_image_2_1.sampling_benchmark_runtime import sample_variant
    from library.models.qwen_image_2_1.preview_memory import validate_config
    validate_config(config)
    args.preview_memory_config = config
    args.runtime_seed = int(config.get('seed', 0) or 0)
    telemetry.phase = 'cache'
    started = time.perf_counter()
    prompts, conditions = cache_conditions(args, config, recorder)
    sync()
    cache_s = time.perf_counter() - started
    text_s = next(row['wall_s'] for row in reversed(recorder.summary['events'])
                  if row['event'] == 'text_conditions_cached')
    recorder.event('cache_complete', wall_s=cache_s, text_cache_s=text_s,
                   reference_cache_and_vae_load_s=cache_s - text_s,
                   formal_prompt_count=2, cached_prompt_count=len(prompts))
    telemetry.phase = 'model_load'
    dit, network, base_parameters = build_model(config, args, recorder)
    counter = PrepareCounter(dit)
    try:
        recorder.event('compile_counters', phase='before_warmup', counters=compile_counters())
        telemetry.phase = 'warmup'
        sample_variant(dit, network, generate_preview, prompts[:2], conditions,
                       args.warmup_steps, 'warmup_B', args, recorder, counter)
        recorder.event('compile_counters', phase='after_warmup', counters=compile_counters())
        telemetry.phase = 'B'
        rows = sample_variant(dit, network, generate_preview, prompts[:2], conditions,
                              args.steps, 'B', args, recorder, counter)
        recorder.event('compile_counters', phase='after_B', counters=compile_counters())
        recorder.summary['formal_results'] = rows
    finally:
        release_model(dit)
        del counter, dit, network, base_parameters
        clean()
    telemetry.phase = 'decode'
    decode(config, args, recorder, rows)


def main():
    args = parser().parse_args()
    if not 1 <= args.steps <= 1000 or not 1 <= args.warmup_steps <= 1000:
        raise ValueError('Steps must be in 1..1000')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Used only for the source manifest; the original baseline is never imported.
    args.baseline_sampling = ROOT / 'library/models/qwen_image_2_1/sampling.py'
    # This benchmark measures the selected training topology. The historical
    # A/B harness remains strict at 24; only this new benchmark accepts the
    # validated config value and records it in model_ready/summary events.
    args.allow_configured_blocks = True
    recorder = Recorder(args.output)
    telemetry = Telemetry(args)
    started = time.perf_counter()
    try:
        telemetry.thread.start()
        recorder.event('gpu_preflight', **gpu_preflight(args))
        config = load_inputs(args, recorder)
        recorder.summary['source_sha256'][str(Path(__file__))] = digest(__file__)
        for name in ('sampling_benchmark_runtime.py',):
            source = Path(__file__).with_name(name)
            recorder.summary['source_sha256'][str(source)] = digest(source)
        config = snapshot_prompts(config, args, recorder)
        run(args, config, recorder, telemetry)
        recorder.summary['status'] = 'ok'
        recorder.event('run_complete')
    except BaseException as error:
        recorder.summary['status'] = 'error'
        recorder.event('error', error_type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        raise
    finally:
        telemetry.stop()
        recorder.summary.update(total_wall_s=time.perf_counter() - started,
                                telemetry_errors=telemetry.errors,
                                telemetry_thread_stopped=not telemetry.thread.is_alive())
        recorder.flush()


if __name__ == '__main__':
    main()
