#!/usr/bin/env python3
"""Diagnostic wrapper for real training; synchronizes only at phase boundaries.

--trace-steps controls telemetry, not training duration: set max_train_steps in
an isolated input config. Peak counters are reset for independent phase peaks;
public CUDA max-memory APIs retain cumulative peaks via in-process wrappers.
This diagnostic run changes timing and is not a throughput benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import sys
import threading
import time
import weakref
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402 - the standalone entry point needs the project root


class CumulativePeaks:
    """Keep public max APIs cumulative while raw CUDA peaks measure each phase."""

    def __init__(self, cuda):
        self.cuda = cuda
        self.raw_allocated = cuda.max_memory_allocated
        self.raw_reserved = cuda.max_memory_reserved
        self.raw_reset = cuda.reset_peak_memory_stats
        self.peaks = defaultdict(lambda: [0, 0])

    def key(self, device):
        if isinstance(device, int):
            return device
        value = torch.device(device) if device is not None else torch.device('cuda')
        return value.index if value.index is not None else self.cuda.current_device()

    def capture(self, device=None):
        key = self.key(device)
        pair = self.peaks[key]
        pair[0] = max(pair[0], int(self.raw_allocated(device)))
        pair[1] = max(pair[1], int(self.raw_reserved(device)))
        return tuple(pair)

    def reset(self, device=None):
        self.capture(device)
        self.raw_reset(device)

    def install(self):
        self.cuda.max_memory_allocated = lambda device=None: self.capture(device)[0]
        self.cuda.max_memory_reserved = lambda device=None: self.capture(device)[1]
        self.cuda.reset_peak_memory_stats = self.reset


def nested_tensors(value, seen=None):
    """QuantState and optimizer containers, without traversing module graphs."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from nested_tensors(child, seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from nested_tensors(child, seen)
    elif type(value).__name__ == 'QuantState':
        yield from nested_tensors(vars(value), seen)


class StorageLedger:
    """Unique backing-storage bytes; aliases are charged to their first owner."""

    def __init__(self):
        self.seen = set()
        self.groups = defaultdict(Counter)
        self.devices = Counter()

    def add(self, group, kind, tensor):
        if tensor.device.type == 'meta':
            return
        storage = tensor.untyped_storage()
        size = int(storage.nbytes())
        if not size:
            return
        device = str(tensor.device)
        key = (device, storage.data_ptr(), size)
        if key in self.seen:
            return
        self.seen.add(key)
        self.groups[group][f'{device}/{kind}'] += size
        self.devices[device] += size

    def module(self, label, module):
        if module is None:
            return
        for name, child in module.named_modules():
            parts = name.split('.')
            block = '.'.join(parts[:2]) if parts[0] == 'blocks' and len(parts) > 1 else 'other'
            group = f'{label}/{block}'
            for param in child.parameters(recurse=False):
                state = getattr(param, 'quant_state', None)
                kind = 'nf4_packed' if state is not None else str(param.dtype).removeprefix('torch.')
                if param.requires_grad:
                    kind = f'trainable_{kind}'
                self.add(group, kind, param)
                if param.grad is not None:
                    self.add(group, 'gradient', param.grad)
                for tensor in nested_tensors(state):
                    self.add(group, 'nf4_quant_state', tensor)
            for tensor in child.buffers(recurse=False):
                self.add(group, 'buffer', tensor)

    def result(self):
        return {'unique_storage_bytes_by_component': {k: dict(v) for k, v in self.groups.items()},
                'unique_storage_bytes_by_device': dict(self.devices)}


def network_state_sha256(network):
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PhaseTracker:
    STARTS = {'before_forward': 'forward', 'before_backward': 'backward',
              'before_optimizer': 'optimizer'}
    ENDS = {'after_forward': 'forward', 'after_backward': 'backward',
            'after_optimizer': 'optimizer', 'forward_exception': 'forward',
            'backward_exception': 'backward', 'optimizer_exception': 'optimizer'}

    def __init__(self, path, max_steps, cuda=torch.cuda):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.cuda = cuda
        self.peaks = CumulativePeaks(cuda)
        self.components = {}
        self.starts = {}
        self.microstep = -1
        self.seq = 0
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.swap_counts = Counter()
        self.swap_pairs = Counter()
        self.reset_seed = None

    def write(self, event):
        self.seq += 1
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'seq': self.seq, 'elapsed_s': time.monotonic() - self.t0,
                                     **event}, ensure_ascii=False, default=str) + '\n')

    def remember(self, **components):
        for name, value in components.items():
            if value is not None:
                self.components[name] = weakref.ref(value)

    def memory(self, device):
        self.cuda.synchronize(device)
        cumulative = self.peaks.capture(device)
        stats = self.cuda.memory_stats(device)
        allocated = int(self.cuda.memory_allocated(device))
        active = int(stats.get('active_bytes.all.current', 0))
        return {'allocated_bytes': allocated,
                'reserved_bytes': int(self.cuda.memory_reserved(device)),
                'active_bytes': active,
                'allocated_minus_active_bytes': allocated - active,
                'inactive_split_bytes': int(stats.get('inactive_split_bytes.all.current', 0)),
                'peak_allocated_bytes': int(self.peaks.raw_allocated(device)),
                'peak_reserved_bytes': int(self.peaks.raw_reserved(device)),
                'cumulative_peak_allocated_bytes': cumulative[0],
                'cumulative_peak_reserved_bytes': cumulative[1]}

    def component_stats(self):
        ledger = StorageLedger()
        objects = {name: ref() for name, ref in self.components.items()}
        for name in ('unet', 'network'):
            ledger.module(name, objects.get(name))
        optimizer = objects.get('optimizer')
        if optimizer is not None:
            for tensor in nested_tensors(getattr(optimizer, 'state', {})):
                ledger.add('optimizer', 'state', tensor)
        unet = objects.get('unet')
        offloader = getattr(unet, 'offloader', None)
        if offloader is not None:
            for tensor in nested_tensors(getattr(offloader, '_swap_gpu_slab_cache', {})):
                ledger.add('offloader', 'cached_slab', tensor)
        result = ledger.result()
        with self.lock:
            result['swap_plan_query_counts'] = dict(self.swap_counts)
            result['fallback_plan_calls_by_pair'] = dict(self.swap_pairs)
        return result

    def boundary(self, label, device=None, step=None, **fields):
        phase = self.STARTS.get(label) or self.ENDS.get(label)
        if label != 'loop_ready' and (phase is None or step is None or step >= self.max_steps):
            return
        if not self.cuda.is_available() or torch.device(device or 'cuda').type != 'cuda':
            return
        if label == 'loop_ready' and self.reset_seed is not None:
            from accelerate.utils import set_seed
            set_seed(self.reset_seed)
            network = self.components.get('network', lambda: None)()
            self.write({'ev': 'controlled_initial_state', 'seed': self.reset_seed,
                        'network_sha256': network_state_sha256(network) if network is not None else None})
        if label == 'before_forward':
            self.microstep += 1
        memory = self.memory(device)
        payload = {'ev': 'phase_boundary', 'label': label, 'step': step,
                   'microstep': self.microstep, 'stage': phase, **fields, **memory}
        if label in self.STARTS:
            self.starts[phase] = memory
        elif phase in self.starts:
            start = self.starts.pop(phase)
            payload['start_allocated_bytes'] = start['allocated_bytes']
            payload['start_reserved_bytes'] = start['reserved_bytes']
            payload['allocated_delta_bytes'] = memory['allocated_bytes'] - start['allocated_bytes']
            payload['reserved_delta_bytes'] = memory['reserved_bytes'] - start['reserved_bytes']
            payload['peak_allocated_increment_bytes'] = memory['peak_allocated_bytes'] - start['allocated_bytes']
            payload['peak_reserved_increment_bytes'] = memory['peak_reserved_bytes'] - start['reserved_bytes']
            payload['phase_complete'] = not label.endswith('_exception')
        try:
            payload.update(self.component_stats())
        except Exception as exc:
            payload['component_summary_error'] = f'{type(exc).__name__}: {exc}'
        self.write(payload)
        # Reading happens before reset; the public max APIs retain this interval.
        self.peaks.reset(device)


def install_hooks(tracker):
    from library.training import loop, memory_probe
    from library.runtime.offloading import ModelOffloader

    tracker.peaks.install()
    original_snapshot = memory_probe.MemoryProbe.snapshot
    original_components = memory_probe.MemoryProbe.component_summary
    original_step = loop._probe_step
    original_plan = ModelOffloader._get_swap_plan

    def snapshot(probe, label, **fields):
        # The existing event sees cumulative peaks before the diagnostic reset.
        original_snapshot(probe, label, **fields)
        try:
            tracker.boundary(label, **fields)
        except Exception as exc:
            tracker.write({'ev': 'diagnostic_error', 'label': label,
                           'exception_type': type(exc).__name__, 'exception_message': str(exc)[:1000]})

    def components(probe, label, **fields):
        tracker.remember(**{k: fields.get(k) for k in ('unet', 'network', 'optimizer')})
        original_components(probe, label, **fields)

    def probe_step(probe, state, label, **fields):
        tracker.remember(unet=state.accelerator.unwrap_model(state.unet),
                         network=state.accelerator.unwrap_model(state.network),
                         optimizer=state.optimizer)
        original_step(probe, state, label, **fields)

    def from_args(cls, args, *, is_main_process, t0=None):
        if not is_main_process:
            return None
        path = cls.resolve_path(args) or str(tracker.path) + '.original.jsonl'
        return cls(path, max_steps=tracker.max_steps, t0=t0)

    def get_plan(offloader, *args, **kwargs):
        plan = original_plan(offloader, *args, **kwargs)
        # No CUDA synchronization or tensor access in the per-swap hook.
        with tracker.lock:
            tracker.swap_counts['plan_calls'] += 1
            tracker.swap_counts['dense_jobs'] += len(plan[0])
            tracker.swap_counts['fallback_jobs'] += len(plan[1])
            tracker.swap_counts['nf4_jobs'] += len(plan[2])
            if plan[1]:
                source = args[0] if args else kwargs.get('block_idx_to_cpu')
                target = args[2] if args else kwargs.get('block_idx_to_cuda')
                tracker.swap_pairs[f'{source}->{target}'] += 1
        return plan

    memory_probe.MemoryProbe.snapshot = snapshot
    memory_probe.MemoryProbe.component_summary = components
    memory_probe.MemoryProbe.from_args = classmethod(from_args)
    loop._probe_step = probe_step
    ModelOffloader._get_swap_plan = get_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--phase-jsonl', required=True)
    parser.add_argument('--trace-steps', type=int, default=8)
    parser.add_argument('--offloader-source', help='Isolated process only: load a saved baseline offloader source')
    parser.add_argument('--reset-seed-at-loop-ready', type=int)
    args, extra = parser.parse_known_args(argv)
    if args.trace_steps < 1:
        parser.error('--trace-steps must be positive')
    config = Path(args.config_file).resolve()
    if not config.is_file():
        parser.error(f'configuration does not exist: {config}')
    tracker = PhaseTracker(args.phase_jsonl, args.trace_steps)
    tracker.reset_seed = args.reset_seed_at_loop_ready
    if args.offloader_source:
        source = Path(args.offloader_source).resolve()
        if not source.is_file():
            parser.error(f'offloader source does not exist: {source}')
        module = types.ModuleType('library.runtime.offloading')
        module.__file__ = str(source)
        module.__package__ = 'library.runtime'
        sys.modules[module.__name__] = module
        exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
        tracker.write({'ev': 'baseline_offloader_source', 'path': str(source),
                       'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    tracker.write({'ev': 'diagnostic_start', 'config_file': str(config),
                   'trace_steps': args.trace_steps, 'training_duration_unchanged': True,
                   'phase_peak_resets': True, 'public_max_apis_cumulative': True,
                   'storage_attribution': 'first owner; unet before network; storage aliases deduplicated',
                   'timing': 'CUDA synchronization at phase boundaries only'})
    install_hooks(tracker)
    sys.argv = [str(ROOT / 'train.py'), '--config_file', str(config), *extra]
    try:
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    except BaseException as exc:
        tracker.write({'ev': 'diagnostic_exit', 'status': 'exception',
                       'exception_type': type(exc).__name__, 'exception_message': str(exc)[:1000]})
        raise
    else:
        tracker.write({'ev': 'diagnostic_exit', 'status': 'ok'})


if __name__ == '__main__':
    main()
