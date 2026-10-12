"""CPU checks for the standalone real-training phase diagnostic."""
import importlib.util
import json
from pathlib import Path

import torch


_PATH = Path(__file__).resolve().parents[1] / 'scripts/anima/probe_mixed_nf4_memory.py'
_SPEC = importlib.util.spec_from_file_location('mixed_nf4_memory_probe', _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


class FakeCuda:
    def __init__(self):
        self.allocated = 100
        self.reserved = 200
        self.peak_allocated = 500
        self.peak_reserved = 700
        self.resets = 0
        self.syncs = 0

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def max_memory_allocated(self, device=None):
        return self.peak_allocated

    def max_memory_reserved(self, device=None):
        return self.peak_reserved

    def reset_peak_memory_stats(self, device=None):
        self.resets += 1
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def synchronize(self, device):
        self.syncs += 1

    def memory_stats(self, device):
        return {'active_bytes.all.current': self.allocated + 20,
                'inactive_split_bytes.all.current': 30}

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.reserved


def test_phase_peaks_are_independent_and_public_peaks_remain_cumulative(tmp_path):
    cuda = FakeCuda()
    tracker = probe.PhaseTracker(tmp_path / 'phases.jsonl', 2, cuda=cuda)
    tracker.peaks.install()
    tracker.boundary('before_forward', device='cuda:0', step=0)
    cuda.allocated = 180
    cuda.peak_allocated = 300
    cuda.peak_reserved = 400
    tracker.boundary('after_forward', device='cuda:0', step=0)
    tracker.boundary('before_backward', device='cuda:0', step=0)
    cuda.allocated = 120
    cuda.peak_allocated = 220
    tracker.boundary('after_backward', device='cuda:0', step=0)
    events = [json.loads(line) for line in tracker.path.read_text().splitlines()]
    forward = events[1]
    assert forward['peak_allocated_bytes'] == 300
    assert forward['peak_allocated_increment_bytes'] == 200
    assert forward['allocated_delta_bytes'] == 80
    assert forward['allocated_minus_active_bytes'] == -20
    assert forward['cumulative_peak_allocated_bytes'] == 500
    assert events[3]['peak_allocated_bytes'] == 220
    assert events[3]['peak_allocated_increment_bytes'] == 40
    assert cuda.max_memory_allocated('cuda:0') == 500
    assert cuda.max_memory_reserved(0) == 700
    tracker.boundary('before_forward', device='cuda:0', step=2)
    assert cuda.syncs == cuda.resets == 4


def test_exception_reports_partial_stage_peak(tmp_path):
    cuda = FakeCuda()
    tracker = probe.PhaseTracker(tmp_path / 'failure.jsonl', 8, cuda=cuda)
    tracker.boundary('before_backward', device='cuda', step=0)
    cuda.peak_allocated = 350
    tracker.boundary('backward_exception', device='cuda', step=0,
                     exception_type='RuntimeError')
    event = json.loads(tracker.path.read_text().splitlines()[-1])
    assert event['peak_allocated_increment_bytes'] == 250
    assert event['phase_complete'] is False
    assert event['exception_type'] == 'RuntimeError'


def test_ledger_deduplicates_shared_storage_and_nested_quant_state():
    base = torch.zeros(20, dtype=torch.bfloat16)
    ledger = probe.StorageLedger()
    ledger.add('block', 'bfloat16', base[:10])
    ledger.add('alias', 'bfloat16', base[10:])
    assert ledger.result()['unique_storage_bytes_by_device'] == {'cpu': 40}
    assert 'alias' not in ledger.result()['unique_storage_bytes_by_component']
    QuantState = type('QuantState', (), {})
    state = QuantState()
    state.absmax = torch.zeros(2)
    state.state2 = {'code': state.absmax, 'nested': [base]}
    assert len(list(probe.nested_tensors(state))) == 2


def test_component_stats_find_nf4_state_without_counting_aliases(tmp_path):
    module = torch.nn.Linear(4, 2, bias=False)
    QuantState = type('QuantState', (), {})
    module.weight.quant_state = QuantState()
    module.weight.quant_state.absmax = torch.zeros(3)
    tracker = probe.PhaseTracker(tmp_path / 'storage.jsonl', 8, cuda=FakeCuda())
    tracker.remember(unet=module, network=module)
    result = tracker.component_stats()
    assert result['unique_storage_bytes_by_device']['cpu'] == 8 * 4 + 3 * 4
    values = result['unique_storage_bytes_by_component']['unet/other']
    assert values['cpu/trainable_nf4_packed'] == 32
    assert values['cpu/nf4_quant_state'] == 12


def test_initial_state_hash_handles_scalar_alpha_and_bfloat16():
    network = torch.nn.Linear(4, 2, dtype=torch.bfloat16)
    network.register_buffer('alpha', torch.tensor(32.0))
    initial = probe.network_state_sha256(network)
    copy = torch.nn.Linear(4, 2, dtype=torch.bfloat16)
    copy.register_buffer('alpha', torch.tensor(32.0))
    copy.load_state_dict(network.state_dict())
    assert probe.network_state_sha256(copy) == initial
    copy.alpha.fill_(16.0)
    assert probe.network_state_sha256(copy) != initial
