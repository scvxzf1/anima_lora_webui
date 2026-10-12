"""CPU-only checks of independent ABC measurement and acceptance contracts."""
from types import SimpleNamespace

import pytest
import torch

from scripts.qwen_image_2_1 import sampling_kv_abc as cli
from scripts.qwen_image_2_1 import sampling_kv_abc_runtime as runtime


def test_gate_requires_both_thresholds_and_shape_finiteness():
    assert runtime.tensor_metrics(torch.ones(8), torch.ones(8))['provisional_gate_pass']
    # Pass absolute tolerance, fail relative tolerance. Threshold is not widened.
    result = runtime.tensor_metrics(torch.ones(8), torch.full((8,), 1.01))
    assert result['max_abs'] < 0.0625
    assert result['relative_l2'] > 1e-3
    assert not result['provisional_gate_pass']
    assert not runtime.tensor_metrics(torch.ones(8), torch.ones(7))['provisional_gate_pass']
    assert not runtime.tensor_metrics(torch.ones(8), torch.full((8,), float('nan')))['provisional_gate_pass']


def test_cli_default_contract():
    args = cli.parser().parse_args(['--config', 'a', '--checkpoint', 'b', '--output', 'c'])
    assert args.order == list('ABCCBA')
    assert (args.steps, args.warmup_steps) == (28, 2)
    assert args.gpu_uuid == 'GPU-806debd9-d254-da57-d515-ccae15b4ac6a'
    assert args.conditions == cli.FIXTURE / 'conditions.pt'


def test_trace_preserves_original_a_module_and_restores_forward(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, 'sync', lambda: None)
    original = lambda **kwargs: (torch.ones(1, 6, 64),)
    dit = SimpleNamespace(forward=original)
    trace = runtime.ForwardTrace(dit)

    def sample(dit, prompt, positive, negative, reference, **kwargs):
        dit.forward(hidden_states=positive)
        dit.forward(hidden_states=positive)
    sample.__module__ = 'qwen_sampling_prepare_original'
    wrapped = trace.wrap(sample)
    assert wrapped.__module__ == sample.__module__
    wrapped(dit, {'width': 16, 'height': 32, 'enum': 0, 'sample_steps': 2}, None, None, None)
    assert trace.predictions[0].shape == (1, 2, 64)
    events = []
    recorder = SimpleNamespace(event=lambda event, **row: events.append((event, row)))
    trace.report(tmp_path, 'A', recorder, 'A')
    trace.close()
    assert dit.forward is original
    assert events[0][1]['counts'] == {'no_cache': 2}
    assert events[0][1]['caches_cleared']


def test_trace_rejects_silent_c_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, 'sync', lambda: None)
    trace = runtime.ForwardTrace(SimpleNamespace(forward=lambda **kw: (torch.ones(1, 2, 64),)))
    with pytest.raises(RuntimeError, match='silently fell back'):
        trace.report(tmp_path, 'C', SimpleNamespace(event=lambda *a, **kw: None), 'C')
    trace.close()


def test_exception_probe_requires_real_cached_mode(monkeypatch):
    monkeypatch.setattr(runtime, 'sync', lambda: None)
    trace = runtime.ForwardTrace(SimpleNamespace(forward=lambda **kw: (torch.ones(1, 2, 64),)))
    trace.fail_at = 1
    with pytest.raises(RuntimeError, match='did not traverse real cached KV'):
        trace.forward()
    trace.close()


def test_compile_delta_keeps_changed_counters():
    assert runtime.compile_delta({'stats': {'unique_graphs': 2}},
                                 {'stats': {'unique_graphs': 3}, 'frames': {'total': 2}}) == {
                                     'stats': {'unique_graphs': 1}, 'frames': {'total': 2}}


def test_real_mode_trace_clears_both_prompt_caches_after_cached_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, 'sync', lambda: None)
    monkeypatch.setattr(runtime, 'DEVICE', torch.device('cpu'))

    def model_forward(**kwargs):
        cache = kwargs['kv_cache']
        if kwargs['kv_cache_mode'] == 'extract':
            cache.layer_caches[0].k = torch.ones(1, 3, 2, 4)
            cache.layer_caches[0].v = torch.ones(1, 3, 2, 4)
        return (torch.ones(1, 2, 64),)

    dit = SimpleNamespace(forward=model_forward)
    trace = runtime.ForwardTrace(dit)
    trace.fail_at = 4

    def sample(dit, prompt, positive, negative, reference):
        cache = SimpleNamespace(layer_caches=[SimpleNamespace(k=None, v=None)])
        try:
            dit.forward(kv_cache=cache, kv_cache_mode='extract')
            dit.forward(kv_cache=cache, kv_cache_mode='cached')
        finally:
            for layer in cache.layer_caches:
                layer.k = layer.v = None

    wrapped = trace.wrap(sample)
    wrapped(dit, {'width': 16, 'height': 32, 'enum': 0, 'sample_steps': 2}, None, None, None)
    with pytest.raises(RuntimeError, match='intentional second-prompt cached failure'):
        wrapped(dit, {'width': 16, 'height': 32, 'enum': 1, 'sample_steps': 2}, None, None, None)
    events = []
    trace.report(tmp_path, 'exception', SimpleNamespace(event=lambda name, **row: events.append(row)), 'C')
    trace.close()
    assert events[0]['cache_count'] == 2
    assert events[0]['caches_cleared']
    assert events[0]['counts'] == {'extract': 2, 'cached': 1}
    assert events[0]['cache_records'][0]['layers'][0]['bytes'] == 192
