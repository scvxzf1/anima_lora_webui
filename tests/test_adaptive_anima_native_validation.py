"""CPU checks for the narrow native replay boundary and evidence contract."""
import copy
from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime import anima_native_validation as native
from bench.adaptive_runtime import anima_native_acceptance as acceptance
from bench.adaptive_runtime import anima_validation_updates as updates
from bench.adaptive_runtime.replay import contract


def test_boundary_preserves_canonical_leaf_gradient_domain():
    noisy = torch.tensor([1.0001, 2.0002], dtype=torch.float32).reshape(1, 1, 1, 1, 2).requires_grad_()
    original = noisy.detach().clone()
    tensors = native.boundary_tensors(noisy, torch.tensor([0.2]), torch.ones(1, 3, 4))
    assert all(t.dtype == torch.float16 for t in tensors.values())
    assert tensors['padding_mask'].shape == (1, 1, 1, 2)
    tensors['noisy'].float().square().sum().backward()
    assert noisy.is_leaf and noisy.grad.dtype == torch.float32
    assert torch.equal(noisy.detach(), original)
    assert torch.equal(noisy.grad, 2 * original.half().float())
    assert native.POLICY['input_gradient_domain'] == native.DOMAIN


def test_boundary_rejects_half_leaf():
    with pytest.raises(ValueError, match='Canonical FP32'):
        native.boundary_tensors(torch.ones(1, 1, 1, 1, 1, dtype=torch.float16, requires_grad=True),
                                torch.tensor([0.5]), torch.ones(1, 1, 1))


def fixed_args():
    return SimpleNamespace(precision=native.PRECISION, swap=20, loss_scale=1024, steps=6,
                           validation_states='state', replay_reference='replay', record_replay=False,
                           fp32_pattern=[], fp16_only_pattern=[])


@pytest.mark.parametrize('key,value', [('precision', 'fp16-islands'), ('swap', 19),
    ('loss_scale', 2048), ('steps', 5), ('record_replay', True), ('fp16_only_pattern', ['*'])])
def test_fixed_candidate_cannot_expand(key, value):
    args = fixed_args()
    setattr(args, key, value)
    with pytest.raises(ValueError, match='single frozen'):
        native.validate_native_args(args, lambda safe: None)


def test_common_validator_receives_copy_with_islands_label_only():
    args = fixed_args()
    observed = []
    native.validate_native_args(args, lambda safe: observed.append(safe))
    assert args.precision == native.PRECISION
    assert observed[0] is not args
    assert observed[0].precision == 'fp16-islands'
    assert observed[0].loss_scale == 1024


def complete_result():
    execution = {'role': 'representative', 'plan_sha256': 'a' * 64}
    steps = [{'step': i, 'case_id': updates.case_id(label, sigma), 'state_label': label,
              'effective_update': True, 'loss_scale_before': 1024, 'loss_scale_after': 1024}
             for i, (label, sigma) in enumerate(updates.CASES, 1)]
    boundaries = [{'step': s['step'], 'case_id': s['case_id'], 'input_gradient_domain': native.DOMAIN}
                  for s in steps]
    obs = {f'blocks.0.linear{i}': {'output_dtype': 'torch.float16', 'autocast_enabled': True}
           for i in range(4)}
    obs['final_layer._fp32_project'] = {'input_dtype': 'torch.float32', 'weight_compute_dtype': 'torch.float32',
                                     'output_dtype': 'torch.float32', 'internal_autocast_enabled': False}
    result = {'status': 'ok', 'precision': native.PRECISION, 'uuid': native.GPU_UUID,
              'swap': 20, 'loss_scale': 1024, 'replay_mode': 'replay', 'updates': steps,
              'signature': {'precision': native.PRECISION, 'loss_scale': 1024},
              'native_execution': {**execution, 'policy': native.POLICY.copy(), 'boundaries': boundaries,
                  'signature_semantics': 'canonical_source_replay_comparison_contract',
                  'observation_scope': 'first_case_initial_sigma_0.2_forward_only',
                  'dtype_snapshot': {'linears': {str(i): 'torch.float16' for i in range(377)},
                                     'guards': {str(i): True for i in range(29)}, 'adapter_dtype': 'torch.float32'},
                  'observations': obs}}
    return result, execution


def test_actual_native_label_and_provenance_required():
    result, execution = complete_result()
    native.validate_native_result(result, execution)
    for mutation in ('label', 'provenance', 'skip', 'scale', 'duplicate', 'missing', 'amp'):
        trial = copy.deepcopy(result)
        if mutation == 'label':
            trial['signature']['precision'] = 'fp16-islands'
        if mutation == 'provenance':
            trial['native_execution']['plan_sha256'] = 'b' * 64
        if mutation == 'skip':
            trial['updates'][0]['effective_update'] = False
        if mutation == 'scale':
            trial['updates'][0]['loss_scale_after'] = 512
        if mutation == 'duplicate':
            trial['native_execution']['boundaries'][1] = trial['native_execution']['boundaries'][0]
        if mutation == 'missing':
            trial['updates'].pop()
        if mutation == 'amp':
            trial['native_execution']['observations']['blocks.0.linear0']['autocast_enabled'] = False
        with pytest.raises(ValueError):
            native.validate_native_result(trial, execution)


def test_exact_twelve_case_inventory_and_unchanged_public_replay_contract():
    rows = [{'role': role, 'state_label': label, 'sigma': sigma}
            for role in acceptance.ROLES for label, sigma in updates.CASES]
    acceptance.validate_rows(rows)
    for bad in (rows[:-1], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match='twelve unique'):
            acceptance.validate_rows(bad)
    ref = {'precision': 'fp32-reference', 'loss_scale': 1, 'fp32_modules': [], 'timestep_policy': 'old'}
    candidate = {**ref, 'precision': native.PRECISION, 'loss_scale': 1024}
    assert contract(ref) == contract(candidate)
    assert contract(candidate) != contract({**candidate, 'timestep_policy': 'native'})


def test_frozen_plan_verification_rehashes_without_reloading_tensors(tmp_path, monkeypatch):
    import json
    plan = {'sources': {'source.py': 'a' * 64}, 'source_evidence': {'reference': {'tensor': 'b' * 64}}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    calls = []
    def frozen_preflight(args, *, validate_tensors=True):
        calls.append(validate_tensors)
        return copy.deepcopy(plan)
    monkeypatch.setattr(acceptance, 'preflight', frozen_preflight)
    args = SimpleNamespace(output_dir=tmp_path)
    assert acceptance.verify_plan(args) == plan
    assert calls == [False]
    changed = copy.deepcopy(plan)
    changed['source_evidence']['reference']['tensor'] = 'c' * 64
    monkeypatch.setattr(acceptance, 'preflight', lambda args, **kwargs: changed)
    with pytest.raises(ValueError, match='Frozen plan'):
        acceptance.verify_plan(args)


def test_independent_numpy_audit_metrics_and_boundary_encoding():
    import importlib.util
    from pathlib import Path
    import numpy as np
    script = Path(__file__).resolve().parents[1] / 'output/runs/anima-adaptive-precision-90hx-20261001-delivery-batch2-legacy-fp16-protocol/independent_audit_cpu.py'
    if not script.is_file():
        pytest.skip('Local preregistered audit artifact is absent')
    spec = importlib.util.spec_from_file_location('independent_native_audit', script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    class Tensors:
        def __init__(self, values):
            self.values = np.array(values, dtype=np.float32)
        def get_tensor(self, name):
            return self.values
    metric = audit.measure(Tensors([3, 4]), Tensors([0, 5]), ['x'])
    assert metric['relative_l2'] == pytest.approx(10 ** .5 / 5)
    assert metric['cosine'] == pytest.approx(.8)
    assert audit.measure(Tensors([0]), Tensors([1]), ['x'])['relative_l2'] is None
    assert audit.measure(Tensors([0]), Tensors([0]), ['x'])['cosine'] == 1
    tensor = torch.tensor([1.0001, -2.0002]).half()
    assert audit.describe('noisy', tensor.numpy()) == native.describe({'noisy': tensor})['noisy']
