"""CPU header and explicit acknowledgement regressions; no numerical claims."""
import asyncio
import json
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from library.quantization import MANIFEST_KEY, inspect_mixed_nf4
from library.quantization.policy import POLICY, policy_paths
from library.quantization.training import validate_mixed_nf4_training
from web.services.config.preflight_nf4 import check_mixed_nf4
from web.services.training.risk_acknowledgement import FLAG, acknowledge_request_risks, pending_preflight


def header_checkpoint(path, family='krea2_raw', mutate=None):
    paths = policy_paths(family, 1)
    inventory = {p + '.weight': {'shape': [64, 64], 'dtype': 'bfloat16'} for p in sorted(paths)}
    manifest = dict(version=1, family=family, policy=POLICY, compute_dtype='bfloat16',
                    blocksize=64, double_quant=True, modules=paths, source_tensors=inventory,
                    source_sha256='a' * 64, bitsandbytes_version='test')
    if mutate:
        mutate(manifest)
    header = {'__metadata__': {MANIFEST_KEY: json.dumps(manifest)}}
    offset = 0
    for p in paths:
        specs = {'': ([2048, 1], 'U8', 2048), '.absmax': ([64], 'U8', 64),
                 '.quant_map': ([16], 'F32', 64), '.nested_absmax': ([1], 'F32', 4),
                 '.nested_quant_map': ([256], 'F32', 1024),
                 '.quant_state.bitsandbytes__nf4': ([1], 'U8', 1)}
        for suffix, (shape, dtype, size) in specs.items():
            header[p + '.weight' + suffix] = dict(shape=shape, dtype=dtype, data_offsets=[offset, offset + size])
            offset += size
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(offset))
    return path


def collect(cfg, path):
    checks = []
    def add(level, key, message, path=None, **flags):
        checks.append(dict(level=level, key=key, message=message, **flags))
    check_mixed_nf4(cfg, path, add)
    return checks


def test_actual_manifest_risk_and_eager(tmp_path):
    path = header_checkpoint(tmp_path / 'base.safetensors')
    cfg = dict(model_family='krea2_raw', base_compute='bf16', gradient_checkpointing=True, torch_compile=True)
    checks = collect(cfg, path)
    assert len(checks) == 1 and checks[0]['overridable'] is True
    assert checks[0]['risk_code'] == 'mixed_nf4_compile'
    cfg['torch_compile'] = False
    assert collect(cfg, path) == []
    plain = tmp_path / 'plain.safetensors'
    raw = b'{"__metadata__":{"krea2_nf4_format":"krea2_raw_nf4"}}'
    plain.write_bytes(struct.pack('<Q', len(raw)) + raw)
    assert inspect_mixed_nf4(plain) is None


@pytest.mark.parametrize('change', ['family', 'manifest', 'precision', 'routing', 'fused'])
def test_hard_errors_cannot_be_overridden(tmp_path, change):
    path = header_checkpoint(tmp_path / 'mixed.safetensors', 'qwen_image_2_1',
                             (lambda m: m.update(compute_dtype='float16')) if change == 'manifest' else None)
    cfg = dict(model_family='qwen_image_2_1', gradient_checkpointing=True, torch_compile=True)
    if change == 'family':
        cfg['model_family'] = 'krea2_raw'
    if change == 'precision':
        cfg['mixed_precision'] = 'fp16'
    if change == 'routing':
        cfg['router_source'] = 'fei'
    if change == 'fused':
        cfg['qwen_fused_projections'] = 'on'
    checks = collect(cfg, path)
    result = {'ok': False, 'errors': checks}
    remaining, _ = acknowledge_request_risks({'ignore_risks': True, 'confirmed': True}, result, [])
    assert not remaining['ok'] and remaining['errors']


def test_runtime_acknowledgement_only_compile(caplog):
    model = SimpleNamespace(_mixed_nf4_family='qwen_image_2_1')
    args = dict(gradient_checkpointing=True, torch_compile=True)
    with pytest.raises(ValueError, match='torch_compile disabled'):
        validate_mixed_nf4_training(model, args)
    args['allow_unverified_nf4_compile'] = True
    validate_mixed_nf4_training(model, args)
    assert '梯度差异' in caplog.text
    args['mixed_precision'] = 'fp16'
    with pytest.raises(ValueError, match='mixed_precision=bf16'):
        validate_mixed_nf4_training(model, args)
    model._mixed_nf4_family = 'z_image'
    with pytest.raises(ValueError, match='torch_compile disabled'):
        validate_mixed_nf4_training(model, args)


def test_header_and_rules_remain_torchless(tmp_path):
    path = header_checkpoint(tmp_path / 'mixed.safetensors')
    script = f"from library.quantization import inspect_mixed_nf4; from library.quantization.training import mixed_nf4_training_errors; import sys; inspect_mixed_nf4({str(path)!r}); mixed_nf4_training_errors('krea2_raw', {{'gradient_checkpointing': True}}); assert not any(x in sys.modules for x in ('torch', 'bitsandbytes'))"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('route', ['handle_start', 'handle_queue_start', 'handle_queue_batch_start'])
@pytest.mark.parametrize('accepted', [True, False, 'false'])
def test_routes_require_explicit_confirmed_risk(monkeypatch, route, accepted):
    from web.routes import training as routes
    risk = dict(level='error', key='mixed_nf4_compile', overridable=True, risk_code='mixed_nf4_compile', message='risk')
    monkeypatch.setattr(routes, 'preflight_training_config', lambda *a, **k: dict(ok=False, errors=[risk]))
    monkeypatch.setattr(routes, 'is_web_runtime_config', lambda value: True)
    calls = []
    class Service:
        async def start(self, *args, **kwargs):
            calls.append(args[2])

        async def enqueue_training(self, *args, **kwargs):
            calls.append(kwargs['extra_args'])
            return {'ok': True}

        async def enqueue_training_batch(self, items, **kwargs):
            calls.extend(item['extra_args'] for item in items)
            return {'ok': True}
    item = dict(variant='lora', config_file='runtime.toml', confirmed=True, ignore_risks=accepted, extra_args=[FLAG])
    payload = {'items': [item]} if route == 'handle_queue_batch_start' else item
    class Request:
        app = {'training_service': Service()}
        async def json(self): return payload
    response = asyncio.run(getattr(routes, route)(Request()))
    assert response.status == (200 if accepted is True else 400)
    assert calls == ([[FLAG]] if accepted is True else [])


def test_pending_and_client_flags_do_not_bypass_hard_errors():
    risk = dict(overridable=True, risk_code='mixed_nf4_compile')
    pf = dict(ok=False, errors=[risk])
    assert pending_preflight(pf, [FLAG])['ok']
    assert not pending_preflight(dict(ok=False, errors=[risk, {'key': 'schema'}]), [FLAG])['ok']
    remaining, args = acknowledge_request_risks({'confirmed': True}, pf, [FLAG])
    assert not remaining['ok'] and args == []


def test_preprocess_filters_flag_and_pending_preserves_it(tmp_path, monkeypatch):
    from aiohttp import web
    from web.services import training_service
    from web.services.training_service import TrainingService
    runtime = dict(runtime_config_file='runtime.toml', output_dir=str(tmp_path),
                   sample_dir=str(tmp_path), sample_config={}, data_dirs={}, run_dir=str(tmp_path))
    monkeypatch.setattr(training_service, '_prepare_web_runtime_config', lambda *a, **k: runtime)
    service = TrainingService(web.Application())
    captured = {}
    async def launch(cmd, env, **kwargs): captured['cmd'] = cmd
    service._launch_job = launch
    asyncio.run(service.start_preprocess('lora', 'default', extra_args=[FLAG], train_after=True))
    assert FLAG not in captured['cmd']
    assert service._pending_train_after_preprocess['extra_args'] == [FLAG]
    risk = dict(overridable=True, risk_code='mixed_nf4_compile', message='risk')
    monkeypatch.setattr(training_service, 'preflight_training_config', lambda *a, **k: dict(ok=False, errors=[risk]))
    async def start(*args, **kwargs): captured['train_args'] = args[2]
    service.start = start
    asyncio.run(service._start_pending_training(service._pending_train_after_preprocess))
    assert captured['train_args'] == [FLAG]


def test_corrupt_header_is_hard(tmp_path):
    path = tmp_path / 'broken.safetensors'
    path.write_bytes(b'bad')
    checks = collect(dict(model_family='qwen_image_2_1'), path)
    assert len(checks) == 1 and not checks[0].get('overridable')


def test_extensionless_krea_weight_still_reports_compile_risk(tmp_path):
    path = header_checkpoint(tmp_path / 'base-without-extension')
    cfg = dict(model_family='krea2_raw', gradient_checkpointing=True, torch_compile=True)
    checks = collect(cfg, path)
    assert len(checks) == 1 and checks[0]['risk_code'] == 'mixed_nf4_compile'


def test_existing_anima_format_does_not_use_three_family_contract(tmp_path):
    # Existing Anima weights have their own manifest and packed tensor namespace.
    header = {
        '__metadata__': {'anima.mixed_nf4.manifest': '{}'},
        'net.blocks.0.mlp.layer1.weight.absmax': {
            'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1],
        },
    }
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    path = tmp_path / 'anima.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0')
    assert collect(dict(model_family='anima', torch_compile=True), path) == []


def test_web_preflight_detects_nf4_despite_bf16_compute_config(tmp_path, monkeypatch):
    from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
    from web.services import config_service

    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, [
        'model_family = "qwen_image_2_1"',
        'base_compute = "bf16"',
        'mixed_precision = "bf16"',
        'gradient_checkpointing = true',
        'torch_compile = true',
        'allow_unverified_nf4_compile = true',
    ])
    header_checkpoint(tmp_path / 'models' / 'anima.safetensors', 'qwen_image_2_1')
    result = config_service.preflight_training_config(
        'lora', 'default', 'imported', config_file='configs/imported/selected.toml',
    )
    assert not result['ok'] and result['can_override']
    assert len(result['errors']) == 1
    assert result['errors'][0]['risk_code'] == 'mixed_nf4_compile'
