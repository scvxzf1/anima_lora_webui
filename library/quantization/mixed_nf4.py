"""CPU conversion and direct restoration of canonical mixed NF4 models."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import tempfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .format import MANIFEST_KEY, inspect_mixed_nf4
from .nf4 import STATE_KEYS, quant_state, restore_nf4_linear
from .policy import POLICY, model_paths


def _model_inventory(model, paths, *, conversion):
    params = dict(model.named_parameters())
    state = model.state_dict()
    if any('quant_state' in key or 'lora_' in key for key in state):
        raise ValueError('Expected ordinary model without packed weights or adapters')
    inventory = {}
    # Keep the manifest inventory canonical.  Stable key order makes the
    # metadata reproducible regardless of module registration order.
    for key in sorted(state):
        value = state[key]
        if conversion:
            if value.device.type != 'cpu':
                raise ValueError('Mixed NF4 conversion requires a CPU model')
            if value.is_floating_point() and (
                    (key in params and value.dtype != torch.bfloat16)
                    or (key not in params and value.dtype not in (torch.bfloat16, torch.float32))):
                raise ValueError(f'Invalid source floating tensor dtype: {key}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite source tensor: {key}')
        inventory[key] = {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    return inventory


def _validate_model(model, manifest, paths):
    expected = _model_inventory(model, paths, conversion=False)
    actual = manifest['source_tensors']
    if set(expected) != set(actual) or any(expected[k]['shape'] != actual[k]['shape'] for k in expected):
        raise ValueError('Mixed NF4 source inventory differs from model')
    if paths != manifest['modules']:
        raise ValueError('Mixed NF4 policy differs from model')
    params = dict(model.named_parameters())
    for key, entry in actual.items():
        if key in params and params[key].is_floating_point() and entry['dtype'] != 'bfloat16':
            raise ValueError(f'Mixed NF4 floating parameters must be BF16: {key}')
        if key not in params and expected[key]['dtype'] != entry['dtype']:
            raise ValueError(f'Mixed NF4 buffer dtype differs from model: {key}')


def load_mixed_nf4_into(model, path, *, family, dtype, device):
    """Load into a plain meta model after exact header and qstate validation."""
    if dtype != torch.bfloat16:
        raise ValueError('Mixed NF4 requires BF16 compute')
    path = Path(path).expanduser()
    manifest = inspect_mixed_nf4(path, family=family)
    if manifest is None:
        raise ValueError('Not a canonical mixed NF4 checkpoint')
    paths = model_paths(model, family, manifest['policy'])
    if any(v.device.type != 'meta' for v in model.state_dict().values()):
        raise ValueError('Mixed NF4 loading requires a meta model')
    _validate_model(model, manifest, paths)
    restored, retained = {}, {}
    targets = {p + '.weight' for p in paths}
    biases = {p + '.bias' for p in paths if model.get_submodule(p).bias is not None}
    with safe_open(str(path), framework='pt', device='cpu') as f:
        # Validate every small qstate before installing modules or reading retained weights.
        states = {}
        for p in paths:
            key = p + '.weight'
            qs = {key + '.' + s: f.get_tensor(key + '.' + s) for s in STATE_KEYS}
            states[p], _ = quant_state(qs, key, manifest['source_tensors'][key]['shape'])
        for p in paths:
            target = model.get_submodule(p)
            restored[p] = restore_nf4_linear(f.get_tensor(p + '.weight'), states[p],
                in_features=target.in_features, out_features=target.out_features,
                bias=f.get_tensor(p + '.bias') if target.bias is not None else None, device=device)
        for key in manifest['source_tensors']:
            if key not in targets | biases:
                retained[key] = f.get_tensor(key).to(device)
    for p, module in restored.items():
        if '.' in p:
            parent, attr = p.rsplit('.', 1)
            target_parent = model.get_submodule(parent)
        else:
            attr, target_parent = p, model
        setattr(target_parent, attr, module)
    missing, unexpected = model.load_state_dict(retained, strict=False, assign=True)
    if set(missing) != targets | biases or unexpected:
        raise ValueError(f'Mixed NF4 state mismatch: missing={missing}, unexpected={unexpected}')
    model._mixed_nf4_family = family
    model._mixed_nf4_policy = manifest['policy']
    return model


def save_mixed_nf4(model, path, *, family, source_sha256, policy=None):
    """Quantize CPU BF16 MLP weights without changing the source model."""
    import bitsandbytes
    from bitsandbytes.functional import quantize_4bit
    output = Path(path)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('source_sha256 must be 64 lowercase hexadecimal characters')
    policy = POLICY if policy is None else policy
    paths = model_paths(model, family, policy)
    inventory = _model_inventory(model, paths, conversion=True)
    manifest = {'version': 1, 'family': family, 'policy': policy, 'compute_dtype': 'bfloat16',
        'blocksize': 64, 'double_quant': True, 'modules': paths, 'source_tensors': inventory,
        'source_sha256': source_sha256, 'bitsandbytes_version': bitsandbytes.__version__}
    payload = dict(model.state_dict())  # References retained tensors; no full model clone.
    for p in paths:
        key = p + '.weight'
        packed, qs = quantize_4bit(payload[key], blocksize=64, compress_statistics=True,
                                    quant_type='nf4', quant_storage=torch.uint8)
        payload[key] = packed.contiguous()
        payload.update({key + '.' + k: v.contiguous() for k, v in qs.as_dict(packed=True).items()})
        quant_state(payload, key, inventory[key]['shape'])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + output.name + '.', dir=output.parent)
    os.close(fd)
    try:
        save_file(payload, temporary, metadata={MANIFEST_KEY: json.dumps(manifest, sort_keys=True)})
        inspect_mixed_nf4(temporary, family=family)
        os.link(temporary, output)  # Atomic publication with no clobber, including races.
        return {'quantized_weights': len(paths), 'bytes': output.stat().st_size,
                'output': str(output), 'source_sha256': source_sha256}
    finally:
        Path(temporary).unlink(missing_ok=True)
