"""Fail-closed manifest and bitsandbytes state validation."""
from __future__ import annotations
import hashlib
import json
import re
from safetensors import safe_open
from library.quantization.constants import STATE_KEYS

MANIFEST_KEY = 'anima.mixed_nf4.manifest'
POLICY = 'mlp-nf4-except-layer1-blocks14-20'
PREFIXES = ('net.', 'model.diffusion_model.', '')


def quantized_paths() -> list[str]:
    return [f'blocks.{i}.mlp.layer{layer}' for i in range(28) for layer in (1, 2)
            if layer != 1 or not 14 <= i <= 20]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_mixed_nf4(path) -> dict | None:
    with safe_open(str(path), framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
        marked = any(k.startswith('anima.mixed_nf4') for k in metadata)
        packed = any('.weight.quant_state.' in k or '.weight.absmax' in k for k in f.keys())
        if not marked and not packed:
            return None
        raw = metadata.get(MANIFEST_KEY)
        if not isinstance(raw, str) or len(raw) > 200000:
            raise ValueError('Missing or oversized Anima mixed NF4 manifest')
        try:
            m = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid mixed NF4 manifest JSON') from exc
        required = {'version', 'family', 'policy', 'prefix', 'modules', 'source_sha256',
                    'bitsandbytes_version', 'tensor_shapes', 'compute_dtype', 'blocksize', 'double_quant'}
        if not isinstance(m, dict) or set(m) != required:
            raise ValueError('Invalid mixed NF4 manifest fields')
        if (type(m['version']) is not int or m['version'] != 1 or m['family'] != 'anima'
                or m['policy'] != POLICY or m['prefix'] not in PREFIXES
                or m['modules'] != quantized_paths() or m['compute_dtype'] != 'bfloat16'
                or type(m['blocksize']) is not int or m['blocksize'] != 64 or m['double_quant'] is not True
                or not isinstance(m['source_sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}', m['source_sha256'])
                or not isinstance(m['bitsandbytes_version'], str)):
            raise ValueError('Unsupported mixed NF4 version, architecture or policy')
        shapes = m['tensor_shapes']
        if not isinstance(shapes, dict) or len(shapes) != 685:
            raise ValueError('Invalid mixed NF4 source tensor inventory')
        if any(not isinstance(k, str) or not isinstance(s, list) or not s
               or any(type(d) is not int or d <= 0 for d in s) for k, s in shapes.items()):
            raise ValueError('Invalid mixed NF4 tensor shapes')
        expected = set(shapes)
        for p in m['modules']:
            w = m['prefix'] + p + '.weight'
            shape = [8192, 2048] if p.endswith('layer1') else [2048, 8192]
            if shapes.get(w) != shape:
                raise ValueError(f'Invalid mixed NF4 module shape: {p}')
            expected.update(w + '.' + k for k in STATE_KEYS)
        if set(f.keys()) != expected:
            raise ValueError('Mixed NF4 tensor/qstate inventory mismatch')
        for key, shape in shapes.items():
            sl = f.get_slice(key)
            if key[:-7].removeprefix(m['prefix']) in m['modules'] and key.endswith('.weight'):
                if sl.get_dtype() != 'U8' or sl.get_shape() != [8192 * 2048 // 2, 1]:
                    raise ValueError(f'Invalid packed NF4 weight: {key}')
            elif sl.get_dtype() != 'BF16' or sl.get_shape() != shape:
                raise ValueError(f'Invalid retained BF16 tensor: {key}')
        from .conversion import validate_model_shapes
        validate_model_shapes(shapes)
        return m


def quant_state(state: dict, weight_key: str, shape) -> tuple[dict, object]:
    """Preserve the Anima API while loading bitsandbytes only for packed weights."""
    from library.quantization.nf4 import quant_state as validate
    return validate(state, weight_key, shape)
