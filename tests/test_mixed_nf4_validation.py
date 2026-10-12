"""Optional dependency and malformed nested-scale regression checks."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def test_plain_header_and_training_imports_do_not_require_bitsandbytes():
    script = '''
import importlib.abc
import sys
class RejectBnb(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'bitsandbytes' or fullname.startswith('bitsandbytes.'):
            raise ModuleNotFoundError('bitsandbytes intentionally unavailable')
sys.meta_path.insert(0, RejectBnb())
import library.quantization
import library.quantization.format
import library.quantization.training
import library.anima.mixed_nf4.format
import library.models.z_image.weights
import library.models.qwen_image_2_1.weights
assert 'bitsandbytes' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', script], check=True,
                   cwd=Path(__file__).resolve().parents[1], timeout=30)


@pytest.mark.parametrize('nested_dtype', ['float16', 'bfloat16', 'int64'])
def test_invalid_nested_decode_dtype_rejected_before_model_mutation(tmp_path, nested_dtype):
    from library.quantization import MANIFEST_KEY, load_mixed_nf4_into, save_mixed_nf4

    def construct(device):
        with torch.device(device):
            model = torch.nn.Module()
            model.blocks = torch.nn.ModuleList([torch.nn.Module()])
            model.blocks[0].mlp = torch.nn.Module()
            for name in ('gate', 'up', 'down'):
                setattr(model.blocks[0].mlp, name, torch.nn.Linear(64, 64, dtype=torch.bfloat16))
        return model

    path = tmp_path / 'mixed.safetensors'
    save_mixed_nf4(construct('cpu'), path, family='krea2_raw', source_sha256='f' * 64)
    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata()
        assert MANIFEST_KEY in metadata
    state = load_file(str(path))
    key = 'blocks.0.mlp.down.weight.quant_state.bitsandbytes__nf4'
    encoded = json.loads(bytes(state[key].tolist()))
    encoded['nested_dtype'] = nested_dtype
    state[key] = torch.tensor(list(json.dumps(encoded).encode()), dtype=torch.uint8)
    save_file(state, str(path), metadata=metadata)
    target = construct('meta')
    originals = tuple(target.parameters())
    with pytest.raises(ValueError, match='quant_state'):
        load_mixed_nf4_into(target, path, family='krea2_raw', dtype=torch.bfloat16, device='cpu')
    assert all(a is b and b.is_meta for a, b in zip(originals, target.parameters()))
    assert not hasattr(target, '_mixed_nf4_family')
