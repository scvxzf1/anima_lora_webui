"""Isolated benchmark inherits runtime topology without relaxing historical A/B."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.qwen_image_2_1 import sampling_prepare_ab as cli
from scripts.qwen_image_2_1 import sampling_benchmark_runtime as runtime
from library.models.qwen_image_2_1.preview_conditions import condition_key


def fixture_args(tmp_path, count, inherited):
    asset = tmp_path / 'asset-step400.bin'
    asset.write_bytes(b'fixture')
    config = dict(model_family='qwen_image_2_1', network_module='networks.lora_anima',
        network_dim=16, network_alpha=16, blocks_to_swap=count, mixed_precision='bf16',
        base_compute='bf16', attn_mode='torch', gradient_checkpointing=True, torch_compile=True,
        block_swap_transfer_dtype='bf16', block_swap_restore_mode='slab')
    for key in ('pretrained_model_name_or_path', 'qwen3', 'vae', 'sample_prompts'):
        config[key] = str(asset)
    path = tmp_path / 'config.toml'
    path.write_text('\n'.join(f'{key} = {json.dumps(value)}' for key, value in config.items()))
    return SimpleNamespace(config=path, checkpoint=asset, baseline_sampling=asset,
                           allow_configured_blocks=inherited)


@pytest.mark.parametrize('count', [0, 16, 24, 30])
def test_new_benchmark_accepts_configured_swap_while_historical_ab_remains_fixed(tmp_path, monkeypatch, count):
    monkeypatch.setattr(cli, 'digest', lambda path: 'test-only-hash')
    args = fixture_args(tmp_path, count, True)
    recorder = SimpleNamespace(summary={}, event=lambda *a, **kw: None)
    assert cli.load_inputs(args, recorder)['blocks_to_swap'] == count
    args.allow_configured_blocks = False
    if count == 24:
        assert cli.load_inputs(args, recorder)['blocks_to_swap'] == 24
    else:
        with pytest.raises(ValueError, match='blocks_to_swap=24'):
            cli.load_inputs(args, recorder)


@pytest.mark.parametrize('training,sample', [(0, 0), (0, 2), (2, 0), (2, 3), (2, 2)])
def test_benchmark_policy_restores_topology_and_counts(tmp_path, monkeypatch, training, sample):
    counter = SimpleNamespace(counts={}, phase='idle')
    counter.reset = lambda: counter.counts.clear()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks_to_swap = training
            self.transformer_blocks = []
            self.offloader = SimpleNamespace(forward_only=False, supports_backward=True, transfer_dtype='bf16',
                reconfigure=lambda *args: None, set_forward_only=lambda mode: setattr(self.offloader, 'forward_only', mode))
        def prepare(self):
            if self.blocks_to_swap:
                counter.counts[counter.phase] = counter.counts.get(counter.phase, 0) + 1
        def switch_block_swap_for_inference(self):
            self.offloader.forward_only = bool(self.blocks_to_swap)
            self.prepare()
        def switch_block_swap_for_training(self):
            self.offloader.forward_only = False
            self.prepare()
    dit, network = Model(), torch.nn.Linear(1, 1).eval()
    monkeypatch.setattr(runtime, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(runtime, 'sync', lambda: None)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 0)
    cfg = dict(model_family='qwen_image_2_1', blocks_to_swap=training,
               qwen_sample_swap_mode='custom', qwen_sample_blocks_to_swap=sample)
    args = SimpleNamespace(preview_memory_config=cfg, output=tmp_path, checkpoint='step400', runtime_seed=42)
    prompt, events = dict(prompt='test', enum=0), []
    def generate(*a, **kw):
        assert dit.blocks_to_swap == sample and not dit.training and not network.training
        assert kw['use_kv_cache'] is False
        return torch.zeros(1, 64, 2, 2)
    rows = runtime.sample_variant(dit, network, generate, [prompt], {condition_key(prompt): ()}, 2,
        'B', args, SimpleNamespace(event=lambda name, **row: events.append(row)), counter)
    assert dit.blocks_to_swap == training and dit.training and not network.training
    assert not dit.offloader.forward_only
    assert len(rows) == 1 and events[-1]['effective_blocks_to_swap'] == sample
