"""CPU-only Qwen preview residency and shared configuration contracts."""
import argparse
from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1 import preview_memory as memory
from library.models.qwen_image_2_1.block_swap import QwenImage21BlockSwapAdapter


def config(**changes):
    return {"model_family": "qwen_image_2_1", "blocks_to_swap": 4,
            "qwen_sample_swap_mode": "custom", "qwen_sample_blocks_to_swap": 2,
            "block_swap_transfer_dtype": "bf16", **changes}


class Offloader:
    forward_only = False
    supports_backward = True
    transfer_dtype = "bf16"

    def __init__(self, fail_calls=()):
        self.calls, self.prepares, self.fail_calls = [], [], fail_calls

    def reconfigure(self, blocks, count):
        self.calls.append(count)
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("migration failed")

    def set_forward_only(self, value):
        self.forward_only = value

    def prepare_block_devices_before_forward(self, blocks, free_cache=True):
        self.prepares.append(self.forward_only)


def model(count=4, fail_calls=()):
    dit = SimpleNamespace(blocks_to_swap=count, transformer_blocks=[object()] * 32,
                          parameters=lambda: [], offloader=Offloader(fail_calls))
    adapter = QwenImage21BlockSwapAdapter.__new__(QwenImage21BlockSwapAdapter)
    adapter.model, adapter.blocks, adapter.offloader = dit, dit.transformer_blocks, dit.offloader
    dit.switch_block_swap_for_inference = adapter.switch_block_swap_for_inference
    dit.switch_block_swap_for_training = adapter.switch_block_swap_for_training
    return dit


def test_inherit_uses_live_count_and_ignores_saved_custom_count():
    dit = model(7)
    with memory.preview_swap_scope(dit, config(qwen_sample_swap_mode="inherit",
                                             qwen_sample_blocks_to_swap=9)):
        assert dit.blocks_to_swap == 7 and dit.offloader.forward_only
    assert dit.blocks_to_swap == 7
    assert dit.offloader.calls == [] and dit.offloader.prepares == [True, False]


@pytest.mark.parametrize("old,new,prepares", [(4, 2, [True, False]), (4, 0, [False]),
                                               (0, 2, [True]), (4, 4, [True, False])])
def test_custom_residency_and_training_prepare_once(old, new, prepares):
    dit = model(old)
    with memory.preview_swap_scope(dit, config(blocks_to_swap=old, qwen_sample_blocks_to_swap=new)):
        assert dit.blocks_to_swap == new
        assert dit.offloader.forward_only == bool(new)
    assert dit.blocks_to_swap == old and not dit.offloader.forward_only
    assert dit.offloader.calls == ([new, old] if old != new else [])
    assert dit.offloader.prepares == prepares


@pytest.mark.parametrize("entry_failure", [False, True])
def test_entry_and_body_failures_restore_training(entry_failure):
    dit = model(fail_calls=(1,) if entry_failure else ())
    with pytest.raises(RuntimeError, match="migration failed" if entry_failure else "generation failed"):
        with memory.preview_swap_scope(dit, config()):
            raise RuntimeError("generation failed")
    assert dit.blocks_to_swap == 4 and not dit.offloader.forward_only
    assert dit.offloader.calls == [2, 4]
    assert dit.offloader.prepares == ([False] if entry_failure else [True, False])


def test_restore_failure_is_fatal_and_keeps_original_error():
    dit = model(fail_calls=(2,))
    with pytest.raises(RuntimeError, match="failed to restore training block swap") as caught:
        with memory.preview_swap_scope(dit, config()):
            raise ValueError("generation failed")
    assert str(caught.value.__cause__) == "migration failed"
    assert any("generation failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("attribute,value,match", [
    ("transfer_dtype", "int8", "lossless BF16"),
    ("forward_only", True, "forward_only=False"),
    ("supports_backward", False, "forward_only=False"),
])
def test_runtime_rejects_unsafe_offloader(attribute, value, match):
    dit = model()
    setattr(dit.offloader, attribute, value)
    with pytest.raises((ValueError, RuntimeError), match=match):
        with memory.preview_swap_scope(dit, config()):
            pytest.fail("unsafe scope entered")
    assert dit.offloader.calls == []


def test_custom_rejects_uncleared_trainable_gradients():
    dit = model()
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones(1)
    dit.parameters = lambda: [parameter]
    with pytest.raises(RuntimeError, match="cleared optimizer gradients"):
        with memory.preview_swap_scope(dit, config()):
            pass
    assert dit.offloader.calls == []


@pytest.mark.parametrize("count", [-1, 31, True, 1.5, "2"])
def test_count_contract_rejects_invalid_values_through_cli_and_compat(count):
    from library.training.cli_args import verify_training_args
    from library.training.compat_matrix import check_training_compat
    cfg = config(qwen_sample_blocks_to_swap=count)
    with pytest.raises(ValueError, match="qwen_sample_blocks_to_swap"):
        verify_training_args(SimpleNamespace(**cfg))
    assert any(item.code == "qwen_preview_memory_contract" and
               item.key == "qwen_sample_blocks_to_swap"
               for item in check_training_compat(cfg).errors)


@pytest.mark.parametrize("override", [dict(qwen_sample_swap_mode="custom"),
                                      dict(qwen_sample_blocks_to_swap=1)])
def test_non_qwen_overrides_rejected(override):
    cfg = {**memory.DEFAULTS, "model_family": "anima", **override}
    assert any("require qwen_image_2_1" in message for _, message in memory.configuration_errors(cfg))


@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image"])
@pytest.mark.parametrize("decode", ["auto", "tiled"])
def test_non_qwen_accepts_legacy_and_current_shared_decode_defaults(family, decode):
    cfg = {**memory.DEFAULTS, "model_family": family, "qwen_sample_vae_decode": decode}
    assert not memory.configuration_errors(cfg)


@pytest.mark.parametrize("transfer", ["int8", "fp8_e4m3"])
def test_custom_count_change_requires_lossless_transfer(transfer):
    assert any("lossless BF16" in message for _, message in
               memory.configuration_errors(config(block_swap_transfer_dtype=transfer)))
    assert not memory.configuration_errors(config(block_swap_transfer_dtype=transfer,
                                                  qwen_sample_blocks_to_swap=4))


def test_cli_and_schema_expose_all_preview_controls(monkeypatch):
    from library.config import schema
    parser = argparse.ArgumentParser()
    memory.add_arguments(parser)
    assert parser.parse_args([]).sample_vae_decode == "tiled"
    args = parser.parse_args(["--qwen_sample_swap_mode", "custom", "--qwen_sample_blocks_to_swap", "0",
                              "--qwen_sample_vae_decode", "cpu"])
    assert vars(args) == {"qwen_sample_swap_mode": "custom", "qwen_sample_blocks_to_swap": 0,
                          "sample_vae_decode": "cpu"}
    monkeypatch.setattr(schema, "CONFIG_SCHEMA", {})
    monkeypatch.setattr(schema, "ALIAS_MAP", {})
    schema.populate_schema(parser)
    for key, value in vars(args).items():
        assert schema.validate_entry(key, value, strict=True) == (key, value)
    with pytest.raises(schema.ConfigSchemaError):
        schema.validate_entry("qwen_sample_blocks_to_swap", True, strict=True)
    with pytest.raises(schema.ConfigSchemaError):
        schema.validate_entry("qwen_sample_vae_decode", "normal", strict=True)
    assert memory.needs_preview_offloader(config(blocks_to_swap=0))
    assert not memory.needs_preview_offloader(config(qwen_sample_blocks_to_swap=0))


@pytest.mark.parametrize("fail", [False, True])
def test_state_scope_restores_rng_consumed_by_mode_hooks(fail):
    class RandomMode(torch.nn.Linear):
        def train(self, mode=True):
            torch.rand(3)
            return super().train(mode)
    dit, network = RandomMode(1, 1), RandomMode(1, 1).eval()
    state = torch.get_rng_state().clone()
    def run():
        with memory.preview_state_scope(dit, network, torch.device("cpu")):
            assert not dit.training and not network.training
            if fail:
                raise RuntimeError("sample failed")
    if fail:
        with pytest.raises(RuntimeError, match="sample failed"):
            run()
    else:
        run()
    assert dit.training and not network.training
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("remote", [False, True])
def test_decode_error_is_coordinated_before_final_barrier(monkeypatch, remote):
    from library.models.qwen_image_2_1 import training_preview
    from library.anima import training
    calls, barriers = [], []
    monkeypatch.setattr(training_preview, "should_sample", lambda *args: True)
    monkeypatch.setattr(training_preview, "_sample_local", lambda *args: None)
    def failed(accelerator, local, processes):
        calls.append(local)
        return len(calls) == 2
    def decode(*args, **kwargs):
        if not remote:
            raise RuntimeError("VAE restore failed")
    monkeypatch.setattr(training_preview, "failed_on_any_process", failed)
    monkeypatch.setattr(training, "decode_samples_for_live_preview", decode)
    accelerator = SimpleNamespace(num_processes=2, process_index=int(remote),
        unwrap_model=lambda value: value, wait_for_everyone=lambda: barriers.append(1))
    with pytest.raises(RuntimeError, match="another process" if remote else "VAE restore failed"):
        training_preview.sample_images(accelerator, SimpleNamespace(), None, 1,
            object(), object(), None, None, None, sample_prompts_te_outputs={}, sample_prompts_snapshot=[])
    assert calls == [False, not remote] and barriers == [1]
