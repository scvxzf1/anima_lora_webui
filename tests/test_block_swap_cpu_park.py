"""CPU ownership and failure contracts for training/preview parking."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.runtime.block_swap_masters import Params4bitBlockSwapCpuMaster
from library.runtime.block_swap_payload import BlockSwapManagedTensor
from library.runtime.offloading import ModelOffloader


class TestBlock(nn.Module):
    __test__ = False

    def forward(self, value):
        return value + .01 * (self.projection(value) + self.adapter(value))


def blocks(mixed=False):
    result = nn.ModuleList()
    for index in range(6):
        projection = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        if mixed and index % 2:
            pytest.importorskip("bitsandbytes")
            from bitsandbytes.functional import quantize_4bit
            from library.anima.mixed_nf4.runtime import restore_linear

            packed, state = quantize_4bit(projection.weight, compress_statistics=True, quant_type="nf4")
            projection = restore_linear(packed, state.as_dict(packed=True), in_features=64, out_features=64)
        block = TestBlock()
        block.projection = projection
        block.payload = BlockSwapManagedTensor(torch.randn(8, 8))
        block.adapter = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        result.append(block)
    return result


def close(offloader):
    offloader.thread_pool.shutdown(wait=True)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("scope", ["all", "participating"])
def test_park_preserves_masters_parameters_and_payload_protection(mixed, forward_only, scope):
    layers = blocks(mixed)
    offloader = ModelOffloader(layers, 3, torch.device("cpu"))
    offloader.master_scope = scope
    try:
        offloader.prepare_block_devices_before_forward(layers, free_cache=False)
        offloader.forward_only = forward_only
        parameters = list(layers.parameters())
        masters = [dict(entry) for entry in offloader._cpu_weight_masters]
        original = [block.projection.weight.detach().clone() for block in layers]
        offloader._swap_gpu_slab_cache[0] = torch.ones(8)
        offloader._swap_gpu_scale_slab_cache[0] = torch.ones(8)
        offloader._int8_resident_materialized.add(0)
        assert offloader.park_blocks_on_cpu(layers)
        assert not offloader.park_blocks_on_cpu(layers)
        assert not offloader._mixed_representation_slots.buffers
        assert not offloader._mixed_representation_slots.owners
        assert not offloader._swap_gpu_slab_cache
        assert not offloader._swap_gpu_scale_slab_cache
        assert not offloader._int8_resident_materialized
        for index, block in enumerate(layers):
            assert block.payload._block_swap_placement_active
            assert block.payload.weight.device.type == "cpu"
            for name, master in masters[index].items():
                assert offloader._cpu_weight_masters[index][name] is master
                data = master.params4bit.data if isinstance(master, Params4bitBlockSwapCpuMaster) else master
                assert block.get_submodule(name).weight.data_ptr() == data.data_ptr()
        with pytest.raises(RuntimeError, match="Restore CPU-parked"):
            offloader.prepare_block_devices_before_forward(layers)
        with pytest.raises(RuntimeError, match="Restore CPU-parked"):
            offloader.wait_for_block(0)
        with pytest.raises(RuntimeError, match="Restore CPU-parked"):
            offloader.reconfigure(layers, 2)
        assert offloader.restore_parked_blocks(layers)
        assert not offloader.restore_parked_blocks(layers)
        assert offloader.forward_only == forward_only
        assert offloader.blocks_to_swap == 3
        assert all(actual is expected for actual, expected in zip(layers.parameters(), parameters))
        for index, block in enumerate(layers):
            torch.testing.assert_close(block.projection.weight, original[index], rtol=0, atol=0)
        # Repeating the entire lifecycle must not leave old mixed owners behind.
        assert offloader.park_blocks_on_cpu(layers)
        assert offloader.restore_parked_blocks(layers)
    finally:
        close(offloader)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA park/restore residency")
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_cuda_park_releases_carriers_and_preserves_next_backward(mixed, checkpointed):
    import copy
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(7)
    layers = blocks(mixed)
    baseline = copy.deepcopy(layers).cuda()
    offloader = ModelOffloader(layers, 3, torch.device("cuda"))
    inputs = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    def run(layers, swapped):
        for block in layers:
            block.adapter.zero_grad(set_to_none=True)
        x = inputs.clone().requires_grad_()
        for index, block in enumerate(layers):
            if swapped:
                offloader.wait_for_block(index)
            x = checkpoint(block, x, use_reentrant=False) if checkpointed else block(x)
            if swapped:
                offloader.submit_move_blocks(layers, index)
        x.float().square().mean().backward()
        return x.detach(), [block.adapter.weight.grad.clone() for block in layers]
    try:
        offloader.prepare_block_devices_before_forward(layers, free_cache=False)
        expected = run(baseline, False)
        torch.testing.assert_close(run(layers, True), expected, rtol=0, atol=0)
        parameters = list(layers.parameters())
        offloader.park_blocks_on_cpu(layers)
        assert all(block.payload.weight.device.type == "cpu" for block in layers)
        assert all(block.projection.weight.device.type == "cpu" for block in layers)
        for block in layers:
            if hasattr(block.projection.weight, "quant_state"):
                assert block.projection.weight.quant_state.absmax.device.type == "cpu"
        assert not offloader._mixed_representation_slots.buffers
        offloader.restore_parked_blocks(layers)
        assert all(block.payload.weight.device.type == "cuda" for block in layers[:3])
        assert all(block.payload.weight.device.type == "cpu" for block in layers[3:])
        assert all(actual is expected for actual, expected in zip(layers.parameters(), parameters))
        torch.testing.assert_close(run(layers, True), expected, rtol=0, atol=0)
    finally:
        for index in list(offloader.futures):
            offloader._wait_blocks_move(index, phase="test_cleanup")
        close(offloader)


@pytest.mark.parametrize("transfer_dtype", ["bf16", "fp8_e4m3", "int8"])
@pytest.mark.parametrize("count", [0, 3])
def test_restore_execution_dtype_for_compressed_cpu_masters(transfer_dtype, count):
    layers = blocks()
    offloader = ModelOffloader(layers, count, torch.device("cpu"), transfer_dtype=transfer_dtype)
    try:
        if count:
            offloader.prepare_block_devices_before_forward(layers, free_cache=False)
        parameters = list(layers.parameters())
        offloader.park_blocks_on_cpu(layers)
        assert offloader.restore_parked_blocks(layers)
        for block in layers[:len(layers) - count]:
            assert block.projection.weight.dtype == torch.bfloat16
            assert block.payload.weight.dtype == torch.float32
        assert all(actual is expected for actual, expected in zip(layers.parameters(), parameters))
    finally:
        close(offloader)


def test_failed_restore_reparks_and_propagates_with_explicit_retry(monkeypatch):
    layers = blocks(True)
    offloader = ModelOffloader(layers, 3, torch.device("cpu"))
    try:
        offloader.prepare_block_devices_before_forward(layers, free_cache=False)
        offloader.park_blocks_on_cpu(layers)
        original = offloader.prepare_block_devices_before_forward
        def failed(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("injected canonical prepare failure")
        monkeypatch.setattr(offloader, "prepare_block_devices_before_forward", failed)
        with pytest.raises(RuntimeError, match="restoration failed") as caught:
            offloader.restore_parked_blocks(layers)
        assert isinstance(caught.value.__cause__, torch.cuda.OutOfMemoryError)
        assert offloader._cpu_park_state is not None
        assert not offloader._cpu_park_state["restoring"]
        assert not offloader._mixed_representation_slots.buffers
        monkeypatch.setattr(offloader, "prepare_block_devices_before_forward", original)
        assert offloader.restore_parked_blocks(layers)
    finally:
        close(offloader)


def test_drain_attempts_every_future_before_any_parking(monkeypatch):
    layers = blocks()
    offloader = ModelOffloader(layers, 3, torch.device("cpu"))
    waited = []
    def wait(index, **kwargs):
        waited.append(index)
        offloader.futures.pop(index)
        if index == 0:
            raise RuntimeError("failed transfer")
    offloader.futures = {0: None, 1: None}
    monkeypatch.setattr(offloader, "_wait_blocks_move", wait)
    try:
        with pytest.raises(RuntimeError, match="transfer failed"):
            offloader.park_blocks_on_cpu(layers)
        assert waited == [0, 1]
        assert getattr(offloader, "_cpu_park_state", None) is None
    finally:
        close(offloader)


@pytest.mark.parametrize("failure", ["decode", "restore", "vae_restore"])
@pytest.mark.parametrize("block_attribute", ["blocks", "layers", "transformer_blocks"])
def test_live_decode_restores_on_oom_and_never_swallows_restore_failure(tmp_path, monkeypatch, failure, block_attribute):
    from library.anima import training
    from library.models.qwen_image_2_1.preview_decode_policy import PreviewDecodeRestoreError

    latents = tmp_path / "sample" / "latents"
    latents.mkdir(parents=True)
    (latents / "sample.pt").touch()
    layers = blocks()
    offloader = ModelOffloader(layers, 3, torch.device("cpu"))
    dit = nn.Module()
    setattr(dit, block_attribute, layers)
    dit.blocks_to_swap = 3
    dit.offloader = offloader
    network = nn.Linear(2, 2)
    restored = []
    original = offloader.restore_parked_blocks
    def restore(layers):
        restored.append(True)
        if failure == "restore":
            raise RuntimeError("restore must propagate")
        return original(layers)
    def decode(*args):
        assert offloader._cpu_park_state is not None
        if failure == "vae_restore":
            raise PreviewDecodeRestoreError("VAE state restore failed")
        raise torch.cuda.OutOfMemoryError("VAE OOM")
    monkeypatch.setattr(training, "decode_pending_samples", decode)
    monkeypatch.setattr(offloader, "restore_parked_blocks", restore)
    try:
        args = SimpleNamespace(output_dir=str(tmp_path))
        accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cpu"))
        if failure == "vae_restore":
            with pytest.raises(PreviewDecodeRestoreError, match="VAE state restore failed"):
                training.decode_samples_for_live_preview(accelerator, args, object(), dit=dit, network=network)
            assert offloader._cpu_park_state is None
        elif failure == "restore":
            with pytest.raises(RuntimeError, match="restore must propagate"):
                training.decode_samples_for_live_preview(accelerator, args, object(), dit=dit, network=network)
        else:
            training.decode_samples_for_live_preview(accelerator, args, object(), dit=dit, network=network)
            assert offloader._cpu_park_state is None
        assert restored == [True]
    finally:
        close(offloader)


def test_block_resolution_rejects_ambiguous_or_wrong_collection():
    from library.runtime.block_swap_lifecycle import model_swap_blocks

    layers = blocks()
    dit = SimpleNamespace(offloader=SimpleNamespace(num_blocks=len(layers)), blocks=layers)
    assert model_swap_blocks(dit) is layers
    dit.layers = layers
    with pytest.raises(ValueError, match="one model block collection"):
        model_swap_blocks(dit)
    del dit.layers
    dit.offloader.num_blocks += 1
    with pytest.raises(ValueError, match="one model block collection"):
        model_swap_blocks(dit)


def test_deferred_decode_uses_the_same_park_restore_boundary(tmp_path, monkeypatch):
    from library.anima import training
    from library.training.train_bootstrap import decode_deferred_samples_safely

    staging = tmp_path / "sample" / "latents"
    staging.mkdir(parents=True)
    (staging / "sample.pt").touch()
    layers = blocks()
    offloader = ModelOffloader(layers, 3, torch.device("cpu"))
    dit = nn.Module()
    dit.blocks, dit.offloader, dit.blocks_to_swap = layers, offloader, 3
    network = nn.Linear(2, 2)
    observed = []
    def decode(*args):
        assert offloader._cpu_park_state is not None
        observed.append("parked_decode")
    monkeypatch.setattr(training, "decode_pending_samples", decode)
    accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cpu"),
                                  unwrap_model=lambda model: model)
    args = SimpleNamespace(output_dir=str(tmp_path), sample_prompts="fixed", sample_every_n_steps=4)
    try:
        decode_deferred_samples_safely(
            accelerator, args, SimpleNamespace(unet=dit, network=network), object(),
            optimizer_eval_fn=lambda: observed.append("optimizer_eval"))
        assert observed == ["optimizer_eval", "parked_decode"]
        assert offloader._cpu_park_state is None
    finally:
        close(offloader)
