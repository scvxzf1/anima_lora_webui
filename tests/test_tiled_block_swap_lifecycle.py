"""Tiled inference uses a single, exception-safe forward-only swap scope."""

from contextlib import contextmanager
import copy
import os
from types import SimpleNamespace

import pytest
import torch

from library.inference import generation
from library.inference.block_swap_lifecycle import inference_block_swap


class FakeNetwork:
    use_sigma_router = True
    use_fei_router = True
    cfg = SimpleNamespace(fei_sigma_low_div=8.0)

    def __init__(self):
        self.sigma = self.fei = None
        self.sigma_clears = self.fei_clears = 0
        self.enabled_calls = []

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_fei(self, fei):
        self.fei = fei

    def clear_sigma(self):
        self.sigma = None
        self.sigma_clears += 1

    def clear_fei(self):
        self.fei = None
        self.fei_clears += 1

    def set_enabled(self, enabled):
        self.enabled_calls.append(enabled)


class FakeAnima:
    patch_spatial = 1

    def __init__(self, *, swap=2, mode=False, fail_forward=None, fail_prepare=None):
        self.blocks_to_swap = swap
        self.offloader = SimpleNamespace(forward_only=mode)
        self._hydra_network = self._pgraft_network = FakeNetwork()
        self.prepare_count = 0
        self.switches = []
        self.forward_modes = []
        self.fail_forward = fail_forward
        self.fail_prepare = fail_prepare

    def prepare_block_swap_before_forward(self):
        self.prepare_count += 1
        if self.prepare_count == self.fail_prepare:
            raise RuntimeError("prepare failure")

    def switch_block_swap_for_inference(self):
        self.switches.append("inference")
        self.offloader.forward_only = True
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        self.switches.append("training")
        self.offloader.forward_only = False
        self.prepare_block_swap_before_forward()

    def __call__(self, latents, timesteps, embed, **kwargs):
        self.forward_modes.append(self.offloader.forward_only)
        if len(self.forward_modes) == self.fail_forward:
            raise RuntimeError("forward failure")
        return latents * 0.125 + embed.mean() * 0.01


def _args(**overrides):
    values = dict(
        image_size=(64, 64), infer_steps=2, flow_shift=1.0, sampler="euler",
        seed=7, guidance_scale=2.0, tile_size=4, tile_overlap=0,
        postfix_weight=None, lora_cutoff_step=0, runtime_dtype="fp32",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(value):
    return {"embed": [torch.full((1, 4, 8), value), None, None,
                      torch.ones(1, 4, dtype=torch.bool)]}


def _run(model, **overrides):
    return generation.generate_body_tiled(
        _args(**overrides), model, _context(1.0), _context(-1.0),
        torch.device("cpu"), seed=7,
    )


def _assert_clean(model):
    network = model._hydra_network
    assert network.sigma is None and network.fei is None
    assert network.sigma_clears == network.fei_clears == 1
    assert network.enabled_calls[-1] is True


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("guidance,forwards", [(2.0, 16), (1.0, 8)])
def test_tiled_reuses_forward_only_layout(mode, guidance, forwards):
    model = FakeAnima(mode=mode)
    _run(model, guidance_scale=guidance)
    assert model.forward_modes == [True] * forwards
    assert model.prepare_count == 2
    assert model.switches == ["inference", "inference" if mode else "training"]
    assert model.offloader.forward_only is mode
    _assert_clean(model)


def test_no_swap_does_not_read_offloader_or_switch():
    class NoSwap(FakeAnima):
        def __getattribute__(self, name):
            if name in {"offloader", "switch_block_swap_for_inference",
                        "switch_block_swap_for_training"}:
                raise AssertionError("no-swap scope touched swap state")
            return super().__getattribute__(name)

        def __call__(self, latents, timesteps, embed, **kwargs):
            return torch.zeros_like(latents)

    model = NoSwap(swap=0)
    _run(model)
    assert model.prepare_count == 0
    _assert_clean(model)


@pytest.mark.parametrize("forward", [1, 2, 5, 10])
@pytest.mark.parametrize("mode", [False, True])
def test_forward_failure_restores_mode_and_router_state(forward, mode):
    model = FakeAnima(mode=mode, fail_forward=forward)
    with pytest.raises(RuntimeError, match="forward failure"):
        _run(model)
    assert model.offloader.forward_only is mode
    assert model.prepare_count == 2
    assert model.forward_modes == [True] * forward
    _assert_clean(model)


def test_sampler_failure_restores_mode_and_router_state(monkeypatch):
    def fail(*args):
        raise RuntimeError("sampler failure")
    monkeypatch.setattr(generation.inference_utils, "step", fail)
    model = FakeAnima()
    with pytest.raises(RuntimeError, match="sampler failure"):
        _run(model)
    assert model.offloader.forward_only is False
    assert model.prepare_count == 2
    _assert_clean(model)


@pytest.mark.parametrize("prepare", [1, 2])
@pytest.mark.parametrize("mode", [False, True])
def test_transition_failure_still_attempts_restore_and_cleans_routers(prepare, mode):
    model = FakeAnima(mode=mode, fail_prepare=prepare)
    with pytest.raises(RuntimeError, match="prepare failure"):
        _run(model)
    assert model.offloader.forward_only is mode
    assert model.prepare_count == 2
    assert len(model.forward_modes) == (0 if prepare == 1 else 16)
    _assert_clean(model)


def test_restore_failure_does_not_replace_forward_failure():
    model = FakeAnima(fail_forward=2, fail_prepare=2)
    with pytest.raises(RuntimeError, match="forward failure") as caught:
        _run(model)
    assert "prepare failure" in caught.value.__notes__[0]
    _assert_clean(model)


@pytest.mark.parametrize("model", [
    SimpleNamespace(blocks_to_swap=2),
    SimpleNamespace(blocks_to_swap=2, offloader=SimpleNamespace(forward_only=1)),
    SimpleNamespace(blocks_to_swap=2, offloader=SimpleNamespace(forward_only=False)),
])
def test_invalid_swap_contract_fails_closed(model):
    with pytest.raises(RuntimeError, match="requires an offloader"):
        with inference_block_swap(model):
            pytest.fail("invalid contract entered inference")


def test_participating_master_rejection_is_propagated_and_restored():
    model = FakeAnima()
    def reject():
        model.switches.append("rejected")
        raise ValueError("Sparse CPU masters support training only")
    model.switch_block_swap_for_inference = reject
    with pytest.raises(ValueError, match="Sparse CPU masters"):
        _run(model)
    assert model.switches == ["rejected", "training"]
    assert model.offloader.forward_only is False
    _assert_clean(model)


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_output_matches_legacy_prepare_each_forward_exactly(monkeypatch, guidance):
    actual = _run(FakeAnima(), guidance_scale=guidance)
    original_forward = FakeAnima.__call__

    class LegacyAnima(FakeAnima):
        def __call__(self, *args, **kwargs):
            self.prepare_block_swap_before_forward()
            return original_forward(self, *args, **kwargs)

    @contextmanager
    def legacy_scope(model):
        yield

    monkeypatch.setattr(generation, "inference_block_swap", legacy_scope)
    legacy_model = LegacyAnima()
    expected = _run(legacy_model, guidance_scale=guidance)
    assert legacy_model.prepare_count == (16 if guidance != 1.0 else 8)
    assert torch.equal(actual, expected)


@pytest.fixture
def explicit_cuda_device():
    # Require a single UUID mask to avoid accidentally using a user's GPU 0.
    mask = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if os.getenv("ANIMA_RUN_BLOCK_SWAP_CUDA_TESTS") != "1":
        pytest.skip("set ANIMA_RUN_BLOCK_SWAP_CUDA_TESTS=1 with an idle GPU UUID")
    if not mask.startswith("GPU-") or "," in mask:
        pytest.skip("CUDA_VISIBLE_DEVICES must contain one explicit GPU UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        pytest.skip("one explicitly masked CUDA device is required")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 hardware is required")
    return torch.device("cuda:0")


class TinySwapBlock(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        self.linear.weight.requires_grad_(False)
        self.adapter = torch.nn.Parameter(
            (0.01 + (torch.arange(16) * 3 + index).remainder(11) / 1024).to(torch.bfloat16)
        )
        with torch.no_grad():
            # Channel mixing makes neighboring blocks noncommutative. Check
            # payload ownership separately: output equality alone can hide a
            # misplaced weight behind BF16 rounding.
            channels = torch.arange(16)
            weight = torch.diag(0.82 + (channels + index).remainder(5) / 100)
            weight[channels, (channels + 1 + index % 7).remainder(16)] += 0.04 + index / 2048
            weight[(3 * index) % 16, (5 * index + 3) % 16] += 0.08
            self.linear.weight.copy_(weight)

    def forward(self, x):
        return self.linear(x) + self.adapter


class TinySwapModel(torch.nn.Module):
    patch_spatial = 1

    def __init__(self, device, count, swap):
        super().__init__()
        from library.runtime.offloading import ModelOffloader
        self.blocks = torch.nn.ModuleList([TinySwapBlock(i) for i in range(count)])
        self.expected_weights = [block.linear.weight.detach().clone() for block in self.blocks]
        self.blocks.to(device)
        self.blocks_to_swap = swap
        self.offloader = ModelOffloader(
            self.blocks, swap, device, transfer_dtype="bf16", restore_mode="slab",
        )
        self.prepare_count = 0
        self.forward_modes = []
        self.fail_block = None

    def prepare_block_swap_before_forward(self):
        self.prepare_count += 1
        self.offloader.prepare_block_devices_before_forward(self.blocks, free_cache=False)

    def switch_block_swap_for_inference(self):
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    def forward(self, latents, timesteps=None, embed=None, **kwargs):
        self.forward_modes.append(self.offloader.forward_only)
        x = latents.movedim(1, -1)
        for i, block in enumerate(self.blocks):
            self.offloader.wait_for_block(i)
            assert torch.equal(block.linear.weight, self.expected_weights[i].to(latents.device))
            if i == self.fail_block:
                raise RuntimeError("middle block failure")
            x = block(x)
            self.offloader.submit_move_blocks(self.blocks, i)
        return x.movedim(-1, 1)

    def close(self):
        for i in list(self.offloader.futures):
            self.offloader._wait_blocks_move(i)
        self.offloader.thread_pool.shutdown(wait=True)
        for handle in self.offloader.remove_handles:
            handle.remove()


def _assert_canonical(model, device):
    resident = len(model.blocks) - model.blocks_to_swap
    assert not model.offloader.futures
    for i, block in enumerate(model.blocks):
        assert block.linear.weight.device.type == (device.type if i < resident else "cpu")
        assert block.adapter.device == device
        assert torch.equal(block.linear.weight.detach().cpu(), model.expected_weights[i])


def test_tiny_block_weights_distinguish_neighbor_owners():
    weights = [TinySwapBlock(i).linear.weight.detach().float() for i in range(30)]
    for left, right in zip(weights, weights[1:]):
        assert not torch.equal(left, right)
        assert not torch.equal(left @ right, right @ left)


@pytest.mark.hardware
@pytest.mark.parametrize("count,swap", [(6, 2), (30, 26)])
def test_real_cuda_tiled_swap_matches_all_gpu_and_recovers(explicit_cuda_device, count, swap):
    device = explicit_cuda_device
    model = TinySwapModel(device, count, swap)
    baseline = copy.deepcopy(model.blocks)
    try:
        args = _args(runtime_dtype="bf16")
        context, negative = _context(1.0), _context(-1.0)
        actual = generation.generate_body_tiled(args, model, context, negative, device, seed=7)
        assert model.forward_modes == [True] * 16
        assert model.prepare_count == 2
        assert model.offloader.forward_only is False
        _assert_canonical(model, device)

        class AllGpu:
            blocks_to_swap = 0
            patch_spatial = 1
            def __call__(self, latents, *args, **kwargs):
                x = latents.movedim(1, -1)
                for block in baseline:
                    x = block(x)
                return x.movedim(-1, 1)

        expected = generation.generate_body_tiled(args, AllGpu(), context, negative, device, seed=7)
        assert torch.equal(actual, expected)
        model.fail_block = 3
        with pytest.raises(RuntimeError, match="middle block failure"):
            with inference_block_swap(model), torch.no_grad():
                model(torch.ones(1, 16, 1, 4, 4, device=device, dtype=torch.bfloat16))
        assert model.offloader.forward_only is False
        _assert_canonical(model, device)
        model.fail_block = None
        # Backward scheduling supports this small conventional training layout.
        if count == 6:
            x = torch.ones(1, 16, 1, 4, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
            model(x).float().square().mean().backward()
            for block in model.blocks:
                assert block.linear.weight.grad is None
                assert block.adapter.grad is not None
                assert torch.isfinite(block.adapter.grad).all()
                assert torch.count_nonzero(block.adapter.grad) > 0
            model.switch_block_swap_for_training()
            _assert_canonical(model, device)
    finally:
        model.close()


@pytest.mark.hardware
@pytest.mark.parametrize("family_name", ["anima", "krea2", "z_image"])
@pytest.mark.parametrize("failure", [False, True])
def test_real_cuda_family_batch_continues_and_restores(
    explicit_cuda_device, monkeypatch, tmp_path, family_name, failure,
):
    from contextlib import nullcontext
    from library.anima import training as anima
    from library.models.krea2_raw import training_preview as krea
    from library.models.z_image import training_preview as zimage

    family = {"anima": anima, "krea2": krea, "z_image": zimage}[family_name]
    device = explicit_cuda_device
    model = TinySwapModel(device, 6, 2)
    baseline = copy.deepcopy(model.blocks)
    network = torch.nn.Linear(1, 1).eval()
    barriers, decoded, predictions = [], [], []
    accelerator = SimpleNamespace(
        device=device, process_index=0, num_processes=1, is_main_process=True,
        unwrap_model=lambda value: value, autocast=nullcontext,
        wait_for_everyone=lambda: barriers.append(1),
    )
    monkeypatch.setattr(anima, "_ensure_sample_compile_range", lambda *args: False)
    monkeypatch.setattr(anima, "decode_samples_for_live_preview", lambda *a, **k: decoded.append(1))

    def sample(*args):
        prompt = args[7 if family_name == "z_image" else 8]
        torch.rand(3)
        torch.rand(3, device=device)
        x = torch.full((1, 16, 1, prompt["height"], prompt["width"]),
                       0.5 + prompt["enum"], device=device, dtype=torch.bfloat16)
        expected = x.movedim(1, -1)
        for block in baseline:
            expected = block(expected)
        expected = expected.movedim(-1, 1)
        for _ in range(6):  # Three denoising steps, positive/negative predictions.
            actual = model(x)
            assert torch.equal(actual, expected)
            predictions.append(prompt["enum"])
        if failure and prompt["enum"] == 1:
            raise RuntimeError("second CUDA prompt failed")

    monkeypatch.setattr(family, "_sample_image_inference" if family_name == "anima" else "_sample_image", sample)
    args = SimpleNamespace(sample_at_first=True, sample_every_n_epochs=None,
        sample_every_n_steps=None, sample_prompts="unused", output_dir=str(tmp_path),
        disable_block_swap_for_eval=False)
    rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    try:
        def run():
            family.sample_images(accelerator, args, None, 0, model, None, None,
                None, None, {"cached": []}, network=network,
                sample_prompts_snapshot=[{"enum": 0, "height": 4, "width": 4},
                                         {"enum": 1, "height": 6, "width": 8}])
        if failure:
            with pytest.raises(RuntimeError, match="second CUDA prompt failed"):
                run()
            assert not barriers and not decoded
        else:
            run()
            assert len(barriers) == 2 and len(decoded) == 1
        assert len(predictions) == 12
        assert model.prepare_count == 2
        assert model.forward_modes == [True] * 12
        assert not model.offloader.forward_only and not network.training
        assert torch.equal(rng_cpu, torch.get_rng_state())
        assert torch.equal(rng_cuda, torch.cuda.get_rng_state())
        _assert_canonical(model, device)
    finally:
        model.close()
